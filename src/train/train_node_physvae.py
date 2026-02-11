import numpy as np

import torch
import shutil
import os
import logging
import time
import yaml
from omegaconf import OmegaConf, DictConfig

import wandb

from src.evaluate.evaluate import (
    evaluate_model,
    EVALUATING_METHODS,
    evaluate_exp as full_evaluation,
)
from src.metrics.loss import kl_gaussians
from src.metrics import mse_loss
from src.utils.wandb import wandb_util
from src.utils.hash import get_exp_dir_path


def train_physvae(
    node_model,
    enc_model,
    train_loader,
    optimizer,
    exp_config,
    wandb_config={},
    val_loader=None,
    test_loader=None,
    num_epochs=10,
    val_step=200,
    early_stopping=True,
    logger=logging.getLogger(__name__),
):

    out_dir = get_exp_dir_path(exp_config, exp_config["group_exp_name"])
    os.makedirs(
        out_dir,
        exist_ok=True,
    )
    logger.info(f"Experiment directory created at: {out_dir}")

    # Set SEED
    exp_config["seed"] = exp_config.get("seed", 0)
    torch.manual_seed(exp_config["seed"])
    np.random.seed(exp_config["seed"])
    torch.cuda.manual_seed_all(exp_config["seed"])

    exp_config = wandb_util.init_wandb(
        exp_config=exp_config, wandb_config=wandb_config, logger=logger
    )
    # Watch model parameters if wandb is enabled
    if (
        wandb.run is not None
        and len(wandb_config) > 0
        and wandb_config.get("wandb_model_watch", False)
    ):
        wandb_util.watch(
            {
                "node model": node_model,
                "encoder": enc_model,
            },
            log=wandb_config.get("wandb_watch_log", "all"),
            log_freq=wandb_config.get("log_freq_batches", 20),
        )
        logger.info("Wandb watch initialized")

    train_losses = {
        "epoch": [],
        "node_loss": [],
        "std_node_loss": [],
        "enc_kl": [],
        "rda_1": [],
        "rda_2": [],
        "ppc": [],
        "N_ELBO": [],
        "total_loss": [],
    }
    val_metrics = {
        "epoch": [],
        "rec_score": [],
        "rec_std": [],
        "params_score": [],
        "params_std": [],
        "forecast_score": [],
    }
    chkp_score = {
        "score": None,
        "epoch": None,
        "model": None,
        "final_model": None,
        "optimizer": None,
    }

    # model hyperparameters
    ALPHA, BETA, GAMMA = (
        exp_config["optimization"].get("alpha", 0.0),
        exp_config["optimization"].get("beta", 0.0),
        exp_config["optimization"].get("gamma", 0.0),
    )

    MAX_PATIENCE_COUNTER = exp_config.get("max_patience_counter", 5)
    max_time = exp_config["training"].get("max_time_tr", 0)  # in minutes
    patience_counter = 0

    training_time = time.perf_counter()
    times_tr_opt = []
    max_length_traj = exp_config["enc_model"].get("len_episode", -1)

    for epoch in range(num_epochs):
        node_model.train()
        enc_model.train()
        history_enc_loss = []
        history_dec_loss = []
        history_losses = {
            "rda_1": [],
            "rda_2": [],
            "ppc": [],
            "N_ELBO": [],
        }
        tr_loss = []
        only_tr_opt_time = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            batch = [x.to(node_model.device) for x in batch]

            x_traj, t_traj = (
                batch[0],
                batch[1],
            )
            optimizer.zero_grad()

            total_length = x_traj.shape[1]
            if max_length_traj > 0 and total_length > max_length_traj:
                start = torch.randint(
                    0, total_length - max_length_traj + 1, (1,)
                ).item()
                end = start + max_length_traj
                x_traj, t_traj = x_traj[:, start:end, :], t_traj[:, start:end]

            # ENCODE
            p_mean, p_logstd, z_mean, z_logstd = enc_model.encode(x_traj)
            p_sample, z_sample = enc_model.sample(
                p_mean, p_logstd, z_mean, z_logstd
            )
            init_x = (
                x_traj[:, 0].clone().view(x_traj.shape[0], -1)
            )  # initial state

            # Decode
            mu_x_pred, mu_x_pred_fp = node_model.sample_trajectory(
                x0=init_x,
                phys_params=p_sample,
                z_sample=z_sample,
                t_span=t_traj[0],
                ode_method=exp_config["vf_model"].get("ode_method", "rk4"),
            )

            mse_traj = (
                (x_traj.flatten(start_dim=1) - mu_x_pred.flatten(start_dim=1))
                .norm(2, dim=1)
                .mean()
            )
            # ELBO
            p_kl, z_kl = enc_model.kl_divergence(
                p_mean, p_logstd, z_mean, z_logstd
            )
            enc_kl = (p_kl + z_kl).mean()
            N_ELBO = mse_traj + enc_kl

            if node_model.phys_eq is not None:
                # R-PPC (Decoder Regularization - Least Action principle)
                # (|| x_pred - x_pred_fp||^2)
                if ALPHA > 0.0:
                    ppc = (
                        (
                            mu_x_pred.flatten(start_dim=1)
                            - mu_x_pred_fp.flatten(start_dim=1)
                        )
                        .pow(2)
                        .sum(1)
                        .mean()
                    )
                else:
                    ppc = torch.tensor(0.0, device=node_model.device)

                # RDA_1
                if BETA > 0.0:
                    # x_cleansed = Encoder(x)
                    # || x_cleased - mu_x_pred_fp||^2
                    x_r_detached = mu_x_pred_fp.detach().requires_grad_(True)
                    R_da_1 = (
                        (
                            enc_model.x_cleansed.flatten(start_dim=1)
                            - x_r_detached.flatten(start_dim=1)
                        )
                        .pow(2)
                        .sum(1)
                        .mean()
                    )
                else:
                    R_da_1 = torch.tensor(0.0, device=node_model.device)

                # RDA_2 (phyiscs only with cleansed data)
                if GAMMA > 0.0:
                    node_model.eval()
                    with torch.no_grad():
                        p_samples, z_samples = enc_model.sample_priors(
                            x_traj.shape[0]
                        )
                        _, mu_x_pred_fp = node_model.sample_trajectory(
                            x0=init_x,
                            phys_params=p_samples,
                            z_sample=z_samples,
                            t_span=t_traj[0],
                            ode_method=exp_config["vf_model"].get(
                                "ode_method", "rk4"
                            ),
                        )
                    node_model.train()
                    aug_f_phy = enc_model.p_proj(
                        enc_model.nnet_h_p(mu_x_pred_fp.detach())
                    )
                    diff = (aug_f_phy - p_samples).pow(2).sum(1)
                    R_da_2 = torch.nan_to_num(
                        diff.mean(), nan=50, posinf=50, neginf=50
                    )

                    nans_mask = torch.isnan(diff)
                    inf_mask = torch.isinf(diff)
                    if nans_mask.any() or inf_mask.any():
                        mask = ~nans_mask & ~inf_mask
                        # log the number of nans and infs and their ratios
                        logger.info(
                            f"\n"
                            f"Number of NaNs: {nans_mask.sum().item()}, Ratio of nan samples: {nans_mask.sum().item()}/{diff.shape[0]} \n"
                            f"Number of Infs: {inf_mask.sum().item()}, Ratio of inf samples: {inf_mask.sum().item()}/{diff.shape[0]} \n"
                            f"Ratio of Valid samples: {mask.sum().item()}/{diff.shape[0]}"
                        )

                else:
                    R_da_2 = torch.tensor(0.0, device=node_model.device)

            else:
                ppc = torch.tensor(0.0, device=node_model.device)
                R_da_1 = torch.tensor(0.0, device=node_model.device)
                R_da_2 = torch.tensor(0.0, device=node_model.device)

            loss_tot = N_ELBO + ALPHA * ppc + BETA * R_da_1 + GAMMA * R_da_2
            loss_tot.backward()
            if exp_config["optimization"].get("grad_clip", 0.0) > 0:
                torch.nn.utils.clip_grad_value_(
                    node_model.parameters(),
                    exp_config["optimization"]["grad_clip"],
                )
                torch.nn.utils.clip_grad_value_(
                    enc_model.parameters(),
                    exp_config["optimization"]["grad_clip"],
                )

            optimizer.step()

            # Log losses
            history_enc_loss.append(enc_kl.detach())
            history_dec_loss.append(mse_traj.detach())
            history_losses["N_ELBO"].append(N_ELBO.detach())
            tr_loss.append(loss_tot.detach())
            if node_model.phys_eq is not None:
                history_losses["ppc"].append(ppc.detach())
                history_losses["rda_1"].append(R_da_1.detach())
                history_losses["rda_2"].append(R_da_2.detach())

        # Record training optimization time
        only_tr_opt_time = time.perf_counter() - only_tr_opt_time
        times_tr_opt.append(only_tr_opt_time)
        if wandb.run is not None:
            wandb.log({"train/only_tr_opt": only_tr_opt_time}, step=epoch)

        # record average batch loss
        # wandb record batch loss
        tr_enc_kl = torch.stack(history_enc_loss).mean().item()
        tr_node_loss = torch.stack(history_dec_loss).mean().item()
        tr_std_node_loss = torch.stack(history_dec_loss).std().item()
        tr_total_loss = torch.stack(tr_loss).mean().item()

        if wandb.run is not None:
            wandb.log(
                {
                    "train/enc_kl": tr_enc_kl,
                    "train/node_loss": tr_node_loss,
                    "train/total_loss": tr_total_loss,
                    "train/time": time.perf_counter() - training_time,
                },
                step=epoch,
            )
            if node_model.phys_eq is not None:
                wandb.log(
                    {
                        "train/rda_1": torch.stack(history_losses["rda_1"])
                        .mean()
                        .item(),
                        "train/rda_2": torch.stack(history_losses["rda_2"])
                        .mean()
                        .item(),
                        "train/ppc": torch.stack(history_losses["ppc"])
                        .mean()
                        .item(),
                    },
                    step=epoch,
                )
        train_losses["node_loss"].append(tr_node_loss)
        train_losses["std_node_loss"].append(tr_std_node_loss)
        train_losses["enc_kl"].append(tr_enc_kl)
        train_losses["N_ELBO"].append(
            torch.stack(history_losses["N_ELBO"]).mean().item()
        )
        train_losses["total_loss"].append(tr_total_loss)
        train_losses["epoch"].append(epoch)
        if epoch % exp_config["training"].get("log_train_step", 20) == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: \n"
                f"\t\t NODE Loss: {tr_node_loss:.4f}, \n"
                f"\t\t Std NODE Loss: {tr_std_node_loss:.4f}, \n"
                f"\t\t Enc KL: {tr_enc_kl:.8f}, \n"
                f"\t\t N_ELBO: {train_losses['N_ELBO'][-1]:.4f}, \n"
                f"\t\t Total Loss: {tr_total_loss:.4f}"
            )

        if (
            epoch > exp_config["training"].get("after_val_step", 50)
            and epoch % val_step == 0
        ) or epoch == num_epochs - 1:
            node_model.eval()
            enc_model.eval()
            if val_loader is not None:
                time_val = time.perf_counter()
                metric_score = evaluate_model(
                    node_model,
                    enc_model,
                    exp_config=exp_config,
                    val_loader=val_loader,
                    metrics=["mse", "std"],
                    eval_method=[EVALUATING_METHODS[0], EVALUATING_METHODS[1]],
                    ode_method=exp_config["vf_model"].get(
                        "val_ode_method", "dopri5"
                    ),
                    do_plot=True,
                )
                rec_score, rec_std = (
                    metric_score["rec_score"]["mse"],
                    metric_score["rec_score"]["std"],
                )
                time_val = time.perf_counter() - time_val
                logger.debug(f"Validation time: {time_val:.4f} seconds")
                val_metrics["rec_score"].append(rec_score)
                val_metrics["rec_std"].append(rec_std)
                val_metrics["epoch"].append(epoch)
                if "forecast_score" in metric_score:
                    forecast_score = metric_score["forecast_score"]["mse"]
                    val_metrics["forecast_score"].append(forecast_score)
                logger.info(
                    f"Validation metric:\n\tMean {rec_score:.4f}\n\t"
                    f"Std: {rec_std:.4f}"
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "val/rec_mse": rec_score,
                            "val/rec_std": rec_std,
                            "val/time": time_val,
                        },
                        step=epoch,
                    )
                    if "forecast_score" in metric_score:
                        wandb.log(
                            {
                                "val/forecast_mse": forecast_score,
                            },
                            step=epoch,
                        )
                if (
                    "params_score" in metric_score
                    and metric_score["params_score"] is not None
                ):
                    params_mean, params_std = (
                        metric_score["params_score"]["mse"],
                        metric_score["params_score"]["std"],
                    )
                    val_metrics["params_score"].append(params_mean)
                    val_metrics["params_std"].append(params_std)
                    logger.info(
                        f"Parameter metric:\n\tMean {params_mean:.4f}\n\t"
                        f" Std: {params_std:.4f}"
                    )
                    if wandb.run is not None:
                        wandb.log(
                            {
                                "val/params_mse": params_mean,
                                "val/params_std": params_std,
                            },
                            step=epoch,
                        )
                # Save the model checkpoint if validation loss improves
                actual_score = (
                    forecast_score
                    if "forecast_score" in metric_score
                    else rec_score
                )
                if (
                    chkp_score["score"] is None
                    or actual_score < chkp_score["score"]
                ):
                    if chkp_score["score"] is not None:
                        logger.info(
                            f"Validation loss improved to {actual_score:.4f} by {chkp_score['score'] - actual_score:.4f} at epoch {epoch + 1}"
                        )
                    chkp_score["score"] = actual_score
                    chkp_score["epoch"] = epoch

                    # Save the model checkpoint
                    chkp_score["model"] = node_model.state_dict()
                    chkp_score["optimizer"] = optimizer.state_dict()
                    # Save best model
                    torch.save(
                        {
                            "vf_model": chkp_score["model"],
                            "enc_model": enc_model.state_dict(),
                            "optimizer": chkp_score["optimizer"],
                            "exp_config": exp_config,
                        },
                        f"{out_dir}/best_chkpt.pth",
                    )
                    logger.info(
                        f"Best model saved at epoch {epoch + 1} with score {actual_score:.4f}"
                    )

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "val/best_score": actual_score,
                                "val/best_epoch": epoch,
                            },
                            step=epoch,
                        )

                # Early stopping:
                # if the validation loss does not improve, stop training
                key_score = (
                    "forecast_score"
                    if "forecast_score" in metric_score
                    else "rec_score"
                )
                timeout = (
                    max_time > 0
                    and (time.perf_counter() - training_time) > max_time * 60
                )
                if timeout:
                    logger.info(
                        f"Early stopping by time at epoch {epoch + 1}. Exceeded {max_time} minutes."
                    )
                    break
                if early_stopping and (
                    len(val_metrics[key_score]) > 2 * MAX_PATIENCE_COUNTER
                    or timeout
                ):
                    if np.mean(
                        val_metrics[key_score][-MAX_PATIENCE_COUNTER:]
                    ) < np.mean(
                        val_metrics[key_score][
                            -2 * MAX_PATIENCE_COUNTER : -MAX_PATIENCE_COUNTER
                        ]
                    ):
                        logger.info(
                            f"Validation loss improved to {actual_score:.4f} at epoch {epoch + 1} -> Resetting patience counter"
                        )
                        patience_counter = 0
                    else:
                        logger.info(
                            f"Validation loss did not improve at epoch {epoch + 1}"
                        )
                        patience_counter += 1
                        if patience_counter >= MAX_PATIENCE_COUNTER:
                            logger.info(
                                f"Early stopping at epoch {epoch + 1} "
                                f"with patience counter {patience_counter}"
                            )
                            break

    # Log training time
    if wandb.run is not None:
        wandb.log(
            {
                "train/time": time.perf_counter() - training_time,
            }
        )

        wandb.log({"train/total_tr_time": np.array(times_tr_opt).sum()})
        wandb.log({"train/mean_tr_time": np.array(times_tr_opt).mean()})

    training_time = time.perf_counter() - training_time
    logger.info(f"Total training time: {training_time / 60:.4f} minutes")
    logger.info(
        f"Only training optimization time: {np.array(times_tr_opt).sum() / 60:.4f} minutes"
    )
    logger.info(
        f"Average training per optimization time: {np.array(times_tr_opt).mean() / 60:.4f} minutes"
    )

    # Save the final model
    chkp_score["final_model"] = node_model.state_dict()
    if wandb.run is not None:
        wandb.finish()

        # Delete all wandb logs
        # shutil.rmtree(f"/dev/shm/{wandb_group}")
        # remove the files of the directory if exists of ./wandb/* using os
        #if os.path.exists(f"./wandb/"):
        #    shutil.rmtree(f"./wandb/")
        #    logger.info("Wandb logs removed")

    # save the final model and optimizer state
    torch.save(
        {
            "vf_model": chkp_score["final_model"],
            "enc_model": enc_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "exp_config": exp_config,
        },
        f"{out_dir}/final_chkpt.pth",
    )
    full_evaluation(
        model_path=f"{out_dir}/best_chkpt.pth",
        val_loader=val_loader if test_loader is None else test_loader,
        device=node_model.device,
        save_to_file=True,
    )

    return (
        node_model,
        enc_model,
        optimizer,
        train_losses,
        val_metrics,
        chkp_score,
    )
