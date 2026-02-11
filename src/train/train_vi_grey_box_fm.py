import torch
import numpy as np

import shutil
import os
import logging
import time
import yaml
from omegaconf import OmegaConf, DictConfig

import wandb

from src.interpolants.interpolants_fabric import linear_interpolation
from src.interpolants.interpolants_fabric import lagrange_interpolation
from src.evaluate.evaluate import (
    evaluate_model,
    EVALUATING_METHODS,
    evaluate_exp as full_evaluation,
)
from src.metrics import mse_loss
from src.utils.wandb import wandb_util
from src.utils.hash import get_exp_dir_path

from src.models.vf_models.grey_box_fm import PhysicsWeightScheduler

class KLAnnealingScheduler:
    def __init__(self, beta_target: float = 0.1, anneal_steps: int = 5000):
        self.beta_target = beta_target
        self.anneal_steps = anneal_steps
        self.current_step = 0
    
    def get_beta(self):
        progress = min(self.current_step / self.anneal_steps, 1.0)
        return self.beta_target * progress
    
    def step(self):
        self.current_step += 1


def train_model(
    vf_model,
    enc_model,
    train_loader,
    optimizer,
    exp_config,
    scheduler=None,
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

    scheduler_type = None
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_type = "plateau"
        elif isinstance(
            scheduler,
            (
                torch.optim.lr_scheduler.StepLR,
                torch.optim.lr_scheduler.CosineAnnealingLR,
                torch.optim.lr_scheduler.LambdaLR,
                torch.optim.lr_scheduler.ExponentialLR,
            ),
        ):
            scheduler_type = "step_based"
        else:
            scheduler_type = "step_based"

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
        models_to_watch = {
            "vf model": vf_model,
        }
        if enc_model is not None:
            models_to_watch["encoder"] = enc_model
        wandb_util.watch(
            models_to_watch,
            log=wandb_config.get("wandb_watch_log", "all"),
            log_freq=wandb_config.get("log_freq_batches", 20),
        )
        logger.info("Wandb watch initialized")

    train_losses = {
        "epoch": [],
        "mean_fm_loss": [],
        "std_fm_loss": [],
        "enc_kl": [],
        "fm_loss": [],
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

    # Model hyperparameters
    gamma_acc = exp_config["optimization"].get("gamma_acc", 0.0)  # 0.1
    beta_phys_kl = exp_config["optimization"].get("beta_p", 1)
    beta_z_kl = exp_config["optimization"].get("beta_z", 1.0)
    coeff_R = exp_config["optimization"].get(
        "coeff_R", 0.0
    )  # Regularization coefficient
    if exp_config["optimization"].get("use_kl_annealing", False) and enc_model is not None:
        kl_scheduler = KLAnnealingScheduler(
            beta_target=exp_config["optimization"].get("beta_p", 0.1),
            anneal_steps=exp_config["optimization"].get("kl_anneal_steps", 5000),
        )
    else:
        kl_scheduler = None

    # Training hyperparameters
    MAX_PATIENCE_COUNTER = exp_config.get("max_patience_counter", 5)
    patience_counter = 0
    max_time = exp_config["training"].get("max_time_tr", 0)  # in minutes

    training_time = time.perf_counter()
    times_tr_opt = []
    second_order_opt = (
        exp_config["vf_model"].get("second_order", False) and gamma_acc > 0
    )
    history_size = exp_config["vf_model"].get("history_size", 0)
    flag_lagrange = (
        exp_config["vf_model"].get("interpolation", "linear") == "lagrange"
    )

    logger.info("Starting training...")
    logger.info(f"Using device: {vf_model.device}")

    # Initialize physics weight scheduler
    if exp_config["vf_model"].get("phys_model", False) and exp_config["optimization"].get("phys_weight_scheduler", False):
        physics_scheduler = PhysicsWeightScheduler(warmup_steps=exp_config["optimization"].get("phys_warming_steps", 100))
    else:
        physics_scheduler = None

    for epoch in range(num_epochs):
        vf_model.train()
        if enc_model is not None:
            enc_model.train()
        history_enc_loss = []
        history_acc_loss = []
        history_fm = []
        tr_loss = []
        only_tr_opt_time = time.perf_counter()

        for batch in train_loader:

            if (
                exp_config["sampler"]["sampler_mode"] == "seq-pairs"
                and flag_lagrange
            ):
                x_pairs, t_pairs, x_traj, times = [
                    b.to(vf_model.device) for b in batch
                ]
                x_pairs = x_pairs.flatten(0, 1)
                t_pairs = t_pairs.flatten(0, 1)
                x0 = x_pairs[:, -2]
                x1 = x_pairs[:, -1]
                t0 = t_pairs[:, -2]
                t1 = t_pairs[:, -1]
                x_c = x_pairs[:, :-2].squeeze(dim=1)
                t_x_c = t_pairs[:, :-2].squeeze(dim=1)
                x_history = None
            elif exp_config["sampler"]["sampler_mode"] == "seq-pairs":
                x0, x1, x_traj, t0, t1, times = [
                    b.to(vf_model.device) for b in batch
                ]
                # Flatten to (batch*seq, dim_state)
                x0 = x0.flatten(0, 1)
                x1 = x1.flatten(0, 1)
                t0 = t0.flatten(0, 1)
                t1 = t1.flatten(0, 1)
                if flag_lagrange:
                    x_c = x_c.flatten(0, 1)
                    t_x_c = t_x_c.flatten(0, 1)
                x_history = None
            elif exp_config["sampler"]["sampler_mode"] == "pairs-history":
                if flag_lagrange:
                    x_pairs, t_pairs, x_traj, times, x_history = [
                        b.to(vf_model.device) for b in batch
                    ]
                    x_pairs = x_pairs.flatten(0, 1)
                    t_pairs = t_pairs.flatten(0, 1).squeeze()
                    x_history = x_history.flatten(0, 1)
                    x_traj = x_traj.flatten(0, 1)
                    x0 = x_pairs[:, -2]
                    x1 = x_pairs[:, -1]
                    t0 = t_pairs[:, -2]
                    t1 = t_pairs[:, -1]
                    x_c = x_pairs[:, :-2].squeeze(dim=1)
                    t_x_c = t_pairs[:, :-2].squeeze(dim=1)
                else:
                    x0, x1, x_traj, t0, t1, times, x_history = [
                        b.to(vf_model.device) for b in batch
                    ]
                    x_traj = x_traj.flatten(0, 1)
                    x_history = x_history.flatten(0, 1)
                    x0 = x0.flatten(0, 1)
                    x1 = x1.flatten(0, 1)
                    t0 = t0.flatten(0, 1)
                    t1 = t1.flatten(0, 1)

            optimizer.zero_grad()

            # Encoder: encode, sample, and compute KL divergence
            if enc_model is not None:
                p_mean, p_logstd, z_mean, z_logstd = enc_model.encode(x_traj)
                p_sample, z_sample = enc_model.sample(
                    p_mean, p_logstd, z_mean, z_logstd
                )
                p_kl, z_kl = enc_model.kl_divergence(
                    p_mean, p_logstd, z_mean, z_logstd
                )
                if kl_scheduler is not None:
                    beta_p_kl = kl_scheduler.get_beta()
                    beta_z_kl = kl_scheduler.get_beta()
                else:
                    beta_p_kl, beta_z_kl = beta_phys_kl, beta_z_kl
                enc_kl = (beta_p_kl * p_kl + beta_z_kl * z_kl).mean()
            else:
                # No encoder: set to segment trajectory
                z_sample, p_sample = None, None
                enc_kl = torch.tensor(0.0, device=vf_model.device)

            if exp_config["sampler"]["sampler_mode"] == "seq-pairs":
                # If using sequence pairs, we need to repeat the samples for each pair
                # padding = (x_pairs.shape[1] - 1) if flag_lagrange else 1
                repeats = x0.shape[0] // z_sample.shape[0]
                p_sample = (
                    (p_sample.unsqueeze(1).repeat(1, repeats, 1).flatten(0, 1))
                    if p_sample is not None
                    else None
                )
                z_sample = (
                    z_sample.unsqueeze(1).repeat(1, repeats, 1).flatten(0, 1)
                )

            # Flow Vector Field
            t = torch.rand(x0.shape[0], 1).to(x0.device)
            x0_input = x0
            x1_input = x1

            # Intepolate
            if flag_lagrange:
                mu_t, x_t, u_t, t_ut, u_t2 = lagrange_interpolation(
                    x0_input,
                    x1_input,
                    t,
                    t0=t0,
                    t1=t1,
                    x_c=x_c,
                    t_x_c=t_x_c,
                    sigma=exp_config["vf_model"]["sigma"],
                )
            else:
                mu_t, x_t, u_t, t_ut, u_t2 = linear_interpolation(
                    x0_input,
                    x1_input,
                    t,
                    t0=t0,
                    t1=t1,
                    sigma=exp_config["vf_model"]["sigma"],
                )
            # Forward pass
            if physics_scheduler is not None:
                vf_model.set_physics_weight(physics_scheduler.get_weight())  
            out_vt = vf_model.forward_train(
                x=x_t,
                p_sample=p_sample,
                z_sample=z_sample,
                t=t_ut,
                dt_xt=u_t if second_order_opt else None,
                x_history=x_history if history_size > 0 else None,
            )

            if not second_order_opt:
                # First order optimization
                fm_loss = mse_loss(out_vt, u_t)
            else:
                # Second order optimization
                vel_out, acc_out = out_vt.chunk(2, dim=-1)
                fm_loss = mse_loss(vel_out, u_t)
                acc_loss = mse_loss(acc_out, u_t2)
                fm_loss += gamma_acc * acc_loss

            # ll_zp_augm = torch.distributions.Normal(loc=p_mean, scale=p_std).log_prob(p_sample).sum(1) - From robust hybdrid model (wehenkel)
            N_ELBO = fm_loss + enc_kl
            loss = N_ELBO

            if coeff_R > 0:
                # Add regularization term on the function evaluation
                loss += coeff_R * torch.norm(vf_model.fv_net, p=2)
            loss.backward()
            optimizer.step()
            # Schedulers steps
            if kl_scheduler is not None:
                kl_scheduler.step()
                
                if wandb.run is not None:
                    wandb.log({"beta/kl": kl_scheduler.get_beta()}, step=epoch)
            if scheduler is not None and scheduler_type == "step_based":
                scheduler.step()
                logger.debug(
                    f"Scheduler update: learning rate: {scheduler.get_last_lr()}"
                )
            if physics_scheduler is not None:
                physics_scheduler.step()  

            history_enc_loss.append(enc_kl.detach())
            history_fm.append(fm_loss.detach())
            if second_order_opt:
                history_acc_loss.append(acc_loss.detach())
            tr_loss.append(loss.detach())

        # Record training optimization time
        only_tr_opt_time = time.perf_counter() - only_tr_opt_time
        times_tr_opt.append(only_tr_opt_time)
        if wandb.run is not None:
            wandb.log({"train/only_tr_opt": only_tr_opt_time}, step=epoch)

        # record average batch loss
        # wandb record batch loss
        tr_enc_kl = torch.stack(history_enc_loss).mean().item()
        tr_fm_loss = torch.stack(history_fm).mean().item()
        if second_order_opt:
            tr_acc_loss = torch.stack(history_acc_loss).mean().item()
        tr_std_fm_loss = torch.stack(history_fm).std().item()
        tr_total_loss = torch.stack(tr_loss).mean().item()

        if wandb.run is not None:
            wandb.log(
                {
                    "train/enc_kl": tr_enc_kl,
                    "train/fm_loss": tr_fm_loss,
                    "train/N_ELBO": tr_total_loss,
                    "train/total_loss": tr_total_loss,
                    "train/time": time.perf_counter() - training_time,
                    "lr/optimizer": (
                        scheduler.get_last_lr()[0]
                        if scheduler is not None
                        else 0
                    ),
                },
                step=epoch,
            )
            if second_order_opt:
                wandb.log({"train/acc_loss": tr_acc_loss}, step=epoch)
        train_losses["mean_fm_loss"].append(tr_fm_loss)
        train_losses["std_fm_loss"].append(tr_std_fm_loss)
        train_losses["enc_kl"].append(tr_enc_kl)
        train_losses["total_loss"].append(tr_total_loss)
        train_losses["epoch"].append(epoch)
        if epoch % exp_config["training"].get("log_train_step", 20) == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: \n"
                f"\t\t FM Loss: {tr_fm_loss:.4f}, \n"
                f"\t\t Enc KL: {tr_enc_kl:.8f}, \n"
                f"\t\t Total Loss: {tr_total_loss:.4f}"
            )

        if (
            epoch >= exp_config["training"].get("after_val_step", 50)
            and epoch % val_step == 0
        ) or epoch == num_epochs - 1:
            vf_model.eval()
            if enc_model is not None:
                enc_model.eval()
            if val_loader is not None:
                time_val = time.perf_counter()
                metric_score = evaluate_model(
                    vf_model,
                    enc_model,
                    exp_config=exp_config,
                    val_loader=val_loader,
                    metrics=["mse", "std"],
                    eval_method=[EVALUATING_METHODS[0], EVALUATING_METHODS[1]],
                    do_plot=True,
                )
                rec_score, rec_std = (
                    metric_score["rec_score"]["mse"],
                    metric_score["rec_score"]["std"],
                )

                time_val = time.perf_counter() - time_val
                logger.debug(
                    f"Validation time: {time.perf_counter() - time_val:.4f} seconds"
                )
                val_metrics["rec_score"].append(rec_score)
                val_metrics["rec_std"].append(rec_std)
                val_metrics["epoch"].append(epoch)
                if "forecast_score" in metric_score:
                    forecast_score = metric_score["forecast_score"]["mse"]
                    val_metrics["forecast_score"].append(forecast_score)

                logger.info(
                    f"Validation metric:\n\t Rec score: Mean {rec_score:.4f} \pm {rec_std:.4f} \n\t"
                )
                if "forecast_score" in metric_score:
                    logger.info(
                        f"\t Forecast score: Mean {forecast_score:.4f} \n\t"
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
                    and enc_model is not None
                    and enc_model.p_dim is not None
                ):
                    params_mean, params_std = (
                        metric_score["params_score"]["mse"],
                        metric_score["params_score"]["std"],
                    )
                    print(params_mean, params_std)
                    val_metrics["params_score"].append(params_mean)
                    val_metrics["params_std"].append(params_std)
                    logger.info(
                        f"Parameter metric:\n\tMean {params_mean:.4f}\n\t"
                        f"Std: {params_std:.4f}"
                    )
                    if wandb.run is not None:
                        wandb.log(
                            {
                                "val/params_mse": params_mean,
                                "val/params_std": params_std,
                            },
                            step=epoch,
                        )

                if scheduler is not None and scheduler_type == "plateau":
                    old_lrs = [group["lr"] for group in optimizer.param_groups]
                    scheduler.step(rec_score)  # Use validation loss as metric
                    new_lrs = [group["lr"] for group in optimizer.param_groups]

                    # Check if learning rate was reduced
                    if any(
                        new_lr < old_lr
                        for old_lr, new_lr in zip(old_lrs, new_lrs)
                    ):
                        logger.info(f"Learning rate reduced at epoch {epoch}")
                        for i, (old_lr, new_lr) in enumerate(
                            zip(old_lrs, new_lrs)
                        ):
                            param_group_name = optimizer.param_groups[i].get(
                                "name", f"group_{i}"
                            )
                            logger.info(
                                f"  {param_group_name}: {old_lr:.2e} -> {new_lr:.2e}"
                            )

                    # Log current learning rates
                    if wandb.run is not None:
                        for i, lr in enumerate(new_lrs):
                            param_group_name = optimizer.param_groups[i].get(
                                "name", f"group_{i}"
                            )
                            wandb.log(
                                {f"lr/{param_group_name}": lr}, step=epoch
                            )

                # check if validation score is better than previous best
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
                    chkp_score["model"] = vf_model.state_dict()
                    chkp_score["optimizer"] = optimizer.state_dict()

                    final_path = out_dir + "/best_chkpt.pth"
                    # Save best model
                    torch.save(
                        {
                            "vf_model": chkp_score["model"],
                            "enc_model": (
                                enc_model.state_dict()
                                if enc_model is not None
                                else None
                            ),
                            "optimizer": chkp_score["optimizer"],
                            "exp_config": exp_config,
                        },
                        final_path,
                    )
                    logger.info(
                        f"Best model saved at epoch {epoch + 1} with score {actual_score:.4f}"
                    )

                    # wandb log best loss
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
                            f"Validation loss improved to {val_metrics[key_score][-1]:.4f} at epoch {epoch + 1} -> Resetting patience counter"
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
    chkp_score["final_model"] = vf_model.state_dict()
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
            "enc_model": (
                enc_model.state_dict() if enc_model is not None else None
            ),
            "optimizer": optimizer.state_dict(),
            "exp_config": exp_config,
        },
        f"{out_dir}/final_chkpt.pth",
    )

    full_evaluation(
        model_path=f"{out_dir}/best_chkpt.pth",
        val_loader=val_loader if test_loader is None else test_loader,
        device=vf_model.device,
        save_to_file=True,
    )

    return (
        vf_model,
        enc_model,
        optimizer,
        train_losses,
        val_metrics,
        chkp_score,
    )
