import torch
from typing import Union, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import random
import wandb

matplotlib.use("Agg")


import os
import random
import numpy as np

from src.metrics.metrics import mse_loss, mae_loss, n_mse_loss, mape_loss
from src.models.encoders import get_encoder
from src.data.tensor_dataset.dataset_factory import get_dataset
from torch.utils.data import DataLoader
from src.utils.hash import get_exp_dir_path
from src.evaluate.utils import average_seed_evaluations
from src.interpolants.interpolants_fabric import linear_interpolation
from src.interpolants.interpolants_fabric import lagrange_interpolation

from src.models.vf_models.vf_factory import get_vf

EVALUATING_METHODS = ["reconstruction", "forecasting", "generation", "elbo-terms"]


def compute_scores(
    X,
    Y,
    metrics=["mse", "mae", "n_mse", "std", "mape"],
):
    """
    Args:
        X: Tensor of shape (batch_size, memory_length, dim) representing the input data.
        Y: Tensor of shape (batch_size, memory_length, dim) representing the model's predictions.
        metric: The metric to use for evaluating the reconstruction error (e.g., "mse", "mae", "mape").
    Returns:
        scores: The computed reconstruction error based on the specified metric.
        result: A dictionary containing the reconstructed data and model parameters.
    """

    dict_scores = {}

    for metric in metrics:
        if metric == "mse":
            scores = mse_loss(pred=Y, true=X)
        elif metric == "n_mse":
            scores = n_mse_loss(pred=Y, true=X)
        elif metric == "mae":
            scores = mae_loss(pred=Y, true=X)
        elif metric == "mape":
            scores = mape_loss(pred=Y, true=X)
        elif metric == "std":
            scores = torch.mean(torch.std(Y - X, dim=-1))

        dict_scores[metric] = scores

    return dict_scores


def reconstruction_err(
    x_batch,
    t_batch,
    params_batch,
    max_length,
    enc_model,
    vf_model,
    metrics=["mse", "n_mse"],
    ode_method="dopri5",
    z_size=1,
    seed=3,
):
    """
    Compute reconstruction error for trajectories.

    Args:
        x_batch: Batch of trajectories (batch_size, seq_len, state_dim)
        t_batch: Batch of timestamps (batch_size, seq_len)
        params_batch: Batch of physical parameters (batch_size, param_dim)
        max_length: Maximum sequence length for encoding
        enc_model: Encoder model
        vf_model: Vector field model
        metrics: List of metrics to compute
        ode_method: ODE solver method
        z_size: Number of latent samples
        seed: Random seed

    Returns:
        Dictionary with trajectory scores, parameter scores, and predictions
    """
    total_length = x_batch.shape[1]
    if total_length > max_length:
        splits = int(np.floor(total_length / max_length))
        extra = total_length % max_length
        x_chunks = list(x_batch.chunk(splits, dim=1))
        t_chunks = list(t_batch.chunk(splits, dim=1))
        # correct chunks if less than max_length
        for i in range(len(x_chunks)):
            len_episode = x_chunks[i].shape[1]
            if len_episode < max_length:
                offset = max_length - len_episode
                x_chunks[i] = torch.cat(
                    [x_chunks[i - 1][:, -offset:], x_chunks[i]], dim=1
                )
                t_chunks[i] = torch.cat(
                    [t_chunks[i - 1][:, -offset:], t_chunks[i]], dim=1
                )
    else:
        # x_in, t_in = (x_batch, t_batch)
        x_chunks = [x_batch]
        t_chunks = [t_batch]

    final_scores = {}
    X_in = []
    Y_recon = []

    # Get history_size from model if available
    history_size = getattr(vf_model, "history_size", 0)
    for i in range(1 if history_size > 0 else 0, len(x_chunks)):
        # Determine starting point and history based on history_size
        x_in = x_chunks[i]
        t_in = t_chunks[i]
        prev_traj = None
        if history_size > 0:
            prev_traj = x_chunks[i - 1]
            x0 = x_in[:, 0, :]
        else:
            x0 = x_in[:, 0, :]

        # Encode trajectory: first max_length points
        if enc_model is not None:
            if history_size > 0 and prev_traj is not None:
                p_mean, p_logstd, z_mean, z_logstd = enc_model.encode(
                    prev_traj[:, :max_length]
                )
            else:
                p_mean, p_logstd, z_mean, z_logstd = enc_model.encode(
                    x_in[:, :max_length]
                )
            if z_size == 1:
                P_sample, Z_sample = p_mean, z_mean
            else:
                P_sample, Z_sample = enc_model.sample(
                    p_mean, p_logstd, z_mean, z_logstd, z_size=z_size
                )  # shape (batch, z_size, dim)
                P_sample, Z_sample = (
                    P_sample.flatten(0, 1) if p_mean is not None else None
                ), Z_sample.flatten(0, 1)
                # Repeat x0 and x_history for multiple z samples
                x0 = x0.unsqueeze(1).repeat(1, z_size, 1).flatten(0, 1)
                if prev_traj is not None:
                    prev_traj = (
                        prev_traj.unsqueeze(1)
                        .repeat(1, z_size, 1, 1)
                        .flatten(0, 1)
                    )
        else:
            Z_sample, P_sample = None, None

        # Adjust t_span to start from the correct index
        t_span_adjusted = t_in[0].squeeze().tolist()

        Y = vf_model.sample_trajectory(
            x0=x0,
            phys_params=P_sample,
            z_sample=Z_sample,
            t_span=t_span_adjusted,
            x_history=(
                prev_traj[:, -history_size:] if prev_traj is not None else None
            ),
            ode_method=ode_method,
        )
        Y, _ = Y if isinstance(Y, tuple) else (Y, None)

        Y_recon.append(Y)
        X_in.append(x_in)

        scores = compute_scores(
            X=x_in.flatten(start_dim=1),
            Y=Y.flatten(start_dim=1),
            metrics=metrics,
        )
        for key in scores:
            if key not in final_scores:
                final_scores[key] = [scores[key]]
            else:
                final_scores[key].append(scores[key])

    for k in scores:
        scores[k] = torch.mean(torch.stack(final_scores[k], dim=0))

    if params_batch is not None and P_sample is not None:
        param_scores = compute_scores(
            X=params_batch,
            Y=P_sample,
            metrics=metrics,
        )

    result = {
        "traj_scores": scores,
        "params_score": (
            param_scores
            if params_batch is not None and P_sample is not None
            else None
        ),
        "Y": Y_recon,
        "X": X_in,
    }
    return result


def forecast_error(
    x_batch,
    t_batch,
    max_length,
    enc_model,
    vf_model,
    metrics,
    ode_method="dopri5",
    z_size=1,
    max_forecast_length=None,
):
    """
    Compute forecasting error for trajectories.

    Args:
        x_batch: Batch of trajectories (batch_size, seq_len, state_dim)
        t_batch: Batch of timestamps (batch_size, seq_len)
        max_length: Maximum sequence length for encoding
        enc_model: Encoder model
        vf_model: Vector field model
        metrics: List of metrics to compute
        ode_method: ODE solver method
        z_size: Number of latent samples
        max_forecast_length: Maximum forecasting length

    Returns:
        Dictionary with trajectory scores and predictions
    """
    x_in, t_in = (
        (x_batch[:, :max_length, :], t_batch[:, :max_length])
        if x_batch.shape[1] > max_length
        else (x_batch, t_batch)
    )
    # future trajectory
    x_f, t_f = x_batch[:, max_length:, :], t_batch[:, max_length:]

    # Get history_size from model if available
    history_size = getattr(vf_model, "history_size", 0)

    if enc_model is not None:
        p_mean, p_logstd, z_mean, z_logstd = enc_model.encode(x_in)
        if z_size == 1:
            P_sample, Z_sample = p_mean, z_mean
            if history_size > 0:
                x0 = x_f[
                    :, 0, :
                ]  # as it has trained on next point predictions
            else:
                x0 = x_in[:, -1, :]
        else:
            P_sample, Z_sample = enc_model.sample(
                p_mean, p_logstd, z_mean, z_logstd, z_size=z_size
            )  # shape (batch, z_size, dim)
            P_sample, Z_sample = P_sample.flatten(0, 1), Z_sample.flatten(0, 1)
            xf = xf.unsqueeze(1).repeat(1, z_size, 1).flatten(0, 1)
            tf = tf.unsqueeze(1).repeat(1, z_size, 1).flatten(0, 1)
            x0 = x_in[:, -1, :].unsqueeze(1).repeat(1, z_size, 1).flatten(0, 1)
    else:
        P_sample, Z_sample = None, None
        if history_size > 0:
            x0 = x_f[:, 0, :]
        else:
            x0 = x_in[:, -1, :]

    # Prepare history for forecasting
    # History should be the last history_size points from input sequence
    x_history = None
    if history_size > 0:
        if x_in.shape[1] >= history_size:
            # Use last history_size points from the input sequence as history
            x_history = x_in[:, -history_size:]
        else:
            # Pad with x0 (first point) if not enough history
            padding_needed = history_size - x_in.shape[1]
            padding = x_in[:, 0:1, :].repeat(1, padding_needed, 1)
            x_history = torch.cat([padding, x_in], dim=1)[:, -history_size:, :]

    t_forecast = t_f
    if (
        max_forecast_length is not None
        and t_forecast.shape[1] > max_forecast_length
    ):
        t_forecast = t_forecast[:, :max_forecast_length]
        x_f = x_f[:, :max_forecast_length, :]

    Y = vf_model.sample_trajectory(
        x0=x0,
        phys_params=P_sample,
        z_sample=Z_sample,
        t_span=t_forecast[0]
        .squeeze()
        .tolist(),  # timestep should be the same for each batch
        x_history=x_history,
        ode_method=ode_method,
    )
    Y, _ = Y if isinstance(Y, tuple) else (Y, None)
    scores = compute_scores(
        X=x_f.flatten(start_dim=1),
        Y=Y.flatten(start_dim=1),
        metrics=metrics,
    )
    result = {
        "traj_scores": scores,
        "Y": Y,
        "X": x_f,
    }
    return result


def generation_error(
    x_batch,
    t_batch,
    enc_model,
    vf_model,
    metrics,
    forecast_length=None,
    z_size=1,
    ode_method="dopri5",
):

    return 0.0

def loss_elbo_terms(
    vf_model,
    enc_model,
    x_batch,
    t_batch,
    max_length,
    metric_score,
    z_size=1,
):
    
    if enc_model is not None:    
        x_in = x_batch[:, :max_length, :]
        p_mean, p_logstd, z_mean, z_logstd = enc_model.encode(x_in)
        p_kl, z_kl = enc_model.kl_divergence(
                        p_mean, p_logstd, z_mean, z_logstd
                    )
        p_kl, z_kl = p_kl.mean().cpu().numpy(), z_kl.mean().cpu().numpy()
    else: 
        p_mean, z_mean = None, None
        p_kl, z_kl = 0.0, 0.0
    
    # interpolant on the data 
    linear_interp = not getattr(vf_model, "second_order", False)
    latent_repeats = -1


    # sample randomly windows in the trajectory 20 times
    
    if linear_interp: 
        start_idx = torch.randint(low=max_length, high=x_batch.shape[1]-1, size=(50,))
        x_k = x_batch[:, start_idx]
        x_k_plus_1 = x_batch[:, start_idx + 1]
        t_k = t_batch[:, start_idx]
        t_k_plus_1 = t_batch[:, start_idx + 1]
        x_c, t_x_c = None, None
    else:
        start_idx = torch.randint(low=max_length, high=x_batch.shape[1]-2, size=(50,))
        x_c = x_batch[:, start_idx]
        t_x_c = t_batch[:, start_idx]
        x_k = x_batch[:, start_idx+1]
        x_k_plus_1 = x_batch[:, start_idx + 2]
        t_k = t_batch[:, start_idx+1]
        t_k_plus_1 = t_batch[:, start_idx + 2]

    # for history 
    history_size = getattr(vf_model, "history_size", 0)
    if history_size > 0:
        traj_windows = x_batch.unfold(1, max_length, 1).movedim(-1, 2)
        x_traj = traj_windows[:, start_idx - max_length]
        x_history = x_traj[:, :, -history_size :]

    latent_repeats = x_k.shape[1]
    x_k = x_k.flatten(start_dim=0, end_dim=1)
    x_k_plus_1 = x_k_plus_1.flatten(start_dim=0, end_dim=1)
    t_k = t_k.flatten(start_dim=0, end_dim=1)
    t_k_plus_1 = t_k_plus_1.flatten(start_dim=0, end_dim=1)
    x_history = x_history.flatten(start_dim=0, end_dim=1) if history_size > 0 else None

    t = torch.rand(x_k.shape[0], 1).to(x_batch.device)
    if linear_interp:  
        mu_t, x_t, u_t, t_ut, u_t2 = linear_interpolation(
            x_k,
            x_k_plus_1,
            t,
            t0=t_k,
            t1=t_k_plus_1,
            sigma=getattr(vf_model, "sigma", 0.0),
        )
    else:
        x_c = x_c.flatten(start_dim=0, end_dim=1)
        t_x_c = t_x_c.flatten(start_dim=0, end_dim=1)
        mu_t, x_t, u_t, t_ut, u_t2 = lagrange_interpolation(
            x_k,
            x_k_plus_1,
            t,
            t0=t_k,
            t1=t_k_plus_1,
            x_c=x_c,
            t_x_c=t_x_c,
            sigma=getattr(vf_model, "sigma", 0.0),
        )
    
    # repeat p_sample, z_sample if needed
    if enc_model is not None:
        if p_mean is not None:
            p_mean = p_mean.unsqueeze(1).repeat(1, latent_repeats, 1).flatten(0, 1)        
        z_mean = z_mean.unsqueeze(1).repeat(1, latent_repeats, 1).flatten(0, 1)
        
    out_vt = vf_model.forward_train(
                x=x_t,
                p_sample=p_mean,
                z_sample=z_mean,
                t=t_ut,
                dt_xt=None if linear_interp else u_t,
                x_history=x_history if history_size > 0 else None,
            )
    if linear_interp:
        out_vt = out_vt
        acc_loss = 0.0
    else:
        out_vt, out_at = out_vt[:, [0]], out_vt[:, [1]]
        acc_loss = mse_loss(pred=out_at, true=u_t2).cpu().numpy()
        
    fm_loss = mse_loss(pred=out_vt, true=u_t).cpu().numpy()
    res_score = {
        "fm_loss": fm_loss,
        "p_kl": p_kl,
        "z_kl": z_kl,
        "acc_loss": acc_loss,
    }
    
    return res_score
    


def evaluate_model(
    vf_model,
    enc_model,
    exp_config,
    val_loader=None,
    metrics=["mse", "n_mse", "mae", "std"],
    eval_method=[EVALUATING_METHODS[0]],
    do_plot=False,
    z_size=1,
    ode_method="dopri5",
    seed=3,
):

    vf_model.eval()
    if enc_model is not None:
        enc_model.eval()
    metric_score = {
        "rec_score": {},
        "params_score": {},
        "forecast_score": {},
        "gen_score": {},
        "elbo_terms_score": {},
    }

    with torch.no_grad():
        for batch in val_loader:
            # x0 [,dim], x1 [,dim], x_trajs [,memory_length, dim]
            batch = [x.to(vf_model.device) for x in batch]
            if enc_model is not None:
                max_length = enc_model.len_episode
            else:
                max_length = exp_config["sampler"].get("enc_len_episode", 1)

            x_traj, t_traj, params = (
                batch[0],
                batch[1],
                batch[2],
            )  # shape (n, memory_length, dim)

            res = reconstruction_err(
                x_traj,
                t_traj,
                params,
                max_length,
                enc_model,
                vf_model,
                metrics,
                ode_method,
                z_size,
                seed=seed,
            )
            for key in res["traj_scores"]:
                if key not in metric_score["rec_score"]:
                    metric_score["rec_score"][key] = []
                    metric_score["params_score"][key] = []
                    metric_score["forecast_score"][key] = []

                metric_score["rec_score"][key].append(
                    res["traj_scores"][key].cpu().numpy()
                )
                if res["params_score"] is not None:
                    metric_score["params_score"][key].append(
                        res["params_score"][key].cpu().numpy()
                    )

            if (
                max_length < x_traj.shape[1]
                and EVALUATING_METHODS[1] in eval_method
            ):
                res = forecast_error(
                    x_traj,
                    t_traj,
                    max_length,
                    enc_model,
                    vf_model,
                    metrics,
                    ode_method,
                    z_size,
                )
                for key in res["traj_scores"]:
                    if key not in metric_score["forecast_score"]:
                        metric_score["forecast_score"][key] = []
                    metric_score["forecast_score"][key].append(
                        res["traj_scores"][key].cpu().numpy()
                    )

            Y_pred = res["Y"]
            X_true = res["X"]

            if EVALUATING_METHODS[3] in eval_method and exp_config["algorithm"] == "dyn-fm":
                res_score = loss_elbo_terms(
                    vf_model,
                    enc_model,
                    x_traj,
                    t_traj,
                    max_length,
                    metric_score,
                    z_size,
                )
                for key in res_score:
                    if key not in metric_score["elbo_terms_score"]:
                        metric_score["elbo_terms_score"][key] = []
                        
                    metric_score["elbo_terms_score"][key].append(
                        res_score[key]
                    )
                    

    for key in metric_score:
        for key_metric in metric_score[key]:
            if len(metric_score[key][key_metric]) > 0:
                metric_score[key][key_metric] = np.mean(
                    metric_score[key][key_metric]
                )

    if do_plot and (
        exp_config["name_exp"].lower() == "rlc"
        or exp_config["name_exp"].lower() == "pendulum"
    ):
        out_dir = exp_config["out_path"]
        plot_dir = os.path.join(out_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        Y_pred = Y_pred.cpu().numpy()
        X_true = X_true.cpu().numpy()
        # plot trajectories for n random trajectories
        n = 10
        for i in range(n):
            idx = random.randint(0, Y_pred.shape[0] - 1)
            fig = plt.figure(figsize=(10, 5))
            x_in = X_true[idx]
            Y = Y_pred[idx]
            for d in range(x_in.shape[-1]):
                plt.plot(
                    x_in[..., d], label=f"True-{d}", alpha=0.5, color="red"
                )
                plt.plot(Y[..., d], label=f"Pred-{d}", alpha=0.5, color="blue")
                plt.legend()

            plt.title(f"Trajectory {i+1}")
            plt.xlabel("Time")
            plt.ylabel("State")
            plt.legend()

            # The title should include the average metric score  and std
            avg_score = metric_score["rec_score"]["mse"]
            std_score = (
                metric_score["rec_score"]["std"]
                if "std" in metric_score["rec_score"]
                else 0.0
            )

            fig_title = f"Rec. Traj. {i+1}"
            title = f"{fig_title} - Avg Score: {avg_score:.4f}, Std: {std_score:.4f}"
            plt.title(title)
            #plt.savefig(os.path.join(plot_dir, f"trajectory_{i+1}.png"))

            # plot figure on wandb if wandb is enabled
            if wandb.run is not None:
                step = wandb.run.step if wandb.run.step is not None else 0
                wandb.log(
                    {fig_title: wandb.Image(fig)},
                    step=step,
                )
            plt.close()

    return metric_score


def evaluate_exp(
    model_path,
    val_loader=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dataset_path="../../../experiments/dataset/RLC/",
    save_to_file=True,
    metrics=["mse", "n_mse", "mae"],
    save_dir=None,
    seed=3,
):
    print(f"Final evaluation model from {model_path} on device {device}")
    # Load Experiments config and model
    state_dict = torch.load(
        model_path, map_location=device, weights_only=False
    )
    exp_config = state_dict["exp_config"]

    # Build GB-VF model
    vf_model, phys_eq = get_vf(
        exp_config,
        device,
        logger=None,
    )  # Ensure vf_config is defined
    # Build Encoder model
    if (
        exp_config["enc_model"]["z_dim"] == 0
        and exp_config["enc_model"]["p_dim"] == 0
    ):
        enc_model = None
    else:
        enc_model = get_encoder(exp_config, device)

    # backward compatibility
    dic_vf_state = state_dict["vf_model"]
    
    vf_model.load_state_dict(dic_vf_state)
    
    vf_model.eval()
    if enc_model is not None:
        enc_model.load_state_dict(state_dict["enc_model"])
        enc_model.eval()
    print("Models loaded and set to eval mode.")

    # Dataset
    if val_loader is None:
        exp_config["dataset"]["data_path_va"] = dataset_path
        _, val_dataset = get_dataset(exp_config, only_testset=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=exp_config["optimization"].get("batch_size", 64),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

    print(f"Experiment config: {exp_config}")

    res = evaluate_model(
        vf_model,
        enc_model,
        exp_config,
        val_loader=val_loader,
        eval_method=[EVALUATING_METHODS[0], EVALUATING_METHODS[1], EVALUATING_METHODS[3]],
        z_size=1,
        do_plot=False,
        metrics=metrics,
        seed=seed,
    )

    # save results to a pands dataframe and into a csv file
    if save_dir is None:
        save_dir = get_exp_dir_path(exp_config, exp_config["group_exp_name"])
    else:
        save_dir = os.path.join(save_dir, exp_config["model_exp_name"], "seed_" + str(exp_config["seed"]))
    
    os.makedirs(save_dir, exist_ok=True)
    import pandas as pd

    # Make dataset and set index as model exp name
    res["model_exp_name"] = exp_config["model_exp_name"]
    dict_df = {"model_exp_name": exp_config["model_exp_name"]}
    for category in res:
        if isinstance(res[category], dict):
            dict_df.update(
                {
                    f"{category}-{metric}": value
                    for metric, value in res[category].items()
                }
            )
    df = pd.DataFrame([dict_df])
    df.set_index("model_exp_name", inplace=True)
    print(df)
    if save_to_file:
        df.to_csv(
            os.path.join(save_dir, f"evaluation_results.csv"),
            index=True,
            sep=",",
        )
        print(
            f"Results saved to {os.path.join(save_dir, f'evaluation_results.csv')}"
        )
    return df


# Write argparse to take model path and dataset path from command line
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./outputs/pendulum/re_eval/6870aee67349ef6f60dc1eb74d8dc778/dyn-fm/grey_box/",
        help="Path to the trained models checkpoint.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./experiments/dataset/pendulum/one_to_many/friction/test/data_863c006a9cdeb07ab35bfd4c6f92a9d5",
        help="Path to the dataset for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation.",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_checkpoints = []
    for dirpath, dirnames, filenames in os.walk(args.root_dir):
        if "best_chkpt.pth" in filenames:
            model_path = os.path.join(dirpath, "best_chkpt.pth")
            all_checkpoints.append(model_path)

    # sort all checkpoints
    all_checkpoints.sort()
    print(f"Found {len(all_checkpoints)} checkpoints to evaluate.")

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    random.seed(0)
    print(f"Setting random seed to {args.seed}")
    print(f"Using device: {device}")

    for model_path in all_checkpoints:
        print(f"Evaluating model at {model_path}")
        df_res = evaluate_exp(
            model_path=model_path,
            dataset_path=args.dataset_path,
            device=device,
            seed=args.seed,
            save_dir=args.root_dir,
        )
        print(df_res)

    
        
    average_seed_evaluations(root_dir=args.root_dir, group_by_label="model_exp_name")
    print(f"Saved all evaluations in {args.root_dir}")


if __name__ == "__main__":
    main()
