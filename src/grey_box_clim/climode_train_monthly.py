import warnings
import os
from src.grey_box_clim.model_function import *
from src.grey_box_clim.model_utils import *
from src.grey_box_clim.utils import *
from torch.utils.data import DataLoader
import torch.nn.functional as Fin
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint as odeint
import matplotlib

matplotlib.use("Agg")
import argparse
import sys
import time
import torch

torch.manual_seed(42)
torch.cuda.empty_cache()
import torch.optim as optim
import random
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)
import sys
import wandb
from src.utils.wandb import wandb_util
import yaml

set_seed(42)
cwd = os.getcwd()
OUT_CHKPT_DIR = os.path.join(cwd, "outputs/climate_monthly_phys_exp/checkpoints")


SOLVERS = [
    "dopri8",
    "dopri5",
    "bdf",
    "rk4",
    "midpoint",
    "adams",
    "explicit_adams",
    "fixed_adams",
    "adaptive_heun",
    "euler",
]
parser = argparse.ArgumentParser("ClimODE")
parser.add_argument("--solver", type=str, default="euler", choices=SOLVERS)
parser.add_argument("--atol", type=float, default=5e-3)
parser.add_argument("--rtol", type=float, default=5e-3)
parser.add_argument(
    "--step_size", type=float, default=None, help="Optional fixed step size."
)
parser.add_argument("--niters", type=int, default=2000)
parser.add_argument("--teacher", type=int, default=1, choices=[0, 1])
parser.add_argument("--scale", type=int, default=0)
parser.add_argument("--days", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--spectral", type=int, default=0, choices=[0, 1])
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--loss_type", type=int, default=0, choices=[0, 1])
parser.add_argument(
    "--force_reload",
    action="store_true",
    default=True,
    help="Force reload data even if cached version exists",
)

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_time_scale = slice("2006", "2016")
val_time_scale = slice("2016", "2016")
test_time_scale = slice("2017", "2018")


# Setup paths based on whether CUDA is available
if not torch.cuda.is_available():
    paths_to_data = [
        "./experiments/dataset"
        + "/experiments/dataset/era5_data/geopotential_500/*.nc",
        "./experiments/dataset" + "/experiments/dataset/era5_data/temperature_850/*.nc",
        "./experiments/dataset" + "/experiments/dataset/era5_data/2m_temperature/*.nc",
        "./experiments/dataset"
        + "/experiments/dataset/era5_data/10m_u_component_of_wind/*.nc",
        "./experiments/dataset"
        + "/experiments/dataset/era5_data/10m_v_component_of_wind/*.nc",
    ]
    const_info_path = [
        "./experiments/dataset"
        + "/experiments/dataset/era5_data/constants/constants_5.625deg.nc"
    ]
else:
    paths_to_data = [
        str(cwd) + "/experiments/dataset/era5_data/geopotential_500/*.nc",
        str(cwd) + "/experiments/dataset/era5_data/temperature_850/*.nc",
        str(cwd) + "/experiments/dataset/era5_data/2m_temperature/*.nc",
        str(cwd) + "/experiments/dataset/era5_data/10m_u_component_of_wind/*.nc",
        str(cwd) + "/experiments/dataset/era5_data/10m_v_component_of_wind/*.nc",
    ]
    const_info_path = [
        str(cwd) + "/experiments/dataset/era5_data/constants/constants_5.625deg.nc"
    ]

levels = ["z", "t", "t2m", "u10", "v10"]
paths_to_data = paths_to_data[0:5]
levels = levels[0:5]
assert len(paths_to_data) == len(
    levels
), "Paths to different type of data must be same as number of types of observations"


def get_cache_filename(spectral):
    """Generate cache filename based on parameters"""
    return f"climate_data_monthly_spectral_{spectral}.pt"


def save_datasets(
    cache_path,
    train_data,
    val_data,
    test_data,
    time_steps,
    lat,
    lon,
    max_lev,
    min_lev,
    time_stamp,
):
    """Save all datasets and metadata to cache"""
    cache_data = {
        "Final_train_data": train_data,
        "Final_val_data": val_data,
        "Final_test_data": test_data,
        "time_steps": time_steps,
        "lat": lat,
        "lon": lon,
        "max_lev": max_lev,
        "min_lev": min_lev,
        "time_stamp": time_stamp,
    }
    torch.save(cache_data, cache_path)
    print(f"Datasets saved to cache: {cache_path}")


# Check if cached datasets exist
cache_dir = os.path.join("./experiments/dataset/era5_data", "cached_datasets")
os.makedirs(cache_dir, exist_ok=True)
cache_filename = get_cache_filename(args.spectral)
cache_path = os.path.join(cache_dir, cache_filename)

if os.path.exists(f"{cache_path}") and not args.force_reload:
    print(
        "############################ Loading cached datasets ###########################"
    )
    (
        Final_train_data,
        Final_val_data,
        Final_test_data,
        time_steps,
        lat,
        lon,
        max_lev,
        min_lev,
        time_stamp,
    ) = load_datasets(cache_path)
    print("Cached datasets loaded successfully!")
else:
    print(
        "############################ Data is loading from source ###########################"
    )
    Final_train_data = 0
    Final_val_data = 0
    Final_test_data = 0
    max_lev = []
    min_lev = []

    for idx, data in enumerate(paths_to_data):
        (
            Train_data,
            Val_data,
            Test_data,
            time_steps,
            lat,
            lon,
            mean_pred,
            std,
            time_stamp,
        ) = get_train_test_data_without_scales_batched_monthly(
            data,
            train_time_scale,
            val_time_scale,
            test_time_scale,
            levels[idx],
            args.spectral,
        )
        max_lev.append(mean_pred)
        min_lev.append(std)
        if idx == 0:
            Final_train_data = Train_data
            Final_val_data = Val_data
            Final_test_data = Test_data
        else:
            Final_train_data = torch.cat([Final_train_data, Train_data], dim=2)
            Final_val_data = torch.cat([Final_val_data, Val_data], dim=2)
            Final_test_data = torch.cat([Final_test_data, Test_data], dim=2)

    # Save datasets to cache
    save_datasets(
        cache_path,
        Final_train_data,
        Final_val_data,
        Final_test_data,
        time_steps,
        lat,
        lon,
        max_lev,
        min_lev,
        time_stamp,
    )

print("Length of training data", len(Final_train_data))
print("Length of validation data", len(Final_val_data))
print("Length of testing data", len(Final_test_data))


const_channels_info, lat_map, lon_map = add_constant_info(const_info_path)
if args.spectral == 1:
    print("############## Running the Model in Spectral Domain ####################")
H, W = Final_train_data.shape[3], Final_train_data.shape[4]
val_clim = torch.mean(Final_val_data, dim=0)
test_clim = torch.mean(Final_test_data, dim=0)
Train_loader = DataLoader(
    Final_train_data[2:],
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)
Val_loader = DataLoader(
    Final_val_data[2:],
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)
Test_loader = DataLoader(
    Final_test_data[2:],
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)
time_loader = DataLoader(
    time_steps[2:], batch_size=args.batch_size, shuffle=False, pin_memory=False
)
time_idx_steps = torch.tensor([i for i in range(12)]).view(-1, 1)
time_idx = DataLoader(
    time_idx_steps[2:],
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)

num_years = len(range(2006, 2016))
model = Climate_encoder_free_uncertain_monthly(
    len(paths_to_data),
    2,
    out_types=len(paths_to_data),
    method=args.solver,
    use_att=True,
    use_err=True,
    use_pos=False,
).to(device)
param = count_parameters(model)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.niters)

best_loss = float("inf")
train_best_loss = float("inf")
best_epoch = float("inf")

print(
    "############################ Data is loaded, Fitting the velocity #########################"
)

kernel = torch.from_numpy(
    np.load("./experiments/dataset/era5_data" + "/cached_datasets/kernel.npy")
)
# Check if velocity is cached
if os.path.exists(cache_dir + "/train_monthly_vel.npy") and not args.force_reload:
    vel_train, vel_val = load_velocity(["train_monthly", "val_monthly"], cache_dir)
    print("Velocity loaded from cache!")
else:
    fit_velocity(
        time_idx,
        time_loader,
        Final_train_data,
        Train_loader,
        torch.device("cpu"),
        num_years,
        paths_to_data,
        args.scale,
        H,
        W,
        types="train_monthly",
        vel_model=Optim_velocity,
        kernel=kernel,
        cache_dir=cache_dir,
    )
    fit_velocity(
        time_idx,
        time_loader,
        Final_val_data,
        Val_loader,
        torch.device("cpu"),
        1,
        paths_to_data,
        args.scale,
        H,
        W,
        types="val_monthly",
        vel_model=Optim_velocity,
        kernel=kernel,
        cache_dir=cache_dir,
    )
    fit_velocity(
        time_idx,
        time_loader,
        Final_test_data,
        Test_loader,
        torch.device("cpu"),
        2,
        paths_to_data,
        args.scale,
        H,
        W,
        types="test_monthly",
        vel_model=Optim_velocity,
        kernel=kernel,
        cache_dir=cache_dir,
    )
    vel_train, vel_val = load_velocity(["train_monthly", "val_monthly"], cache_dir)

print(
    "############################ Velocity loaded, Model starts to train #########################"
)
print(model)
print(
    "####################### Total Parameters",
    param,
    "################################",
)


# initialize wandb and logging
logger = logging.getLogger("ClimODE")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)

wandb_config_file = os.path.join(
    cwd, "experiments/scripts/configs/wandb/wandb_config.yaml"
)
with open(wandb_config_file, "r") as f:
    wandb_config = yaml.safe_load(f)

if wandb_config["run_wandb_log"] == False:
    wandb_config = {}

exp_config = {
    "algorithm": "ClimODE",
    "model": "ClimODE_monthly",
    "group_exp_name": "climate_monthly_phys_exp",
    "model_exp_name": f"fm_phys_bs{args.batch_size}_lr{args.lr}",
    "args": f"{str(vars(args))}",
}
exp_config = wandb_util.init_wandb(
    exp_config=exp_config, wandb_config=wandb_config, logger=logger
)

if (
    wandb.run is not None
    and len(wandb_config) > 0
    and wandb_config.get("wandb_model_watch", False)
):
    models_to_watch = {
        "model_phys": model,
    }
    wandb_util.watch(
        models_to_watch,
        log=wandb_config.get("wandb_watch_log", "all"),
        log_freq=wandb_config.get("log_freq_batches", 20),
    )
    logger.info("Wandb watch initialized")


model.train()

VAL_INTERVAL = 5
WARMUP_EPOCHS = args.warmup_epochs if hasattr(args, "warmup_epochs") else -1

for epoch in range(args.niters):
    total_train_loss = 0
    val_loss = 0
    test_loss = 0
    model.train()
    if epoch == 0:
        var_coeff = 0.001
    else:
        var_coeff = 2 * scheduler.get_last_lr()[0]

    for entry, (time_steps, batch) in enumerate(zip(time_loader, Train_loader)):
        optimizer.zero_grad()
        data = (
            batch[0]
            .to(device)
            .view(num_years, 1, len(paths_to_data) * (args.scale + 1), H, W)
        )
        past_sample = (
            vel_train[entry]
            .view(num_years, 2 * len(paths_to_data) * (args.scale + 1), H, W)
            .to(device)
        )
        model.update_param(
            [
                past_sample,
                const_channels_info.to(device),
                lat_map.to(device),
                lon_map.to(device),
            ]
        )
        t = time_steps.float().to(device).flatten()

        mean_pred, std, _ = model(t, data)

        loss = nll(mean_pred, std, batch.float().to(device), lat, var_coeff)

        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        loss.backward()
        optimizer.step()
        print("Loss for batch is ", loss.item())
        if torch.isnan(loss):
            print("Quitting due to Nan loss")
            quit()
        total_train_loss = total_train_loss + loss.item()

    lr_val = scheduler.get_last_lr()[0]
    scheduler.step()

    print("|Iter ", epoch, " | Total Train Loss ", total_train_loss, "|")

    last_val_loss = float("inf")
    if epoch % VAL_INTERVAL == 0 and epoch >= WARMUP_EPOCHS:
        Lead_ACC_val = {lev: [[] for _ in range(args.batch_size - 1)] for lev in levels}
        Lead_RMSD_val = {
            lev: [[] for _ in range(args.batch_size - 1)] for lev in levels
        }
        val_loss = 0
        model.eval()
        for entry, (time_steps, batch) in enumerate(zip(time_loader, Val_loader)):
            truth_val = batch.to(device)
            data = (
                batch[0]
                .to(device)
                .view(1, 1, len(paths_to_data) * (args.scale + 1), H, W)
            )
            past_sample = (
                vel_val[entry]
                .view(1, 2 * len(paths_to_data) * (args.scale + 1), H, W)
                .to(device)
            )
            model.update_param(
                [
                    past_sample,
                    const_channels_info.to(device),
                    lat_map.to(device),
                    lon_map.to(device),
                ]
            )
            t = time_steps.float().to(device).flatten()

            with torch.no_grad():
                mean_pred, std, _ = model(t, data)

            loss = nll(mean_pred, std, batch.float().to(device), lat, var_coeff)
            if torch.isnan(loss):
                print("Quitting due to Nan loss")
                quit()

            print("Val Loss for batch is ", loss.item())
            val_loss = val_loss + loss.item()

            for t_step in range(mean_pred.shape[0]):
                for yr in range(0, mean_pred.shape[1]):
                    pred2 = mean_pred[t_step, yr, :, :, :].cpu()
                    truth = truth_val[t_step, yr, :, :, :].cpu()
                    # Assuming clim is [B, C, H, W], we need to select the right one
                    mean_data = val_clim[yr, :, :, :].cpu().detach().numpy()

                    evaluate_rmsd = evaluation_rmsd_mm(
                        pred2, truth, lat, lon, max_lev, min_lev, H, W, levels
                    )
                    evaluate_acc = evaluation_acc_mm(
                        pred2,
                        truth,
                        lat,
                        lon,
                        max_lev,
                        min_lev,
                        H,
                        W,
                        levels,
                        mean_data,
                    )

                    # Store per-timestep, per-observable metrics
                    if (
                        t_step - 1 < 7
                    ):  # Ensure we don't go out of bounds for Lead_..._val
                        for idx, lev in enumerate(levels):
                            Lead_RMSD_val[lev][t_step - 1].append(evaluate_rmsd[idx])
                            Lead_ACC_val[lev][t_step - 1].append(evaluate_acc[idx])

        val_loss = val_loss / len(Val_loader)
        logger.info(f"Epoch {epoch} Validation Loss: {val_loss}")

        for t_idx in range(1):
            acc_values = []
            mse_values = []
            for idx, lev in enumerate(levels):
                if Lead_ACC_val[lev][t_idx]:
                    mean_acc = np.mean(Lead_ACC_val[lev][t_idx])
                    std_acc = np.std(Lead_ACC_val[lev][t_idx])
                    mean_rmsd = np.mean(Lead_RMSD_val[lev][t_idx])
                    std_rmsd = np.std(Lead_RMSD_val[lev][t_idx])
                    acc_values.append(mean_acc)
                    mse_values.append(mean_rmsd)
                else:
                    acc_values.append(0)
                    mse_values.append(0)

        acc = f"{acc_values[0]:.4f}, {acc_values[1]:.4f}, {acc_values[2]:.4f}, {acc_values[3]:.4f}, {acc_values[4]:.4f}"
        mse = f"{mse_values[0]:.4f}, {mse_values[1]:.4f}, {mse_values[2]:.4f}, {mse_values[3]:.4f}, {mse_values[4]:.4f}"

        if np.mean(mse_values) > 0:
            overall_mse = np.mean(mse_values)
        else:
            overall_mse = float("inf")

        logger.info(f"Epoch {epoch} Validation Overall MSE: {overall_mse}")
        logger.info(f"Epoch {epoch} Validation RMSE: {mse}")
        logger.info(f"Epoch {epoch} Validation ACC: {acc}")

        if wandb.run is not None:
            log_dict = {
                "epoch": epoch,
                "val/val_loss": float(val_loss),
                "val/acc_overall": float(np.mean(acc_values)),
                "val/mse_overall": float(overall_mse),
            }
            # Log per-variable metrics
            for i, lvl in enumerate(levels):
                log_dict[f"val/acc_{lvl}"] = float(acc_values[i])
                log_dict[f"val/mse_{lvl}"] = float(mse_values[i])
            wandb.log(log_dict)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            dict_to_save = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
            }
            os.makedirs(OUT_CHKPT_DIR, exist_ok=True)
            torch.save(
                dict_to_save,
                OUT_CHKPT_DIR
                + f"/ClimODE_monthly_{args.solver}_{args.spectral}"
                + "_model_"
                + ".pt",
            )
            if wandb.run is not None:
                wandb.save(
                    OUT_CHKPT_DIR
                    + f"/ClimODE_monthly_{args.solver}_{args.spectral}"
                    + "_model_"
                    + ".pt"
                )
                wandb.log(
                    {
                        "val/best_model_epoch": epoch,
                        "val/best_val_loss": best_loss,
                    }
                )
            logger.info(
                f"Best model saved at epoch {epoch} with validation loss {best_loss} \n \t saved at {OUT_CHKPT_DIR}/ClimODE_monthly_{args.solver}_{args.spectral}_model_.pt"
            )

        if total_train_loss < train_best_loss:
            train_best_loss = total_train_loss
            dict_to_save = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_loss": train_best_loss,
            }
            torch.save(
                dict_to_save,
                OUT_CHKPT_DIR
                + f"/ClimODE_monthly_overfit_{args.solver}_{args.spectral}_model_.pt",
            )
            if wandb.run is not None:
                wandb.save(
                    OUT_CHKPT_DIR
                    + f"/ClimODE_monthly_overfit_{args.solver}_{args.spectral}_model_.pt"
                )
                wandb.log(
                    {
                        "best_overfit_model_epoch": epoch,
                        "best_overfit_model_train_loss": train_best_loss,
                    }
                )
            logger.info(
                f"Best overfit model saved at epoch {epoch} with training loss {train_best_loss} \n\t saved at {OUT_CHKPT_DIR}/ClimODE_monthly_overfit_{args.solver}_{args.spectral}_model_.pt"
            )
