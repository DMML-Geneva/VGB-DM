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
import matplotlib
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
import yaml
import hashlib

set_seed(42)

# Setup logger
logger = logging.getLogger("Global Training")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)
cwd = os.getcwd()
OUT_CHKPT_DIR = os.path.join(cwd, "outputs/climate_hourly_phys_exp/checkpoints")

# data_path = {'z500':str(cwd) + '/experiments/dataset/era5_data/geopotential_500/*.nc','t850':str(cwd) + '/experiments/dataset/era5_data/temperature_850/*.nc'}
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
parser.add_argument("--niters", type=int, default=300)
parser.add_argument("--scale", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--spectral", type=int, default=0, choices=[0, 1])
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument(
    "--use_att",
    action="store_true",
    default=True,
    help="Use attention mechanism in the model",
)
parser.add_argument(
    "--use_err",
    action="store_true",
    default=False,
    help="Use error estimation in the model",
)
parser.add_argument(
    "--use_pos",
    action="store_true",
    default=False,
    help="Use positional encoding in the model",
)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument(
    "--force_reload",
    action="store_true",
    help="Force reload data even if cached version exists",
)
parser.add_argument(
    "--time_budget",
    type=int,
    default=60,
    help="Time budget in minutes for training",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_time_scale = slice("2006", "2016")
val_time_scale = slice("2016", "2016")
test_time_scale = slice("2017", "2018")

cache_dir = os.path.join(cwd, "cached_datasets")
os.makedirs(cache_dir, exist_ok=True)

train_time_scale = slice("2006", "2016")
val_time_scale = slice("2016", "2016")
test_time_scale = slice("2017", "2018")

# paths_to_data = [str(cwd) + '/experiments/dataset/era5_data/geopotential_500/*.nc',str(cwd) + '/experiments/dataset/era5_data/temperature_850/*.nc',str(cwd) + '/experiments/dataset/era5_data/2m_temperature/*.nc',str(cwd) + '/experiments/dataset/era5_data/10m_u_component_of_wind/*.nc',str(cwd) + '/experiments/dataset/era5_data/10m_v_component_of_wind/*.nc']
# const_info_path = [str(cwd) +  '/experiments/dataset/era5_data/constants/constants_5.625deg.nc']

if not torch.cuda.is_available():
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

# Initialize CSV file with manual string formatting
os.makedirs("./logs", exist_ok=True)
log_file = f"./logs/training_log_node.csv"
header = [
    "epoch",
    "train_loss",
    "val_loss",
    "overall_accuracy",
    "overall_mse",
    "epoch_time_min",
    "total_time_min",
]
acc_levels = [f"{lvl}_acc" for lvl in levels]
mse_levels = [f"{lvl}_mse" for lvl in levels]
header.extend(acc_levels)
header.extend(mse_levels)

with open(log_file, "w") as f:
    f.write(",".join(header) + "\n")


def get_cache_filename(spectral):
    """Generate cache filename based on parameters"""
    return f"climate_data_spectral_{spectral}.pt"


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


def load_datasets(cache_path):
    """Load all datasets and metadata from cache"""
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    return (
        cache_data["Final_train_data"],
        cache_data["Final_val_data"],
        cache_data["Final_test_data"],
        cache_data["time_steps"],
        cache_data["lat"],
        cache_data["lon"],
        cache_data["max_lev"],
        cache_data["min_lev"],
        cache_data["time_stamp"],
    )


# Check if cached datasets exist
cache_dir = os.path.join(cwd, "experiments/dataset/era5_data/cached_datasets")
os.makedirs(cache_dir, exist_ok=True)
cache_filename = get_cache_filename(args.spectral)
cache_path = os.path.join(cache_dir, cache_filename)

# start_y=2010 if not torch.cuda.is_available() else 2006
start_y = 2006
num_years = len(range(start_y, 2016))

if os.path.exists(f"{cache_path}"):
    # print("############################ Loading cached datasets ###########################")
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
    # print("Cached datasets loaded successfully!")
else:
    # print("############################ Data is loading from source ###########################")
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
            mean,
            std,
            time_stamp,
        ) = get_train_test_data_without_scales_batched(
            data,
            train_time_scale,
            val_time_scale,
            test_time_scale,
            levels[idx],
            start_y,
        )
        max_lev.append(mean)
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


# print("Length of training data",len(Final_train_data))
# print("Length of validation data",len(Final_val_data))
# print("Length of testing data",len(Final_test_data))

const_channels_info, lat_map, lon_map = add_constant_info(const_info_path)
H, W = Final_train_data.shape[3], Final_train_data.shape[4]

if torch.cuda.is_available():
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
else:
    Train_loader = DataLoader(
        Final_train_data[1452:],
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
    )
    Val_loader = DataLoader(
        Final_val_data[1452:],
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
    )
    Test_loader = DataLoader(
        Final_test_data[1452:],
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
    )

clim = torch.mean(Final_test_data, dim=0)

time_loader = DataLoader(
    time_steps[2:], batch_size=args.batch_size, shuffle=False, pin_memory=False
)
time_idx_steps = torch.tensor([i for i in range(365 * 4)]).view(-1, 1)
time_idx = DataLoader(
    time_idx_steps[2:],
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)
# Model declaration

num_years = len(range(2006, 2016))
model = Climate_encoder_free_uncertain(
    len(paths_to_data),
    2,
    out_types=len(paths_to_data),
    method=args.solver,
    use_att=args.use_att,
    use_err=args.use_err,
    use_pos=args.use_pos,
).to(device)
# model.apply(weights_init_uniform_rule)
param = count_parameters(model)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

best_composite_score = float("inf")
best_epoch = float("inf")

# print("############################ Data is loaded, Fitting the velocity #########################", flush=True)
kernel_path = cache_dir + "/kernel.npy"
if os.path.exists(kernel_path):
    kernel_np = np.load(kernel_path, allow_pickle=True)
else:
    kernel_np = get_gauss_kernel((32, 64), lat, lon)
    np.save(kernel_path, kernel_np)
kernel = torch.from_numpy(kernel_np).float().to(device)

#
if os.path.exists(cache_dir + "/train_10year_2day_mm_vel.npy"):
    vel_train, vel_val, vel_test = load_velocity(
        ["train_10year_2day_mm", "val_10year_2day_mm", "test_10year_2day_mm"],
        cache_dir,
    )
else:
    print("Velocity not found in cache, fitting velocity and saving to cache...")
    vel_train = fit_velocity(
        time_idx,
        time_loader,
        Final_train_data,
        Train_loader,
        device,
        num_years,
        paths_to_data,
        args.scale,
        H,
        W,
        types="train_10year_2day_mm",
        vel_model=Optim_velocity,
        kernel=kernel,
        cache_dir=cache_dir,
    )
    vel_val = fit_velocity(
        time_idx,
        time_loader,
        Final_val_data,
        Val_loader,
        device,
        1,
        paths_to_data,
        args.scale,
        H,
        W,
        types="val_10year_2day_mm",
        vel_model=Optim_velocity,
        kernel=kernel,
        cache_dir=cache_dir,
    )
    vel_test = fit_velocity(
        time_idx,
        time_loader,
        Final_test_data,
        Test_loader,
        device,
        2,
        paths_to_data,
        args.scale,
        H,
        W,
        types="test_10year_2day_mm",
        vel_model=Optim_velocity,
        kernel=kernel,
        cache_dir=cache_dir,
    )

print(
    "############################ Velocity loaded, Model starts to train #########################"
)
print(model)
print(
    "####################### Total Parameters",
    param,
    "################################",
)

clim = torch.mean(Final_test_data, dim=0)
mse_loss = nn.MSELoss()

# Initialize wandb
wandb_config_file = os.path.join(
    cwd, "experiments/scripts/configs/wandb/wandb_config.yaml"
)
with open(wandb_config_file, "r") as f:
    wandb_config = yaml.safe_load(f)

if wandb_config["run_wandb_log"] == False:
    wandb_config = {}

import wandb
from src.utils.wandb import wandb_util

# Hash also args to ensure uniqueness
args_str = str(vars(args))
hashed_args = hashlib.md5(args_str.encode()).hexdigest()[:8]
hashed_exp_name = f"{hashed_args}"

exp_config = {
    "algorithm": "ClimODE_Global",
    "model": "Climate_encoder_free_uncertain",
    "group_exp_name": "climate_phys_exp",
    "model_exp_name": f"climode_bs{args.batch_size}_lr{args.lr}_{hashed_exp_name}",
    "args": f"{str(vars(args))}",
}
exp_config = wandb_util.init_wandb(
    exp_config=exp_config, wandb_config=wandb_config, logger=logger
)


logger.info(f"Experiment Run Name (hashed): {hashed_exp_name}")

if (
    wandb.run is not None
    and len(wandb_config) > 0
    and wandb_config.get("wandb_model_watch", False)
):
    models_to_watch = {
        "global_model": model,
    }
    wandb_util.watch(
        models_to_watch,
        log=wandb_config.get("wandb_watch_log", "all"),
        log_freq=wandb_config.get("log_freq_batches", 20),
    )
    logger.info("Wandb watch initialized")

training_start_time = time.perf_counter()
for epoch in range(args.niters):
    epoch_start_time = time.perf_counter()
    total_train_loss = 0
    total_nll_loss = 0
    total_l2_loss = 0
    val_loss = 0
    if epoch == 0:
        var_coeff = 0.001
    else:
        var_coeff = 2 * scheduler.get_last_lr()[0]

    # ---- Training ----
    last_train_loss = None
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
        mean, std, _ = model(t, data)
        nll_loss = nll(mean, std, batch.float().to(device), lat, var_coeff)
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = nll_loss + l2_lambda * l2_norm
        loss.backward()
        optimizer.step()

        if torch.isnan(loss):
            logger.error("Quitting due to NaN loss")
            quit()

        total_train_loss += loss.item()
        total_nll_loss += nll_loss.item()
        total_l2_loss += (l2_lambda * l2_norm).item()
        last_train_loss = loss.item()

        # carriage return update (does not add a new line)
        sys.stdout.write(
            f"\rEpoch {epoch}, "
            f"Train Batch {entry}: "
            f"Loss = {loss.item():.3e}, "
            f"NLL = {nll_loss.item():.3e}, "
            f"L2 = {(l2_lambda * l2_norm).item():.3e}, "
            f"time= {(time.perf_counter() - epoch_start_time) / 60.0:.2f} min"
        )
        sys.stdout.flush()

    lr_val = scheduler.get_last_lr()[0]
    scheduler.step()

    # Log epoch-level training metrics
    avg_train_loss = total_train_loss / len(Train_loader)
    avg_nll_loss = total_nll_loss / len(Train_loader)
    avg_l2_loss = total_l2_loss / len(Train_loader)

    logger.info(
        f"Epoch {epoch} | Train Loss = {last_train_loss:.3e} "
        f"| Avg Train Loss = {avg_train_loss:.3e} "
        f"| Avg NLL Loss = {avg_nll_loss:.3e} "
        f"| Avg L2 Loss = {avg_l2_loss:.3e}"
    )

    if wandb.run is not None:
        wandb.log(
            {
                "epoch": epoch,
                "train/avg_total_loss": float(avg_train_loss),
                "train/avg_nll_loss": float(avg_nll_loss),
                "train/avg_l2_loss": float(avg_l2_loss),
                "train/lr": float(lr_val),
            }
        )

    # ---- Validation ----
    acc_timesteps = []
    mse_timesteps = []
    mse = ""
    acc = ""
    last_val_loss = float("inf")

    if epoch % 5 == 0:
        # Initialize storage for per-timestep, per-observable metrics
        Lead_ACC_val = {lev: [[] for _ in range(7)] for lev in levels}
        Lead_RMSD_val = {lev: [[] for _ in range(7)] for lev in levels}
        total_val_loss = 0

        for entry, (time_steps, batch) in enumerate(zip(time_loader, Test_loader)):
            data = (
                batch[0]
                .to(device)
                .view(2, 1, len(paths_to_data) * (args.scale + 1), H, W)
            )
            past_sample = (
                vel_test[entry]
                .view(2, 2 * len(paths_to_data) * (args.scale + 1), H, W)
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
                mean, std, _ = model(t, data)
            loss = nll(mean, std, batch.float().to(device), lat, var_coeff)

            if torch.isnan(loss):
                logger.error("Quitting due to NaN loss")
                quit()

            val_loss += loss.item()
            total_val_loss += loss.item()
            last_val_loss = loss.item()

            # Loop over 2 years
            for yr in range(2):
                for t_step in range(1, len(time_steps), 1):
                    pred2 = mean[t_step, yr, :, :, :].cpu()
                    truth = batch[t_step, yr, :, :, :].cpu()
                    mean_data = clim[yr, :, :, :].cpu().detach().numpy()

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
                    for idx, lev in enumerate(levels):
                        Lead_RMSD_val[lev][t_step - 1].append(evaluate_rmsd[idx])
                        Lead_ACC_val[lev][t_step - 1].append(evaluate_acc[idx])

            # Compute running averages for display (optional)
            avg_acc_display = np.mean(
                [
                    np.mean(Lead_ACC_val[lev][0])
                    for lev in levels
                    if Lead_ACC_val[lev][0]
                ]
            )
            avg_mse_display = np.mean(
                [
                    np.mean(Lead_RMSD_val[lev][0])
                    for lev in levels
                    if Lead_RMSD_val[lev][0]
                ]
            )

            # carriage return update for validation
            sys.stdout.write(
                f"\rEpoch {epoch}, Val Batch {entry}: "
                f"Loss = {loss.item():.3e}, "
                f"time= {(time.perf_counter() - epoch_start_time) / 60.0:.2f} min"
            )
            sys.stdout.flush()

        # After all batches, compute per-timestep, per-observable averages
        for t_idx in range(1):
            acc_values = []
            mse_values = []
            for idx, lev in enumerate(levels):
                mean_acc = np.mean(Lead_ACC_val[lev][t_idx])
                std_acc = np.std(Lead_ACC_val[lev][t_idx])
                mean_rmsd = np.mean(Lead_RMSD_val[lev][t_idx])
                std_rmsd = np.std(Lead_RMSD_val[lev][t_idx])

                acc_values.append(mean_acc)
                mse_values.append(mean_rmsd)

            # Format for this timestep (matching your original format)
            acc = f"{acc_values[0]:.4f}, {acc_values[1]:.4f}, {acc_values[2]:.4f}, {acc_values[3]:.4f}, {acc_values[4]:.4f}"
            mse = f"{mse_values[0]:.4f}, {mse_values[1]:.4f}, {mse_values[2]:.4f}, {mse_values[3]:.4f}, {mse_values[4]:.4f}"

        # Final summary
        sys.stdout.write(
            f"\rEpoch {epoch} "
            f"| Last Train Loss = {last_train_loss:.3e} "
            f"| Total Train Loss = {total_train_loss:.3e} "
            f"| Last Val Loss = {last_val_loss:.3e} "
            f"| Total Val Loss = {val_loss:.3e} "
            f"| RMSE = {mse} "
            f"| ACC = {acc} "
            f"| Tot-time {(time.perf_counter() - training_start_time) / 60:.2f}\n"
        )
        sys.stdout.flush()

        # Log validation metrics to wandb
        if wandb.run is not None:
            avg_val_loss = val_loss / len(Test_loader)
            overall_mse = np.mean(mse_values)

            log_dict = {
                "epoch": epoch,
                "val/avg_val_loss": float(avg_val_loss),
                "val/last_val_loss": float(last_val_loss),
                "val/acc_overall": float(np.mean(acc_values)),
                "val/mse_overall": float(overall_mse),
            }
            # Log per-variable metrics
            for i, lvl in enumerate(levels):
                log_dict[f"val/acc_{lvl}"] = float(acc_values[i])
                log_dict[f"val/mse_{lvl}"] = float(mse_values[i])

            wandb.log(log_dict)

            # Compute weighted composite score for checkpoint saving
            composite_score = 0.5 * avg_val_loss + 0.5 * overall_mse

            # Save checkpoint if composite score improved
            if composite_score < best_composite_score:
                best_composite_score = composite_score
                best_epoch = epoch
                os.makedirs(OUT_CHKPT_DIR, exist_ok=True)
                ckpt_path = (
                    f"{OUT_CHKPT_DIR}/"
                    + f"ClimODE_hourly_{'with_noise' if args.use_err else 'node'}_{hashed_exp_name}.pt"
                )
                dict_to_save = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_composite_score": best_composite_score,
                    "best_val_loss": avg_val_loss,
                    "best_mse": overall_mse,
                    "args": vars(args),
                }
                torch.save(dict_to_save, ckpt_path)
                logger.info(
                    f"Checkpoint saved at epoch {epoch} | "
                    f"Composite Score: {composite_score:.3e} "
                    f"(Val Loss: {avg_val_loss:.3e}, MSE: {overall_mse:.3e})"
                )
                if wandb.run is not None:
                    wandb.save(ckpt_path)
                    wandb.log(
                        {
                            "epoch": epoch,
                            "val/best_composite_score": float(best_composite_score),
                            "val/best_val_loss": float(avg_val_loss),
                            "val/best_mse": float(overall_mse),
                        }
                    )

    # Check time budget
    elapsed_time_minutes = (time.perf_counter() - training_start_time) / 60
    if elapsed_time_minutes >= args.time_budget:
        logger.info(
            f"Time budget of {args.time_budget} minutes reached. Stopping training at epoch {epoch}."
        )
        break
