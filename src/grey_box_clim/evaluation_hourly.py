import warnings
import os

from utils import add_constant_info
from torch.utils.data import DataLoader
import torch.nn.functional as Fin
import timeit
import pandas as pd
import numpy as np


from torchdiffeq import odeint as odeint

import argparse
import sys
import time
import torch
import torch.optim as optim
import random

from src.grey_box_clim.utils import (
    evaluation_rmsd_mm,
    evaluation_acc_mm,
)
from src.grey_box_clim.fm_phys_func import Climate_VFM_ENC
from src.grey_box_clim.model_function import (
    Climate_encoder_free_uncertain as ClimODE,
)

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
]
parser = argparse.ArgumentParser("ClimODE")

parser.add_argument("--solver", type=str, default="euler", choices=SOLVERS)
parser.add_argument(
    "--model", type=str, default="Models_ClimODE_with_noise_758f1b2d"
)  # "data", "Models_fm_phys_683e4c13"
parser.add_argument("--atol", type=float, default=5e-3)
parser.add_argument("--rtol", type=float, default=5e-3)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument(
    "--step_size", type=float, default=None, help="Optional fixed step size."
)
parser.add_argument("--scale", type=int, default=0)
parser.add_argument("--days", type=int, default=3)
parser.add_argument("--spectral", type=int, default=0, choices=[0, 1])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()
cache_dir = os.path.join("./experiments/dataset/era5_data", "cached_datasets")

train_time_scale = slice("2006", "2016")
val_time_scale = slice("2016", "2016")
test_time_scale = slice("2017", "2018")

parent_dir = os.path.dirname(cwd)
paths_to_data = [
    "./experiments/dataset" + "/experiments/dataset/era5_data/geopotential_500/*.nc",
    "./experiments/dataset" + "/experiments/dataset/era5_data/temperature_850/*.nc",
    "./experiments/dataset" + "/experiments/dataset/era5_data/2m_temperature/*.nc",
    "./experiments/dataset"
    + "/experiments/dataset/era5_data/10m_u_component_of_wind/*.nc",
    "./experiments/dataset"
    + "/experiments/dataset/era5_data/10m_v_component_of_wind/*.nc",
    "./experiments/dataset" + "/experiments/dataset/era5_data/v_component_of_wind/*.nc",
    "./experiments/dataset" + "/experiments/dataset/era5_data/u_component_of_wind/*.nc",
    "./experiments/dataset" + "/experiments/dataset/era5_data/relative_humidity/*.nc",
    "./experiments/dataset" + "/experiments/dataset/era5_data/specific_humidity/*.nc",
]

const_info_path = [
    cwd + "/experiments/dataset/era5_data/constants/constants_5.625deg.nc"
]

levels = ["z", "t", "t2m", "u10", "v10"]
paths_to_data = paths_to_data[0:5]
levels = levels[0:5]
assert len(paths_to_data) == len(
    levels
), "Paths to different type of data must be same as number of types of observations"


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


cache_dir = os.path.join("./experiments/dataset/era5_data", "cached_datasets")
os.makedirs(cache_dir, exist_ok=True)


def get_cache_filename(spectral):
    """Generate cache filename based on parameters"""
    return f"climate_data_spectral_{spectral}.pt"


# Check if cached datasets exist
cache_filename = get_cache_filename(args.spectral)
cache_path = os.path.join(cache_dir, cache_filename)

if os.path.exists(cache_path):
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


print("Length of training data", len(Final_train_data))
print("Length of validation data", len(Final_val_data))
print("Length of testing data", len(Final_test_data))
const_channels_info, lat_map, lon_map = add_constant_info(const_info_path)

if args.spectral == 1:
    print("############## Running the Model in Spectral Domain ####################")
H, W = Final_train_data.shape[3], Final_train_data.shape[4]
clim = torch.mean(Final_test_data, dim=0)
Test_loader = DataLoader(Final_test_data[2:], batch_size=args.batch_size, shuffle=False)
time_loader = DataLoader(time_steps[2:], batch_size=args.batch_size, shuffle=False)
time_idx_steps = torch.tensor([i for i in range(365 * 4)]).view(-1, 1)
time_idx = DataLoader(
    time_idx_steps[2:],
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)

# total_time_len = len(time_steps[2:])
# total_time_steps = time_steps[2:].numpy().flatten().tolist()
num_years = 2
Final_train_data = 0
Final_val_data = 0

mod_str = args.model

if "climode" in mod_str.lower():
    vel_test = torch.from_numpy(np.load(f"{cache_dir}/test_10year_2day_mm_vel.npy"))

if mod_str != "data":
    model_dict = torch.load(mod_str, map_location=device, weights_only=False)
    exp_args = model_dict["args"]
    if "gb_dm" in mod_str:
        exp_args = model_dict["args"]
        model_state_dict = model_dict["model_state_dict"]
        model = Climate_VFM_ENC(
            num_channels=5,
            history_size=exp_args.get("history_size", 0),
            const_channels=2,
            out_types=5,
            method="dopri5",
            use_att=True,
            use_err=False,
            use_pos=False,
            use_history_attention=exp_args.get("use_history_attention", False),
        ).to(device)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()
    elif "climode" in mod_str.lower():
        exp_args = model_dict["args"]
        model_state_dict = model_dict["model_state_dict"]
        model = ClimODE(
            num_channels=5,
            const_channels=2,
            out_types=5,
            method=exp_args["solver"],
            use_att=exp_args["use_att"],
            use_err=exp_args["use_err"],
            use_pos=exp_args["use_pos"],
        )
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()


# print(model)

print(f"Model Path {mod_str}", flush=True)

org_time = 1
RMSE = []
RMSE_lat_lon = []
Pred = []
Truth = []
Lead_RMSE_arr = {
    "z": [[] for _ in range(args.batch_size - 1)],
    "t": [[] for _ in range(args.batch_size - 1)],
    "t2m": [[] for _ in range(args.batch_size - 1)],
    "u10": [[] for _ in range(args.batch_size - 1)],
    "v10": [[] for _ in range(args.batch_size - 1)],
}
Lead_ACC = {
    "z": [[] for _ in range(args.batch_size - 1)],
    "t": [[] for _ in range(args.batch_size - 1)],
    "t2m": [[] for _ in range(args.batch_size - 1)],
    "u10": [[] for _ in range(args.batch_size - 1)],
    "v10": [[] for _ in range(args.batch_size - 1)],
}
Lead_CRPS = {
    "z": [[] for _ in range(args.batch_size - 1)],
    "t": [[] for _ in range(args.batch_size - 1)],
    "t2m": [[] for _ in range(args.batch_size - 1)],
    "u10": [[] for _ in range(args.batch_size - 1)],
    "v10": [[] for _ in range(args.batch_size - 1)],
}

total_entries = len(time_loader)

for entry, (time_steps, batch) in enumerate(zip(time_loader, Test_loader)):
    data = batch[0].to(
        device
    )  # .view(num_years,1,len(paths_to_data)*(args.scale+1),H,W)

    t = time_steps.float().to(device).flatten()
    with torch.no_grad():
        if "climode" in mod_str.lower():
            init_velocity_sample = (
                vel_test[entry]
                .view(num_years, 2 * len(paths_to_data) * (args.scale + 1), H, W)
                .to(device)
            )
            model.update_param(
                [
                    init_velocity_sample,
                    const_channels_info.to(device),
                    lat_map.to(device),
                    lon_map.to(device),
                ]
            )
            if "climode" in mod_str.lower():
                mean_pred, _, ode_out = model(t, data.unsqueeze(1))
            else:
                mean_pred = model(t, data)
        elif "gb_dm" in mod_str:
            model.update_param_phys(
                [
                    torch.zeros(data.shape[0], 1).to(device),
                    const_channels_info.to(device),
                    lat_map.to(device),
                    lon_map.to(device),
                ]
            )
            if exp_args is not None and exp_args.get("history_size", 0) > 0:
                history_size = exp_args["history_size"]
                # repeat the first frame history_size times for initial history
                x_history = data.unsqueeze(1).repeat(1, history_size, 1, 1, 1)
            else:
                x_history = None
            mean_pred = model.predict_trajectory(
                batch[0].to(device),
                t_timesteps=t,
                x_history=x_history,
                solver_method="dopri5",
            )
        elif "data" in mod_str:
            mean_pred = data.repeat(args.batch_size, 1, 1, 1, 1)
    # mean_avg = mean_pred.view(-1,len(paths_to_data)*(args.scale+1),H,W)
    # std_avg = std_pred.view(-1,len(paths_to_data)*(args.scale+1),H,W)

    for yr in range(2):
        for t_step in range(1, len(time_steps), 1):  # ``
            msg = f"\rEntry {entry + 1}/{total_entries} | Year {yr} | Step {t_step}/{len(time_steps) - 1}"
            sys.stdout.write(msg)
            sys.stdout.flush()
            evaluate_rmsd = evaluation_rmsd_mm(
                mean_pred[t_step, yr, :, :, :].cpu(),
                batch[t_step, yr, :, :, :].cpu(),
                lat,
                lon,
                max_lev,
                min_lev,
                H,
                W,
                levels,
            )
            evaluate_acc = evaluation_acc_mm(
                mean_pred[t_step, yr, :, :, :].cpu(),
                batch[t_step, yr, :, :, :].cpu(),
                lat,
                lon,
                max_lev,
                min_lev,
                H,
                W,
                levels,
                clim[yr, :, :, :].cpu().detach().numpy(),
            )
            # evaluate_crps = evaluation_crps_mm(mean_pred[t_step,yr,:,:,:].cpu(),batch[t_step,yr,:,:,:].cpu(),lat,lon,max_lev,min_lev,H,W,levels,std_pred[t_step,yr,:,:,:].cpu())
            for idx, lev in enumerate(levels):
                Lead_RMSE_arr[lev][t_step - 1].append(evaluate_rmsd[idx])
                Lead_ACC[lev][t_step - 1].append(evaluate_acc[idx])
                # Lead_CRPS[lev][t_step-1].append(evaluate_crps[idx])


for t_idx in range(args.batch_size - 1):
    for idx, lev in enumerate(levels):
        print(
            "Lead Time ",
            (t_idx + 1) * 6,
            "hours ",
            "| Observable ",
            lev,
            "| Mean RMSE ",
            np.mean(Lead_RMSE_arr[lev][t_idx]),
            "| Std RMSE ",
            np.std(Lead_RMSE_arr[lev][t_idx]),
        )
        print(
            "Lead Time ",
            (t_idx + 1) * 6,
            "hours ",
            "| Observable ",
            lev,
            "| Mean ACC ",
            np.mean(Lead_ACC[lev][t_idx]),
            "| Std ACC ",
            np.std(Lead_ACC[lev][t_idx]),
        )
        # print("Lead Time ",(t_idx+1)*6, "hours ","| Observable ",lev, "| Mean CRPS ", np.mean(Lead_CRPS[lev][t_idx]), "| Std CRPS ", np.std(Lead_CRPS[lev][t_idx]))
