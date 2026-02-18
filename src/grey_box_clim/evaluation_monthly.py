import os
from src.grey_box_clim.utils import (
    load_datasets,
    set_seed,
    add_constant_info,
    evaluation_rmsd_mm,
    evaluation_acc_mm,
)
from torch.utils.data import DataLoader
import numpy as np
import matplotlib

matplotlib.use("Agg")
import argparse
import torch

from src.grey_box_clim.fm_phys_func import Climate_VFM_Monthly
from src.grey_box_clim.model_function import (
    Climate_encoder_free_uncertain_monthly,
)

set_seed(42)
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
parser.add_argument("--atol", type=float, default=5e-3)
parser.add_argument("--rtol", type=float, default=5e-3)
parser.add_argument("--batch_size", type=int, default=7)
parser.add_argument(
    "--model",
    type=str,
    default="src/grey_box_clim/Models/bests/gb_dm_monthly_7a303583.pt",
)
parser.add_argument(
    "--step_size", type=float, default=None, help="Optional fixed step size."
)
parser.add_argument("--teacher", type=int, default=1, choices=[0, 1])
parser.add_argument("--scale", type=int, default=0)
parser.add_argument("--days", type=int, default=3)
parser.add_argument("--spectral", type=int, default=0, choices=[0, 1])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()
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
]
pdir = str(cwd)
const_info_path = [
    pdir + "/experiments/dataset/era5_data/constants/constants_5.625deg.nc"
]
levels = ["z", "t", "t2m", "u10", "v10", "v", "u", "r", "q"]
paths_to_data = paths_to_data[0:5]
levels = levels[0:5]
assert len(paths_to_data) == len(
    levels
), "Paths to different type of data must be same as number of types of observations"

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
) = load_datasets(
    "./experiments/dataset/era5_data/cached_datasets/climate_data_monthly_spectral_0.pt"
)
print("Cached datasets loaded successfully!")

print("Length of training data", len(Final_train_data))
print("Length of validation data", len(Final_val_data))
print("Length of testing data", len(Final_test_data))
const_channels_info, lat_map, lon_map = add_constant_info(const_info_path)

if args.spectral == 1:
    print("############## Running the Model in Spectral Domain ####################")
H, W = 32, 64
clim = torch.mean(Final_test_data, dim=0)
Test_loader = DataLoader(Final_test_data[2:], batch_size=args.batch_size, shuffle=False)
time_loader = DataLoader(time_steps[2:], batch_size=args.batch_size, shuffle=False)
time_idx_steps = torch.tensor([i for i in range(12)]).view(-1, 1)
time_idx = DataLoader(
    time_idx_steps[2:],
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)

otal_time_len = len(time_steps[2:])
total_time_steps = time_steps[2:].numpy().flatten().tolist()
num_years = 2

vel_test = torch.from_numpy(
    np.load("./experiments/dataset/era5_data/cached_datasets/test_monthly_vel.npy")
)
model_str = args.model
if model_str != "data":
    model_dict = torch.load(
        model_str,
        map_location=device,
        weights_only=False,
    )
    if "gb_dm" in model_str:
        exp_args = model_dict["args"]
        model_state_dict = model_dict["model_state_dict"]
        model = Climate_VFM_Monthly(
            num_channels=5,
            history_size=exp_args.get("history_size", 0),
            const_channels=2,
            out_types=5,
            use_att=True,
            use_err=False,
            use_pos=False,
            use_history_attention=exp_args.get("use_history_attention", False),
        ).to(device)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()
    elif "climode" in model_str.lower():
        model_state_dict = model_dict["model_state_dict"]
        model = Climate_encoder_free_uncertain_monthly(
            len(paths_to_data),
            2,
            out_types=len(paths_to_data),
            method=args.solver,
            use_att=True,
            use_err=True,
            use_pos=False,
        )
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()

RMSE = []
RMSE_lat_lon = []
Pred = []
Truth = []

org_time = 1
RMSE = []
RMSE_lat_lon = []
Mean_pred = 0
Truth_pred = 0
Std_pred = 0
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
for entry, (time_steps, batch) in enumerate(zip(time_loader, Test_loader)):
    x0 = batch[0].to(device)
    past_sample = (
        vel_test[entry]
        .view(num_years, 2 * len(paths_to_data) * (args.scale + 1), H, W)
        .to(device)
    )

    t = time_steps.float().to(device).flatten()
    with torch.no_grad():
        if "climode" in model_str.lower():
            model.update_param(
                [
                    past_sample,
                    const_channels_info.to(device),
                    lat_map.to(device),
                    lon_map.to(device),
                ]
            )
            mean_pred, _, _ = model(t, x0.unsqueeze(1))
        elif "gb_dm" in model_str.lower():
            model.update_param_phys(
                [
                    past_sample,
                    const_channels_info.to(device),
                    lat_map.to(device),
                    lon_map.to(device),
                ]
            )
            if exp_args is not None and exp_args.get("history_size", 0) > 0:
                history_size = exp_args["history_size"]
                # repeat the first frame history_size times for initial history
                x_history = x0.unsqueeze(1).repeat(1, history_size, 1, 1, 1)
            else:
                x_history = None
            mean_pred = model.predict_trajectory(
                x0.to(device), t_timesteps=t, x_history=x_history
            )
        elif "data" in model_str.lower():
            mean_pred = x0.repeat(args.batch_size, 1, 1, 1, 1)

    for yr in range(2):
        for t_step in range(1, len(time_steps), 1):
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

            for idx, lev in enumerate(levels):
                Lead_RMSE_arr[lev][t_step - 1].append(evaluate_rmsd[idx])
                Lead_ACC[lev][t_step - 1].append(evaluate_acc[idx])


for t_idx in range(args.batch_size - 1):
    for idx, lev in enumerate(levels):
        print(
            "Lead Time ",
            (t_idx + 1),
            "Month ",
            "| Observable ",
            lev,
            "| Mean RMSE ",
            np.mean(Lead_RMSE_arr[lev][t_idx]),
            "| Std RMSE ",
            np.std(Lead_RMSE_arr[lev][t_idx]),
        )
