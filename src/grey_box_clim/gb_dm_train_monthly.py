from src.grey_box_clim.fm_phys_func import Climate_VFM_Monthly
from src.grey_box_clim.model_function import Optim_velocity

from src.grey_box_clim.utils import *
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
torch.cuda.empty_cache()

import sys
import logging
import argparse
import time, yaml, os, hashlib
import wandb
from src.utils.wandb import wandb_util

logging.propagate = False
logging.getLogger().setLevel(logging.INFO)

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

parser.add_argument("--atol", type=float, default=5e-3)
parser.add_argument("--rtol", type=float, default=5e-3)
parser.add_argument(
    "--step_size", type=float, default=None, help="Optional fixed step size."
)
parser.add_argument("--spectral", type=int, default=0, choices=[0, 1])
parser.add_argument("--niters", type=int, default=2000)
parser.add_argument("--scale", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--val_batch_size", type=int, default=3)
parser.add_argument("--history_size", type=int, default=3)
parser.add_argument(
    "--use_history_attention",
    action="store_true",
    help="Use history attention mechanism",
    default=True,
)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument(
    "--initial_noise",
    type=float,
    default=0.01,
    help="Initial noise variance for annealing schedule.",
)
parser.add_argument(
    "--final_noise",
    type=float,
    default=0.001,
    help="Final noise variance for annealing schedule.",
)
parser.add_argument(
    "--noise_annealing_epochs",
    type=int,
    default=100,
    help="Number of epochs to anneal noise over.",
)

parser.add_argument("--lambda_grad", type=float, default=1.0)
parser.add_argument("--lambda_output", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument(
    "--warmup_epochs",
    type=int,
    default=-1,
    help="Number of warmup epochs before validation starts",
)
parser.add_argument(
    "--clip_grad_norm",
    type=float,
    default=1.0,
    help="Clip gradient norm to this value",
)
parser.add_argument(
    "--force_reload",
    action="store_true",
    default=False,
    help="Force reload data even if cached version exists",
)
parser.add_argument(
    "--time_budget",
    type=int,
    default=300,
    help="Time budget in minutes for training",
)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
logger = logging.getLogger("GB Matching")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)

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
        print(data)
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
        ) = get_train_test_data_without_scales_batched_monthly(
            data,
            train_time_scale,
            val_time_scale,
            test_time_scale,
            levels[idx],
            args.spectral,
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
    Final_train_data,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)
Val_loader = DataLoader(
    Final_val_data[2:],
    batch_size=args.val_batch_size,
    shuffle=False,
    pin_memory=False,
)
Test_loader = DataLoader(
    Final_test_data[2:],
    batch_size=args.val_batch_size,
    shuffle=False,
    pin_memory=False,
)
time_loader = DataLoader(
    time_steps, batch_size=args.batch_size, shuffle=False, pin_memory=False
)
val_time_loader = DataLoader(
    time_steps[2:],
    batch_size=args.val_batch_size,
    shuffle=False,
    pin_memory=False,
)
test_time_loader = DataLoader(
    time_steps[2:],
    batch_size=args.val_batch_size,
    shuffle=False,
    pin_memory=False,
)

num_years = (
    len(range(2006, 2016)) if torch.cuda.is_available() else len(range(2010, 2016))
)
model = Climate_VFM_Monthly(
    len(paths_to_data),
    args.history_size,
    2,
    out_types=len(paths_to_data),
    use_att=True,
    use_err=False,
    use_pos=False,
    use_history_attention=args.use_history_attention,
).to(device)
param = count_parameters(model)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.niters)

best_composite_score = float("inf")
best_epoch = float("inf")

print(
    "############################ Data is loaded, Fitting the velocity #########################"
)
kernel_path = cache_dir + "/kernel.npy"
if os.path.exists(kernel_path):
    kernel_np = np.load(kernel_path, allow_pickle=True)
else:
    kernel_np = get_gauss_kernel((32, 64), lat, lon)
    np.save(kernel_path, kernel_np)
kernel = torch.from_numpy(kernel_np)


time_idx_steps = torch.tensor([i for i in range(12)]).view(-1, 1)

time_idx = DataLoader(
    time_idx_steps[2:],
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)

fit_velocity(
    time_idx,
    val_time_loader,
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
    test_time_loader,
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

print(
    "############################ Velocity loaded, Model starts to train #########################"
)
print(model)
print(
    "####################### Total Parameters",
    param,
    "################################",
)
model.train()

mse_loss_fn = nn.MSELoss()

# initialize wandb
wandb_config_file = os.path.join(
    cwd, "experiments/scripts/configs/wandb/wandb_config.yaml"
)
with open(wandb_config_file, "r") as f:
    wandb_config = yaml.safe_load(f)

if wandb_config["run_wandb_log"] == False:
    wandb_config = {}

args_str = str(vars(args))
hashed_args = hashlib.md5(args_str.encode()).hexdigest()[:8]
hashed_exp_name = f"{hashed_args}"


exp_config = {
    "algorithm": "GB-DM-ClimODE",
    "model": "Climate_VFM_ENC",
    "group_exp_name": "climate_monthly_phys_exp",
    "model_exp_name": f"fm_phys_bs{args.batch_size}_lr{args.lr}_{hashed_exp_name}",
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
        "model_phys": model,
    }
    wandb_util.watch(
        models_to_watch,
        log=wandb_config.get("wandb_watch_log", "all"),
        log_freq=wandb_config.get("log_freq_batches", 20),
    )
    logger.info("Wandb watch initialized")


def to_device(*tensors, device):
    return [t.to(device) for t in tensors]


training_start_time = time.perf_counter()


VAL_INTERVAL = 5
WARMUP_EPOCHS = args.warmup_epochs if hasattr(args, "warmup_epochs") else -1

for epoch in range(args.niters):
    # --- Noise Annealing ---
    if args.initial_noise > 0:
        if epoch < args.noise_annealing_epochs:
            # Linearly anneal from initial_noise to final_noise
            current_noise = args.initial_noise - (
                args.initial_noise - args.final_noise
            ) * (epoch / args.noise_annealing_epochs)
        else:
            current_noise = args.final_noise
        model.var_noise = current_noise
    else:
        # Use fixed noise if not annealing
        model.var_noise = args.var_noise

    epoch_start_time = time.perf_counter()
    total_train_loss = 0
    total_fm_loss = 0
    total_v_loss = 0
    total_l2_loss = 0
    total_grad_reg_loss = 0
    total_output_reg_loss = 0
    mse_val_loss = 0.0
    model.train()

    # ---- Training ----
    last_train_loss = None
    for entry, (time_steps, batch) in enumerate(zip(time_loader, Train_loader)):
        optimizer.zero_grad()
        seq_len = batch.shape[0]

        x_history = [] if args.history_size > 0 else None
        if args.history_size > 0:
            num_windows = batch.shape[0] - args.history_size
            for i in range(args.history_size):
                history_step = batch[i : i + num_windows]
                x_history.append(history_step)

            x_history = torch.stack(x_history, dim=1).to(
                device
            )  # [T, B, hist_size, C, H, W]

        # x0 is the state at the end of the history window
        if args.history_size > 0:
            x0 = batch[args.history_size - 1 : -1].to(device)
        else:
            x0 = batch[:-1].to(device)
        # x1 is the state one step after x0
        if args.history_size > 0:
            x1 = batch[args.history_size :].to(device)
        else:
            x1 = batch[1:].to(device)
        n_x0 = x0.shape[0]

        # Reshape for loss function
        if args.history_size > 0:
            x_history = x_history.reshape(
                -1,
                args.history_size,
                len(paths_to_data) * (args.scale + 1),
                H,
                W,
            )
        x0 = x0.reshape(-1, len(paths_to_data) * (args.scale + 1), H, W)
        x1 = x1.reshape(-1, len(paths_to_data) * (args.scale + 1), H, W)

        t = time_steps.float().to(device).flatten()
        if args.history_size > 0:
            t0, t1 = t[args.history_size - 1 : -1], t[args.history_size :]
        else:
            t0, t1 = t[:-1], t[1:]
        t0 = t0.unsqueeze(1).repeat(1, batch.shape[1]).view(-1)
        t1 = t1.unsqueeze(1).repeat(1, batch.shape[1]).view(-1)

        model.update_param_phys(
            [
                torch.zeros(n_x0, 1),
                const_channels_info.to(device),
                lat_map.to(device),
                lon_map.to(device),
            ]
        )

        (
            combined_loss,
            fm_loss,
            v_loss,
            x_t,
            v_target,
            t_fm,
            output_reg,
            grad_reg,
            l2_norm,
        ) = model.flow_matching_loss(
            x0=x0,
            x1=x1,
            t0=t0,
            t1=t1,
            x_history=x_history,
            lambda_output=args.lambda_output,
            lambda_grad=args.lambda_grad,
        )

        combined_loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.clip_grad_norm
            )
        optimizer.step()

        if torch.isnan(combined_loss):
            print("Quitting due to NaN loss")
            quit()

        # Accumulate loss components
        total_train_loss += combined_loss.item()
        total_fm_loss += fm_loss.item()
        total_v_loss += v_loss.item()
        total_l2_loss += l2_norm
        total_grad_reg_loss += grad_reg.item()
        total_output_reg_loss += output_reg.item()
        last_train_loss = combined_loss.item()

        # carriage return update (does not add a new line)
        sys.stdout.write(
            f"\rEpoch {epoch}, "
            f"Train Batch {entry}: "
            f"Loss = {combined_loss.item():.3e}, "
            f"FM Loss = {fm_loss.item():.3e}, "
            f"V Loss = {v_loss.item():.3e}, "
            f"L2 Loss = {l2_norm:.3e}, "
            f"time= {(time.perf_counter() - epoch_start_time) / 60.0:.2f} min"
        )
        sys.stdout.flush()

    lr_val = scheduler.get_last_lr()[0]
    scheduler.step()

    # Log epoch-level training metrics
    avg_train_loss = total_train_loss / len(Train_loader)
    avg_fm_loss = total_fm_loss / len(Train_loader)
    avg_v_loss = total_v_loss / len(Train_loader)

    avg_l2_loss = total_l2_loss / len(Train_loader)
    avg_grad_reg_loss = total_grad_reg_loss / len(Train_loader)
    avg_output_reg_loss = total_output_reg_loss / len(Train_loader)

    logger.info(
        f"Epoch {epoch} | Train Loss = {last_train_loss:.3e} "
        f"| Avg Train Loss = {avg_train_loss:.3e} "
        f"| Avg FM Loss = {avg_fm_loss:.3e} "
        f"| Avg L2 Loss = {avg_l2_loss:.3e}"
    )

    if wandb.run is not None:
        log_data = {
            "epoch": epoch,
            "train/avg_total_loss": float(avg_train_loss),
            "train/avg_fm_loss": float(avg_fm_loss),
            "train/avg_v_loss": float(avg_v_loss),
            "train/avg_l2_loss": float(avg_l2_loss),
            "train/avg_grad_reg_loss": float(avg_grad_reg_loss),
            "train/avg_output_reg_loss": float(avg_output_reg_loss),
            "train/lr": float(lr_val),
        }
        if args.initial_noise > 0:
            log_data["train/var_noise"] = model.var_noise
        wandb.log(log_data)

    # ---- Validation ----
    acc_timesteps = []
    mse_timesteps = []
    mse = ""
    acc = ""

    last_val_loss = float("inf")
    if epoch % VAL_INTERVAL == 0 and epoch >= WARMUP_EPOCHS:
        # Initialize storage for per-timestep, per-observable metrics
        Lead_ACC_val = {
            lev: [[] for _ in range(args.val_batch_size - 1)] for lev in levels
        }
        Lead_RMSD_val = {
            lev: [[] for _ in range(args.val_batch_size - 1)] for lev in levels
        }
        total_val_loss = 0
        model.eval()

        for entry, (time_steps, batch) in enumerate(zip(val_time_loader, Val_loader)):
            # Validation batch [B, T, C, H, W]

            # Prepare history and initial state for validation
            if args.history_size > 0:
                x0 = (
                    batch[0]
                    .view(-1, len(paths_to_data) * (args.scale + 1), H, W)
                    .to(device)
                )
                x_history = x0.unsqueeze(1).repeat(1, args.history_size, 1, 1, 1)
            else:
                x_history = None
                x0 = (
                    batch[0]
                    .view(-1, len(paths_to_data) * (args.scale + 1), H, W)
                    .to(device)
                )

            model.update_param_phys(
                [
                    torch.zeros(x0.shape[0], 1).to(device),
                    const_channels_info.to(device),
                    lat_map.to(device),
                    lon_map.to(device),
                ]
            )

            t_val = time_steps.float().squeeze().to(device)
            t_val = torch.atleast_1d(t_val)

            with torch.no_grad():
                # Predict trajectory starting from the end of the history window
                pred = model.predict_trajectory(
                    x0, t_val, x_history=x_history, solver_method="dopri5"
                )

            # Ground truth for comparison starts from the same point
            truth_val = batch.to(device)

            mse_loss = mse_loss_fn(pred, truth_val)

            if torch.isnan(mse_loss):
                logger.warning(
                    f"\nNaN encountered at batch {entry} during validation, skipping this batch"
                )
                continue

            mse_val_loss += mse_loss.item()
            total_val_loss += mse_loss.item()
            last_val_loss = mse_loss.item()

            for t_step in range(pred.shape[0]):
                for yr in range(0, pred.shape[1]):
                    pred2 = pred[t_step, yr, :, :, :].cpu()
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

            # carriage return update for validation
            sys.stdout.write(
                f"\rEpoch {epoch}, Val Batch {entry}: "
                f"Loss = {mse_loss.item():.3e}, "
                f"time= {(time.perf_counter() - epoch_start_time) / 60.0:.2f} min"
            )
            sys.stdout.flush()

        # After all batches, compute per-timestep, per-observable averages
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

            # Format for this timestep (matching your original format)
            acc = f"{acc_values[0]:.4f}, {acc_values[1]:.4f}, {acc_values[2]:.4f}, {acc_values[3]:.4f}, {acc_values[4]:.4f}"
            mse = f"{mse_values[0]:.4f}, {mse_values[1]:.4f}, {mse_values[2]:.4f}, {mse_values[3]:.4f}, {mse_values[4]:.4f}"

            # print(f"Lead Time {(t_idx + 1) * 6} hours | Acc = {acc} | MSE = {mse}")

        # Final summary
        sys.stdout.write(
            f"\rEpoch {epoch} "
            f"| Last Train Loss = {last_train_loss:.3e} "
            f"| Total Train Loss = {total_train_loss:.3e} "
            f"| Last Val Loss = {last_val_loss:.3e} "
            f"| Total Val Loss = {mse_val_loss:.3e} "
            f"| RMSE = {mse} "
            f"| ACC = {acc} "
            f"| Tot-time {(time.perf_counter() - training_start_time) / 60:.2f}\n"
        )

        sys.stdout.flush()

        if mse_val_loss > 0:
            avg_val_loss = mse_val_loss / len(Val_loader)
            overall_mse = np.mean(mse_values)
        else:
            avg_val_loss = float("inf")
            overall_mse = float("inf")

        # logger info for validation  with avg val mse traj, loss acc, mse overall
        logger.info(
            f"---------------- Validation Epoch {epoch} ----------------\n"
            f"Epoch {epoch} | Avg Val MSE Traj = {avg_val_loss:.3e} "
            f"| Last Val Loss = {last_val_loss:.3e} "
            f"| Acc Overall = {np.mean(acc_values):.3e} "
            f"| MSE Overall = {overall_mse:.3e}"
        )

        # Log validation metrics to wandb
        log_dict = {
            "epoch": epoch,
            "val/avg_val_mse_traj": float(avg_val_loss),
            "val/last_val_loss": float(last_val_loss),
            "val/acc_overall": float(np.mean(acc_values)),
            "val/mse_overall": float(overall_mse),
        }
        for i, lvl in enumerate(levels):
            log_dict[f"val/acc_{lvl}"] = float(acc_values[i])
            log_dict[f"val/mse_{lvl}"] = float(mse_values[i])
        if wandb.run is not None:
            # Log per-variable metrics
            wandb.log(log_dict)

        # Compute weighted composite score for checkpoint saving
        # Weight: 0.5 for loss, 0.5 for MSE (adjust as needed)
        composite_score = avg_val_loss * 0.5 + 0.5 * overall_mse

        # Save checkpoint if composite score improved
        if composite_score < best_composite_score:
            best_composite_score = composite_score
            best_epoch = epoch
            os.makedirs(OUT_CHKPT_DIR, exist_ok=True)
            ckpt_path = OUT_CHKPT_DIR + f"/gb_dm_best_monthly_{hashed_exp_name}.pt"
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
                    }
                )

    epoch_time_min = (time.perf_counter() - epoch_start_time) / 60.0
    total_time_min = (time.perf_counter() - training_start_time) / 60.0

    avg_train_loss = total_train_loss / len(Train_loader)

    # Check time budget
    elapsed_time_minutes = (time.perf_counter() - training_start_time) / 60
    if elapsed_time_minutes >= args.time_budget:
        logger.info(
            f"Time budget of {args.time_budget} minutes reached. Stopping training at epoch {epoch}."
        )
        break

# save also the final model at the end of training also in wandb
os.makedirs(OUT_CHKPT_DIR, exist_ok=True)
ckpt_path = OUT_CHKPT_DIR + f"/gb_dm_monthly_end_{hashed_exp_name}.pt"
dict_to_save = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "args": vars(args),
}
torch.save(dict_to_save, ckpt_path)
logger.info(f"Final model checkpoint saved at epoch {epoch} to {ckpt_path}")
wandb.save(ckpt_path)
