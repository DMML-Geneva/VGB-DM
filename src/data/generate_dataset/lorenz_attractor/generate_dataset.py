from typing import Any, Sequence

import torch
from torch import nn, Tensor
import numpy as np
from torchdyn.core import NeuralODE as ode_solver

import matplotlib.pyplot as plt
from tqdm import trange
import os
import json

from src.simulators.lorenz_attractor.lorenz_model import LorenzAttractorDynamics
from src.utils.hash import dict_hash

def generate_lorenz_dataset(
    n_trajectories: int=1000,
    dt: float=0.025,
    tmin: float=0.0,
    tmax: float=4.0,
    seed: int = 42,
    irregular_sampling=False,
    n_rnd_points=50,
    output_path: str = ".",
) -> Sequence[Tensor]:
    # set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.makedirs(output_path, exist_ok=True)
    
    time_points = torch.arange(tmin, tmax, dt)
    n_points = len(time_points)
    
    lorenz_dynamics = LorenzAttractorDynamics(params=torch.zeros(1, 3), full=True)
    node = ode_solver(
                lorenz_dynamics,
                sensitivity="adjoint",
                solver="dopri5",
                atol=1e-10,
                rtol=1e-10,
            )
    time_points = torch.linspace(tmin, tmax, n_points)
    full_traj = []
    trajs = []
    parameters = []
    full_ts = []
    ts = []


    #init_cond = sample_initial_conditions()
    with torch.no_grad():
        for _ in trange(n_trajectories):
            params = lorenz_dynamics.sample_parameters()
            params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
            lorenz_dynamics.params = params_tensor
            init_cond = lorenz_dynamics.sample_initial_conditions()
            
            init_tensor = torch.tensor(init_cond, dtype=torch.float32).unsqueeze(0)
            traj = node.trajectory(init_tensor, time_points)
            full_traj.append(traj.squeeze(0).numpy())
            full_ts.append(time_points.numpy())
            parameters.append(params)
        
        
    # make hash out of the arguments used to generate the dataset
    dict_args = {
        "n_trajectories": n_trajectories,
        "dt": dt,
        "n_points": n_points,
        "tmin": tmin,
        "tmax": tmax,
        "seed": seed,
    }
    hashed_exp = dict_hash(dict_args)
    # save dataset into pytorch file with hash as filename

    filename = f"lorenz_data_seed_{seed}_{hashed_exp}.pt"
    
    if irregular_sampling:
        filename = f"lorenz_data_irr_{n_rnd_points}_seed_{seed}_{hashed_exp}.pt"
        x_full = np.array(full_traj)
        t_full = np.array(full_ts)
        x = []
        ts = []
        for i in range(x_full.shape[0]):
            idxs = np.sort(
                np.random.choice(
                    n_points, size = (n_rnd_points), replace=False
                )
            )
            x.append(x_full[i, idxs, :])
            ts.append(t_full[i, idxs])
        
        x = np.array(x)
        ts = np.array(ts)
    else:
        x = np.array(full_traj)
        ts = np.array(full_ts)
        x_full = np.array(full_traj)
        t_full = np.array(full_ts)


    save_dict = {
        "exp_config": dict_args,
        "x": x,
        "t": ts,
        "x_full": x_full,
        "t_full": t_full,
        "params": np.array(parameters),
    }

    # print all shapes 
    print(f"x shape: {x.shape}")
    print(f"t shape: {ts.shape}")
    print(f"x_full shape: {x_full.shape}")
    print(f"t_full shape: {t_full.shape}")
    print(f"params shape: {np.array(parameters).shape}")

    torch.save(save_dict, f"{output_path}/{filename}")
    with open(f"{output_path}/config_seed_{seed}_{hashed_exp}.json", "w") as f:
        json.dump(dict_args, f, indent=4)

    print(f"Dataset saved at: {output_path}/{filename}")
    return full_traj, parameters, full_ts


# Parser arguments

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_trajectories", type=int, default=1000, help="Number of trajectories"
)
parser.add_argument(
    "--dt", type=float, default=0.0338, help="Time step" 
)
parser.add_argument(
    "--tmin", type=float, default=0.0, help="Minimum time"
)
parser.add_argument(
    "--tmax", type=float, default=2.0, help="Maximum time"
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed"
)
parser.add_argument(
    "--irregular_sampling",
    action="store_true",
    help="Whether to use irregular sampling",
    default=False,
)
parser.add_argument(
    "--n_rnd_points",
    type=int,
    default=50,
    help="Number of random points for irregular sampling",
)
parser.add_argument(
    "--output_path", type=str, default="./experiments/dataset/lorenz_attractor", help="Output path"
)

if __name__ == "__main__":

    args = parser.parse_args()

    tmin = args.tmin
    tmax = args.tmax
    dt = args.dt
    time_points = torch.arange(tmin, tmax, dt)
    n_points = len(time_points)

    generate_lorenz_dataset(
        n_trajectories=args.n_trajectories,
        dt=args.dt,
        tmin=tmin,
        tmax=tmax,
        seed=args.seed,
        irregular_sampling=args.irregular_sampling,
        n_rnd_points=args.n_rnd_points,
        output_path=args.output_path,
    )
    

    
    
    