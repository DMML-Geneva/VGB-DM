import os
import pathlib
import yaml
import numpy as np
import torch
import pandas as pd
from src.simulators.reactdiff.reactdiff_options import (
    ReactDiffOptionParser,
)
from src.simulators.reactdiff.reactdiff_model import ReactDiffSolver

from src.utils import hash
import json

exptname = "src/data/generate_dataset/reactdiff/"
output_dir = os.path.join("experiments/dataset/reactdiff/")


def sample_init_cond_params(
    size,
    stoch_samples=0,
    conf_data=None,
):
    """
    Sample initial conditions and parameters for the reaction-diffusion model.
    """

    a = (
        torch.rand(size, 1, device=device)
        * (conf_data["a_range"][1] - conf_data["a_range"][0])
        + conf_data["a_range"][0]
    )  # a shape (batch, 1)
    b = (
        torch.rand(size, 1, device=device)
        * (conf_data["b_range"][1] - conf_data["b_range"][0])
        + conf_data["b_range"][0]
    )  # b shape (batch, 1)

    k = conf_data["k"]  # currently one-to-one mapping

    if isinstance(k, list):
        # random sample from k[0], k[1] bounds

        if stoch_samples > 0:
            k = (
                torch.rand(size, stoch_samples, device=device) * (k[1] - k[0])
                + k[0]
            ).unsqueeze(-1)
        else:
            k = (
                torch.rand(size, 1, device=device) * (k[1] - k[0]) + k[0]
            ).unsqueeze(-1)
        a = a.unsqueeze(1).repeat(1, stoch_samples, 1)
        b = b.unsqueeze(1).repeat(1, stoch_samples, 1)
    else:
        k = torch.tensor(k, device=device).expand(size, 1)

    params = torch.cat([a, b, k], dim=-1)  # shape (batch, 3)
    init_cond = torch.rand(
        size,
        2,
        conf_data["grid_size"],
        conf_data["grid_size"],
        device=device,
    )  # initial conditions shape (batch, 2, grid_size, grid_size), all uniform random in [0, 1]
    # batchify and solve

    if stoch_samples > 0:
        init_cond = init_cond.unsqueeze(1).repeat(1, stoch_samples, 1, 1, 1)
        init_cond = init_cond.reshape(
            -1, 2, conf_data["grid_size"], conf_data["grid_size"]
        )
        params = params.reshape(-1, 3)

    return init_cond, params


if __name__ == "__main__":
    # Load configuration
    with open(os.path.join(exptname, "params.yaml"), "r") as fd:
        params_data = yaml.safe_load(fd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # seed
    seed = params_data["seed"]
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)

    len_episode = torch.arange(
        params_data["t_span"][0],
        params_data["t_span"][1] + params_data["dt"],
        params_data["dt"],
    ).shape[0]

    reactdiff_solver = ReactDiffSolver(
        t0=params_data["t_span"][0],
        t1=params_data["t_span"][1],
        dx=params_data["mesh_step"],
        noise_std=params_data["sigma"],
        dt=params_data["dt"],
        ode_stepsize=params_data["ode_stepsize"],
        method=params_data["ode_solver"],
        device=device,
    )

    max_samples = params_data["n_samples"]
    stoch_samples = params_data["n_stoch_samples"]

    mini_batch_size = (
        params_data["batch_size"] if "batch_size" in params_data else -1
    )
    if mini_batch_size == -1:
        mini_batch_size = max_samples

    res = {
        "x": [],  # shape: (n_samples, 2, grid_size, grid_size, len_episode)
        "sol": [],
        "init_conds": [],
        "params": [],
        "t": [],
        "time_eval": [],
    }

    print(f"Mini batch size: {mini_batch_size}")

    # Name experiment and configuration
    min_conf = ReactDiffOptionParser.get_min_dict(params_data)
    params_data["all_args"] = params_data.copy()

    hashed_exp_name = hash.uuid_hash(min_conf)
    print(f"Hashed experiment name: {hashed_exp_name}")
    filename = f"{hashed_exp_name}"

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = os.path.join(output_dir, f"{filename}")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    params_data["filename"] = filename
    params_data["batch_size"] = mini_batch_size
    params_data["all_args"]["filename"] = filename

    data_annotations = []
    j = 0
    res = {
        "x": [],
        "init_conds": [],
        "params": [],
        "t": [],
        "time_eval": [],
    }
    for i in range(0, max_samples, mini_batch_size):
        print(f"Processing {i} to {i + mini_batch_size}")
        x_batch, params_batch = sample_init_cond_params(
            size=mini_batch_size,
            stoch_samples=stoch_samples,
            conf_data=params_data,
        )

        if (
            i + mini_batch_size > max_samples
        ):  # e.g if max_samples=1000 and i=960, then i+mini_batch_size=1024
            x_batch = x_batch[: max_samples - i]
            params_batch = params_batch[: max_samples - i]

        with torch.no_grad():
            batch_res = reactdiff_solver(
                init_conds=x_batch,
                params=params_batch,
            )

        if stoch_samples > 0:
            if not batch_res["params"][:, -1].all():
                batch_res["params"] = batch_res["params"][:, :-1]
            # reshape back
            res["x"].append(
                batch_res["x"].reshape(
                    -1,
                    len_episode,
                    2,
                    params_data["grid_size"],
                    params_data["grid_size"],
                )
            )

            res["params"].append(batch_res["params"].reshape(-1, 3))
            # res["init_conds"].append(
            #    batch_res["init_conds"].reshape(
            #        -1,
            #        2,
            #        params_data["grid_size"],
            #        params_data["grid_size"],
            #    )
            # )
            t_times = (
                batch_res["t"]
                .unsqueeze(0)
                .repeat(batch_res["params"].shape[0], 1)
            )
            res["t"].append(t_times)
        else:
            # if last column all zeros, remove it
            if not batch_res["params"][:, -1].all():
                batch_res["params"] = batch_res["params"][:, :-1]
            res["x"].append(batch_res["x"])
            res["params"].append(batch_res["params"])
            t_times = (
                batch_res["t"]
                .unsqueeze(0)
                .repeat(batch_res["params"].shape[0], 1)
            )
            res["t"].append(t_times)

    # Concatenate results and save
    res["x"] = torch.cat(res["x"], dim=0)
    res["params"] = torch.cat(res["params"], dim=0)
    # res["init_conds"] = torch.cat(res["init_conds"], dim=0)
    res["t"] = torch.cat(res["t"], dim=0)
    with open(os.path.join(output_dir, f"dataset_{filename}.pt"), "wb") as f:
        torch.save(res, f)

    # save the configuration
    with open(os.path.join(output_dir, f"args_{filename}.json"), "w") as fd:
        json.dump(params_data, fd, indent=4)
