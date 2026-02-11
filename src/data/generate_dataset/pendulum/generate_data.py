import torch
from torch import nn
import numpy as np

import argparse
import json
from src.io import utils as io_utils
import time

from src.simulators.pendulum.pendulum_model import PendulumSolver
from src.simulators.pendulum.options import PendulumOptionParser
from src.sampler.distributions import dist_sampler

from src.utils import hash

from scipy.integrate import solve_ivp

import logging


def generate_traj_scipy(
    init_cond,
    omega,
    gamma,
    A,
    f,
    noise_std,
    dt,
    len_episode,
    rng,
    method="DOP853",
):
    A_val = A if A is not None else 0.0
    gamma_val = gamma if gamma is not None else 0.0
    phi_val = f if f is not None else 0.0

    # ODE function
    def fun(t, s):
        th, thdot = s
        force = A_val * omega * omega * np.cos(2.0 * np.pi * phi_val * t)
        return [thdot, force - gamma_val * thdot - omega * omega * np.sin(th)]

    start_t = time.perf_counter()
    sol = solve_ivp(
        fun,
        (0.0, dt * (len_episode - 1)),
        init_cond,
        dense_output=True,
        method=method,
    )
    t = np.linspace(0.0, dt * (len_episode - 1), len_episode)
    x = sol.sol(t).T
    sol_ode = x.copy()
    x = x[:, 0]
    end_t = time.perf_counter()
    if noise_std is not None:
        x = x + rng.normal(loc=0.0, scale=noise_std, size=x.shape)

    return t, x, sol_ode, (end_t - start_t) + 1e-9


def generate_data(args):
    # Set the CUDA device
    device_name = args.device
    device = torch.device(device_name)
    if device_name == "cuda":
        if not torch.cuda.is_available():
            device = torch.device("cpu")

    # check inputs
    assert args.range_init[0] <= args.range_init[1]
    assert args.range_omega[0] <= args.range_omega[1]

    if args.range_gamma is not None:
        assert args.range_gamma[0] <= args.range_gamma[1]
    if args.range_A is not None:
        assert args.range_A[0] <= args.range_A[1]
    if args.range_f is not None:
        assert args.range_f[0] <= args.range_f[1]

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("Pendulum Data Generation")
    logger.info(f"\nArguments: {args}")

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # set random seed
    rng = np.random.default_rng(args.seed)
    logger.info(f"Using random seed {args.seed}")

    fwd_model = PendulumSolver(
        args.len_episode,
        dt=args.dt,
        noise_loc=args.noise_loc,
        noise_std=args.noise_std,
        with_grad=False,
    )

    # Generate random uniform samples using torch of range_init, range_omega, range_gamma, range_A, range_f#

    # range_init: initial angle
    init_cond_samples = (args.range_init[1] - args.range_init[0]) * torch.rand(
        args.n_samples
    ) + args.range_init[0]
    init_cond_samples = init_cond_samples.reshape((args.n_samples, 1))
    # add zero vector to init_cond_samples
    init_cond_samples = torch.cat(
        (init_cond_samples, torch.zeros(args.n_samples, 1)), dim=1
    )

    # range_omega:
    omega_samples = (args.range_omega[1] - args.range_omega[0]) * torch.rand(
        args.n_samples
    ) + args.range_omega[0]

    # safety check for positive number of samples
    n_stoch_samples = args.n_stoch_samples if args.n_stoch_samples > 1 else 1
    gamma_samples = None


    if args.range_gamma is not None:
        if args.range_gamma[1] == args.range_gamma[0]:
            gamma_samples = (
                torch.ones(args.n_samples, n_stoch_samples)
                * args.range_gamma[0]
            )
            gamma_samples = gamma_samples.reshape(
                (args.n_samples, n_stoch_samples)
            )
        else:
            
            # n_samples \times  n_stoch_samples
            gamma_samples = dist_sampler(
                args.gamma_dist,
                args.range_gamma[0],
                args.range_gamma[1],
            ).sample((args.n_samples, n_stoch_samples))
            gamma_samples = gamma_samples.reshape((args.n_samples, -1))

        A_samples = None
        if args.range_A is not None:
            if args.range_A[1] == args.range_A[0]:
                A_samples = (
                    torch.ones(args.n_samples, n_stoch_samples)
                    * args.range_A[0]
                )
                A_samples = A_samples.reshape(
                    (args.n_samples, n_stoch_samples)
                )
            else:
                A_samples = dist_sampler(
                    args.A_dist,
                    args.range_A[0],
                    args.range_A[1],
                ).sample((args.n_samples, n_stoch_samples))
                A_samples = A_samples.reshape((args.n_samples, -1))

        f_samples = None
        if args.range_f is not None:
            if args.range_f[1] == args.range_f[0]:
                f_samples = (
                    torch.ones(args.n_samples, n_stoch_samples)
                    * args.range_f[0]
                )
                f_samples = f_samples.reshape(
                    (args.n_samples, n_stoch_samples)
                )
            else:
                
                f_samples = dist_sampler(
                    args.f_dist,
                    args.range_f[0],
                    args.range_f[1],
                ).sample((args.n_samples, n_stoch_samples))
                f_samples = f_samples.reshape((args.n_samples, -1))

    # Initialize trajectory (solutions, thdot) tensor
    data = torch.empty((args.n_samples, n_stoch_samples, args.len_episode))
    # ODE solutions solutions

    ts = torch.zeros((args.n_samples, n_stoch_samples, args.len_episode))
    func_eval_time_hist = torch.zeros((args.n_samples, n_stoch_samples))

    # Get true parameters dimension
    param_dim = 1
    if gamma_samples is not None:
        param_dim += 1
    if A_samples is not None:
        param_dim += 1
    if f_samples is not None:
        param_dim += 1
    # Initialize parameters tensor
    true_params = torch.empty((args.n_samples, n_stoch_samples, param_dim))
    logger.info(f"Parameters shape: {true_params.shape}")
    
    logger.info(f"Generating data using the forward model")
    for i in range(args.n_samples):
        for j in range(args.n_stoch_samples):
            if "library" in args and "torch" in args.library:
                params_sim = [omega_samples[i]]
                if gamma_samples is not None:
                    params_sim.append(gamma_samples[i, j])
                if A_samples is not None:
                    params_sim.append(A_samples[i, j])
                if f_samples is not None:
                    params_sim.append(f_samples[i, j])
                params_sim = torch.tensor(params_sim)
                fwd_res = fwd_model(
                    init_conds=init_cond_samples[i],
                    params=params_sim,
                )
                data[i, j] = fwd_res["x"].flatten()

                ts[i, j] = fwd_res["t"]
                func_eval_time_hist[i, j] = fwd_res["time_eval"]

            elif "library" in args and "scipy" in args.library:
                ts = ts.numpy()
                data = data.numpy()
                (
                    ts[i, j],
                    data[i, j],
                    func_eval_time_hist[i, j],
                ) = generate_traj_scipy(
                    init_cond_samples[i],
                    omega_samples[i],
                    gamma_samples[i, j] if gamma_samples is not None else None,
                    A_samples[i, j] if A_samples is not None else None,
                    f_samples[i, j] if f_samples is not None else None,
                    noise_std=args.noise_std,
                    dt=args.dt,
                    len_episode=args.len_episode,
                    rng=rng,
                )
                ts = torch.tensor(ts)
                data = torch.tensor(data)
            else:
                raise ValueError(
                    f"Library {args.library} not supported. Use 'torch' or 'scipy'"
                )

            # Filter the simulation that diverge, by checking that the trajectories is less or major than twice the external force, like ((c_data < -A * 2) | (c_data > A * 2)).sum()
            # check if the trajectory is diverging assign nans
            if (
                A_samples is not None
                and args.range_A[0] != 0.0
                and args.div_eps is not None
            ):
                if (
                    (
                        data[i, j]
                        < args.range_init[0] * args.range_A[0] * args.div_eps
                    )
                    | (
                        data[i, j]
                        > args.range_init[1] * args.range_A[1] * args.div_eps
                    )
                ).sum() > 0:
                    logger.warning(
                        f"Trajectory is likely to be diverging, setting to nan."
                    )
                    data[i, j] = torch.full_like(data[i, j], torch.nan)

            param_sample = [omega_samples[i]]
            if gamma_samples is not None:
                param_sample.append(gamma_samples[i, j])
            if A_samples is not None:
                param_sample.append(A_samples[i, j])
            if f_samples is not None:
                param_sample.append(f_samples[i, j])

            true_params[i, j] = torch.tensor(param_sample)

    # reshape  data, func_eval_time_hist to (n_samples * n_stoch_samples, len_episode)
    data = data.reshape(-1, args.len_episode)
    func_eval_time_hist = func_eval_time_hist.reshape(-1)
    ts = ts.reshape(-1, args.len_episode)
    if n_stoch_samples > 1:
        dim_init_cond = init_cond_samples.shape[1]
        init_cond_samples = init_cond_samples.repeat(1, n_stoch_samples)
        init_cond_samples = init_cond_samples.reshape(-1, dim_init_cond)
    true_params = true_params.reshape(-1, param_dim)

    # shuffle data
    if n_stoch_samples > 1 and args.shuffle_data:
        logger.info(f"Shuffling data")
        perm = torch.randperm(args.n_samples * args.n_stoch_samples)
        data = data[perm]
        ts = ts[perm]
        init_cond_samples = init_cond_samples[perm]
        true_params = true_params[perm]
        func_eval_time_hist = func_eval_time_hist[perm]

    dic_data = {
        "sims": data,
        "init_conds": init_cond_samples,
        "params": true_params,
        "t": ts,
    }

    # save data
    if args.outdir is not None or str(args.outdir).lower() != "none":
        # check if directory exists and create it if not
        io_utils.create_dir_if_ne(args.outdir)

        all_args = vars(args)
        keys = PendulumOptionParser.HASHED_KEYS
        keys = list(filter(lambda key: key in all_args, keys))
        hash_keys = {key: all_args[key] for key in keys}
        args_exp = {
            "all_args": all_args,
            PendulumOptionParser.HASHED_KEYS_NAME: hash_keys,
        }

        # compute the hash of the experiment and use it as the file name
        min_conf = PendulumOptionParser.get_min_dict(args)
        hashed_exp_name = hash.uuid_hash(min_conf)
        logger.info(f"Hashed experiment name: {hashed_exp_name}")

        args.filename = f"{hashed_exp_name}"

        # save args
        with open(
            "{}/args_{}.json".format(args.outdir, hashed_exp_name), "w"
        ) as f:
            json.dump(args_exp, f, sort_keys=True, indent=4)
            logger.info(
                f"Saved args to {args.outdir}/args_{hashed_exp_name}.json"
            )

        torch.save(dic_data, "{}/data_{}".format(args.outdir, hashed_exp_name))
        logger.info(f"Saved data to {args.outdir}/data_{hashed_exp_name}")
        logger.info(
            "Simulations range: min(abs(x))={}, max(abs(x))={}".format(
                torch.min(torch.abs(data), dim=1)[0],
                torch.max(torch.abs(data), dim=1)[0],
            )
        )

        if fwd_model.store_time_eval:
            torch.save(
                func_eval_time_hist,
                "{}/func_eval_time_{}".format(args.outdir, hashed_exp_name),
            )
            logger.info(
                f"Saved function evaluation time to {args.outdir}/func_eval_time_{hashed_exp_name}"
            )

        logger.info(
            "time per sample: {:.5f} sec, out of {} samples".format(
                torch.mean(func_eval_time_hist).item(),
                args.n_samples,
            )
        )
    return dic_data


if __name__ == "__main__":
    # Parse arguments
    parser = PendulumOptionParser()
    args = parser.parse_args()
    dic_result = generate_data(args)
    print("Data and true parameters generated successfully!")
