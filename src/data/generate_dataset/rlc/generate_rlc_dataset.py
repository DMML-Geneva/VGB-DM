#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

from src.simulators.rlc.rlc_circuit import RLCCircuit
import torch
from tqdm import tqdm
import os
import pickle


def gen_data(
    n=100,
    timesteps=200,
    T0=0.0,
    T1=20.0,
    irregular_sampling=False,
    n_rnd_points=50,
):
    distributions = {
        "R": lambda x: torch.distributions.Uniform(1.0, 3.0).sample((x,)),
        "L": lambda x: torch.distributions.Uniform(1.0, 3.0).sample((x,)),
        "C": lambda x: torch.distributions.Uniform(0.5, 1.5).sample((x,)),
    }
    s = RLCCircuit()
    s.n_timesteps = timesteps
    s.T0 = T0
    s.T1 = T1
    if irregular_sampling:
        dataset = torch.zeros(n, n_rnd_points, 1, 2)
        timestamps = torch.zeros(n, n_rnd_points)
        full_dataset = torch.zeros(n, s.n_timesteps, 1, 2)
        full_timestamps = torch.zeros(n, s.n_timesteps)
    else:
        dataset = torch.zeros(n, s.n_timesteps, 1, 2)
        full_dataset = torch.zeros(n, s.n_timesteps, 1, 2)
        timestamps = torch.zeros(n, s.n_timesteps)
        full_timestamps = torch.zeros(n, s.n_timesteps)
    true_param = {
        "R": torch.zeros(n),
        "L": torch.zeros(n),
        "C": torch.zeros(n),
        "V_a": torch.zeros(n),
        "V_c": torch.zeros(n),
        "omega": torch.zeros(n),
    }
    for i in tqdm(range(n)):
        true_param["R"][i] = distributions["R"](1)[0]
        true_param["L"][i] = distributions["L"](1)[0]
        true_param["C"][i] = distributions["C"](1)[0]
        true_param["V_a"][i] = 2.5
        true_param["V_c"][i] = 1.0
        true_param["omega"][i] = 2.0
        t, x = s.sample_sequences({x: t[i] for x, t in true_param.items()})
        if irregular_sampling:
            rnd_indices = torch.randperm(s.n_timesteps)[:n_rnd_points]
            selected_x = x[rnd_indices, :].clone()
            selected_t = t[rnd_indices].clone()
            selected_t, sorted_indices = torch.sort(selected_t)
            selected_x = selected_x[sorted_indices, :]
            dataset[i, :, :, :] = selected_x
            timestamps[i, :] = selected_t
        
        full_dataset[i, :, :, :] = x
        full_timestamps[i, :] = t

    full_dataset = full_dataset.permute(0, 1, 3, 2).unsqueeze(3)
    if not irregular_sampling:
        dataset = full_dataset.clone()
        timestamps = full_timestamps.clone()
    else:
        dataset = dataset.permute(0, 1, 3, 2).unsqueeze(3)

    return timestamps, dataset, true_param, full_timestamps, full_dataset


import argparse

if __name__ == "__main__":
    path = "experiments/dataset/RLC"
    if not os.path.exists(path):
        os.makedirs(path)
    # define a parser for train and test size
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tr_size", type=int, default=1000, help="Training size"
    )
    parser.add_argument(
        "--val_size", type=int, default=100, help="Validation size"
    )
    parser.add_argument(
        "--test_size", type=int, default=100, help="Test size"
    )
    parser.add_argument(
        "--timesteps", type=int, default=200, help="Number of timesteps"
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
        default=25,
        help="Number of random points for irregular sampling",
    )

    with torch.no_grad():
        args = parser.parse_args()
        with open(
            r"%s/train_size_%d_%s.pkl" % (path, args.tr_size, f"irregular_{args.n_rnd_points}" if args.irregular_sampling else "regular"), "wb"
        ) as output_file:
            pickle.dump(
                gen_data(
                    args.tr_size,
                    timesteps=args.timesteps,
                    irregular_sampling=args.irregular_sampling,
                    n_rnd_points=args.n_rnd_points,
                ),
                output_file,
            )
        if not args.irregular_sampling:
            with open(
                r"%s/valid_size_%d.pkl" % (path, args.val_size), "wb"
            ) as output_file:
                pickle.dump(gen_data(args.val_size), output_file)

            with open(
                r"%s/test_size_%d.pkl" % (path, args.test_size), "wb"
            ) as output_file:
                pickle.dump(gen_data(args.test_size), output_file)
