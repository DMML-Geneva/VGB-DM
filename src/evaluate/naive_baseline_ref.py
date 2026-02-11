import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import numpy as np


from src.data.tensor_dataset.dataset_factory import get_dataset
from torch.utils.data import DataLoader

from src.evaluate.evaluate import compute_scores




def evaluate_exp(
    val_loader=None,
    dataset_path="./experiments/dataset/RLC/test_size_100.pkl",
    exp_name = "rlc",
    max_length: int = 25,
    seed=3,
):
    exp_config = {}
    exp_config["dataset"] = {}
    exp_config["dataset"]["data_path_tr"] = dataset_path
    exp_config["dataset"]["data_path_va"] = dataset_path
    exp_config["name_exp"] = exp_name
    exp_config["sampler"]= {} 
    exp_config["sampler"]["max_length"] = max_length
    exp_config["vf_model"] = {}
    


    _, val_dataset = get_dataset(exp_config, only_testset=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    device = torch.device("cpu")
    for batch in val_loader:
        batch = [x.to(device) for x in batch]

        x_traj, t_traj, params = (
            batch[0],
            batch[1],
            batch[2],
        )  # shape (n, memory_length, dim)

        x_in, t_in = (
            (x_traj[:, :max_length, :], t_traj[:, :max_length])
            if x_traj.shape[1] > max_length
            else (x_traj, t_traj)
        )
        # future trajectory
        x_f, t_f = x_traj[:, max_length:, :], t_traj[:, max_length:]

        x_last = x_in[:, -1, :]
        x_last = x_last.flatten(start_dim=1).unsqueeze(1).repeat(1, t_f.shape[1], 1)
        x_last = x_last.reshape_as(x_f)
        res = compute_scores(X=x_f, Y=x_last,metrics=["mse", "mae", "n_mse"])

    # compute the mean 
    score = {}
    for key in res:
        if key not in score:
            score[key] = []

        score[key].append(
            res[key].cpu().numpy()
        )
        

    print(f"Evaluation scores: {score}")
    return score


# Write argparse to take model path and dataset path from command line
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./experiments/dataset/reactdiff/test/3e7cf06d9323caaea6c4f566b7d328f2/dataset_3e7cf06d9323caaea6c4f566b7d328f2.pt",
        help="Path to the dataset for evaluation.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="reactdiff",
        help="Name of the experiment (e.g., pendulum, rlc).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=5,
        help="Maximum length of the input sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Random seed for evaluation.",
    )
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    scores = evaluate_exp(
        dataset_path=args.dataset_path,
        exp_name=args.exp_name,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
