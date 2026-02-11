import torch
import torch.utils.data as data
import numpy as np
from typing import Tuple, Optional, Union

from src.data.dataset_utils import splits_trajectory


class PendulumDataset(data.Dataset):
    """
    PyTorch Dataset for RLC time series data.

    Depending on the mode, each sample is:
    - 'pairs': a tuple (x_k, x_{k+1}) from a trajectory.
    - 'trajectory': the full trajectory of shape (n_timesteps, n_features).
    """

    def __init__(
        self,
        data_path: str,
        mode: str = "seq-pairs",
        sample_params: bool = False,
        max_length: Optional[int] = None,
        n_cons_points: int = 1,
        history_size: int = 0,
        sampler_config: dict = None,
        device: str = "cpu",
    ):
        """
        Args:
            data_path: Path to the .pt file containing the dataset
            mode: 'seq-pairs' to sample consecutive points, 'trajectory' to sample full trajectories
        """

        self.sampler_config = sampler_config
        dataset = torch.load(
            data_path, weights_only=False, map_location=device
        )
        x = dataset["sims"]
        x = x.reshape(-1, x.shape[1], 1)
        self.dim_state = x.shape[2]
        self.X = x
        # self.X = x.flatten(start_dim=2, end_dim=-1)  # shape: (n, length, dim*grid_size*grid_size)
        self.timestamps = dataset["t"]
        self.params = dataset.get("params", None)[:, :1]  # p_dim=1
        self.sample_params = sample_params

        self.max_length = max_length

        self.n_trajectories, self.n_timesteps = self.X.shape[:2]
        self.n_cons_points = n_cons_points

        if (
            self.max_length is not None
            and self.n_timesteps > self.max_length
            and self.max_length > 0
        ):
            self.X, self.timestamps, _ = splits_trajectory(
                self.n_timesteps, self.max_length, self.X, self.timestamps
            )
            self.n_trajectories, self.n_timesteps = self.X.shape[:2]

        if self.n_timesteps < self.n_cons_points + 1:
            raise ValueError(
                f"Need at least {self.n_cons_points + 1} time steps to sample consecutive points"
            )

        if mode not in ["pairs", "trajectory", "seq-pairs", "pairs-history"]:
            raise ValueError("mode must be either 'pairs' or 'trajectory'")

        self.mode = mode
        self.history_size = history_size

        if self.mode == "pairs":
            # Precompute all possible (trajectory_idx, time_idx) pairs
            self.index_pairs = [
                (traj_idx, time_idx)
                for traj_idx in range(self.n_trajectories)
                for time_idx in range(self.n_timesteps - self.n_cons_points)
            ]

    def __len__(self) -> int:
        if self.mode == "pairs":
            return len(self.index_pairs)
        else:  # 'trajectory'
            return self.n_trajectories

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.mode == "pairs":
            traj_idx, time_idx = self.index_pairs[idx]
            traj = self.X[traj_idx]
            time = self.timestamps[traj_idx]
            x_k = traj[time_idx]
            x_k_plus_1 = traj[time_idx + 1]
            t_k = time[time_idx].squeeze()
            t_k_plus_1 = time[time_idx + 1].squeeze()
            return x_k, x_k_plus_1, traj, t_k, t_k_plus_1, time

        elif self.mode == "seq-pairs":
            # return all concatenated pairs from a trajectory by flattening, idx is the trajectory index
            # it should return x_k, x_k_plus_1, traj
            traj = self.X[idx]
            time = self.timestamps[idx]
            if self.n_cons_points == 1:
                x_k = traj[:-1]
                x_k_plus_1 = traj[1:]
                t_k = time[:-1]
                t_k_plus_1 = time[1:]
                return x_k, x_k_plus_1, traj, t_k, t_k_plus_1, time
            else:
                x_pairs = [traj[: -self.n_cons_points]]
                t_pairs = [time[: -self.n_cons_points]]
                for i in range(1, self.n_cons_points + 1):
                    if i == self.n_cons_points:
                        x_pairs.append(traj[i:])
                        t_pairs.append(time[i:])
                    else:
                        x_pairs.append(traj[i : -(self.n_cons_points - i)])
                        t_pairs.append(time[i : -(self.n_cons_points - i)])
                return (
                    torch.stack(
                        x_pairs, dim=1
                    ),  # shape (n_timesteps-n_cons_points, n_cons_points-1, dim)
                    torch.stack(t_pairs, dim=1),
                    traj,
                    time,
                )  # shape (n_timesteps, dim)
        elif self.mode == "pairs-history":
            # randomly sample an index inside the trajectory; satisfying history_size constraint
            traj = self.X[idx]
            times = self.timestamps[idx]
            enc_len_episode = self.sampler_config.get("enc_len_episode", None)
            if enc_len_episode is not None and enc_len_episode > 0:
                traj_length = enc_len_episode
                if traj_length <= self.history_size:
                    traj_length = self.history_size
            else:
                traj_length = self.history_size
            n_pairs_per_traj = self.sampler_config.get("n_pairs_per_traj", 1)
            start_idx = np.random.randint(
                low=traj_length,
                high=self.n_timesteps - self.n_cons_points,
                size=n_pairs_per_traj,
            )

            traj_windows = traj.unfold(0, traj_length, 1).permute(
                0, 2, 1
            )  # shape:(T-history_size+1, traj_length, dim)
            # Index with start_idx - history_size to get the correct windows
            x_traj = traj_windows[start_idx - traj_length]
            x_history = x_traj[:, -self.history_size :]

            if self.n_cons_points == 1:
                x_k = traj[start_idx]
                x_k_plus_1 = traj[start_idx + 1]
                t_k = times[start_idx]
                t_k_plus_1 = times[start_idx + 1]
                return (
                    x_k,
                    x_k_plus_1,
                    x_traj,
                    t_k,
                    t_k_plus_1,
                    times,
                    x_history,
                )
            else:
                idx_pairs = [
                    start_idx + i for i in range(0, self.n_cons_points + 1)
                ]
                idx_pairs = np.stack(idx_pairs, axis=1).flatten()
                x_pairs = traj[idx_pairs].reshape(
                    n_pairs_per_traj, self.n_cons_points + 1, -1
                )
                t_pairs = times[idx_pairs].reshape(
                    n_pairs_per_traj, self.n_cons_points + 1, -1
                )
                return (
                    x_pairs,
                    t_pairs,
                    x_traj,
                    times,
                    x_history,
                )  # shape (n_pairs_per_traj, n_cons_points, dim)
        else:  # 'trajectory'
            if self.sample_params and self.params is not None:
                return (self.X[idx], self.timestamps[idx], self.params[idx])
            else:
                return (
                    self.X[idx],
                    self.timestamps[idx],
                )  # shape: (n_timesteps, n_features)

    def get_data_statistics(self) -> dict:
        return {
            "shape": self.X.shape,
            "mean": self.X.mean(dim=(0, 1)),
            "std": self.X.std(dim=(0, 1)),
            "min": self.X.min(),
            "max": self.X.max(),
            "n_trajectories": self.n_trajectories,
            "n_timesteps": self.n_timesteps,
            "dim_state": self.dim_state,
            "grid_size": self.grid_size,
            "n_features": self.dim_state * self.grid_size * self.grid_size,
            "total_pairs": (
                len(self.index_pairs) if self.mode == "pairs" else "N/A"
            ),
        }
