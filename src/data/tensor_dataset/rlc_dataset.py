import torch
import torch.utils.data as data
import numpy as np
from typing import Tuple, Optional, Union


class RLCDataset(data.Dataset):
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
        history_size: int = 0,
        sampler_config: dict = None,
    ):
        """
        Args:
            data_path: Path to the .pt file containing the dataset
            mode: 'seq-pairs' to sample consecutive points, 'trajectory' to sample full trajectories
            sample_params: Whether to return physical parameters
            max_length: Maximum length of trajectory segments
            history_size: Number of previous points to include in history (for seq-pairs mode)
        """
        self.sampler_config = sampler_config
        dataset = np.load(data_path, allow_pickle=True)
        x = dataset[1]
        self.X = x.flatten(start_dim=2, end_dim=-1)  # shape: (n, length, 2)
        print(self.X.shape, x.shape)
        self.timestamps = dataset[0]
        self.params = torch.concatenate(
            [dataset[2][key].unsqueeze(1) for key in dataset[2].keys()], axis=1
        )
        self.sample_params = sample_params
        self.params_keys = list(dataset[2].keys())  # R, L, C, Va, Vc, omega
        # filter params for L and C
        self.params = self.params[:, [1, 2]]  # L and C

        if len(self.X.shape) != 3:
            raise ValueError(
                f"Expected 3D data (n, 11, 2), got shape {self.X.shape}"
            )

        self.max_length = max_length

        self.n_trajectories, self.n_timesteps, self.dim_state = self.X.shape
        if (
            self.max_length is not None
            and self.n_timesteps > self.max_length
            and self.max_length > 0
        ):
            # split the trajectories into segments of max_length
            n_segments = np.floor(self.n_timesteps / self.max_length).astype(
                int
            )
            even_splits = self.n_timesteps % self.max_length

            if even_splits > 0:
                last_X = self.X[:, -self.max_length :, :]
                last_timestamps = self.timestamps[:, -self.max_length :]
                max_offset = n_segments * self.max_length
                self.X = self.X[:, :max_offset, :]
                self.X = torch.cat([self.X[:, :max_offset], last_X], dim=1)
                self.timestamps = torch.cat(
                    [self.timestamps[:, :max_offset], last_timestamps], dim=1
                )

            self.X = self.X.reshape(
                -1,
                self.max_length,
                self.dim_state,
            )
            self.timestamps = self.timestamps.reshape(
                -1,
                self.max_length,
            )
            self.n_trajectories, self.n_timesteps, self.dim_state = (
                self.X.shape
            )

            # self.X = self.X[:, : self.max_length, :]
            # self.timestamps = self.timestamps[:, : self.max_length]
            # self.n_timesteps = self.max_length

        if self.n_timesteps < 2:
            raise ValueError(
                "Need at least 2 time steps to sample consecutive points"
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
                for time_idx in range(self.n_timesteps - 1)
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
            x_k = traj[time_idx]
            x_k_plus_1 = traj[time_idx + 1]
            times = self.timestamps[traj_idx]
            t_k = times[time_idx].squeeze()
            t_k_plus_1 = times[time_idx + 1].squeeze()
            return x_k, x_k_plus_1, traj, t_k, t_k_plus_1, times
        elif self.mode == "seq-pairs":
            # return all concatenated pairs from a trajectory by flattening, idx is the trajectory index
            traj = self.X[idx]
            times = self.timestamps[idx]
            x_k = traj[:-1]
            x_k_plus_1 = traj[1:]
            t_k = times[:-1]
            t_k_plus_1 = times[1:]
            return x_k, x_k_plus_1, traj, t_k, t_k_plus_1, times
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

            start_idx = np.random.randint(
                low=traj_length,
                high=self.n_timesteps - 1,
                size=self.sampler_config.get("n_pairs_per_traj", 1),
            )

            traj_windows = traj.unfold(0, traj_length, 1).permute(
                0, 2, 1
            )  # shape:(T-history_size+1, history_size, dim)
            # Index with start_idx - history_size to get the correct windows
            x_traj = traj_windows[start_idx - traj_length]
            x_history = x_traj[:, -self.history_size :]

            x_k = traj[start_idx]
            x_k_plus_1 = traj[start_idx + 1]
            t_k = times[start_idx]
            t_k_plus_1 = times[start_idx + 1]

            return x_k, x_k_plus_1, x_traj, t_k, t_k_plus_1, times, x_history
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
            "n_features": self.dim_state,
            "total_pairs": (
                len(self.index_pairs) if self.mode == "pairs" else "N/A"
            ),
        }


# Example usage
if __name__ == "__main__":
    dataset = RLCDataset(
        data_path="./experiments/dataset/RLC/train.pkl",
        mode="seq-pairs",  # Change to 'trajectory' to sample full sequences
    )

    dataloader = data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, prefetch_factor=2
    )

    # Dataset stats
    print("Dataset Statistics:")
    for k, v in dataset.get_data_statistics().items():
        print(f"{k}: {v}")

    print("\n" + "=" * 50)

    # One batch
    for batch in dataloader:
        if dataset.mode == "pairs":
            x_k, x_k_plus_1, X, t_k, t_k_plus_1 = batch
            print(
                f"x_k shape: {x_k.shape}, x_k_plus_1 shape: {x_k_plus_1.shape}"
            )
            print(f"Sample x_k[0]: {x_k[0]}")
            print(f"Sample x_k_plus_1[0]: {x_k_plus_1[0]}")
            print(f"X shape: {X.shape}")
            print(f"timestamps shape: {t_k.shape}, {t_k_plus_1.shape}")
        elif dataset.mode == "seq-pairs":
            x_k, x_k_plus_1, X, t_k, t_k_plus_1, times, x_history = batch
            print(
                f"x_k shape: {x_k.shape}, x_k_plus_1 shape: {x_k_plus_1.shape}"
            )
            print(f"X shape: {X.shape}")
            print(f"x_history shape: {x_history.shape}")
            print(f"timestamps shape: {t_k.shape}, {t_k_plus_1.shape}")
        else:
            trajectories, t = batch
            print(f"Trajectory batch shape: {trajectories.shape}")
            print(f"Sample trajectory[0]: {trajectories[0]}")
            print(f"Timestamps shape: {t.shape}")
            print(f"Sample timestamp[0]: {t[0]}")
        break

    print("\n" + "=" * 50)

    # Epoch loop
    print("Training loop (5 batches):")
    for i, batch in enumerate(dataloader):
        if dataset.mode == "pairs":
            x_k, x_k_plus_1, X, t_k, t_k_plus_1 = batch
            print(
                f"Batch {i+1}: x_k shape {x_k.shape}, x_k_plus_1 shape {x_k_plus_1.shape}"
            )
        elif dataset.mode == "seq-pairs":
            x_k, x_k_plus_1, X, t_k, t_k_plus_1, times, x_history = batch
            print(
                f"Batch {i+1}: x_k shape {x_k.shape}, x_k_plus_1 shape {x_k_plus_1.shape}"
            )
            print(f"X shape: {X.shape}")
            print(f"x_history shape: {x_history.shape}")
            print(f"Timestamps shape: {t_k.shape}, {t_k_plus_1.shape}")
        else:
            trajectories, t = batch
            print(f"Batch {i+1}: Trajectory batch shape: {trajectories.shape}")
        if i == 4:
            break
