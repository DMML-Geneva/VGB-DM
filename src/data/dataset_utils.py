import torch
import numpy as np


def splits_trajectory(total_length, max_length, X, timestamps):
    # split the trajectories into segments of max_length
    d_dim_shape = X.shape[2:]
    n_segments = np.floor(total_length / max_length).astype(int)
    even_splits = total_length % max_length

    if even_splits > 0:
        last_X = X[:, -max_length:, :]
        last_timestamps = timestamps[:, -max_length:]
        max_offset = n_segments * max_length
        X = X[:, :max_offset, :]
        X = torch.cat([X[:, :max_offset], last_X], dim=1)
        timestamps = torch.cat(
            [timestamps[:, :max_offset], last_timestamps], dim=1
        )

    return (
        X.reshape(
            -1,
            max_length,
            *d_dim_shape,
        ),
        timestamps.reshape(
            -1,
            max_length,
        ),
        n_segments + (1 if even_splits > 0 else 0),
    )
