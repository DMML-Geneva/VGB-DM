import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Dataset that returns consecutive trajectory pairs (x0, x1) from climate data.

    Args:
        data: Tensor of shape (time_steps, years, channels, height, width)
        time_steps: Tensor of time values corresponding to each trajectory
    """

    def __init__(self, data, time_steps):
        """
        Initialize the trajectory dataset.

        Args:
            data: Climate data tensor of shape (time_steps, years, channels, height, width)
            time_steps: Time steps tensor of shape (time_steps, 1)
        """
        self.data = data
        self.time_steps = time_steps

        # Ensure we have at least 2 time steps for consecutive pairs
        if len(data) < 2:
            raise ValueError(
                "Data must have at least 2 time steps for consecutive pairs"
            )

    def __len__(self):
        """Return the number of consecutive pairs available."""
        return len(self.data) - 1

    def __getitem__(self, idx):
        """
        Get a consecutive pair of trajectories.

        Args:
            idx: Index of the first trajectory in the pair

        Returns:
            tuple: (time_step, (x0, x1)) where:
                - time_step: Time value for the transition
                - x0: Current trajectory at time idx
                - x1: Next trajectory at time idx+1
        """
        if idx >= len(self) or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        x0 = self.data[idx]  # Current trajectory
        x1 = self.data[idx + 1]  # Next trajectory

        # Return the time step and the trajectory pair
        time_step = self.time_steps[idx]  # Time step for the transition to x1

        return time_step, x0, x1


# Usage example for your specific case:
def create_trajectory_datasets(
    Final_train_data, Final_val_data, Final_test_data, time_steps, batch_size
):
    """
    Create trajectory datasets for training, validation, and testing.

    Args:
        Final_train_data, Final_val_data, Final_test_data: Climate data tensors
        time_steps: Time steps tensor
        batch_size: Batch size for DataLoader

    Returns:
        tuple: (train_loader, val_loader, test_loader) - DataLoaders for trajectory pairs
    """
    from torch.utils.data import DataLoader

    # Adjust indices based on your current logic
    if torch.cuda.is_available():
        start_idx = 0
    else:
        start_idx = 1442

    # Create datasets
    train_dataset = TrajectoryDataset(
        Final_train_data[start_idx:], time_steps[start_idx:]
    )

    val_dataset = TrajectoryDataset(Final_val_data[start_idx:], time_steps[start_idx:])

    test_dataset = TrajectoryDataset(
        Final_test_data[start_idx:], time_steps[start_idx:]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False
    )

    return train_loader, val_loader, test_loader
