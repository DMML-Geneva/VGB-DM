import torch


class GenerativeDataloader:
    def __init__(self, generator, batch_size, n_iterations):
        self.generator = generator
        self.batch_size = batch_size
        self.num_iterations = n_iterations

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        self.i += 1
        if self.i > self.num_iterations:
            raise StopIteration
        else:
            return self.generator(n=(self.batch_size,))


def batchify_sequence(x: torch.Tensor, window_size: int = 3) -> torch.Tensor:
    """
    Splits a tensor of shape (batch_size, dim_state, seq_len) into
    batches of sequences with length window_size, discarding any remaining
    elements at the end of the sequence if seq_len is not a multiple
    of window_size.

    Args:
        x (torch.Tensor): The input tensor with shape (batch_size, dim_state, seq_len).
        window_size (int): The size of the sequence window (default is 3).

    Returns:
        torch.Tensor: The batchified tensor with shape
                      (new_batch_size, dim_state, window_size),
                      where new_batch_size = batch_size * (seq_len // window_size).
    """
    if x.ndim == 2:
        x = x.unsqueeze(1)

    batch_size, dim_state, seq_len = x.shape
    w = window_size

    # Calculate the number of full windows
    num_full_windows = seq_len // w

    # If there are no full windows, return an empty tensor with the correct shape
    if num_full_windows == 0:
        return torch.empty((0, dim_state, w), dtype=x.dtype, device=x.device)

    # Calculate the length covered by the full windows
    covered_len = num_full_windows * w

    x_new = x[..., :covered_len]

    # Slicing should maintain contiguity of the sliced dimension if the original tensor was contiguous.
    x_new = x_new.unfold(dimension=2, size=w, step=w)

    x_new = x_new.permute(0, 2, 1, 3)

    x_new = x_new.reshape(-1, dim_state, w)

    if covered_len != seq_len:
        # add Last window from the last sequence
        x_new = torch.cat((x_new, x[..., -w:]), dim=0)

    return x_new
