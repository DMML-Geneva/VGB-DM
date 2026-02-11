import torch
from src.utils.torch_utils import vmap_3d, vmap_2d


def lagrange_interpolant(y_points, t_points, q_t):
    """
    Fully vectorized Lagrange interpolation optimized for vector-valued functions.
    Specifically tailored for y_points of shape [n, d] or [d, n] where each point has a d-dimensional vector.

    Args:
        y_points: Tensor of y coordinates of the samples [n, d] or [d, n]
        q_t: Tensor of time query points, at which to evaluate the interpolation [time_batch].
        t_points: Tensor of t time samples samples [n], t_points \in R

    Returns:
        Tensor of interpolated values at x [batch_size, d]
    """
    # Check input dimensions
    n_points = len(t_points)
    Y = y_points
    if y_points.ndim == 1:
        Y = y_points.unsqueeze(1)  # Ensure y_points is 2D
    if y_points.shape[0] != n_points:
        Y = y_points.T
        if Y.shape[0] != n_points:
            raise ValueError("Number of x and y points must be the same")

    # Reshape x for broadcasting: [batch_size] -> [batch_size, 1]
    q_t_reshaped = q_t.view(-1, 1)

    # Create a difference matrix: [batch_size, n]
    diff_matrix = q_t_reshaped - t_points

    # Create pairwise differences between x_points: [n, n]
    x_diffs = t_points.view(-1, 1) - t_points.view(1, -1)

    # Replace diagonal elements with 1s to avoid division by zero
    x_diffs = x_diffs.clone()
    x_diffs[torch.eye(n_points, dtype=torch.bool, device=t_points.device)] = (
        1.0
    )

    # Create a mask to exclude diagonal elements
    mask = ~torch.eye(n_points, dtype=torch.bool, device=t_points.device)

    # Expand diff_matrix to [batch_size, n, n]
    expanded_diffs = diff_matrix.unsqueeze(1).expand(-1, n_points, -1)

    # Replace entries where j == i with 1.0
    expanded_diffs_masked = expanded_diffs.clone()
    expanded_diffs_masked[
        :, torch.eye(n_points, dtype=torch.bool, device=t_points.device)
    ] = 1.0

    # Compute the numerators: [batch_size, n]
    numerators = torch.prod(expanded_diffs_masked, dim=2)

    # Compute the denominators: [n]
    x_diffs_masked = x_diffs[mask].view(n_points, n_points - 1)
    denominators = torch.prod(x_diffs_masked, dim=1)

    # Compute the Lagrange basis functions: [batch_size, n]
    basis_functions = numerators / denominators

    # Special case for 2D y_points: direct and efficient computation
    # Use einsum for the vector case - very efficient for [n, d] case
    # basis_functions: [batch_size, n], y_points: [n, d] → result: [batch_size, d]
    interpolated_values = torch.einsum("bn,nd->bd", basis_functions, Y)

    return interpolated_values


def lagrange_dt_3p(y_points, t_points, qt):
    """
    Lagrange interpolation for the first derivative at a point qt
    using three points.
    Args:
        y_points: Tensor of y coordinates of the samples[d, n] or [n, d]
        t_points: Tensor of t time samples samples [n], t_points \in R
        qt: Tensor of time query points, at which to evaluate the interpolation [time_batch]
    Returns:
        Tensor of interpolated values at x [batch_size, d]
    """
    n_points = len(t_points)
    Y = y_points
    if y_points.ndim == 1:
        Y = y_points.unsqueeze(1)  # Ensure y_points is 2D
    if y_points.shape[-1] != n_points:
        Y = y_points.T
        if Y.shape[-1] != n_points:
            raise ValueError("Number of x and y points must be the same")

    t0, t1, t2 = t_points
    y0, y1, y2 = Y[..., [0]], Y[..., [1]], Y[..., [2]]
    denom0 = (t0 - t1) * (t0 - t2)
    denom1 = (t1 - t0) * (t1 - t2)
    denom2 = (t2 - t0) * (t2 - t1)

    term0 = (2 * qt - t1 - t2) / denom0
    term1 = (2 * qt - t0 - t2) / denom1
    term2 = (2 * qt - t0 - t1) / denom2

    res = y0 * term0 + y1 * term1 + y2 * term2
    return res.T


def lagrange_dt2_3p(y_points, t_points):
    """
    Lagrange interpolation for the second derivative at a point qt
    using three points.
    Args:
        y_points: Tensor of y coordinates of the samples [d, n] or [n, d]
        t_points: Tensor of t time samples samples [n], t_points \in R
    Returns:
        Tensor of interpolated values at x [batch_size, d]
    """
    n_points = len(t_points)
    Y = y_points
    if y_points.ndim == 1:
        Y = y_points.unsqueeze(1)  # Ensure y_points is 2D
    if y_points.shape[-1] != n_points:
        Y = y_points.T
        if Y.shape[-1] != n_points:
            raise ValueError("Number of x and y points must be the same")

    t0, t1, t2 = t_points
    y0, y1, y2 = Y[..., [0]], Y[..., [1]], Y[..., [2]]
    denom0 = (t0 - t1) * (t0 - t2)
    denom1 = (t1 - t0) * (t1 - t2)
    denom2 = (t2 - t0) * (t2 - t1)

    term0 = 2 / denom0
    term1 = 2 / denom1
    term2 = 2 / denom2

    res = y0 * term0 + y1 * term1 + y2 * term2
    return res.T


def make_gamma3p_t():
    gamma = (
        lambda t_points, qt: (qt - t_points[0])
        * (qt - t_points[1])
        * (qt - t_points[2])
    )
    dt_f = (
        lambda t_points, qt: (qt - t_points[1]) * (qt - t_points[2])
        + (qt - t_points[0]) * (qt - t_points[2])
        + (qt - t_points[0]) * (qt - t_points[1])
    )
    dt2_f = lambda t_points, qt: 6 * t - 2 * (
        t_points[0] + t_points[1] + t_points[2]
    )
    return gamma, dt_f, dt2_f


def true_function_damped(x):
    f = torch.exp(-0.1 * x) * torch.sin(x)
    df = -0.1 * torch.exp(-0.1 * x) * torch.sin(x) + torch.exp(
        -0.1 * x
    ) * torch.cos(x)
    d2f = (0.01 - 1) * torch.exp(-0.1 * x) * torch.sin(x) - 0.2 * torch.exp(
        -0.1 * x
    ) * torch.cos(x)
    return f, df, d2f


if __name__ == "__main__":
    # Example usage
    h = 1
    batch_size = 45
    T_points = []
    X = []
    all_X = []
    all_dt = []
    all_dt2 = []
    n_qt = 100
    Q_t = []
    for i in range(batch_size):
        t = torch.rand(1) * 10
        t = t.item()
        t_points = torch.tensor([t - h, t, t + h])
        # 3 points
        x_points = torch.tensor([true_function_damped(x)[0] for x in t_points])
        # add new column of random noise dimension
        x_points = torch.cat(
            (x_points.unsqueeze(1), (x_points * 2).unsqueeze(1)), dim=1
        )

        X.append(x_points)
        T_points.append(t_points)
        if n_qt > 1:
            qt = torch.linspace(t_points[0], t_points[-1], n_qt)
        else:
            qt = (t_points[-1] + t_points[0]) / 2
        qt = torch.linspace(t_points[0], t_points[-1], n_qt)
        f_X, dt_X, dt2_X = true_function_damped(qt)
        f_X = torch.cat((f_X.unsqueeze(1), (f_X * 2).unsqueeze(1)), dim=1)
        all_X.append(f_X)
        all_dt.append(dt_X)
        all_dt2.append(dt2_X)
        Q_t.append(qt)

    all_X = torch.atleast_3d(torch.stack(all_X))

    all_dt = torch.stack(all_dt)
    all_dt2 = torch.stack(all_dt2)
    X = torch.stack(X)
    T_points = torch.stack(T_points)
    Q_t = torch.stack(Q_t)

    # Calculate the values of the true function and the Taylor expansions
    it_fun, dt_fun, dt2_fun = (
        vmap_3d(lagrange_interpolant),
        vmap_3d(lagrange_dt_3p),
        vmap_2d(lagrange_dt2_3p),
    )
    # X.shape = [batch_size, 3, 2]
    # T_points.shape = [batch_size, 3]
    # Q_t.shape = [batch_size, n_qt]
    it, dt, dt2 = (
        it_fun(X, T_points, Q_t),
        dt_fun(X, T_points, Q_t),
        dt2_fun(X, T_points),
    )

    # Assert interpolation
    mid_point = (n_qt // 2) - 1
    assert torch.isclose(
        it[:, [0, mid_point, -1]], all_X[:, [0, mid_point, -1]], atol=1e-2
    ).all()

    # assert derivative dt=0.2 randomly chosen
    # assert torch.isclose(dt[:, [24]], all_dt[:, [24]], atol=0.1).all()
