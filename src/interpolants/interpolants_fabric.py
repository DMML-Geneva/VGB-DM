import numpy as np
import torch
import math

from src.utils.torch_utils import vmap_3d, vmap_2d
from src.interpolants.lagrange_interpolant import (
    make_gamma3p_t,
    lagrange_interpolant,
    lagrange_dt_3p,
    lagrange_dt2_3p,
)


def make_interpolant(path="linear"):
    dt2It = None
    if path == "linear":
        a = lambda t: (1 - t)
        adot = lambda t: -1.0
        b = lambda t: t
        bdot = lambda t: 1.0
        It = lambda x, t1, qt: a(qt) * x[:, 0] + b(qt) * x[:, 1]
        dtIt = lambda x, t1, qt: adot(qt) * x[:, 0] + bdot(qt) * x[:, 1]
    elif path == "lagrange":
        It, dtIt, dt2It = lagrange_interpolant, lagrange_dt_3p, lagrange_dt2_3p
        a = lambda t: 1.0
        adot = lambda t: 0
        b = lambda t: 1.0
        bdot = lambda t: 0.0

    elif path == "encoding-decoding":

        a = lambda t: torch.where(
            t <= 0.5, torch.cos(math.pi * t) ** 2, torch.tensor(0.0)
        )
        adot = lambda t: torch.where(
            t <= 0.5,
            -2 * math.pi * torch.cos(math.pi * t) * torch.sin(math.pi * t),
            torch.tensor(0.0),
        )
        b = lambda t: torch.where(t > 0.5, torch.cos(math.pi * t) ** 2, 0.0)
        bdot = lambda t: torch.where(
            t > 0.5,
            -2 * math.pi * torch.cos(math.pi * t) * torch.sin(math.pi * t),
            torch.tensor(0.0),
        )
        It = lambda x, t1, qt: a(qt) * x[0] + b(qt) * x[1]
        dtIt = lambda x, t1, qt: adot(qt) * x[0] + bdot(qt) * x[1]

    elif path == "custom":
        return None, None, None

    else:
        raise NotImplementedError(
            "The interpolant you specified is not implemented."
        )
    It = vmap_3d(It)
    dtIt = vmap_3d(dtIt)
    if dt2It is not None:
        dt2It = vmap_2d(dt2It)

    return It, dtIt, (a, adot, b, bdot), dt2It


def make_gamma(gamma_type="brownian", a=None):
    """
    returns callable functions for gamma, gamma_dot,
    and gamma(t_points, t)*gamma_dot(t) to avoid numerical divide by 0s,
    e.g. if one is using the brownian (default) gamma.
    """
    if gamma_type == "brownian":
        gamma = lambda t_points, t: torch.sqrt(t * (1 - t))
        gamma_dot = lambda t_points, t: (1 / (2 * torch.sqrt(t * (1 - t)))) * (
            1 - 2 * t
        )
        gg_dot = lambda t_points, t: (1 / 2) * (1 - 2 * t)

    elif gamma_type == "a-brownian":
        gamma = lambda t_points, t: torch.sqrt(a * t * (1 - t))
        gamma_dot = (
            lambda t: (1 / (2 * torch.sqrt(a * t * (1 - t)))) * a * (1 - 2 * t)
        )
        gg_dot = lambda t_points, t: (a / 2) * (1 - 2 * t)

    elif gamma_type == "3points":
        gamma, gamma_dot, gg_dot = make_gamma3p_t()

    elif gamma_type == "zero":
        gamma = gamma_dot = gg_dot = lambda t_points, t: torch.zeros_like(t)

    elif gamma_type == "bsquared":
        gamma = lambda t_points, t: t * (1 - t)
        gamma_dot = lambda t_points, t: 1 - 2 * t
        gg_dot = lambda t_points, t: gamma(t_points, t) * gamma_dot(t)

    elif gamma_type == "sinesquared":
        gamma = lambda t_points, t: torch.sin(math.pi * t) ** 2
        gamma_dot = (
            lambda t: 2
            * math.pi
            * torch.sin(math.pi * t)
            * torch.cos(math.pi * t)
        )
        gg_dot = lambda t_points, t: gamma(t_points, t) * gamma_dot(t)

    elif gamma_type == "sigmoid":
        f = torch.tensor(10.0)
        gamma = (
            lambda t: torch.sigmoid(f * (t - (1 / 2)) + 1)
            - torch.sigmoid(f * (t - (1 / 2)) - 1)
            - torch.sigmoid((-f / 2) + 1)
            + torch.sigmoid((-f / 2) - 1)
        )
        gamma_dot = lambda t_points, t: (-f) * (
            1 - torch.sigmoid(-1 + f * (t - (1 / 2)))
        ) * torch.sigmoid(-1 + f * (t - (1 / 2))) + f * (
            1 - torch.sigmoid(1 + f * (t - (1 / 2)))
        ) * torch.sigmoid(
            1 + f * (t - (1 / 2))
        )
        gg_dot = lambda t_points, t: gamma(t_points, t) * gamma_dot(t)

    elif gamma_type == None:
        gamma = lambda t_points, t: torch.zeros(1)  ### no gamma
        gamma_dot = lambda t_points, t: torch.zeros(1)  ### no gamma
        gg_dot = lambda t_points, t: torch.zeros(1)  ### no gamma

    else:
        raise NotImplementedError(
            "The gamma you specified is not implemented."
        )

    return gamma, gamma_dot, gg_dot


def make_activation(act):
    if act == "elu":
        return torch.nn.ELU()
    if act == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif act == "elu":
        return torch.nn.ELU()
    elif act == "relu":
        return torch.nn.ReLU()
    elif act == "tanh":
        return torch.nn.Tanh()
    elif act == "sigmoid":
        return torch.nn.Sigmoid()
    elif act == "softplus":
        return torch.nn.Softplus()
    elif act == "silu":
        return torch.nn.SiLU()
    elif act == "Sigmoid2Pi":

        class Sigmoid2Pi(torch.nn.Sigmoid):
            def forward(self, input):
                return 2 * np.pi * super().forward(input) - np.pi

        return Sigmoid2Pi()
    elif act == "none" or act is None:
        return None
    else:
        raise NotImplementedError(f"Unknown activation function {act}")


def linear_interpolation(
    x0, x1, t, t1=None, t0=None, x_c=None, t_x_c=None, sigma=1e-4
):
    """
    Linear interpolation between two points x0 and x1 at time t.
    Args:
        x0: Tensor of shape [batch_size, dim] at time t0.
        x1: Tensor of shape [batch_size, dim] at time t1.
        t: Tensor of shape [batch_size, 1] representing the interpolation factor.
        t1: Optional tensor of shape [batch_size, 1] representing the time at which x1 is defined.
        t0: Optional tensor of shape [batch_size, 1] representing the time at which x0 is defined.
        x_c: Optional tensor of shape [batch_size, dim] representing additional points.
        t_x_c: Optional tensor of shape [batch_size, 1] representing the points at time at which x_c is defined.
    Returns:
        mu_t: Tensor of shape [batch_size, dim] representing the interpolated data at time t.
        x_t: Tensor of shape [batch_size, dim] representing the noisy interpolated data at time t.
        ut: Tensor of shape [batch_size, dim] representing the velocity at time t.
        t_ut: Tensor of shape [batch_size, 1] representing the time at which x_t and ut is defined.
        torch.zeros_like(ut): No second-order derivatives are computed in this case.
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)

    t_broadcast = t.reshape(*t.shape, *([1] * (x0.ndim - t.ndim)))
    mu_t = x0 * (1 - t_broadcast) + x1 * t_broadcast  # data interpolation

    if sigma > 0:
        noise = torch.randn_like(x0) * sigma
        x_t = mu_t + noise
    else:
        x_t = mu_t

    ut = x1 - x0  #

    if t1 is not None and t0 is not None:
        t0, t1 = t0.unsqueeze(1) if t0.dim() == 1 else t0, (
            t1.unsqueeze(1) if t1.dim() == 1 else t1
        )

        dt = t1 - t0
        t_ut = t * dt + t0
        next_t = t1 - t_ut
        dt = dt + 1e-4
        dt = dt.reshape(*dt.shape, *([1] * (x0.ndim - dt.ndim)))
        ut = ut / dt

    return mu_t, x_t, ut, t_ut, torch.zeros_like(ut)


lag_It, lag_dtIt, (a, adot, b, bdot), lag_dt2It = make_interpolant("lagrange")


def lagrange_interpolation(x0, x1, t, t1, t0, x_c=None, t_x_c=None, sigma=0.0):
    """
    Lagrange interpolation between between points x0, x_c, x1 at time t.
    Args:
        x0: Tensor of shape [batch_size, dim] at time t0.
        x1: Tensor of shape [batch_size, dim] at time t1.
        t: Tensor of shape [batch_size, 1] representing the interpolation factor.
        t1: Optional tensor of shape [batch_size, 1] representing the time at which x1 is defined.
        t0: Optional tensor of shape [batch_size, 1] representing the time at which x0 is defined.
        x_c: Optional tensor of shape [batch_size, dim] representing additional points (for 3-point interpolation). This will be the
        t_x_c: Optional tensor of shape [batch_size, 1] representing the points at time at which x_c is defined.
    Returns:
        mu_t: Tensor of shape [batch_size, dim] representing the interpolated data at time t.
        x_t: Tensor of shape [batch_size, dim] representing the noisy interpolated data at time t.
        ut: Tensor of shape [batch_size, dim] representing the velocity at time t.
        t_ut: Tensor of shape [batch_size, 1] representing the time at which x_t and ut is defined.
        dt2_x: Tensor of shape [batch_size, dim] representing the acceleration at time t.
    """
    x_input = torch.cat((x_c, x0, x1), dim=-1)
    x_input = x_input.unsqueeze(1)
    t_input = torch.cat(
        (t_x_c.unsqueeze(-1), t0.unsqueeze(-1), t1.unsqueeze(-1)), dim=-1
    )
    qt = t * (t_input[..., [2]] - t_input[..., [0]]) + t_input[..., [0]]
    x = lag_It(x_input, t_input, qt).squeeze(1)
    dt_x = lag_dtIt(x_input, t_input, qt).squeeze(1)
    dt2_x = lag_dt2It(x_input, t_input).squeeze(1)
    if sigma > 0:
        noise = torch.randn_like(x0) * sigma
        x_noisy = x + noise
    else:
        x_noisy = x

    return x, x_noisy, dt_x, qt, dt2_x
