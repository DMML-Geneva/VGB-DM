from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from torchdyn.core import NeuralODE
from torchdyn.numerics import odeint

from typing import *
import math


def log_normal(x: Tensor) -> Tensor:
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2


class BaseGreyBoxNODE(nn.Module):

    def __init__(
        self,
        drift_net: nn.Module,
        phys_eq: Optional[Callable] = None,
        and_only_phys: bool = True,
        state_dim: int = 1,
        phys_params_dim: int = 1,
        latent_dim: int = 1,
        num_freqs: int = 10,
        sigma: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        super(BaseGreyBoxNODE, self).__init__()

        self.state_dim = state_dim
        self.phys_params_dim = phys_params_dim
        self.latent_dim = latent_dim

        self.phys_eq = phys_eq
        self.and_only_phys = and_only_phys

        self.drift_net = drift_net
        self.sigma = sigma
        self.device = next(drift_net.parameters()).device
        PE_BASE = 0.012  # 0.012615662610100801
        freqs = torch.pow(
            PE_BASE,
            -torch.arange(0, num_freqs, dtype=torch.float32) / num_freqs,
        )
        # freqs = torch.arange(1, freqs + 1).to(self.device) * torch.pi
        self.register_buffer("freqs", freqs)
        self.time_norm = nn.LayerNorm(2 * num_freqs).to(self.device)

        self._init_velocity_field()

    def _init_velocity_field(self):
        """Initialize the final layer of velocity field with smaller weights."""
        if hasattr(self.drift_net, "modules"):
            for module in self.drift_net.modules():
                if isinstance(module, nn.Linear):
                    # Use Xavier initialization scaled down
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def time_encoding(self, t: Tensor) -> Tensor:
        """
        Encodes time t using sine and cosine functions.
        Args:
            t: Tensor of shape (..., 1) representing time.
        Returns:
            Encoded time tensor of shape (..., 2 * freqs).
        """
        angles = self.freqs * t[...]
        sin_t = torch.sin(angles)
        cos_t = torch.cos(angles)
        pos_enc = torch.cat([sin_t, cos_t], dim=-1)

        # Apply layer normalization if enabled
        pos_enc = self.time_norm(pos_enc)
        return pos_enc

    def concatenate_input(self, x, p_sample, z_sample, t, t_enc):
        new_x = torch.cat(
            [x[..., : self.state_dim], p_sample, z_sample, t_enc],
            dim=-1,
        )
        return new_x

    def forward(self, t: Tensor, x: Tensor, *args, **kwargs):
        """
        Forward pass for NeuralODE torchdyn.
        Args:
            x: Input tensor of shape (batch_size, state_dim (+ copy state_dim if phys_eq is defined)).
            t: time float
        Returns:
            Tensor of shape (batch_size, state_dim (+ state_dim if phys_eq is defined)
        """
        t = t.repeat(x.shape[0])[:, None]
        t_enc = self.time_encoding(t)

        new_x = self.concatenate_input(
            x[:, : self.state_dim], self.p_sample, self.z_sample, t, t_enc
        )
        fv_net = self.drift_net(new_x)
        self.fv_net = fv_net

        if self.phys_eq is not None:

            self.phys_eq.init_phys_params(self.p_sample)
            phys_eval = self.phys_eq(x=x[:, : self.state_dim], t=t)
            total_velocity = fv_net + phys_eval

            if self.and_only_phys:
                only_phys_x = x[:, self.state_dim :]
                self.phys_eq.init_phys_params(self.p_sample)
                only_phys_eval = self.phys_eq(x=only_phys_x, t=t)
        else:
            total_velocity = fv_net

        vf = total_velocity
        if self.phys_eq is not None and self.and_only_phys:
            # If phys_eq is defined, concatenate the only physical evaluation
            vf = torch.cat([vf, only_phys_eval], dim=1)
        return vf

    def sample_trajectory(
        self,
        x0: Tensor,
        phys_params: Tensor,
        z_sample: Tensor,
        t_span: Tensor,
        x_history: Optional[
            Tensor
        ] = None,  # shape (batch_size, history_size, states...)
        dt_xt: Optional[Tensor] = None,
        ode_method: Literal["dopri5", "euler", "rk4"] = "rk4",
    ):
        vf_2d = x0.shape[1] != self.state_dim and self.phys_eq is not None
        if vf_2d:
            # e.g Pendulum case v(0) is zero
            x_input = torch.cat([x0, torch.zeros_like(x0)], dim=1)
        else:
            x_input = x0

        full_traj = self.sample(
            x0=x_input,
            phys_params=phys_params,
            z_sample=z_sample,
            t_span=t_span,
            ode_method=ode_method,
        )
        if self.phys_eq is None:
            traj = full_traj
            phys_traj = torch.zeros_like(full_traj[:, :, : self.state_dim])
        else:
            traj = full_traj[:, :, : self.state_dim]
            phys_traj = full_traj[:, :, self.state_dim :]

        if vf_2d:
            # e.g Pendulum case v(0) is zero
            traj = traj[:, :, : self.state_dim // 2]
            phys_traj = phys_traj[:, :, : self.state_dim // 2]
        return traj, phys_traj

    def sample(
        self,
        x0: Tensor,
        phys_params: Tensor,
        z_sample: Tensor,
        t_span: Tensor,
        ode_method: Literal["dopri5", "euler"] = "dopri5",
    ):

        node = NeuralODE(
            self,
            solver=ode_method,
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )
        self.p_sample = phys_params
        self.z_sample = z_sample
        input = x0

        if self.phys_eq is not None and self.and_only_phys:
            # If phys_eq is defined, concatenate a copy of the initial state for the only physical evaluation
            input = torch.cat([input, x0], dim=1)
        t_span = (
            torch.tensor(t_span, device=self.device).float()
            if isinstance(t_span, list)
            else t_span
        )
        traj = node.trajectory(
            input,
            t_span=t_span,
        )

        dims = list(range(traj.ndim))
        dims[0], dims[1] = dims[1], dims[0]  # swap first two
        traj = traj.permute(*dims)
        return traj

    def sample_phys_trajectory(
        self,
        x0: Tensor,
        phys_params: Tensor,
        t_span: list,
        ode_method: Literal["dopri5", "euler", "rk4"] = "rk4",
    ) -> Tensor:
        """
        Sample a only a physical trajectory given initial state and physical parameters.
        Args:
            x0: Initial state tensor of shape (batch_size, state_dim).
            phys_params: Physical parameters tensor of shape (batch_size, phys_params_dim).
            t_span: List of time points for the trajectory.
        Returns:
            Tensor of shape (batch_size, T, state_dim) representing the sampled trajectory.
        """

        class phys_eq_wrapper(nn.Module):
            def __init__(self, phys_eq):
                super().__init__()
                self.phys_eq = phys_eq

            def forward(self, t, x, *args, **kwargs):
                return self.phys_eq(x=x, t=t)

        self.phys_eq.init_phys_params(phys_params)
        t_eval, sol = odeint(
            phys_eq_wrapper(self.phys_eq),
            x=x0,
            solver=ode_method,
            t_span=t_span,
            atol=1e-4,
            rtol=1e-4,
        )
        sol = sol.permute(1, 0, 2)  # shape (batch_size, T, state_dim)
        return sol
