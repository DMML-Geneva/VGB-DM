from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from torchdyn.core import NeuralODE

from typing import *
import math


def log_normal(x: Tensor) -> Tensor:
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2


class PhysicsWeightScheduler:
    """
    Manages physics weight scheduling without needing global_step in the model.
    Physics weight ramps: 0 → 1 over specified warmup steps.
    """
    
    def __init__(self, warmup_steps=1000):
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.physics_weight = 0.0
    
    def step(self):
        """Call once per training iteration."""
        self.physics_weight = min(self.current_step / self.warmup_steps, 1.0)
        self.current_step += 1
    
    def get_weight(self):
        """Get current physics weight."""
        return self.physics_weight


class BaseGreyBoxFM(nn.Module):

    def __init__(
        self,
        vf_nnet: nn.Module,
        phys_eq: Optional[Callable] = None,
        state_dim: int = 1,
        phys_params_dim: int = 1,
        latent_dim: int = 1,
        history_size: int = 0,
        second_order: bool = False,
        use_vf_phys: bool = False,
        num_freqs: int = 10,
        sigma: float = 1e-4,
    ):
        super(BaseGreyBoxFM, self).__init__()

        self.state_dim = state_dim
        self.phys_params_dim = phys_params_dim
        self.use_vf_phys = use_vf_phys
        self.latent_dim = latent_dim
        self.history_size = history_size
        self.physics_weight = 1.0 

        self.phys_eq = phys_eq
        self.second_order = second_order

        self.vf_net = vf_nnet
        self.sigma = sigma
        self.device = next(vf_nnet.parameters()).device
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
        if hasattr(self.vf_net, "modules"):
            for module in self.vf_net.modules():
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

    def concatenate_input(
        self,
        x,
        p_sample,
        z_sample,
        t,
        t_enc,
        phys_vf: Tensor = None,
        dt_xt: Optional[Tensor] = None,
        x_history: Optional[
            Tensor
        ] = None,  # shape (batch_size, history_size, state_dim)
    ):
        new_x = x
        if self.second_order and dt_xt is not None:
            # Append velocity to the state if second order
            new_x = torch.cat([new_x, dt_xt], dim=-1)

        if p_sample is not None:
            new_x = torch.cat([new_x, p_sample], dim=-1)

        if z_sample is not None:
            new_x = torch.cat(
                [new_x, z_sample],
                dim=-1,
            )

        new_x = torch.cat(
            [new_x, t_enc],
            dim=-1,
        )

        if self.use_vf_phys and self.phys_eq is not None and phys_vf is not None:
            new_x = torch.cat([new_x, phys_vf], dim=-1)

        if self.history_size > 0 and x_history is not None:
            new_x = torch.cat(
                [new_x, x_history.reshape(x.shape[0], -1)], dim=-1
            )

        return new_x

    def forward_train(
        self,
        x: Tensor,
        p_sample: Tensor,
        z_sample: Tensor,
        t: Tensor,
        dt_xt: Optional[Tensor] = None,
        x_history: Optional[Tensor] = None,
    ):
        """
        Forward pass for training mode.
        Args:
            x: Input tensor of shape (batch_size, state_dim + phys_params_dim + latent_dim + 1).
                The last dimension is time.
        Returns:
            Tensor of shape (batch_size, state_dim + phys_params_dim + latent_dim) representing
            the forward velocity field.
        """

        if t.dim() == 1:
            t = t[:, None]
        t_enc = self.time_encoding(t)
        self.p_sample = p_sample
        self.z_sample = z_sample

        # Physics-based velocity field
        if self.phys_eq is not None:
            self.phys_eq.init_phys_params(self.p_sample)
            phys_input = (
                torch.cat([x, dt_xt], dim=-1)
                if self.second_order and dt_xt is not None
                else x
            )
            self.phys_vf = self.phys_eq(x=phys_input, t=t)

        new_x = self.concatenate_input(
            x,
            self.p_sample,
            self.z_sample,
            t,
            t_enc,
            phys_vf=self.phys_vf if self.use_vf_phys and self.phys_eq is not None else None,
            dt_xt=dt_xt,
            x_history=x_history,
        )  # shape (batch_size, state_dim + phys_params_dim + latent_dim + 2 * num_freqs)

        fv_net = self.vf_net(new_x)
        self.fv_net = fv_net

        if self.phys_eq is not None:
            total_vf = fv_net + self.physics_weight * self.phys_vf
        else:
            total_vf = fv_net

        return total_vf
    
    def set_physics_weight(self, weight: float):
        """
        Set physics weight (called from scheduler during training).
        During inference, this stays at 1.0.
        
        Args:
            weight: Physics weight in range [0, 1]
        """
        self.physics_weight = weight

    def forward(self, t, x, *args, **kwargs):
        """
        Forward pass for inference mode, using NeuralODE torchdyn.
        During ODE integration, the history should remain constant.
        """
        if self.history_size > 0:
            # Extract history from the augmented state
            state = x[:, 0, : self.state_dim]
            dt_xt = (
                x[:, 0, self.state_dim : self.state_dim * 2]
                if self.second_order
                else None
            )
            x_hist = x[:, 1:, : self.state_dim]
            # Compute velocity field for the state using the history
            vf = self.forward_train(
                state,
                self.p_sample,
                self.z_sample,
                t.repeat(x.shape[0])[:, None],
                dt_xt=dt_xt,
                x_history=x_hist,
            )

            # Append zero change dx/dt for the **history** part to maintain dimensions and constant history!
            zero_vf_hist = torch.zeros(
                (x.shape[0], x_hist.shape[1], *x.shape[2:])
            ).to(self.device)
            vf = torch.cat([vf.unsqueeze(1), zero_vf_hist], dim=1)
        else:
            state = x[..., : self.state_dim]
            dt_xt = (
                x[..., self.state_dim : self.state_dim * 2]
                if self.second_order
                else None
            )
            vf = self.forward_train(
                state,
                self.p_sample,
                self.z_sample,
                t.repeat(x.shape[0])[:, None],
                dt_xt=dt_xt,
                x_history=None,
            )

        return vf

    def sample_trajectory(
        self,
        x0: Tensor,
        phys_params: Tensor,
        z_sample: Tensor,
        t_span: list,
        x_history: Optional[
            Tensor
        ] = None,  # shape (batch_size, history_size, states...)
        dt_xt: Optional[Tensor] = None,
        ode_method: Literal["dopri5", "euler", "rk4"] = "rk4",
        sde_method: Literal["euler", "midpoint"] = None,
    ):
        self.p_sample = phys_params
        self.z_sample = z_sample

        t_span = (
            torch.tensor(t_span, device=self.device).float()
            if isinstance(t_span, list)
            else t_span
        )
        # If history is enabled, integrate step by step with sliding window
        if self.history_size > 0 and x_history is not None:
            batch_size = x0.shape[0]
            num_steps = len(t_span)
            if self.second_order:
                if dt_xt is not None:
                    x0 = torch.cat([x0, dt_xt], dim=-1)
                else:
                    x0 = torch.cat(
                        [x0, torch.zeros_like(x0)], dim=-1
                    )  # append zero velocity
                x_history = x_history.expand(
                    -1, -1, 2
                )  # conform history wrt x dim for torch.cat

            traj = torch.zeros(
                batch_size, num_steps, *x0.shape[1:], device=self.device
            )

            # Set first point
            traj[:, 0, :] = x0
            history_buffer = (
                x_history.clone()
            )  # shape (batch_size, history_size, state_dim)

            node = NeuralODE(
                self,
                solver=ode_method,
                sensitivity="adjoint",
                atol=1e-4,
                rtol=1e-4,
            )

            # Integrate step by step
            for i in range(1, num_steps):
                # Prepare input (state + (second order)) with current history
                x_current = traj[:, i - 1]
                x_input = torch.cat(
                    [x_current.unsqueeze(1), history_buffer], dim=1
                )

                # Integrate one step
                t_curr = t_span[i - 1 : i + 1]
                # make a linspace between t_span
                # t_linspace = torch.linspace(t_curr[0], t_curr[1], steps=2).to(self.device)
                step_traj = node.trajectory(
                    x_input, t_span=t_curr
                )  # shape (time, batch, 1+hist , states...)

                # Extract next point (last timestep, remove history part)
                state_end_idx = (
                    self.state_dim
                    if not self.second_order
                    else self.state_dim * 2
                )
                next_point = step_traj[-1, :, 0, :state_end_idx]
                traj[:, i, :] = next_point

                # Update sliding window: remove oldest, add current point
                # The history should contain past observations, so we add x_current
                history_buffer = torch.cat(
                    [
                        history_buffer[:, 1:, :],
                        x_current.unsqueeze(1),
                    ],
                    dim=1,
                )

            if self.second_order:
                traj = traj[
                    ..., : self.state_dim
                ]  # return only state, not velocity

            return traj
        else:
            # Previous unchanged code
            # Standard integration without history
            node = NeuralODE(
                self,
                solver=ode_method,
                sensitivity="adjoint",
                atol=1e-4,
                rtol=1e-4,
            )
            if self.second_order:
                if dt_xt is not None:
                    x0 = torch.cat([x0, dt_xt], dim=-1)
                else:
                    x0 = torch.cat(
                        [x0, torch.zeros_like(x0)], dim=-1
                    )  # append zero velocity

            traj = node.trajectory(
                x0,
                t_span=t_span,
            )
            dims = list(range(traj.ndim))
            dims[0], dims[1] = dims[1], dims[0]  # swap first two
            traj = traj.permute(*dims)
            if self.second_order:
                traj = traj[
                    ..., : self.state_dim
                ]  # return only state, not velocity

            return traj

    def sample_only_physics(
        self,
        x0: Tensor,
        phys_params: Tensor,
        T0: float,
        T1: float,
        steps: int = 10,
        ode_method: Literal["dopri5", "euler", "rk4"] = "dopri5",
    ):

        if self.phys_eq is None:
            raise ValueError("Physics equation is not defined.")

        self.phys_eq.init_phys_params(phys_params)
        node = NeuralODE(
            self.phys_eq,
            solver=ode_method,
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        piece_traj = node.trajectory(
            x0,
            t_span=torch.linspace(T0, T1, steps=steps).to(self.device),
        )
        return piece_traj
