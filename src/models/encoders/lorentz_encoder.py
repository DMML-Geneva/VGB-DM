import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple

from src.metrics.loss import kl_gaussians


class LorenzEncoder(nn.Module):
    """
    Lightweight temporal encoder for Lorenz system.
    - Uses simple GRU to process sequences
    - Separates physics parameters (p) from stochastic latent (z)
    
    Input shape: (batch_size, len_episode, state_dim) -> (batch_size, len_episode, 3)
    """

    def __init__(
        self,
        len_episode: int,
        state_dim: int,
        p_dim: int,
        z_dim: int,
        gru_hidden: int = 64,
        z_prior_type: str = "normal",
        p_prior_type: str = "normal",
        p_prior_mean: torch.Tensor = None,
        p_prior_std: torch.Tensor = None,
        z_prior_std: float = 1.0,
        prior_U_bounds: Optional[torch.Tensor] = None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            len_episode: Length of trajectory sequence
            state_dim: State dimension (3 for Lorenz x, y, z)
            p_dim: Physics parameter latent dimension
            z_dim: Latent variable (stochasticity) dimension
            gru_hidden: GRU hidden size
            z_prior_type: 'normal' or 'uniform'
            p_prior_type: 'normal' or 'uniform'
        """
        super().__init__()
        self.state_dim = state_dim
        self.len_episode = len_episode
        self.d_dim = state_dim * len_episode
        self.p_dim = p_dim
        self.z_dim = z_dim
        self.gru_hidden = gru_hidden
        self.device = device
        self.z_prior_type = z_prior_type
        self.p_prior_type = p_prior_type
        self.z_prior_std = z_prior_std

        # Prior parameters
        if p_prior_mean is not None:
            if isinstance(p_prior_mean, (list, tuple)):
                self.p_prior_mean = torch.tensor(p_prior_mean, device=device, dtype=torch.float32)
            else:
                self.p_prior_mean = p_prior_mean.to(device)
        else:
            self.p_prior_mean = torch.zeros(p_dim, device=device, dtype=torch.float32)

        if p_prior_std is not None:
            if isinstance(p_prior_std, (list, tuple)):
                self.p_prior_std = torch.tensor(p_prior_std, device=device, dtype=torch.float32)
            else:
                self.p_prior_std = p_prior_std.to(device)
        else:
            self.p_prior_std = torch.ones(p_dim, device=device, dtype=torch.float32)

        if prior_U_bounds is not None:
            if isinstance(prior_U_bounds, torch.Tensor):
                self.register_buffer("uniform_low", prior_U_bounds[:, 0].to(device))
                self.register_buffer("uniform_high", prior_U_bounds[:, 1].to(device))
            else:
                bounds_tensor = torch.tensor(prior_U_bounds, device=device, dtype=torch.float32)
                self.register_buffer("uniform_low", bounds_tensor[:, 0])
                self.register_buffer("uniform_high", bounds_tensor[:, 1])

        # GRU for temporal encoding
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            num_layers=1,
        )

        # Z (stochasticity)
        self.z_head = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.Tanh(),
        )
        self.z_mean_proj = nn.Linear(32, z_dim)
        self.z_logstd_proj = nn.Linear(32, z_dim)

        # Physics parameters
        if p_dim > 0:
            # Simple residual connection: use both GRU hidden and z_mean info
            self.p_head = nn.Sequential(
                nn.Linear(gru_hidden + z_dim, 32),
                nn.Tanh(),
            )
            self.p_mean_proj = nn.Linear(32, p_dim)
            self.p_logstd_proj = nn.Linear(32, p_dim)

        self.to(device)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode trajectory to physics and latent parameters.

        Args:
            x: Input tensor of shape (batch_size, len_episode, state_dim)

        Returns:
            Tuple of (p_mean, p_logstd, z_mean, z_logstd)
        """
        batch_size = x.size(0)

        # GRU processes full trajectory
        # Output: (batch, len_episode, gru_hidden), (batch, gru_hidden)
        _, h_gru = self.gru(x)  # h_gru: (1, batch, gru_hidden)
        h_gru = h_gru.squeeze(0)  # (batch, gru_hidden)

        # ========== Z pathway: extract stochasticity ==========
        h_z = self.z_head(h_gru)
        z_mean = self.z_mean_proj(h_z)
        z_logstd = self.z_logstd_proj(h_z)

        # ========== Physics extraction physics 
        if self.p_dim > 0:
            h_p_input = torch.cat([h_gru, z_mean.detach()], dim=1)  # Detach z_mean to reduce coupling
            h_p = self.p_head(h_p_input)
            p_mean = self.p_mean_proj(h_p)
            p_logstd = self.p_logstd_proj(h_p)
        else:
            p_mean = torch.zeros(batch_size, self.p_dim, device=self.device, dtype=x.dtype)
            p_logstd = torch.zeros(batch_size, self.p_dim, device=self.device, dtype=x.dtype)

        return p_mean, p_logstd, z_mean, z_logstd

    def kl_divergence(
        self, p_mean, p_logstd, z_mean, z_logstd
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute KL divergence with fixed priors."""
        batch_size = p_mean.size(0)

        # KL for physics parameters
        if self.p_dim > 0:
            if self.p_prior_type in ["normal", "gaussian"]:
                p_kl = kl_gaussians(
                    p_mean,
                    torch.exp(p_logstd),
                    self.p_prior_mean.unsqueeze(0).expand(batch_size, -1),
                    self.p_prior_std.unsqueeze(0).expand(batch_size, -1),
                )
            elif self.p_prior_type == "uniform":
                p_kl = torch.zeros(batch_size, device=self.device)
            else:
                raise ValueError(f"Unknown prior type: {self.p_prior_type}")
        else:
            p_kl = torch.zeros(batch_size, device=self.device)

        # KL for latent variables
        if self.z_prior_type in ["normal", "gaussian"]:
            z_kl = kl_gaussians(
                z_mean,
                torch.exp(z_logstd),
                torch.zeros_like(z_mean),
                torch.ones_like(z_logstd),
            )
        elif self.z_prior_type == "uniform":
            z_entropy = 0.5 * (
                1 + torch.log(2 * np.pi * torch.exp(z_logstd))
            ).sum(dim=-1)
            uniform_log_prob = (
                -torch.log(self.uniform_high - self.uniform_low) * self.z_dim
            )
            z_kl = -z_entropy - uniform_log_prob
        else:
            raise ValueError(f"Unknown prior type: {self.z_prior_type}")

        return p_kl, z_kl

    def sample(
        self, p_mean, p_logstd, z_mean, z_logstd, z_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample using reparameterization trick."""
        p_std = torch.exp(p_logstd) if p_mean is not None else None
        z_std = torch.exp(z_logstd)

        if z_size == 1:
            p_sample = (
                p_mean + torch.randn_like(p_std) * p_std
                if p_mean is not None
                else None
            )
            z_sample = z_mean + torch.randn_like(z_std) * z_std
        else:
            p_sample = (
                p_mean.unsqueeze(1)
                + torch.randn(p_mean.size(0), z_size, self.p_dim, device=self.device)
                * p_std.unsqueeze(1)
                if p_mean is not None
                else None
            )
            z_sample = z_mean.unsqueeze(1) + torch.randn(
                z_mean.size(0), z_size, self.z_dim, device=self.device
            ) * z_std.unsqueeze(1)

        return p_sample, z_sample

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass returning all relevant outputs."""
        p_mean, p_logstd, z_mean, z_logstd = self.encode(x)
        p_sample, z_sample = self.sample(p_mean, p_logstd, z_mean, z_logstd)
        p_kl, z_kl = self.kl_divergence(p_mean, p_logstd, z_mean, z_logstd)

        return {
            "p_sample": p_sample,
            "z_sample": z_sample,
            "p_mean": p_mean,
            "z_mean": z_mean,
            "p_logstd": p_logstd,
            "z_logstd": z_logstd,
            "p_kl": p_kl,
            "z_kl": z_kl,
        }

    def sample_priors(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from prior distributions."""
        if self.p_prior_type == "uniform":
            p_sample = (
                torch.rand(batch_size, self.p_dim, device=self.device)
                * (self.uniform_high - self.uniform_low)
                + self.uniform_low
            )
        else:
            p_sample = (
                torch.randn(batch_size, self.p_dim, device=self.device)
                * self.p_prior_std
                + self.p_prior_mean
            )

        if self.z_prior_type == "normal":
            z_sample = (
                torch.randn(batch_size, self.z_dim, device=self.device)
                * self.z_prior_std
            )
        elif self.z_prior_type == "uniform":
            z_sample = (
                torch.rand(batch_size, self.z_dim, device=self.device)
                * (self.uniform_high - self.uniform_low)
                + self.uniform_low
            )
        else:
            z_sample = torch.randn(batch_size, self.z_dim, device=self.device)

        return p_sample, z_sample