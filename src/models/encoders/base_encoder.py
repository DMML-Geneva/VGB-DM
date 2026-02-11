import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple, Union
from torch.types import Tensor

from src.metrics.loss import kl_gaussians


class BaseEncoder(nn.Module):
    """
    Base class for encoders. This class can be extended to implement specific encoder architectures.
    It provides a common interface for encoding input data into latent representations.
    """

    def __init__(
        self,
        len_episode,
        state_dim: int,
        d_dim: int,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.d_dim = d_dim
        self.len_episode = len_episode
        self.device = device

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input tensor x into latent representations.
        This method should be implemented by subclasses.

        Args:
            x: Input tensor of shape (batch_size, d_dim)

        Returns:
            Tuple of (mean, logvar)
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def sample(
        self, p_mean, p_logstd, z_mean, z_logstd, z_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the encoder distributions using reparameterization trick.

        Args:

        Returns:
            Tuple of (p_sample, z_sample)
        """
        # Reparameterization trick
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
                + torch.randn(
                    p_mean.size(0), z_size, self.p_dim, device=self.device
                )
                * p_std.unsqueeze(1)
                if p_mean is not None
                else None
            )
            z_sample = z_mean.unsqueeze(1) + torch.randn(
                z_mean.size(0), z_size, self.z_dim, device=self.device
            ) * z_std.unsqueeze(1)

        return p_sample, z_sample

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass returning all relevant outputs.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing samples, statistics, and KL divergences
        """
        p_mean, p_logstd, z_mean, z_logstd = self.encode(x)
        p_sample, z_sample = self.sample(p_mean, p_logstd, z_mean, z_logstd)
        p_kl, z_kl = self.kl_divergence(p_mean, p_logstd, z_mean, z_logstd)

        return {
            "p_sample": p_sample,
            "z_sample": z_sample,
            "p_kl": p_kl,
            "z_kl": z_kl,
        }

    def log_prob(
        self, x: torch.Tensor, p_sample: torch.Tensor, z_sample: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability of samples under the encoder distributions.

        Args:
            x: Input tensor
            p_sample: Physics parameter samples
            z_sample: Latent variable samples

        Returns:
            Tuple of (p_log_prob, z_log_prob)
        """
        p_mean, p_logstd, z_mean, z_logstd = self.encode(x)

        # Create distributions
        p_dist = Normal(p_mean, torch.exp(p_logstd))
        z_dist = Normal(z_mean, torch.exp(z_logstd))

        # Compute log probabilities
        p_log_prob = p_dist.log_prob(p_sample).sum(dim=-1)
        z_log_prob = z_dist.log_prob(z_sample).sum(dim=-1)

        return p_log_prob, z_log_prob

    def kl_divergence(
        self, p_mean, p_logstd, z_mean, z_logstd
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute KL divergence between encoder distributions and fixed priors.
        Supports both normal and uniform priors.

        Args:
            p_mean: Physics parameter mean
            p_logstd: Physics parameter log variance
            z_mean: Latent variable mean
            z_logstd: Latent variable log variance

        Returns:
            Tuple of (p_kl, z_kl) - KL divergences for physics and latent spaces
        """

        # KL divergence for physics parameters
        if self.p_dim is not None:
            if (
                self.p_prior_type == "normal"
                or self.p_prior_type == "gaussian"
            ):
                p_kl = kl_gaussians(
                    p_mean,
                    torch.exp(p_logstd),
                    self.p_prior_mean.unsqueeze(0).repeat(p_mean.shape[0], 1),
                    self.p_prior_std.unsqueeze(0).repeat(p_mean.shape[0], 1),
                )

            elif self.p_prior_type == "uniform":
                # Since the prior is uniform, this is not an informative for regularization as wehenkel ELBO: https://github.com/apple/ml-robust-expert-augmentations/blob/678a9aa0fd9a327711b6d1d645173145deea5327/code/hybrid_models/HVAE.py#L764
                p_kl = torch.zeros(p_mean.size(0), device=self.device)
            else:
                raise ValueError(f"Unknown prior type: {self.p_prior_type}")
        else:
            p_kl = torch.zeros(z_mean.size(0), device=self.device)

        # KL divergence for latent variables
        if self.z_prior_type == "normal" or self.z_prior_type == "gaussian":

            z_kl = kl_gaussians(
                z_mean,
                torch.exp(z_logstd),
                torch.zeros_like(z_mean),
                torch.ones_like(z_logstd),
            )
        elif self.z_prior_type == "uniform":
            # KL(q||uniform) = -H(q) - log(b-a) where H(q) is entropy of q
            z_entropy = 0.5 * (
                1 + torch.log(2 * torch.pi * torch.exp(z_logstd))
            ).sum(dim=-1)
            uniform_log_prob = (
                -torch.log(self.uniform_high - self.uniform_low) * self.z_dim
            )
            z_kl = -z_entropy - uniform_log_prob
        else:
            raise ValueError(f"Unknown prior type: {self.z_prior_type}")

        return p_kl, z_kl

    def sample_prior_p(self, batch_size: int) -> torch.Tensor:
        """
        Sample physics parameters from the prior distribution.

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tensor of shape (batch_size, p_dim) with sampled physics parameters
        """
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
        return p_sample

    def sample_priors(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample physics and latent variables from their prior distributions.
        Args:
            batch_size: Number of samples to generate
        Returns:
            Tuple of (p_sample, z_sample) where:
                p_sample: Tensor of shape (batch_size, p_dim) with sampled physics parameters
                z_sample: Tensor of shape (batch_size, z_dim) with sampled latent variables
        """
        p_sample = self.sample_prior_p(batch_size)
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
        return p_sample, z_sample


class VariationalEncoder(BaseEncoder):
    """
    Variational Encoder that outputs two latent spaces: physics parameters (p_dim) and latent variables (z_dim).
    Both latent variables have fixed priors and the encoder learns mean and variance for each latent space with joint dependence.
    p(z_p, z | x) = p(z | x) * p(z_p | x, z)
    """

    def __init__(
        self,
        len_episode,
        state_dim: int,
        p_dim: int,
        z_dim: int,
        z_hidden_layers: list = [64, 64],
        cleansing_net: list = [128, 128],
        p_hidden_layers: list = [64, 64],
        softplus_mean: bool = False,
        p_prior_type: str = "normal",  # 'normal' or 'uniform'
        z_prior_type: str = "normal",  # 'normal' or 'uniform'
        p_prior_mean: Tensor = [0.0],
        p_prior_std: Tensor = [1.0],
        z_prior_std: float = 1.0,
        prior_U_bounds: Optional[Tensor] = None,
        reflex_clamp: bool = False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            d_dim: Input signal dimension (full trajectory dimensions)
            p_dim: Physics parameter latent dimension
            z_dim: Latent variable dimension
            hidden_dims: Hidden layer dimensions for default MLP (default: [512, 256])
            network: Custom network (nn.Module) or network factory (callable that takes hidden_dims)
            p_prior_type: Prior type for physics parameters ('normal' or 'uniform')
            z_prior_type: Prior type for latent variables ('normal' or 'uniform')
            prior_std: Standard deviation for normal priors (default: 1.0)
            uniform_bounds: Bounds for uniform priors
        """
        super().__init__(
            len_episode=len_episode,
            state_dim=state_dim,
            d_dim=state_dim * len_episode,
            device=device,
        )
        self.device = device
        self.p_dim = p_dim
        self.z_dim = z_dim
        self.p_prior_type = p_prior_type
        self.z_prior_type = z_prior_type
        self.z_prior_std = z_prior_std
        self.p_prior_mean = p_prior_mean
        if self.p_prior_mean is not None and len(self.p_prior_mean) > 0:
            self.p_prior_mean = torch.tensor(p_prior_mean, device=device)
        self.p_prior_std = p_prior_std
        if self.p_prior_std is not None and len(self.p_prior_std) > 0:
            self.p_prior_std = torch.tensor(p_prior_std, device=device)
        self.prior_U_bounds = prior_U_bounds
        self.softplus_mean = softplus_mean
        self.reflex_clamp = reflex_clamp

        # Default MLP with first flattening layer with (batch_size, -1) input
        z_layers = [nn.Flatten(1)]  # Flatten input to (batch_size, d_dim)
        prev_dim = self.d_dim
        for hidden_dim in z_hidden_layers:
            z_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim

        # Latent network Z
        self.nnet_z = nn.Sequential(*z_layers).to(device)
        nnet_out_dim = z_hidden_layers[-1]
        self.z_mean_proj = nn.Linear(nnet_out_dim, z_dim).to(device)
        self.z_logstd_proj = nn.Linear(nnet_out_dim, z_dim).to(device)

        # Physics parameter network
        if self.p_dim > 0:
            p_layers = [nn.Flatten(1)]  # Flatten input to (batch_size, d_dim)
            prev_dim = self.d_dim
            for hidden_dim in p_hidden_layers:
                p_layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.LeakyReLU(),
                        nn.Dropout(0.1),
                    ]
                )
                prev_dim = hidden_dim

            self.nnet_h_p = nn.Sequential(*p_layers).to(device)
            nnet_out_dim = p_hidden_layers[-1]
            self.p_proj = nn.Linear(nnet_out_dim, p_dim).to(device)
            self.p_logstd_proj = nn.Linear(nnet_out_dim, p_dim).to(device)

            # Cleansing network
            # add first layer of flattening
            if cleansing_net is not None and len(cleansing_net) > 0:
                cleansing_layers = [
                    nn.Flatten(1),
                ]
                cleansing_layers = [
                    nn.Sequential(
                        nn.Linear(
                            (
                                (self.d_dim + self.z_dim)
                                if i == 0
                                else cleansing_net[i - 1]
                            ),
                            h,
                        ),
                        nn.LeakyReLU(),
                        nn.Dropout(0.1),
                    )
                    for i, h in enumerate(cleansing_net)
                ]
                cleansing_layers.append(
                    nn.Linear(cleansing_net[-1], self.d_dim)
                )
                self.cleansing_net = nn.Sequential(*cleansing_layers).to(
                    device
                )
            else:
                self.cleansing_net = None

        # Uniform prior bounds
        if self.prior_U_bounds is not None and len(self.prior_U_bounds) > 0:
            self.register_buffer("uniform_low", prior_U_bounds[:, 0])
            self.register_buffer("uniform_high", prior_U_bounds[:, 1])

        self.to(device)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to get mean and log variance for both latent spaces.

        Args:
            x: Input tensor of shape (batch_size, d_dim)

        Returns:
            Tuple of (p_mean, p_logstd, z_mean, z_logstd)
        """
        x = x.flatten(start_dim=1)  # Flatten input to (batch_size, d_dim)
        h_z = self.nnet_z(x)
        z_mean = self.z_mean_proj(h_z)
        z_logstd = self.z_logstd_proj(h_z)

        if self.p_dim > 0:
            # cleansing step
            if self.cleansing_net is not None:
                x_cleansed = self.cleansing_net(torch.cat((x, z_mean), dim=-1))
                x_cleansed = x + x_cleansed.reshape_as(
                    x
                )  # x - x_cleansed.reshape_as(x)
            else:
                x_cleansed = x
            self.x_cleansed = x_cleansed.squeeze()
            h_p = self.nnet_h_p(self.x_cleansed)
            p_mean = self.p_proj(h_p)

            p_logstd = self.p_logstd_proj(h_p)

            if self.p_prior_type == "uniform":
                # Ensure p_mean is within uniform bounds with sigmoid and bounds scaling
                p_mean = (
                    torch.sigmoid(p_mean)
                    * (self.prior_U_bounds[:, 1] - self.prior_U_bounds[:, 0])
                    + self.prior_U_bounds[:, 0]
                )
                p_mean = torch.clamp(
                    p_mean,
                    self.prior_U_bounds[:, 0],
                    self.prior_U_bounds[:, 1],
                )
                sigma_max = (
                    self.prior_U_bounds[:, 1] - self.prior_U_bounds[:, 0]
                ) / 4.0

                sigma_min = torch.tensor(1e-4).to(
                    self.device
                )  # Minimum std to avoid numerical issues
                p_logstd = torch.clamp(
                    p_logstd, torch.log(sigma_min), torch.log(sigma_max)
                )
            else:
                # softplus if softplus_mean is True
                p_mean = F.softplus(p_mean) if self.softplus_mean else p_mean
        else:
            p_mean = torch.zeros(x.size(0), self.p_dim, device=self.device)
            p_logstd = torch.zeros(x.size(0), self.p_dim, device=self.device)

        return p_mean, p_logstd, z_mean, z_logstd

    def clamp(
        self, p_sample: torch.Tensor, reflect: bool = True
    ) -> torch.Tensor:
        if not reflect:
            p_sample = torch.clamp(
                p_sample,
                self.prior_U_bounds[:, 0],
                self.prior_U_bounds[:, 1],
            )
        else:
            bounds_low = self.prior_U_bounds[:, 0]
            bounds_high = self.prior_U_bounds[:, 1]
            bounds_width = bounds_high - bounds_low

            # Shift to [0, width] range
            p_shifted = p_sample - bounds_low

            # Apply reflection using modulo arithmetic
            p_reflected = torch.remainder(p_shifted, 2 * bounds_width)
            mask = p_reflected > bounds_width
            p_reflected[mask] = 2 * bounds_width - p_reflected[mask]

            # Shift back to original range
            p_sample = p_reflected + bounds_low
        return p_sample

    def sample(
        self, p_mean, p_logstd, z_mean, z_logstd, z_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the encoder distributions using reparameterization trick.

        Args:

        Returns:
            Tuple of (p_sample, z_sample)
        """
        p_sample, z_sample = super().sample(
            p_mean, p_logstd, z_mean, z_logstd, z_size=z_size
        )

        # Apply bounds after sampling for uniform priors
        p_sample = self.clamp(p_sample, reflect=self.reflex_clamp)

        return p_sample, z_sample


# Example usage
if __name__ == "__main__":
    # Example with default MLP and normal priors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    encoder = VariationalEncoder(d_dim=128, p_dim=10, z_dim=32, device=device)

    # Example with uniform prior for physics parameters
    encoder_uniform = VariationalEncoder(
        d_dim=128,
        p_dim=10,
        z_dim=32,
        p_prior_type="uniform",
        prior_U_bounds=torch.tensor([[-2.0, 2.0]]),
        device=device,
    )

    # Test all encoders
    x = torch.randn(16, 128).to(device)

    print("=== Testing Default Encoder ===")
    output = encoder(x)
    print("Physics sample shape:", output["p_sample"].shape)
    print("Latent sample shape:", output["z_sample"].shape)

    print("\n=== Testing Uniform Prior Encoder ===")
    output_uniform = encoder_uniform(x)
    print("Physics KL mean:", output_uniform["p_kl"].mean().item())
    print("Latent KL mean:", output_uniform["z_kl"].mean().item())
