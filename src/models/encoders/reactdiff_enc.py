from typing import Tuple
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision

from src.metrics.loss import kl_gaussians

from src.nn.unet import Encoder2d, Decoder2d, Permute
from src.models.encoders.base_encoder import BaseEncoder


class ConditionalUNetReactionDiffusion(nn.Module):
    def __init__(
        self,
        c_dim=1,
        enc_chs=(2, 16, 32, 64, 128),
        dec_chs=(128, 64, 32, 16),
        num_class=2,
        retain_dim=True,
        final_act=None,
    ):
        super().__init__()
        self.encoder = Encoder2d(enc_chs)
        self.decoder = Decoder2d(dec_chs, cond_size=c_dim)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.final_act = (
            nn.Sigmoid() if final_act is not None else nn.Identity()
        )
        self.retain_dim = retain_dim

    def forward(self, x, cond=None):
        enc_ftrs = self.encoder(x)
        if cond is not None:
            cond = (
                cond.unsqueeze(2)
                .unsqueeze(2)
                .expand(-1, -1, enc_ftrs[-1].shape[2], enc_ftrs[-1].shape[3])
            )
            enc_ftrs[-1] = torch.cat((cond, enc_ftrs[-1]), 1)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = self.final_act(out)
        if self.retain_dim:
            out = F.interpolate(out, x.shape[-1])
        return out


class ReactDiffEncoder(BaseEncoder):
    def __init__(
        self,
        grid_size=32,
        state_dim=2,
        z_dim=10,
        p_dim=2,
        len_episode=10,
        p_prior_type="normal",  # 'normal' or 'uniform'
        z_prior_type="normal",  # 'normal' or 'uniform'
        p_prior_mean=[0.0],
        p_prior_std=[1.0],
        z_prior_std=1.0,
        prior_U_bounds=None,
        reflex_clamp=False,
        device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    ):

        super(ReactDiffEncoder, self).__init__(
            len_episode=len_episode,
            state_dim=state_dim,
            d_dim=state_dim * grid_size * grid_size * len_episode,
            device=device,
        )
        self.grid_size = grid_size

        self.z_dim = z_dim
        self.p_dim = p_dim if p_dim is not None and p_dim > 0 else None
        self.scaler = 1
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
        self.reflex_clamp = reflex_clamp

        # Conditional UNet for processing observations
        self.cleansing_net = ConditionalUNetReactionDiffusion(c_dim=self.z_dim)

        # Common encoder backbone
        self.enc_common = nn.Sequential(
            nn.Flatten(0, 1),
            nn.Conv2d(2, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 32, 3),
            nn.ReLU(),
            nn.Unflatten(0, (-1, self.len_episode)),
            Permute((0, 2, 1, 3, 4)),
            nn.Conv3d(32, 16, 2),
            nn.ReLU(),
            nn.Conv3d(16, 16, 2),
            nn.Flatten(1, 4),
        )

        # Calculate the actual output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(
                2, self.len_episode, 2, self.grid_size, self.grid_size
            )
            dummy_output = self.enc_common(dummy_input)
            self.common_output_size = dummy_output.shape[1]
        self.common_output_size = self.common_output_size

        # Encoder for z_a (appearance/augmentation latent)
        self.nnet_z = nn.Sequential(
            self.enc_common,
            nn.Linear(self.common_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * self.z_dim),  # Mean and log variance
        )

        # Encoder for z_p (physical parameters latent)
        if self.p_dim is not None:
            self.nnet_h_p = nn.Sequential(
                self.enc_common,
                nn.Linear(self.common_output_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            # Mean and log variance
            self.p_mean_proj = nn.Linear(256, p_dim).to(device)
            self.p_logstd_proj = nn.Linear(256, p_dim).to(device)
            # Activation function for physical parameters (placeholder)
            self.act_mu_p = (
                nn.Identity()
            )  # Replace with actual activation if needed

    def p_proj(self, h_p):
        return self.act_mu_p(self.p_mean_proj(h_p)) * self.scaler

    def encode(self, x):
        """
        Encode input x into latent variables z_a and z_p

        Args:
            x: Input tensor of shape (batch_size, len_episode, 2, height, width)

        Returns:
        """
        b_size, len_episode, im_size = x.shape[0], x.shape[1], x.shape[-1]

        # Encode z (augmented latent)
        z_mean, z_logstd = torch.chunk(self.nnet_z(x), 2, 1)

        z_std = torch.exp(z_logstd)
        z = z_mean + z_std * torch.randn_like(z_mean)

        # Encode z_p (physical parameters latent)
        if self.p_dim is not None:
            # Process observations with conditional UNet using
            delta = self.cleansing_net(
                x.reshape(-1, 2, im_size, im_size),
                z.unsqueeze(1)
                .expand(-1, self.len_episode, -1)
                .reshape(-1, self.z_dim),
            ).reshape(b_size, self.len_episode, 2, im_size, im_size)

            self.x_cleansed = (
                x + delta
            )  # xp contains x_obs and the augmentation delta given by z_a
            
            h_p = self.nnet_h_p(self.x_cleansed)
            p_mean, p_logstd = self.p_proj(h_p), self.p_logstd_proj(h_p)
            
        else:
            p_mean = p_logstd = None

        return p_mean, p_logstd, z_mean, z_logstd

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
        if self.p_dim is not None:
            p_dist = Normal(p_mean, torch.exp(p_logstd))
            p_log_prob = p_dist.log_prob(p_sample).sum(dim=-1)
        else:
            p_log_prob = p_dist = None
        z_dist = Normal(z_mean, torch.exp(z_logstd))
        z_log_prob = z_dist.log_prob(z_sample).sum(dim=-1)

        return p_log_prob, z_log_prob


import numpy as np

if __name__ == "__main__":
    nrows = 70
    x_dim = (10, 2, 32, 32)

    x = torch.randn(nrows, *x_dim)
    model = ReactDiffEncoder(
        d_dim=np.prod(x_dim.shape),
        z_dim=10,
        p_dim=2,
        len_episode=10,
        device="cpu",
    )
    res = model(x)
    # print shapes
    shapes = {k: v.shape for k, v in res.items()}
    print("Output shapes:")
    for k, v in shapes.items():
        print(f"{k}: {v}")
