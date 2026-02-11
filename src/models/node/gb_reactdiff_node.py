from typing import Callable, Literal, Optional
import torch

import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from src.nn.mlp import act_dict

from src.models.node.grey_box_node import BaseGreyBoxNODE


class GBReactDiffNODE(BaseGreyBoxNODE):
    def __init__(
        self,
        vf_config: dict,
        phys_eq: Optional[Callable] = None,
        state_dim: int = 1,
        phys_params_dim: int = 1,
        latent_dim: int = 1,
        and_only_phys: bool = True,
        num_freqs: int = 10,
        activation: str = "ReLU",
        sigma: float = 1e-4,
        device: Optional[torch.device] = None,
    ):

        # build vf_nnet
        self.device = device if device is not None else torch.device("cpu")
        act_fun = act_dict.get(activation, act_dict["ReLU"])

        self.latent_dim = latent_dim
        input_conv_dim = 2 + latent_dim

        self.t_enc_dim = 2 * num_freqs
        self.t_dependent = vf_config.get("t_dependent", True)
        self.grid_size = vf_config.get("grid_size", 32)
        if self.t_dependent:
            input_conv_dim += 1  # self.t_enc_dim

        vf_nnet = torch.nn.Sequential(
            Conv2d(
                input_conv_dim, 16, 3, padding=1
            ), 
            act_fun(),
            Conv2d(16, 16, 3, padding=1),
            act_fun(),
            Conv2d(16, 2, 3, padding=1),
        ).to(self.device)
        # Initialize the parent class
        super(GBReactDiffNODE, self).__init__(
            drift_net=vf_nnet,
            phys_eq=phys_eq,
            state_dim=state_dim,
            phys_params_dim=phys_params_dim,
            and_only_phys=and_only_phys,
            latent_dim=latent_dim,
            device=device,
        )

    def time_encoding(self, t: Tensor) -> Tensor:
        return None

    def concatenate_input(self, x, p_sample, z_sample, t, t_enc):
        B, C, H, W = x.shape
        new_x = torch.cat(
            [
                x,
                z_sample.unsqueeze(2)
                .unsqueeze(3)
                .expand(-1, -1, x.shape[2], x.shape[3]),
            ],
            dim=1,
        )
        if self.t_dependent:
            new_x = torch.cat(
                [
                    new_x,
                    t.unsqueeze(2)
                    .unsqueeze(3)
                    .expand(-1, 1, x.shape[2], x.shape[3]),
                ],
                dim=1,
            )

        return new_x

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
        x0 = x0.reshape(x0.shape[0], -1, self.grid_size, self.grid_size)
        return super().sample_trajectory(
            x0=x0,
            phys_params=phys_params,
            z_sample=z_sample,
            t_span=t_span,
            ode_method=ode_method,
        )
