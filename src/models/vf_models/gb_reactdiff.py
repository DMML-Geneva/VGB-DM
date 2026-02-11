from typing import Callable, Optional
import torch

import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from src.nn.mlp import act_dict

from src.models.vf_models.grey_box_fm import BaseGreyBoxFM


class GBReactDiffVF(BaseGreyBoxFM):
    def __init__(
        self,
        vf_config,
        phys_eq: Optional[Callable] = None,
        state_dim: int = 1,
        phys_params_dim: int = 1,
        latent_dim: int = 1,
        history_size: int = 0,
        second_order: bool = False,
        use_vf_phys: bool = False,
        activation: str = "ReLU",
        num_freqs: int = 10,
        sigma: float = 1e-4,
        device: Optional[torch.device] = None,
    ):

        self.device = device if device is not None else torch.device("cpu")
        # build vf_nnet
        act_fun = act_dict.get(activation, act_dict["ReLU"])

        self.latent_dim = latent_dim
        self.history_size = history_size

        in_channel_dim = 2 + latent_dim

        self.t_enc_dim = 2 * num_freqs
        self.t_dependent = vf_config.get("t_dependent", True)
        if self.t_dependent:
            in_channel_dim += 1  # , check also self.t_enc_dim
        if self.history_size > 0:
            in_channel_dim += self.history_size * state_dim

        self.n_filters = vf_config.get("n_filters", 16)
        self.n_layers = vf_config.get("hidden_layers", 3)
        convs = []
        for i in range(self.n_layers - 1):
            convs.append(Conv2d(in_channel_dim, self.n_filters, 3, padding=1))
            convs.append(act_fun())
            in_channel_dim = self.n_filters

        vf_nnet = torch.nn.Sequential(
            *convs,
            Conv2d(self.n_filters, 2, 3, padding=1),
        ).to(self.device)

        # Initialize the parent class
        super(GBReactDiffVF, self).__init__(
            vf_nnet=vf_nnet,
            phys_eq=phys_eq,
            state_dim=state_dim,
            phys_params_dim=phys_params_dim,
            latent_dim=latent_dim,
            history_size=history_size,
            use_vf_phys=use_vf_phys,
            second_order=second_order,
        )

    def time_encoding(self, t: Tensor) -> Tensor:
        return None

    def concatenate_input(
        self,
        x,
        p_sample,
        z_sample,
        t,
        t_enc,
        phys_vf: Tensor = None,
        dt_xt: Optional[Tensor] = None,
        x_history: Optional[Tensor] = None,
    ):
        B, C, H, W = x.shape

        if z_sample is not None:
            new_x = torch.cat(
                [
                    x,
                    z_sample.unsqueeze(2)
                    .unsqueeze(3)
                    .expand(-1, -1, x.shape[2], x.shape[3]),
                ],
                dim=1,
            )
        else:
            new_x = x

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

        if self.history_size > 0 and x_history is not None:

            new_x = torch.cat(
                [
                    new_x,
                    x_history.flatten(start_dim=1, end_dim=2),
                ],
                dim=1,
            )

        return new_x
