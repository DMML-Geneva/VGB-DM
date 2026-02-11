import math

import torch
from zuko.nn import LayerNorm

from src.utils.torch_utils import parse_torch_nn_class


class MLP(torch.nn.Module):
    # Based on zuko.nn.MLP

    def __init__(
        self,
        input_dim,
        out_dim,
        hidden_features=(64, 64),
        activation=None,
        normalize: bool = False,
        dropout: float = 0.0,
        inplace: bool = True,
        spectral_norm: bool = False,
        spectral_norm_config: dict = None,
        **kwargs,
    ):
        if spectral_norm_config is None:
            spectral_norm_config = dict()
        super().__init__()
        activation = parse_torch_nn_class(activation, torch.nn.ReLU)

        normalization = LayerNorm if normalize else lambda: None
        dropout_layer = lambda: (
            torch.nn.Dropout(dropout, inplace=inplace) if dropout > 0 else None
        )

        if isinstance(input_dim, (tuple, list)):
            input_dim = math.prod(input_dim)

        layers = []

        for before, after in zip(
            (input_dim, *hidden_features),
            (*hidden_features, out_dim),
        ):
            layers.extend(
                [
                    dropout_layer(),
                    (
                        torch.nn.Linear(before, after, **kwargs)
                        if not spectral_norm
                        else torch.nn.utils.parametrizations.spectral_norm(
                            torch.nn.Linear(before, after, **kwargs),
                            **spectral_norm_config,
                        )
                    ),
                    activation(inplace=inplace),
                    normalization(),
                ]
            )

        layers = layers[1:-2]
        layers = filter(lambda l: l is not None, layers)

        self.layers = torch.nn.Sequential(*layers)

        self.in_features = input_dim
        self.out_features = out_dim
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)


class AugmentedLinear(torch.nn.Linear):
    def __init__(self, n_pass, *args, **kwargs):
        super(AugmentedLinear, self).__init__(*args, **kwargs)
        self.n_pass = n_pass

    def forward(self, input):
        return torch.cat(
            (
                input[..., : self.n_pass],
                super(AugmentedLinear, self).forward(input),
            ),
            dim=-1,
        )


class PartStraightThrough(torch.nn.Module):
    def __init__(self, n_pass, module):
        super(PartStraightThrough, self).__init__()
        self.module = module
        self.n_pass = n_pass

    def forward(self, input):
        return torch.cat(
            (
                input[..., : self.n_pass],
                self.module(input[..., self.n_pass :]),
            ),
            dim=-1,
        )


act_dict = {
    "ReLU": torch.nn.ReLU,
    "Softplus": torch.nn.Softplus,
    "SELU": torch.nn.SELU,
    "SiLU": torch.nn.SiLU,
}


class ConditionalMLP(torch.nn.Module):
    # Modified, backward-compatible version

    def __init__(
        self,
        input_dim,
        condition_dim,
        out_dim,
        hidden_features=(64, 64),
        multiple_heads: bool = False,
        head_layers=(64, 64),
        activation=None,
        normalize: bool = False,
        dropout: float = 0.0,
        inplace: bool = True,
        spectral_norm: bool = False,
        spectral_norm_config: dict = None,
        **kwargs,
    ):
        if spectral_norm_config is None:
            spectral_norm_config = dict()
        super().__init__()
        activation = act_dict.get(activation, activation)
        if isinstance(input_dim, (tuple, list)):
            input_dim = math.prod(input_dim)

        self.in_features = input_dim
        self.out_features = out_dim
        self.condition_dim = condition_dim
        self.multiple_heads = multiple_heads and len(head_layers) > 0

        if not self.multiple_heads:
            head_1 = self.build_layers(
                input_dim,
                condition_dim,
                hidden_features + (out_dim,),
                activation,
                normalize,
                dropout,
                inplace,
                spectral_norm,
                spectral_norm_config,
                kwargs,
            )
            head_1 = head_1[1:-2]  # same filtering as before
            head_1 = filter(lambda l: l is not None, head_1)
            self.head_1 = torch.nn.Sequential(*head_1)
            self.head_2 = None
            self.backbone = None
        else:

            backbone_layers = hidden_features
            # Multi-head: shared backbone``
            backbone = self.build_layers(
                input_dim,
                condition_dim,
                backbone_layers,
                activation,
                normalize,
                dropout,
                inplace,
                spectral_norm,
                spectral_norm_config,
                kwargs,
            )
            backbone = backbone[1:]
            backbone = filter(lambda l: l is not None, backbone)
            self.backbone = torch.nn.Sequential(*backbone)

            # head_1: velocity
            head1 = self.build_layers(
                backbone_layers[-1],
                condition_dim,
                head_layers + (out_dim,),
                activation,
                normalize,
                dropout,
                inplace,
                spectral_norm,
                spectral_norm_config,
                kwargs,
            )
            head1 = head1[:-2]  # drop final activation/normalization
            head1 = filter(lambda l: l is not None, head1)

            # head_2: acceleration
            head2 = self.build_layers(
                input_dim + out_dim,
                condition_dim,
                head_layers + (out_dim,),
                activation,
                normalize,
                dropout,
                inplace,
                spectral_norm,
                spectral_norm_config,
                kwargs,
            )
            head2 = head2[:-2]
            head2 = filter(lambda l: l is not None, head2)

            self.head_1 = torch.nn.Sequential(*head1)
            self.head_2 = torch.nn.Sequential(*head2)

        self.apply(self._init_weights)

    def build_block(
        self,
        condition_dim,
        before,
        after,
        activation,
        normalization,
        dropout_layer,
        inplace,
        spectral_norm,
        spectral_norm_config,
        kwargs,
    ):
        return [
            (
                PartStraightThrough(condition_dim, dropout_layer())
                if dropout_layer() is not None
                else None
            ),
            (
                AugmentedLinear(condition_dim, before, after, **kwargs)
                if not spectral_norm
                else torch.nn.utils.parametrizations.spectral_norm(
                    AugmentedLinear(condition_dim, before, after, **kwargs),
                    **spectral_norm_config,
                )
            ),
            PartStraightThrough(condition_dim, activation(inplace=inplace)),
            (
                PartStraightThrough(condition_dim, normalization())
                if normalization() is not None
                else None
            ),
        ]

    def build_layers(
        self,
        input_dim,
        condition_dim,
        features,
        activation,
        normalize,
        dropout,
        inplace,
        spectral_norm,
        spectral_norm_config,
        kwargs,
    ):
        features_in = [hf + condition_dim for hf in features]
        normalization = LayerNorm if normalize else lambda: None
        dropout_layer = lambda: (
            torch.nn.Dropout(dropout, inplace=inplace) if dropout > 0 else None
        )
        layers = []
        for before, after in zip(
            (input_dim + condition_dim, *features_in),
            (*features,),
        ):
            layers.extend(
                self.build_block(
                    condition_dim,
                    before,
                    after,
                    activation,
                    normalization,
                    dropout_layer,
                    inplace,
                    spectral_norm,
                    spectral_norm_config,
                    kwargs,
                )
            )
        return layers

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        if not self.multiple_heads:
            return self.head_1(x)[..., self.condition_dim :]

        # new backbone + two heads
        backbone_out = self.backbone(x)
        out_1 = self.head_1(backbone_out)[..., self.condition_dim :]
        out_2 = self.head_2(torch.cat((x, out_1.detach()), dim=-1))[
            ..., self.condition_dim :
        ]
        return torch.cat((out_1, out_2), dim=-1)


class ConditionalZeroMLP(ConditionalMLP):
    def __init__(self, init_mean=0.0, init_std=0.01, *args, **kwargs):
        self.init_mean = init_mean
        self.init_std = init_std
        super(ConditionalZeroMLP, self).__init__(*args, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(
                module.weight.data, mean=self.init_mean, std=self.init_std
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias.data)
