import logging
import torch
from torch.nn import Conv2d

from src.nn.mlp import ConditionalMLP, act_dict
from src.simulators.rlc.rlc_circuit import RLCODEDynamic
from src.simulators.pendulum.pendulum_model import PendulumDynamics
from src.simulators.reactdiff.reactdiff_model import ReactDiffDynamics
from src.models.vf_models.grey_box_fm import BaseGreyBoxFM
from src.models.vf_models.gb_reactdiff import GBReactDiffVF
from src.models.node.gb_reactdiff_node import GBReactDiffNODE
from src.models.node.grey_box_node import BaseGreyBoxNODE
from src.simulators.lorenz_attractor.lorenz_model import LorenzAttractorDynamics


from src.models.node.grey_box_node import BaseGreyBoxNODE


def _get_physical_equation(exp_name, vf_config, device, logger):
    """
    Factory function to create physical equation models based on experiment name.

    Args:
        exp_name: Name of the experiment
        vf_config: Vector field configuration dictionary
        device: Device to place the model on
        logger: Logger instance

    Returns:
        Physical equation model or None
    """
    phys_eq_map = {
        "rlc": lambda: RLCODEDynamic(device=device),
        "pendulum": lambda: PendulumDynamics(device=device),
        "reactdiff": lambda: ReactDiffDynamics(
            dx=vf_config.get("dx", 0.1), device=device
        ),
        "lorenz": lambda: LorenzAttractorDynamics(params=None, full=False).to(device),
    }

    if exp_name in phys_eq_map:
        return phys_eq_map[exp_name]()


    return None


def _should_use_and_only_phys(optimization_config):
    """
    Determine if only physical model should be used based on regularization weights.

    Args:
        optimization_config: Optimization configuration dictionary

    Returns:
        Boolean indicating if only physical model should be used
    """
    alpha = optimization_config.get("alpha", 0.0)
    beta = optimization_config.get("beta", 0.0)
    gamma = optimization_config.get("gamma", 0.0)

    return any(
        [
            alpha is not None and alpha > 0.0,
            beta is not None and beta > 0.0,
            gamma is not None and gamma > 0.0,
        ]
    )


def _get_latent_dim(exp_config):
    """
    Calculate the latent dimension based on algorithm type.

    Args:
        exp_config: Full experiment configuration

    Returns:
        Latent dimension size
    """
    enc_config = exp_config["enc_model"]
    return enc_config["z_dim"]


def _build_reactdiff_node_model(exp_config, phys_eq, device):
    """Build ReactDiff NODE model."""
    vf_config = exp_config["vf_model"]
    opt_config = exp_config["optimization"]

    and_only_phys = _should_use_and_only_phys(opt_config)

    return GBReactDiffNODE(
        vf_config=vf_config,
        phys_eq=phys_eq,
        and_only_phys=and_only_phys,
        state_dim=vf_config["dim_state"],
        phys_params_dim=exp_config["enc_model"]["p_dim"],
        latent_dim=exp_config["enc_model"]["z_dim"],
        activation=vf_config.get("activation", "ReLU"),
        num_freqs=vf_config.get("t_freq_dim", 10),
        sigma=vf_config.get("sigma", 1e-4),
        device=device,
    ).to(device)


def _build_reactdiff_fm_model(exp_config, phys_eq, device):
    """Build ReactDiff Flow Matching model."""
    vf_config = exp_config["vf_model"]
    z_dim = _get_latent_dim(exp_config)
    history_size = vf_config.get("history_size", 0)

    return GBReactDiffVF(
        vf_config=vf_config,
        phys_eq=phys_eq,
        state_dim=vf_config["dim_state"],
        phys_params_dim=exp_config["enc_model"]["p_dim"],
        latent_dim=z_dim,
        history_size=history_size,
        second_order=vf_config.get("second_order", False),
        activation=vf_config.get("activation", "ReLU"),
        num_freqs=vf_config.get("t_freq_dim", 10),
        sigma=vf_config.get("sigma", 1e-4),
        device=device,
    ).to(device)


def _build_conditional_mlp(exp_config, device):
    """
    Build a conditional MLP vector field network.

    Args:
        exp_config: Full experiment configuration
        device: Device to place the model on

    Returns:
        ConditionalMLP network
    """
    enc_config = exp_config["enc_model"]
    vf_config = exp_config["vf_model"]

    p_dim = enc_config.get("p_dim", 0)
    z_dim = _get_latent_dim(exp_config)
    dim_state = vf_config["dim_state"]
    t_freq_dim = vf_config["t_freq_dim"]
    history_size = vf_config.get("history_size", 0)
    if p_dim > 0 and vf_config.get("vf_phys", False):
        conditional_dim = (
            z_dim + p_dim + t_freq_dim * 2 + dim_state + (history_size) * dim_state
        )
    else:   
        conditional_dim = (
            z_dim + p_dim + t_freq_dim * 2 + (history_size) * dim_state
        )

    input_dim = (
        dim_state
        if not vf_config.get("second_order", False)
        else 1 + dim_state
    )

    return ConditionalMLP(
        input_dim=input_dim,
        condition_dim=conditional_dim,
        hidden_features=tuple(vf_config["hidden_dims"]),
        activation=vf_config.get("activation", "SELU"),
        multiple_heads=vf_config.get("second_order", False),
        head_layers=tuple(vf_config.get("head_layers", [])),
        normalize=True,
        spectral_norm=True,
        dropout=vf_config.get("dropout", 0.0),
        out_dim=dim_state,
    ).to(device)


def _build_base_grey_box_fm(exp_config, vf_nnet, phys_eq, device):
    """Build base grey-box flow matching model."""
    enc_config = exp_config["enc_model"]
    vf_config = exp_config["vf_model"]
    z_dim = _get_latent_dim(exp_config)

    return BaseGreyBoxFM(
        state_dim=vf_config["dim_state"],
        vf_nnet=vf_nnet,
        phys_params_dim=enc_config["p_dim"],
        latent_dim=z_dim,
        phys_eq=phys_eq,
        second_order=vf_config.get("second_order", False),
        use_vf_phys=vf_config.get("vf_phys", False),
        sigma=vf_config.get("sigma", 0.1),
        history_size=vf_config.get("history_size", 0),
    ).to(device)


def _build_base_grey_box_node(exp_config, drift_net, phys_eq, device):
    """Build base grey-box NODE model."""
    enc_config = exp_config["enc_model"]
    vf_config = exp_config["vf_model"]
    opt_config = exp_config["optimization"]

    and_only_phys = _should_use_and_only_phys(opt_config)

    return BaseGreyBoxNODE(
        state_dim=vf_config["dim_state"],
        drift_net=drift_net,
        phys_params_dim=enc_config["p_dim"],
        latent_dim=enc_config["z_dim"],
        phys_eq=phys_eq,
        and_only_phys=and_only_phys,
        sigma=vf_config.get("sigma", 0.1),
    ).to(device)


def get_vf(exp_config, device, logger):
    """
    Factory function to create vector field models based on experiment configuration.

    Args:
        exp_config: Full experiment configuration dictionary containing:
            - name_exp: Experiment name (e.g., 'rlc', 'pendulum', 'reactdiff')
            - algorithm: Algorithm type ('dyn-fm', or 'node')
            - vf_model: Vector field model configuration
            - enc_model: Encoder model configuration
            - optimization: Optimization configuration (for NODE models)
        device: Device to place models on (CPU or CUDA)
        logger: Logger instance for warnings and errors

    Returns:
        tuple: (vf_gb_model, phys_eq) - Vector field model and physical equation
    """
    if logger is None:
        logger = logging.getLogger("Factory Vector Field Model")

    vf_config = exp_config["vf_model"]
    exp_name = exp_config["name_exp"]
    algorithm = exp_config["algorithm"]

    # Get physical equation model
    enable_phys_model = vf_config.get("phys_model", False)

    if enable_phys_model:
        phys_eq = _get_physical_equation(exp_name, vf_config, device, logger)
    else:
        phys_eq = None
        logger.warning(
            "Physical equation not defined for the experiment. Using None."
        )

    # Build vector field model based on experiment type
    vf_gb_model = None

    # Special case: ReactDiff with specific models
    if exp_name == "reactdiff":
        if algorithm == "node":
            vf_gb_model = _build_reactdiff_node_model(
                exp_config, phys_eq, device
            )
        else:
            vf_gb_model = _build_reactdiff_fm_model(
                exp_config, phys_eq, device
            )

    # Generic models if not already built
    if vf_gb_model is None:
        # Build the neural network backbone
        vf_nnet = _build_conditional_mlp(exp_config, device)

        # Wrap in appropriate grey-box model based on algorithm
        if algorithm in ["dyn-fm"]:
            vf_gb_model = _build_base_grey_box_fm(
                exp_config, vf_nnet, phys_eq, device
            )

        elif algorithm == "node":
            vf_gb_model = _build_base_grey_box_node(
                exp_config, vf_nnet, phys_eq, device
            )

        else:
            raise ValueError(
                f"Vector field model not defined for algorithm '{algorithm}'. "
                "Valid algorithms: 'dyn-fm', 'node'."
            )

    return vf_gb_model, phys_eq
