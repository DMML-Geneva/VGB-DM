import argparse
import contextlib
import importlib
from numbers import Real

import math
import os
import random
import warnings
from functools import partial

import numpy as np
import torch
import torchdiffeq
import wandb
import yaml


def vmap_2d(func):
    return torch.vmap(func, in_dims=(0, 0), out_dims=0)


def vmap_3d(func):
    return torch.vmap(func, in_dims=(0, 0, 0), out_dims=0)


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Copied from: https://stackoverflow.com/a/34333710

    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


def set_seed(seed, seed_shift=1670):
    if seed is not None:
        seed += seed_shift
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        if torch.distributed.is_initialized():
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            env_context = contextlib.nullcontext()
        else:
            env_context = modified_environ(CUBLAS_WORKSPACE_CONFIG=":4096:8")
    else:
        env_context = contextlib.nullcontext()
    return env_context, np.random.RandomState(seed=seed)


def save_model(
    path,
    model,
    optimizer,
    lr_scheduler,
    best_val_metric=None,
    best_weights=None,
    ema_model=None,
    fname="model",
):
    checkpoint = {
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "best_val_metric": best_val_metric,
        "best_model": best_weights,
        "ema_model": (
            {k: v.cpu() for k, v in ema_model.module.state_dict().items()}
            if ema_model is not None
            else None
        ),
    }
    torch.save(checkpoint, os.path.join(path, f"{fname}.pt"))


def load_model(
    path=None,
    checkpoint=None,
    model=None,
    optimizer=None,
    lr_scheduler=None,
    best=False,
    ema=False,
    map_location=None,
    fname="model",
    strict=True,
):
    assert (
        path is not None or checkpoint is not None
    ), "One of `path` and `checkpoint` has to be provided!"
    assert not (
        path is not None and checkpoint is not None
    ), "Cannot prove both`path` and `checkpoint`!"
    if checkpoint is None:
        checkpoint = torch.load(
            os.path.join(path, f"{fname}.pt"), map_location=map_location
        )
    else:
        assert (
            map_location is None
        ), "Cannot provide `map_location` when `checkpoint` passed!"
    assert not (ema and best), "Cannot load ema and best at the same time!"
    if model is not None:
        if ema:
            key = "ema_model"
        elif best:
            key = "best_model"
        else:
            key = "model"
        model_checkpoint = checkpoint[key]
        model.load_state_dict(model_checkpoint, strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


def parse_torch_nn_class(cfg, default):
    if cfg is None:
        return default
    elif isinstance(cfg, dict):
        c = getattr(
            torch.nn,
            cfg.pop(
                "type",
            ),
        )
        if c_params := cfg.pop("parameters", False):
            c = partial(c, **c_params)
        return c
    else:
        return cfg


def torch_all_but_first_dim(tensor, operator=torch.sum):
    dims_to_sum = list(range(1, tensor.ndim))
    return operator(tensor, dim=dims_to_sum)


batched_trace = torch.vmap(torch.trace)


def sample_vf(
    y: torch.tensor,
    vf: callable,
    shape,
    sample_shape,
    device,
    omega: torch.tensor = None,
    solver_cfg=None,
    latent_sampler=None,
):
    if solver_cfg is None:
        solver_cfg = dict(
            method="euler",
            options=dict(step_size=1 / 128),
        )
    if latent_sampler is None:
        latent_sampler = torch.randn
    return torchdiffeq.odeint(
        lambda t, _x: vf(t, _x, y, omega=omega),
        latent_sampler(size=(*sample_shape, *shape), device=device),
        torch.linspace(0, 1, 2, device=device),
        **solver_cfg,
    )[1]


def augmented_forward(
    t,
    x,
    y,
    vf,
    omega=None,
):
    dx = vf(
        t,
        x,
        y,
        omega=omega,
    )
    jac = torch.vmap(
        torch.func.jacrev(
            lambda t, x, y, omega: vf(t, x, y, omega)
            .flatten(start_dim=1)
            .squeeze(dim=0),
            argnums=1,
        ),
        in_dims=(0, 0, 0, 0 if omega is not None else None),
    )(
        t.view(-1).expand(*y.shape),
        x,
        y,
        omega,
    )
    return dx, -batched_trace(jac)


def log_prob_vf(
    x: torch.tensor,
    y: torch.tensor,
    vf: callable,
    device,
    omega: torch.tensor = None,
    solver_cfg=None,
    prior=None,
):
    if solver_cfg is None:
        solver_cfg = dict(
            method="euler",
            options=dict(step_size=1 / 128),
        )
    if prior is None:
        prior = log_normal
    divergence = torch.zeros((x.shape[0],), device=device)
    f_prime = lambda _t, _x_divergence,: augmented_forward(
        _t,
        _x_divergence[0],
        y,
        vf=vf,
        omega=omega,
    )
    z, divergence = tuple(
        map(
            lambda trajectory: trajectory[1],
            torchdiffeq.odeint(
                f_prime,
                (x, divergence),
                torch.linspace(1, 0, 2, device=device),
                **solver_cfg,
            ),
        )
    )
    return prior(z) - divergence


def initialize_experiment(
    args,
    wandb_project,
    job_type,
    wandb_group_path=None,
):
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    os.makedirs(args.out_path, exist_ok=True)
    warnings.filterwarnings("default", category=CorrectorWarning)
    if "cuda" not in args:
        device = torch.device("cpu")
    else:
        use_cuda = torch.cuda.is_available() if args.cuda else False
        device = torch.device("cuda" if use_cuda else "cpu")
    set_torch_defaults(cfg["precision"])
    if wandb_group_path is None:
        if (exp_name := os.getenv("DVC_EXP_NAME")) is not None:
            wandb_group = f"{exp_name}--{args.experiment}--{wandb.util.generate_id()}--{args.model_seed}"
        else:
            wandb_group = f"{args.experiment}--{wandb.util.generate_id()}--{args.model_seed}"
        with open(os.path.join(args.out_path, "wandb_group"), "w") as f:
            f.write(wandb_group)
    else:
        with open(wandb_group_path, "r") as f:
            wandb_group = f.read()
    
    wandb_run = wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=f"{wandb_group}-{job_type}",
        tags=(args.experiment,),
        job_type=job_type,
    )
    return cfg, device, wandb_group, wandb_run


def get_simulator(cfg, device):
    simulator = (
        importlib.import_module(f"src.simulators.{cfg['name']}")
        .Simulator(**cfg["parameters"])
        .to(device)
    )
    return simulator


def get_meshgrid_log_dx(cfg, device):
    meshgrid = torch.meshgrid(
        torch.linspace(
            start=cfg["visualization"]["axis_range"]["xlim"][0],
            end=cfg["visualization"]["axis_range"]["xlim"][1],
            steps=cfg["visualization"]["resolution"],
            device=device,
        ),
        torch.linspace(
            start=cfg["visualization"]["axis_range"]["ylim"][0],
            end=cfg["visualization"]["axis_range"]["ylim"][1],
            steps=cfg["visualization"]["resolution"],
            device=device,
        ),
    )
    meshgrid_flat = (
        torch.stack(meshgrid, dim=0)
        .flatten(
            start_dim=1,
        )
        .t()
    )
    log_dx = (
        np.log(
            cfg["visualization"]["axis_range"]["xlim"][1]
            - cfg["visualization"]["axis_range"]["xlim"][0]
        )
        + np.log(
            cfg["visualization"]["axis_range"]["ylim"][1]
            - cfg["visualization"]["axis_range"]["ylim"][0]
        )
        - (2 * np.log(cfg["visualization"]["resolution"]))
    )
    return log_dx, meshgrid, meshgrid_flat


class CorrectorWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def eye_like(tensor):
    return torch.eye(*tensor.size(), device=tensor.device)


def log_normal(x: torch.tensor, sigma=1) -> torch.tensor:
    var = sigma**2
    log_scale = math.log(sigma) if isinstance(sigma, Real) else sigma.log()
    return torch_all_but_first_dim(
        -(x**2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    )
