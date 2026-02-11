import argparse
import os
import yaml

from omegaconf import OmegaConf, DictConfig
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils import hash
from src.train.train_vi_grey_box_fm import train_model
from src.train.train_node_physvae import train_physvae

from src.models.encoders import get_encoder
from src.models.vf_models.vf_factory import get_vf


from src.data.tensor_dataset.dataset_factory import get_dataset


parser = argparse.ArgumentParser(
    description="Train Corrector of Conditional FM with CFG"
)

parser.add_argument(
    "--config_file",
    type=str,
    metavar="CONFIG",
    default="./experiments/scripts/configs/exp/lorenz.yaml",
    help="Path of the experiment configuration file",
)


# Set up logger
logger = logging.getLogger("GB Matching")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)


def run_task(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config is None or not bool(config):
        raise ValueError("Experiment configuration is not provided or empty.")
    wandb_config = config.get("wandb", {})
    exp_config = config.get("exp", config)
    if isinstance(exp_config, OmegaConf) or isinstance(exp_config, DictConfig):
        exp_config = OmegaConf.to_container(exp_config, resolve=True)
    logger.info(f"Experiment configuration: {config}")

    # load wandb config if not exists
    if not bool(wandb_config):
        wandb_config_file = exp_config.get("wandb_config_file", None)
        if wandb_config_file and os.path.exists(wandb_config_file):
            with open(wandb_config_file, "r") as f:
                wandb_config = yaml.safe_load(f)
            wandb_config["run_wandb_log"] = wandb_config.get(
                "run_wandb_log", False
            )
            if wandb_config["run_wandb_log"] == False:
                wandb_config = {}
        else:
            wandb_config = {}
    # Make out_path if it does not exist
    if not os.path.exists(exp_config["out_path"]):
        os.makedirs(exp_config["out_path"], exist_ok=True)

    # Seed for reproducibility
    exp_config["seed"] = exp_config.get("seed", 0)
    torch.manual_seed(exp_config["seed"])
    np.random.seed(exp_config["seed"])
    torch.cuda.manual_seed_all(exp_config["seed"])
    logger.info(f"Using device: {device}")
    logger.info(f"Using seed: {exp_config['seed']}")

    # get dataset from dataset factory.
    tr_dataset, val_dataset = get_dataset(exp_config)
    logger.info(f"Training dataset: {len(tr_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Define data loaders
    train_loader = DataLoader(
        tr_dataset,
        batch_size=exp_config["optimization"].get("batch_size", 64),
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=exp_config["optimization"].get("batch_size", 64),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Define Models
    # Encoder
    logger.info("Initializing encoder model...")
    # Ensure the encoder is only created if p_dim or z_dim > 0
    if (
        exp_config["enc_model"]["z_dim"] == 0
        and exp_config["enc_model"]["p_dim"] == 0
    ):
        enc_model = None
    else:
        enc_model = get_encoder(exp_config, device)
        logger.info(f"Encoder model: {enc_model}")

    # Vector Field Model
    logger.info("Initializing vector field model...")
    vf_model, phys_eq = get_vf(
        exp_config, device, logger=logger
    )  # Ensure vf_config is defined
    logger.info(f"Vector Field Model: {vf_model}")

    # Define optimizer
    opt_params = [
        {
            "params": vf_model.parameters(),
            "lr": float(exp_config["optimization"]["vf_lr"]),
            "weight_decay": float(
                exp_config["optimization"].get("vf_weight_decay", 0.0)
            ),
        }
    ]

    if enc_model is not None:
        opt_params.append(
            {
                "params": enc_model.parameters(),
                "lr": float(exp_config["optimization"]["enc_lr"]),
                "weight_decay": float(
                    exp_config["optimization"].get("enc_weight_decay", 0.0)
                ),
            }
        )
    optimizer = torch.optim.AdamW(
        params=opt_params,
        betas=(0.9, 0.999),
        eps=1e-08,
        amsgrad=True,
    )
    logger.info(f"Optimizer: {optimizer}")

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=exp_config["optimization"]["n_epochs"],
        eta_min=1e-6,
    )
    logger.info(f"Scheduler: {scheduler}")

    # Hash experiment name and model
    hashed_task, min_conf = hash.get_uuid_hash_from_config(
        exp_config,
        KEYS=["name_exp", "dataset"],
        discard_keys=["sampler"],
    )
    exp_config["group_exp_name"] = hashed_task
    hashed_model, _ = hash.get_uuid_hash_from_config(
        exp_config,
        KEYS=[
            "name_exp",
            "algorithm",
            "dataset",
            "enc_model",
            "vf_model",
            "optimization",
        ],
    )
    exp_config["model_exp_name"] = hashed_model
    logger.info(
        f"\n Hashed exp task uuid: {hashed_task}"
        f"\n Hashed exp training-exp-model uuid: {hashed_model}"
    )

    # Save configuration
    # Train model
    if exp_config["algorithm"] == "dyn-fm":
        (
            vf_model,
            enc_model,
            optimizer,
            train_losses,
            val_losses,
            chkp_score,
        ) = train_model(
            vf_model=vf_model,
            enc_model=enc_model,
            exp_config=exp_config,
            wandb_config=wandb_config,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=exp_config["optimization"]["n_epochs"],
            val_step=exp_config["training"].get("log_val_step", 20),
            early_stopping=exp_config["training"].get("early_stopping", True),
            logger=logger,
        )
    else:
        (
            vf_model,
            enc_model,
            optimizer,
            train_losses,
            val_losses,
            chkp_score,
        ) = train_physvae(
            node_model=vf_model,
            enc_model=enc_model,
            exp_config=exp_config,
            wandb_config=wandb_config,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=exp_config["optimization"]["n_epochs"],
            val_step=exp_config["training"].get("log_val_step", 20),
            early_stopping=exp_config["training"].get("early_stopping", True),
            logger=logger,
        )
    return vf_model, enc_model, optimizer, train_losses, val_losses, chkp_score


def main():
    args, unknown = parser.parse_known_args()
    yaml_path = args.config_file
    exp_config = OmegaConf.load(yaml_path)
    cli_conf = OmegaConf.from_dotlist(unknown)
    exp_config = OmegaConf.merge(exp_config, cli_conf)
    # Convert omegaconf to a dictionary
    exp_config = OmegaConf.to_container(exp_config, resolve=True)
    logger.info(
        f"Configuring experiment {exp_config['name_exp']} with config file {yaml_path}"
    )
    vf_model, enc_model, optimizer, train_losses, val_losses, chkp_score = (
        run_task(exp_config)
    )


if __name__ == "__main__":
    main()
