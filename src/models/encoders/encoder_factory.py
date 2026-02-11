from src.models.encoders import VariationalEncoder
from src.models.encoders.reactdiff_enc import ReactDiffEncoder
from src.models.encoders.lorentz_encoder import LorenzEncoder

import torch


def get_encoder(exp_config, device):
    enc_config = exp_config["enc_model"]
    if exp_config["name_exp"] == "reactdiff":
        enc_model = ReactDiffEncoder(
            len_episode=enc_config["len_episode"],
            grid_size=enc_config["grid_size"],
            state_dim=enc_config["state_dim"],
            z_dim=enc_config["z_dim"],
            p_dim=enc_config["p_dim"],
            p_prior_type=enc_config["p_prior_type"],
            p_prior_mean=enc_config["p_prior_mean"],
            p_prior_std=enc_config["p_prior_std"],
            prior_U_bounds=torch.tensor(
                enc_config["prior_U_bounds"], device=device
            ),
            reflex_clamp=False,
            device=device,
        ).to(device)
    elif exp_config["name_exp"] == "lorenz":
        enc_model = LorenzEncoder(
            len_episode=enc_config["len_episode"],
            state_dim=enc_config["state_dim"],
            p_dim=enc_config["p_dim"],
            z_dim=enc_config["z_dim"],
            gru_hidden=enc_config["gru_hidden"],
            p_prior_type=enc_config["p_prior_type"],
            p_prior_mean=enc_config["p_prior_mean"],
            p_prior_std=enc_config["p_prior_std"],
            z_prior_std=enc_config.get("z_prior_std", 1.0),
            device=device,
        ).to(device)
    else:
        
        if "len_episode" not in enc_config:
            # backward compatibility
            enc_config["len_episode"] = enc_config.get("d_dim", 50) // enc_config.get("p_dim", 2)
        if "state_dim" not in enc_config:
            enc_config["state_dim"] = exp_config["vf_model"]["dim_state"]
            
        enc_model = VariationalEncoder(
            len_episode=enc_config["len_episode"],
            state_dim=enc_config["state_dim"],
            p_dim=enc_config["p_dim"],
            z_dim=enc_config["z_dim"],
            z_hidden_layers=enc_config["z_hidden_dims"],
            p_hidden_layers=enc_config["p_hidden_dims"],
            cleansing_net=enc_config["cleansing_net"],
            p_prior_type=enc_config["p_prior_type"],
            p_prior_mean=enc_config["p_prior_mean"],
            p_prior_std=enc_config["p_prior_std"],
            prior_U_bounds=torch.tensor(
                enc_config["prior_U_bounds"], device=device
            ),
            softplus_mean=enc_config.get("softplus_mean", True),
            device=device,
        ).to(device)
    return enc_model
