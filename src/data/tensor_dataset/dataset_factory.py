from .rlc_dataset import RLCDataset
from .reactdiff_dataset import ReactDiffDataset
from .pendulum_dataset import PendulumDataset
from .lorenz_dataset import LorenzDataset


def get_dataset(exp_config, only_testset=False):
    data_traj_length = exp_config["sampler"].get("max_length", None)
    if data_traj_length is None:
        data_traj_length = exp_config["enc_model"].get("len_episode", None)
        # if data_traj_length is None:
        #    enc_input_dim = exp_config["enc_model"].get("d_dim", None)
        #    vf_state_dim = int(exp_config["vf_model"]["dim_state"])
        #    data_traj_length = enc_input_dim // vf_state_dim

    history_size = exp_config["vf_model"].get("history_size", 0)
    if exp_config["name_exp"] == "rlc":
        tr_dataset = (
            None
            if only_testset
            else RLCDataset(
                data_path=exp_config["dataset"]["data_path_tr"],
                mode=exp_config["sampler"]["sampler_mode"],
                max_length=data_traj_length,
                history_size=history_size,
                sampler_config=exp_config["sampler"],
            )
        )
        val_dataset = RLCDataset(
            data_path=exp_config["dataset"]["data_path_va"],
            mode="trajectory",
            sample_params=True,
            sampler_config=exp_config["sampler"],
        )
    elif exp_config["name_exp"] == "pendulum":
        tr_dataset = (
            None
            if only_testset
            else PendulumDataset(
                data_path=exp_config["dataset"]["data_path_tr"],
                mode=exp_config["sampler"]["sampler_mode"],
                max_length=data_traj_length,
                n_cons_points=(
                    2
                    if (
                        exp_config["sampler"]["sampler_mode"] == "seq-pairs"
                        or exp_config["sampler"]["sampler_mode"]
                        == "pairs-history"
                    )
                    and exp_config["vf_model"].get("interpolation", "linear")
                    == "lagrange"
                    else 1
                ),
                history_size=history_size,
                sampler_config=exp_config["sampler"],
            )
        )
        val_dataset = PendulumDataset(
            data_path=exp_config["dataset"]["data_path_va"],
            mode="trajectory",
            sample_params=True,
        )
    elif exp_config["name_exp"] == "reactdiff":
        tr_dataset = (
            None
            if only_testset
            else ReactDiffDataset(
                data_path=exp_config["dataset"]["data_path_tr"],
                mode=exp_config["sampler"]["sampler_mode"],
                max_length=data_traj_length,
                history_size=history_size,
                sampler_config=exp_config["sampler"],
            )
        )
        val_dataset = ReactDiffDataset(
            data_path=exp_config["dataset"]["data_path_va"],
            mode="trajectory",
            sample_params=True,
        )
    elif exp_config["name_exp"] == "lorenz":
        tr_dataset = (
            None
            if only_testset
            else LorenzDataset(
                data_path=exp_config["dataset"]["data_path_tr"],
                mode=exp_config["sampler"]["sampler_mode"],
                max_length=data_traj_length,
                history_size=history_size,
                sampler_config=exp_config["sampler"],
            )
        )
        val_dataset = LorenzDataset(
            data_path=exp_config["dataset"]["data_path_va"],
            mode="trajectory",
            sample_params=True,
        )
    else:
        raise ValueError(
            "Task not found in the configuration file. Please provide a valid dataset."
        )
    return tr_dataset, val_dataset
