import hashlib
import json

import random
import string
import subprocess
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


def get_exp_dir_path(exp_config, hash_task):
    dir_path = (
        f"{exp_config['out_path']}/{hash_task}/{exp_config['algorithm']}"
    )

    is_grey_box = exp_config["vf_model"].get("phys_model", False)
    dir_path += "/grey_box" if is_grey_box else "/black_box"

    dir_path += f"/{exp_config['model_exp_name']}"
    dir_path += f"/seed_{exp_config['seed']}"

    return dir_path


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def generate_salt(length=4):
    """Generate a random salt with the specified length. Number of possible combinations is 62^4 = 14776336."""
    characters = string.ascii_letters + string.digits
    salt = "".join(random.choice(characters) for _ in range(length))
    return salt


def get_gitsha():
    _gitsha = "gitSHA_{}"
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        gitsha = _gitsha.format(out.strip().decode("ascii"))
    except OSError:
        gitsha = "noGitSHA"
    return gitsha


# Create a unique id for the experiment using the config and the seed, gitsha
def create_uuid(*args, **kwargs):
    """Builds the uuid of the experiment."""
    uuid = uuid_basis(*args, **kwargs)
    config = kwargs.get("config", None)
    seed = kwargs.get("seed", None)

    assert seed is not None, "seed must be specified"
    assert config is not None, "config must be specified"

    algo = config["algo"]

    # Enrich the uuid with extra information
    uuid = f"{config['uuid']}.{algo}.{get_gitsha()}.{config['env_id']}"

    uuid += f".seed{str(seed).zfill(2)}"
    return uuid


def uuid_basis(*args, **kwargs):
    method = kwargs.get("method", "human_hash")
    add_salt = kwargs.get("add_salt", False)

    uuid = None
    if method == "syllables":
        uuid = uuid_syllables(*args, **kwargs)
    elif method == "hash":
        uuid = uuid_hash(*args, **kwargs)
    elif method == "parameters":
        uuid = uuid_parameters(*args, **kwargs)
    else:
        raise NotImplementedError

    if add_salt:
        salt_len = kwargs.get("salt_len", 4)
        salt = generate_salt(salt_len)
        uuid = f"{uuid}-{salt}"
    return uuid


def is_valid_key(key):
    if key.startswith("wandb"):
        return False
    if key in [
        "seed",
        "execution_time",
        "uuid",
        "task",
        "num_timestep",
        "expert_path",
    ]:
        return False

    return True


def uuid_parameters(min_config):
    """
    DO NOT USE THIS FUNCTION YET for uuid_basis
    """
    uuid = ""
    for key, value in min_config.items():
        if is_valid_key(key):
            uuid += f"{key}={value}-"

    uuid = uuid[:-1]
    return uuid


# Create a unique id for the experiment
def uuid_hash(min_config):
    uuid = dict_hash(min_config)
    return uuid


def get_uuid_hash_from_config(config, KEYS, discard_keys=[]):
    # check if config is a dictionary if not convert it to a dictionary
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    elif not isinstance(config, dict):
        config = vars(config)

    def deep_search_keys(dictionary):
        conf_exp_x = {}
        for key, value in dictionary.items():
            if key in KEYS and key not in discard_keys:
                if OmegaConf.is_list(value):
                    value = list(value)
                conf_exp_x[key] = value
            if isinstance(value, dict) or isinstance(value, DictConfig):
                nested_keys = deep_search_keys(value)
                conf_exp_x.update(nested_keys)
        return conf_exp_x

    conf_exp_x = deep_search_keys(config)
    u_x = uuid_hash(conf_exp_x)
    return u_x, conf_exp_x


def uuid_syllables(num_syllables=2, num_parts=3):
    """Randomly create a semi-pronounceable uuid, with num_syllable=2 and num_parts=3, number of possible combinations is 15^3 * 6^2 = 202500."""
    part1 = [
        "s",
        "t",
        "r",
        "ch",
        "b",
        "c",
        "w",
        "z",
        "h",
        "k",
        "p",
        "ph",
        "sh",
        "f",
        "fr",
    ]
    part2 = ["a", "oo", "ee", "e", "u", "er"]
    seps = ["_"]  # [ '-', '_', '.']
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for i in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for i in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2):
            result += part1[i1] + part2[i2]
    return result
