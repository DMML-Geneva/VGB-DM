import os
from os.path import dirname

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from src.io.utils import get_hydra_output_dir

from src.runner.run_task import run_task
from src.utils import hash

config_dir = os.path.join(
    dirname(dirname(dirname(hash.__file__))), "experiments/scripts/configs"
)


@hydra.main(version_base="1.3", config_path=config_dir, config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Run benchmarking.
    """
    run_task(cfg)


if __name__ == "__main__":
    main()
