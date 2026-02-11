import argparse
from src.io.utils import load_yaml, dict_to_argparse_input
from src.simulators.pendulum.options import PendulumOptionParser
from src.data.generate_dataset.pendulum.generate_data import generate_data


if __name__ == "__main__":
    # Parse arguments

    parser = argparse.ArgumentParser()

    # Add optional argument
    parser.add_argument(
        "-f",
        "--data_path_conf",
        default="./src/data/generate_dataset/pendulum/pendulum_experiments.yaml",
        help="Input file path for configuration",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="./experiments/dataset/pendulum/one_to_many/friction/training",
        help="Output file path for data",
    )
    parser.add_argument(
        "-m", "--mode", default="one-to-many", help="Mode for generating data"
    )
    parser.add_argument(
        "-idx",
        "--idx",
        default=0,
        type=int,
        help="Index for configuration generation",
    )
    parser.add_argument(
        "-s", "--seed", default=4321, type=int, help="Seed for data generation"
    )
    parser.add_argument(
        "--n-samples",
        default=1000,
        type=int,
        help="Number of samples to generate",
    )

    parser.add_argument(
        "--dt",
        default=0.1,
        type=float,
        help="Time step for simulation",
    )

    parser.add_argument(
        "--method",
        default="rk4",
        type=str,
        help="Method for simulation",
    )

    parser.add_argument(
        "--library",
        default="torch",
        type=str,
        help="Library for simulation",
    )

    args = parser.parse_args()

    # read yaml conf file
    conf = load_yaml(args.data_path_conf)
    if conf is None:
        raise ValueError("Configuration file not found")
    print("Configuration file loaded successfully!")

    # Load index configuration
    idx = args.idx
    mode = args.mode
    conf_exps = conf[mode]
    key_conf = list(conf_exps.keys())[idx]
    conf_pendulum = conf_exps[key_conf]

    # Add output directory to the configuration
    conf_pendulum["outdir"] = args.outdir
    # add seed and n_samples to the configuration
    conf_pendulum["seed"] = args.seed
    conf_pendulum["n-samples"] = args.n_samples
    conf_pendulum["dt"] = args.dt
    conf_pendulum["method"] = args.method
    conf_pendulum["library"] = args.library

    # create unique filename from configuration using mode and key of the configuration, nsample
    print(f"YAML configuration: \n\t{conf_pendulum}")

    # Parse configuration to the Pendulum
    pend_parser = PendulumOptionParser()
    conf_input = dict_to_argparse_input(conf_pendulum)
    print(f"Parsing configuration...\n\t{conf_input}")
    sims_args = pend_parser.parse_args(conf_input)
    print("Configuration parsed successfully!")

    dic_res = generate_data(sims_args)
    print("Data and true parameters generated successfully!")
