import argparse
from argparse import ArgumentParser


class PendulumOptionParser:
    HASHED_KEYS_NAME = "hashed_keys"

    HASHED_KEYS = [
        "A_dist",
        "dt",
        "f_dist",
        "gamma_dist",
        "len_episode",
        "n_samples",
        "n_stoch_samples",
        "noise_loc",
        "noise_std",
        "one_to_one",
        "range_A",
        "range_f",
        "range_gamma",
        "range_init",
        "range_omega",
        "seed",
        "single_sample_parameters",
        "method",
        "type",
        "library",
    ]

    def __init__(self) -> None:
        """Initialize the class."""
        parser = ArgumentParser()

        parser.add_argument(
            "--outdir", type=str, default="./experiments/dataset/pendulum"
        )
        parser.add_argument(
            "--device", type=str, default="cpu", help="cuda or cpu"
        )

        parser.add_argument("--method", type=str, default="rk4")

        parser.add_argument("--filename", type=str, default="simulations")

        # configurations
        parser.add_argument("--n-samples", type=int, default=2500)
        parser.add_argument("--len-episode", type=int, default=50)
        parser.add_argument("--dt", type=float, default=0.05)
        parser.add_argument(
            "--range-init", type=float, nargs="+", default=[-1.57, 1.57]
        )
        parser.add_argument(
            "--range-omega",
            type=float,
            nargs=2,
            default=[0.785, 3.14],
        )
        parser.add_argument(
            "--range-gamma",
            type=float,
            nargs="+",
            default=[0.0, 0.8],
            # required=False,
        )  # [0.0, 0.8]
        parser.add_argument("--gamma-dist", type=str, default="uniform")
        parser.add_argument(
            "--range-A", type=float, nargs="+", default=None, required=False
        )  # [0.0, 40.0]
        parser.add_argument("--A-dist", type=str, default="uniform")
        parser.add_argument(
            "--range-f", type=float, nargs="+", default=None, required=False
        )  # [3.14, 6.28]
        parser.add_argument("--f-dist", type=str, default="uniform")
        parser.add_argument(
            "--single-sample-parameters",
            type=int,
            default=0,
            help="If 0, each parameter is sampled independently. If 1, only used for bernoulli or categorical dist to sample parameters as single set.",
        )
        parser.add_argument(
            "--single-sample-params-probs",
            type=float,
            default=[0.5],
            nargs="+",
            help="If 0, each parameter is sampled independently. If 1, only used for bernoulli or categorical dist to sample parameters as single set.",
        )
        
        parser.add_argument(
            "--div_eps", type=float, default=None, required=False
        )

        # Trajectory noise
        parser.add_argument(
            "--noise-loc", type=float, required=False, default=None
        )
        parser.add_argument(
            "--noise-std", type=float, required=False, default=0.0
        )
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument(
            "--visualize", type=bool, required=False, default=False
        )
        parser.add_argument(
            "--n-stoch-samples",
            type=int,
            required=False,
            default=1,
            help="Number of stochastic samples for complete parameters, if range is given for them.",
        )
        parser.add_argument(
            "--one-to-one",
            type=int,
            required=False,
            default=None,
            help="If specified, the global parameters (gamma, A, f) are sampled once with the given int seed value",
        )

        parser.add_argument(
            "--shuffle_data",
            type=int,
            default=0,
            help="Shuffle the data before saving it",
        )

        parser.add_argument("--log-level", type=str, default="INFO")
        parser.add_argument("--library", type=str, default="torch")  # scipy

        self.parser = parser

    def parse_args(self, args=None):
        if args is not None:
            return self.parser.parse_args(args)
        return self.parser.parse_args()

    def get_min_dict(parse_args):
        if isinstance(parse_args, argparse.Namespace):
            dict_args = parse_args.__dict__
        else:
            dict_args = vars(parse_args)
        keys = list(
            filter(
                lambda key: key in PendulumOptionParser.HASHED_KEYS,
                dict_args.keys(),
            )
        )
        min_dict = {key: dict_args[key] for key in keys}
        return min_dict
