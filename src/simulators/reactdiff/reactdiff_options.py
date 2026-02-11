import argparse
from argparse import ArgumentParser
from src.simulators.reactdiff.reactdiff_model import HASHED_KEYS


class ReactDiffOptionParser:

    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_argument(
            "--outdir",
            type=str,
            default="./src/simulators/reactdiff/out",
        )

        self.parser.add_argument("--n_samples", type=int, default=1000)
        self.parser.add_argument("--n_stoch_samples", type=int, default=5)
        self.parser.add_argument("--grid_size", type=int, default=32)
        self.parser.add_argument("--len_episode", type=int, default=16)
        self.parser.add_argument("--t_span", nargs=2, default=[0.0, 1.5])
        self.parser.add_argument(
            "--mesh_size", type=float, default=0.06451612903
        )  # 2/31

        self.parser.add_argument(
            "--range-a", type=float, nargs=2, default=[1e-2, 1e-1]
        )
        self.parser.add_argument(
            "--range-b", type=float, nargs=2, default=[1e-2, 1e-1]
        )
        self.parser.add_argument("--k", type=float, default=0.005)
        self.parser.add_argument("--sigma", type=float, default=0.0)
        self.parser.add_argument("--seed", type=int, default=1234)
        self.parser.add_argument("--method", type=str, default="rk4")
        self.parser.add_argument(
            "--library", type=str, default="torch"
        )  # torch

    def parse_args(self):
        return self.parser.parse_args()

    def get_min_dict(args):
        if isinstance(args, argparse.Namespace):
            dict_args = args.__dict__
        if isinstance(args, dict):
            dict_args = args
        else:
            dict_args = vars(args)

        min_dict = {key: dict_args[key] for key in HASHED_KEYS}
        return min_dict
