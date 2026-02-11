import os
import pandas as pd
import numpy as np

import glob
from matplotlib import pyplot as plt


def average_seed_evaluations(root_dir, group_by_label="method"):
    all_dfs = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "evaluation_results.csv" in filenames:
            csv_path = os.path.join(dirpath, "evaluation_results.csv")
            parent_name = os.path.basename(
                dirpath
            )  # unique name from parent folder
            # get the parent directory name of dirpath
            gn_path = os.path.dirname(os.path.dirname(dirpath))
            bb_gb = os.path.basename(gn_path)
            method_name = os.path.basename(os.path.dirname(gn_path))

            df = pd.read_csv(csv_path)
            df["seed"] = parent_name  # keep track of source
            df["method"] = method_name + "-" + bb_gb
            all_dfs.append(df)

    # concatenate all results
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(
            os.path.join(root_dir, "combined_results.csv"),
            index=False,
        )
        # convert any non-numeric and not a string columns to zero
        
        # for the column that starts with param* if the value are not not numeric, convert to zero
        for col in combined_df.columns:
            if col not in [group_by_label, "seed", "model_exp_name", "method"]:
                combined_df[col] = pd.to_numeric(
                    combined_df[col], errors="coerce"
                ).fillna(0)
        
        # select only numeric columns for aggregation
        numeric_cols = combined_df.select_dtypes(include="number").columns

        grouped_df = combined_df.groupby(group_by_label)[numeric_cols].agg(
            ["mean", "std"]
        )

        # flatten multiindex columns
        grouped_df.columns = [
            f"{col[0]}_{col[1]}" for col in grouped_df.columns
        ]
        grouped_df = grouped_df.reset_index()

        grouped_df.to_csv(
            os.path.join(root_dir, "stats_seed_results.csv"),
            index=False,
        )

        return grouped_df