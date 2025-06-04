#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" "
Adapted from: https://github.com/BenevolentAI/MolBERT/blob/main/scripts/run_qsar_test_molbert.py
          and https://github.com/jrwnter/cddd/blob/master/cddd/evaluation.py

Module to to test the performance of the translation model to extract
    meaningfull features for a QSAR modelling. TWO QSAR datasets were extracted
    from literature:
    Ames mutagenicity: K. Hansen, S. Mika, T. Schroeter, A. Sutter, A. Ter Laak,
    T. Steger-Hartmann, N. Heinrich and K.-R. MuÌ´Lller, J. Chem.
    Inf. Model., 2009, 49, 2077–2081.
    Lipophilicity: Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse,
    A. S. Pappu, K. Leswing and V. Pande, Chemical Science, 2018,
    9, 513–530.
"""

import json
import os
import pickle
from datetime import datetime

import chembench
import click
import numpy as np
import pandas as pd

# from cddd.inference import InferenceModel
from chembench import load_data
from featurizer import GiraffeFeaturizer
from sklearn import metrics
from sklearn.svm import SVC, SVR


def get_data(dataset):
    """Check if exists, download if not, save splits return paths to separated splits"""
    df, indices = load_data(dataset)
    df = df.rename(columns={"smiles": "SMILES"})
    df.columns = [col.replace(" ", "_") for col in df.columns]
    return df, indices


def get_summary_df():
    chembench_path = os.path.dirname(chembench.__file__)
    with open(os.path.join(chembench_path, "notebook/summary.pkl"), "rb") as f:
        summary_df = pickle.load(f)

    # filter such that dataframe only contains datasets with a single task
    summary_df = summary_df[summary_df["n_task"] == 1]

    # filter out PDB tasks
    summary_df = summary_df[~summary_df["task_name"].str.contains("PDB")]

    return summary_df


def batchify(iterable, batch_size):
    for ndx in range(0, len(iterable), batch_size):
        batch = iterable[ndx : min(ndx + batch_size, len(iterable))]
        yield batch


def cv(dataset, summary_df, giraffe_model_ckpt, run_name):
    df, indices = get_data(dataset)
    giraffe = GiraffeFeaturizer(giraffe_model_ckpt)
    # ecfp = MorganFPFeaturizer(fp_size=2048, radius=2, use_counts=True, use_features=False)
    # rdkit_norm = PhysChemFeaturizer(normalise=True)

    def giraffe_fn(smiles):
        return giraffe.transform(smiles)[0]

    # def ecfp_fn(smiles):
    #     return ecfp.transform(smiles)[0]

    # def rdkit_norm_fn(smiles):
    #     return rdkit_norm.transform(smiles)[0]

    for i, (train_idx, valid_idx, test_idx) in enumerate(indices):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]

        # combine train and valid set as SVMs don't use a validation set, but NNs do.
        # this way they use the same amount of data.
        train_df = pd.concat([train_df, valid_df])
        test_df = df.iloc[test_idx]

        fn_combos = [
            ("giraffe", giraffe_fn),
            # ("ECFP4", ecfp_fn),
            # ("rdkit_norm", rdkit_norm_fn),
        ]

        for feat_name, feat_fn in fn_combos:
            train_features = np.nan_to_num(
                np.vstack(
                    [feat_fn(batch) for batch in batchify(train_df["SMILES"], 256)]
                ),
                nan=0.0,
            )
            train_labels = train_df[df.columns[-1]]

            test_features = np.nan_to_num(
                np.vstack(
                    [feat_fn(batch) for batch in batchify(test_df["SMILES"], 256)]
                ),
                nan=0.0,
            )
            test_labels = test_df[df.columns[-1]]

            mode = (
                summary_df[summary_df["task_name"] == dataset]
                .iloc[0]["task_type"]
                .strip()
            )

            np.random.seed(i)
            if mode == "regression":
                model = SVR(C=5.0)
            elif mode == "classification":
                model = SVC(C=5.0, probability=True)
            else:
                raise ValueError(
                    f"Mode has to be either classification or regression but was {mode}."
                )

            model.fit(train_features, train_labels)

            predictions = model.predict(test_features)

            if mode == "classification":
                # predict probabilities (needed for some metrics) and get probs of positive class ([:, 1])
                prob_predictions = model.predict_proba(test_features)[:, 1]
                metrics_dict = {
                    "AUROC": lambda: metrics.roc_auc_score(
                        test_labels, prob_predictions
                    ),
                    "AveragePrecision": lambda: metrics.average_precision_score(
                        test_labels, prob_predictions
                    ),
                    "Accuracy": lambda: metrics.accuracy_score(
                        test_labels, predictions
                    ),
                }
            else:
                metrics_dict = {
                    "MAE": lambda: metrics.mean_absolute_error(
                        test_labels, predictions
                    ),
                    "RMSE": lambda: np.sqrt(
                        metrics.mean_squared_error(test_labels, predictions)
                    ),
                    "MSE": lambda: metrics.mean_squared_error(test_labels, predictions),
                    "R2": lambda: metrics.r2_score(test_labels, predictions),
                }

            metric_values = {}
            for name, callable_metric in metrics_dict.items():
                try:
                    metric_values[name] = callable_metric()
                except Exception as e:
                    print(f"unable to calculate {name} metric")
                    print(e)
                    metric_values[name] = np.nan

            default_path = os.path.join("./logs/benchmark/", run_name)
            output_dir = os.path.join(default_path, dataset)
            os.makedirs(output_dir, exist_ok=True)
            with open(
                os.path.join(output_dir, f"{feat_name}_metrics_{str(i)}.json"), "w+"
            ) as fp:
                json.dump(metric_values, fp)


@click.command()
@click.option(
    "-m",
    "--giraffe_model_ckpt",
    default="models/wae_pub_final/atfp_70.pt",
    help="Checkpoint of the trained Giraffe model.",
)
@click.option("-n", "--name", default=None, help="Name of the benchmark run.")
def main(giraffe_model_ckpt, name):
    if not name:
        name = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    summary_df = get_summary_df()

    for dataset in summary_df["task_name"].unique():
        print(f"Running experiment for {dataset}")
        cv(dataset, summary_df, giraffe_model_ckpt, name)


if __name__ == "__main__":
    main()
