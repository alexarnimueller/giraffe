#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from collections import defaultdict

import click
import pandas as pd


def combine_json_files(directory: str = "logs/benchmark"):
    """Walk through the benchmark logs and combine files belonging to the same model.

    :param directory: directory to search for results, defaults to "logs"
    :type directory: str, optional
    :return: nested dictionary containing the contents of the found results files
    :rtype: dictionary
    """
    json_results = defaultdict(lambda: defaultdict(list))
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".json"):
                descriptor = os.path.splitext(file_name)[0]
                descriptor = descriptor.split("_")[0]
                with open(os.path.join(root, file_name)) as f:
                    benchmark = root.split("/")[-1]
                    json_results[descriptor][benchmark].append(json.load(f))
    return json_results


def average_results(json_results: dict):
    """Get the average and std.dev. from the aggregated benchmark results provided in a nested dictionary.

    :param json_results: nested dictionary containing the benchmark results
    :type json_results: dict
    :return: dictionary containing the averaged results per model
    :rtype: dictionary of pandas DataFrames
    """
    rslt = {}
    for k in json_results.keys():
        ind_rslt = {}
        for e in json_results[k].keys():
            means = pd.DataFrame.from_dict(json_results[k][e]).mean()
            stdds = pd.DataFrame.from_dict(json_results[k][e]).std()
            idx = [i + "_avg" for i in means.index] + [i + "_std" for i in stdds.index]
            tmp = pd.concat((means, stdds))
            tmp.index = idx
            ind_rslt[e] = tmp
        rslt[k] = pd.DataFrame.from_dict(ind_rslt).T
    return rslt


def get_dfs(directory, metrics, round):
    json_results = combine_json_files(directory)
    rslt = average_results(json_results)
    out = {}
    for k, v in rslt.items():
        out[k] = v[[c for c in v.columns if any([m in c for m in metrics.split(",")])]].round(round)
    return out


@click.command()
@click.argument("directory")
@click.option("-m", "--metrics", default="AUROC,RMSE", help="Comma-separated list of metrics to keep")
@click.option("-r", "--round", default=3, help="Number of decimal places to round to")
def main(directory, metrics, round):
    # read all benchmarks and build dictionary
    rslt = {m: {} for m in metrics.split(",")}
    for path in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, path)):
            d = get_dfs(os.path.join(directory, path), metrics, round)
            for n, df in d.items():
                if all([c in df.columns for c in [f"{m}_avg" for m in metrics.split(",")]]):
                    df = df.sort_values(by=[f"{m}_avg" for m in metrics.split(",")]).T
                    for m in metrics.split(","):
                        m_out = {}
                        for i, (a, s) in enumerate(zip(df.loc[f"{m}_avg"], df.loc[f"{m}_std"])):
                            if not pd.isna(a):
                                m_out[df.columns[i]] = f"{a} +/- {s}"
                        rslt[m][f"{path}_{n}"] = m_out
    # save aggregated results to csv files
    for m, v in rslt.items():
        df = pd.DataFrame().from_dict(v, orient="index").sort_index()
        df.to_csv(os.path.join(directory, f"{m}.csv"), sep=";", index=True, header=True)


if __name__ == "__main__":
    main()
