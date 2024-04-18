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


@click.command()
@click.option("-d", "--directory", default="logs/benchmark", help="Directory containing benchmark result files")
def main(directory):
    json_results = combine_json_files(directory)
    rslt = average_results(json_results)
    for k, v in rslt.items():
        v.to_csv(os.path.join(directory, f"{k}_summary.csv"))


if __name__ == "__main__":
    main()
