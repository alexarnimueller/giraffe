#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os

import click
import matplotlib.pyplot as plt

from model import anneal_cycle_linear, anneal_cycle_sigmoid

WDIR = os.path.expanduser("~/Code/Generative/GraphGiraffe")


@click.command()
def main():
    maxval = 0.2
    steps = 100000
    beta_lin = anneal_cycle_linear(steps, n_grow=4, n_cycle=10, ratio=0.75)
    beta_sig = anneal_cycle_sigmoid(steps, n_grow=20, n_cycle=20, ratio=0.75)
    beta_lin *= maxval
    beta_sig *= maxval
    best_lin = 45000
    best_sig = 70000

    fig = plt.figure(figsize=(9, 6))
    stride = max(int(steps / 8), 1)

    plt.subplot(211)
    plt.plot(
        range(steps),
        beta_lin,
        "-",
        label="Linear",
        color="r",
        markevery=stride,
        lw=2,
        mec="k",
        mew=1,
        markersize=10,
    )
    plt.vlines(best_lin, 0, maxval, color="k", linestyle="--", lw=2)

    plt.legend(fontsize=16, shadow=True, loc="right")
    plt.grid(True)

    plt.ylabel("$\\beta$", fontsize=16)

    # X-axis label
    plt.xticks(
        list(range(0, steps + 10000, 10000)),
        [""] * (int(steps / 10000) + 1),
        color="k",
        size=14,
    )

    # Left Y-axis labels
    plt.yticks((0.0, maxval / 2, maxval), ("0", str(maxval / 2), str(maxval)), color="k", size=14)

    plt.xlim(-10, steps)
    plt.ylim(maxval - maxval * 1.1, maxval * 1.1)

    plt.subplot(212)
    plt.plot(
        range(steps),
        beta_sig,
        "-",
        label="Sigmoid",
        color="blue",
        markevery=stride,
        lw=2,
        mec="k",
        mew=1,
        markersize=10,
    )
    plt.vlines(best_sig, 0, maxval, color="k", linestyle="--", lw=2)

    plt.legend(fontsize=16, shadow=True, loc="right")
    plt.grid(True)

    plt.xlabel("# Steps", fontsize=16, fontweight="bold")
    plt.ylabel("$\\beta$", fontsize=16, fontweight="bold")

    # X-axis label
    plt.xticks(
        list(range(0, steps + 10000, 10000)),
        (f"{10*i}k" for i in range(0, int(steps / 10000) + 1)),
        color="k",
        size=14,
        rotation=45,
    )

    # Left Y-axis labels
    plt.yticks((0.0, maxval / 2, maxval), ("0", str(maxval / 2), str(maxval)), color="k", size=14)

    plt.xlim(-10, steps)
    plt.ylim(maxval - maxval * 1.1, maxval * 1.1)

    plt.show()

    fig.savefig(f"{WDIR}/paper/figures/annealing_cyclical.png", bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
