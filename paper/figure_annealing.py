import click
import matplotlib.pyplot as plt

from model import anneal_cycle_linear, anneal_cycle_sigmoid

WDIR = "/home/muela115/Code/Generative/GraphGiraffe"


@click.command()
@click.option("-s", "--steps", default=200000)
@click.option("-g", "--grow", default=4)
@click.option("-c", "--cycle", default=10)
@click.option("-r", "--ratio", default=0.75)
def main(steps, grow, cycle, ratio):
    beta_lin = anneal_cycle_linear(steps, n_grow=grow, n_cycle=cycle, ratio=ratio)
    beta_sig = anneal_cycle_sigmoid(steps, n_grow=grow, n_cycle=cycle, ratio=ratio)

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
    plt.yticks((0.0, 0.5, 1.0), ("0", "0.5", "1"), color="k", size=14)

    plt.xlim(-10, steps)
    plt.ylim(-0.1, 1.1)

    plt.subplot(212)
    plt.plot(
        range(steps),
        beta_sig,
        "-",
        label="Sigmoid",
        color="k",
        markevery=stride,
        lw=2,
        mec="k",
        mew=1,
        markersize=10,
    )

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
    plt.yticks((0.0, 0.5, 1.0), ("0", "0.5", "1"), color="k", size=14)

    plt.xlim(-10, steps)
    plt.ylim(-0.1, 1.1)

    plt.show()

    fig.savefig(f"{WDIR}/paper/figures/annealing_cyclical.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
