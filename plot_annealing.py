import matplotlib.pyplot as plt
import numpy as np

from model import anneal_cycle_linear, anneal_cycle_sigmoid

total_steps = 40000

for t in ["sigmoid", "linear"]:
    if t == "sigmoid":
        beta_np_cyc = anneal_cycle_sigmoid(total_steps, 0.0, 1.0, 4)
        beta_np_inc = anneal_cycle_sigmoid(total_steps, 0.0, 1.0, 1, 0.25)
    else:
        beta_np_cyc = anneal_cycle_linear(total_steps, 0.0, 1.0, 4)
        beta_np_inc = anneal_cycle_linear(total_steps, 0.0, 1.0, 1, 0.25)

    beta_np_con = np.ones(total_steps)

    fig = plt.figure(figsize=(6, 4.0))
    stride = max(int(total_steps / 8), 1)

    plt.subplot(211)
    plt.plot(
        range(total_steps),
        beta_np_inc,
        "-",
        label="Monotonic",
        marker="o",
        color="r",
        markevery=stride,
        lw=2,
        mec="k",
        mew=1,
        markersize=10,
    )

    leg = plt.legend(fontsize=16, shadow=True, loc=(0.6, 0.2))
    plt.grid(True)

    plt.ylabel("$\\beta$", fontsize=16)

    ax = plt.gca()

    # X-axis label
    plt.xticks(
        (0, 5 * 1000, 10 * 1000, 15 * 1000, 20 * 1000, 25 * 1000, 30 * 1000, 35 * 1000, 40 * 1000),
        (" ", " ", " ", " ", " ", " ", " ", " ", " "),
        color="k",
        size=14,
    )

    # Left Y-axis labels
    plt.yticks((0.0, 0.5, 1.0), ("0", "0.5", "1"), color="k", size=14)

    plt.xlim(-10, total_steps)
    plt.ylim(-0.1, 1.1)

    plt.subplot(212)
    plt.plot(
        range(total_steps),
        beta_np_cyc,
        "-",
        label="Cyclical",
        marker="s",
        color="k",
        markevery=stride,
        lw=2,
        mec="k",
        mew=1,
        markersize=10,
    )

    leg = plt.legend(fontsize=16, shadow=True, loc=(0.6, 0.2))
    plt.grid(True)

    plt.xlabel("# Iteration", fontsize=16)
    plt.ylabel("$\\beta$", fontsize=16)

    ax = plt.gca()

    # X-axis label
    plt.xticks(
        (0, 5 * 1000, 10 * 1000, 15 * 1000, 20 * 1000, 25 * 1000, 30 * 1000, 35 * 1000, 40 * 1000),
        ("0", "5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"),
        color="k",
        size=14,
    )

    # Left Y-axis labels
    plt.yticks((0.0, 0.5, 1.0), ("0", "0.5", "1"), color="k", size=14)

    plt.xlim(-10, total_steps)
    plt.ylim(-0.1, 1.1)

    plt.show()

    fig.savefig(f"annealing_{t}.png", dpi=300, bbox_inches="tight")
    plt.close()
