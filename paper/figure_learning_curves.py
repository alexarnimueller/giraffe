#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A python script to plot learning curves for the different models from tensorboard logs to matplotlib figures."""

import os

import click
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WDIR = os.path.expanduser("~/Code/Generative/GraphGiraffe")

MAXVALS = {"total": 40, "smiles": 0.3, "props": 0.025, "kld": 200}


def plot_training_curves(event_acc, log_path, name_train, name_val, y_label="Loss"):
    # Check if names are known in logs
    if name_train not in event_acc.Tags()["scalars"] or name_val not in event_acc.Tags()["scalars"]:
        print(f"Skipping unknown name_test or name_val, available: {event_acc.Tags()['scalars']}")
        return None

    smls_train = event_acc.Scalars(name_train)
    smls_val = event_acc.Scalars(name_val)

    plt.plot([i.step for i in smls_train], [i.value for i in smls_train], label="training")
    plt.plot([i.step for i in smls_val], [i.value for i in smls_val], label="validation")
    if name_train.split("_")[-1] in MAXVALS:
        a, _ = plt.ylim()
        plt.ylim([max(0, a), MAXVALS[name_train.split("_")[-1]]])
    plt.xlabel("Steps")
    plt.ylabel(y_label)
    plt.legend(loc="best", frameon=True)
    plt.savefig(f"{WDIR}/paper/figures/learning_curves_{os.path.basename(log_path)}_{name_train.split('_')[-1]}.pdf")
    plt.close()


def plot_single_curve(event_acc, log_path, name, y_label="Loss"):
    # Check if names are known in logs
    if name not in event_acc.Tags()["scalars"]:
        print(f"Skipping unknown name, available: {event_acc.Tags()['scalars']}")
        return None
    vals = event_acc.Scalars(name)

    plt.plot([i.step for i in vals], [i.value for i in vals])
    plt.xlabel("Steps")
    plt.ylabel(y_label)
    plt.savefig(f"{WDIR}/paper/figures/learning_curves_{os.path.basename(log_path)}_{name}.pdf")
    plt.close()


@click.command()
@click.option("-l", "--log_dir", default=f"{WDIR}/logs/pub_vae_lin_final")
@click.option("-n", "--n_steps", default=10000)
def main(log_dir, n_steps):
    print("Loading Tensorflow logs...")
    tf_size_guidance = {"compressedHistograms": 0, "images": 0, "scalars": n_steps, "histograms": 0}
    event_acc = EventAccumulator(log_dir, tf_size_guidance)
    event_acc.Reload()

    for t, v, lab in [
        ("loss_train_total", "loss_val_total", "Total loss"),
        ("loss_train_smiles", "loss_val_smiles", "SMILES reconstruction loss"),
        ("loss_train_props", "loss_val_props", "Property MSE loss"),
        ("loss_train_kld", "loss_val_kld", "KLD Loss"),
    ]:
        print("Plotting", t, "and", v)
        plot_training_curves(event_acc, log_dir, t, v, lab)

    for n in ["kld_weight", "lr", "valid"]:
        print("Plotting", n)
        plot_single_curve(event_acc, log_dir, n, n)


if __name__ == "__main__":
    main()
