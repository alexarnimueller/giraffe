#! /usr/bin/env python
# -*- coding: utf-8 -*-

""""
Script to benchmark GIRAFFE embeddings on tasks from Polaris.
"""

import logging
from copy import deepcopy

import click
import numpy as np
import pandas as pd
import polaris as po
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from featurizer import GiraffeFeaturizer
from model import FFNN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCHMARKS = [
    "graphium/tox21-v1",
    "graphium/zinc12k-v1",
    # "polaris/adme-fang-r-1",
    # "polaris/adme-fang-PERM-1",
    # "polaris/adme-fang-SOLU-1",
    # "polaris/adme-fang-RPPB-1",
    "novartis/adme-novartis-cyp3a4-reg",
    "biogen/adme-fang-reg-v1",
]


class BenchmarkDataset(Dataset):
    def __init__(self, features, labels=None, testing=False):
        super().__init__()
        self.features = features
        self.labels = labels
        self.testing = testing

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if self.testing:
            return self.features[index]
        return self.features[index], self.labels[index]


class EarlyStopping(object):
    def __init__(self, mode="higher", patience=10):
        assert mode in ["higher", "lower"]
        self.mode = mode
        if self.mode == "higher":
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.best_parameters = None
        self.best_score = None
        self.best_step = 0
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.best_step += self.counter + 1
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        self.best_parameters = deepcopy(model.state_dict())

    def load_checkpoint(self, model):
        model.load_state_dict(self.best_parameters)


def get_data(dataset):
    """Get benchmark splits from polaris to tabular format"""
    benchmark = po.load_benchmark(dataset)
    train, test = benchmark.get_train_test_split()
    y = train.y
    if not isinstance(y, dict):
        y = {dataset.split("/")[-1]: y}
    return benchmark, train.X.tolist(), pd.DataFrame.from_dict(y), test.X


def loss_with_nans(y_pred, y_true):
    mask = ~torch.isnan(y_true).to(DEVICE)
    diff = torch.abs(torch.flatten(y_pred[mask]) - torch.flatten(y_true[mask]))
    return torch.sum(diff) / mask.sum() if mask.sum() > 0 else 0.0


def train_one_epoch(model, train_loader, optimizer):
    total_loss = 0.0
    model.train()
    for batch in train_loader:
        feats, labs = batch
        feats, labs = feats.to(DEVICE), labs.to(DEVICE)
        optimizer.zero_grad()
        output = model(feats)
        loss = loss_with_nans(output, labs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def eval_one_epoch(model, valid_loader):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            feats, labs = batch
            feats, labs = feats.to(DEVICE), labs.to(DEVICE)
            output = model(feats)
            loss = loss_with_nans(output, labs)
            total_loss += loss.item()
    return total_loss / len(valid_loader)


def cv(pol_username, dataset, giraffe_model_ckpt, max_epochs, patience, lr, batch_size, n_jobs):
    giraffe = GiraffeFeaturizer(giraffe_model_ckpt, True, n_jobs=n_jobs)

    logger.info(f"Downloading and featurizing {dataset}...")
    benchmark, smls, y, smls_test = get_data(dataset)
    X = np.nan_to_num(giraffe.transform(smls)[0], nan=0.0)
    X_test = np.nan_to_num(giraffe.transform(smls_test)[0], nan=0.0)

    y_all = y.values
    X_train, X_val, y_train, y_val = train_test_split(X, y_all, test_size=0.1, random_state=4070)
    if y.shape[1] == 1:
        y_train, y_val, y_all = y_train.flatten(), y_val.flatten(), y.values.flatten()

    full_set = BenchmarkDataset(X, y_all)
    train_set = BenchmarkDataset(X_train, y_train)
    val_set = BenchmarkDataset(X_val, y_val)
    test_set = BenchmarkDataset(X_test, testing=True)
    full_loader = DataLoader(full_set, batch_size=batch_size, shuffle=False, num_workers=n_jobs, drop_last=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=n_jobs, drop_last=False)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=n_jobs, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=n_jobs, drop_last=False)

    # create new models and optimizers
    model = FFNN(
        input_dim=X.shape[1],
        hidden_dim=int(X.shape[1] / 2),
        output_dim=y.shape[1],
        n_layers=3,
        dropout=0.2,
    ).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(mode="lower", patience=patience)

    # train
    train_losses, eval_losses = [], []
    for e in range(max_epochs):
        loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = eval_one_epoch(model, valid_loader)
        train_losses.append(loss)
        eval_losses.append(val_loss)
        logging.info(f"Loss at epoch {e + 1:03d}: Train: {loss:.4f}, Val: {val_loss:.4f}")

        early_stop = stopper.step(val_loss, model)
        if early_stop:
            logger.info(f"Early stopping at epoch {e + 1}")
            logger.info(f"Lowest validation loss at epoch {stopper.best_step + 1}")
            break

    # recreate model and train with full training set for best_step epochs
    e_retrain = stopper.best_step + patience
    logger.info(f"Retraining on full training set for {e_retrain + 1} epochs...")
    model = FFNN(
        input_dim=model.input_dim,
        hidden_dim=model.hidden_dim,
        output_dim=model.output_dim,
        n_layers=model.n_layers,
        dropout=model.dropout,
    ).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)

    for e in range(e_retrain):
        loss = train_one_epoch(model, full_loader, optimizer)
    logging.info(f"Final training loss at epoch {e + 1:03d}: {loss:.4f}")

    # predict labels for test set
    logger.info("Predicting labels of test set...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            predictions.append(model(batch.to(DEVICE)))
    predictions = torch.cat(predictions).cpu().numpy()

    # evaluate predictions with polaris
    logger.info("Evaluating with polaris...")
    probs = pd.DataFrame(predictions, columns=y.columns)
    try:
        if len(y.columns) == 1:
            results = benchmark.evaluate(predictions.flatten())
        else:
            results = benchmark.evaluate({k: v.values.flatten() for k, v in probs.items()})
    except ValueError:
        cls = probs.map(lambda x: 0 if x < 0.5 else 1)
        probs = probs.map(lambda x: 0.0 if x < 0 else 1.0 if x > 1 else x)
        results = benchmark.evaluate(
            {k: v.values.flatten() for k, v in cls.items()}, {k: v.values.flatten() for k, v in probs.items()}
        )

    results.name = dataset.split("/")[-1] + "-GIRAFFE"
    results.github_url = "https://github.com/alexarnimueller/giraffe"
    results.paper_url = "https://openreview.net/forum?id=7WYcOGds6R"
    results.description = "GIRAFFE embeddings with a MLP."
    results.upload_to_hub(owner=pol_username)


@click.command()
@click.argument("polaris_username")
@click.option("-d", "--dataset", default=None, help="Dataset to use. None: all predfined ones.")
@click.option(
    "-m",
    "--giraffe_model_ckpt",
    default="models/pub_vae_sig_final/atfp_70.pt",
    help="Checkpoint of the trained Giraffe model.",
)
@click.option("-e", "--max_epochs", default=500, help="Maximum number of epochs to train.")
@click.option("-p", "--patience", default=10, help="Early stopping patience (epochs).")
@click.option("-l", "--lr", default=1e-3, help="Learning rate for the optimizer.")
@click.option("-b", "--batch_size", default=128, help="Batch size for model training.")
@click.option("-j", "--n_jobs", default=8, help="Number of cores to use for data loader.")
def main(pol_username, dataset, giraffe_model_ckpt, max_epochs, patience, lr, batch_size, n_jobs):
    if dataset is not None:
        print(f"Running benchmark for {dataset}")
        cv(pol_username, dataset, giraffe_model_ckpt, max_epochs, patience, lr, batch_size, n_jobs)
    else:
        for dataset in BENCHMARKS:
            print(f"Running benchmark for {dataset}")
            cv(pol_username, dataset, giraffe_model_ckpt, max_epochs, patience, lr, batch_size, n_jobs)


if __name__ == "__main__":
    main()
