#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions to run inference on trained giraffe models"""

import os
from multiprocessing import cpu_count

import torch

from giraffe.embedding import smiles2embedding
from giraffe.model import load_models
from giraffe.sampling import embedding2smiles
from giraffe.utils import is_valid_mol

_default_model_name = "pub_wae"
_default_epoch = 70
_default_model_dir = os.path.join(
    os.path.dirname(__file__), f"models/{_default_model_name}"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class InferenceModel(object):
    """Class that handles the inference of a trained model."""

    def __init__(
        self,
        model_dir=_default_model_dir,
        epoch=_default_epoch,
        batch_size=256,
        temp=0.5,
        maxlen=256,
    ):
        """Constructor for the inference model.

        Args:
            model_dir: Path to the model directory.
            epoch: Model epoch to use
            batch_size: Number of samples to process per step.
            temp: Temperature to use for decoding
            maxlen: Maximum SMILES string length for sampling
        Returns:
            None
        """
        self.model_encoder, self.model_decoder, self.config = load_models(
            model_dir, epoch
        )
        self.batch_size = batch_size
        self.temp = temp
        self.maxlen = maxlen

    def random_emb(self, n):
        """Sample random embeddings"""
        return torch.randn(n, self.model_decoder.hidden_dim).detach().cpu().numpy()

    def seq_to_emb(self, smls):
        """Encode SMILES strings to embeddings"""
        if isinstance(smls, str):  # single SMILES
            smls = [smls]
        return smiles2embedding(
            smls,
            self.model_encoder,
            self.batch_size,
            max(cpu_count() - 2, 1),
            self.config["vae"] == "True",
        )

    def emb_to_seq(self, embeddings):
        """Decode embeddings to SMILES strings"""
        seqs = []
        for hn in torch.from_numpy(embeddings):
            hn = torch.reshape(hn, (1, -1)).to(DEVICE)
            s = embedding2smiles(hn, self.model_decoder, self.temp, self.maxlen)
            _, smls = is_valid_mol(s, True)
            seqs.append(smls)
        return seqs
