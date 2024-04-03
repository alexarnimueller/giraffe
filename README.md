# Giraffe

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

![Logo](data/logo.png)

This repository contains training, embedding and inference code for the "Graph Infused Representation Adapted For Future molecule Enhancements" (Giraffe) model used to create meaningful molecular representations for small molecules.

## Quick start

### Training

### Sampling
To randomly sample up to `100` SMILES strings of maximum length `96` at temperature `0.6` from a trained model checkpoint (in this case epoch `140` of the model `irci`), run the following:
```bash
python sampling.py -e 140 -t 0.6 -l 96 -n 100 models/irci
```

Conditional sampling around a SMILES string of interest using epoch `140` of the pretrained model `irci`:
```bash
python sampling.py -e 140 -t 0.6 -l 96 -n 100 -s "CC1(CC(CC(N1)(C)C)OC2=NN=C(C=C2)C3=C(C=C(C=C3)C4=CNN=C4)O)C" models/irci
```
The sampled SMILES strings are stored in `output/sampled.csv` together with the negative log likelihood score.

### Embedding
To embed SMILES strings using the pretrained GNN, proceed as follows:
```bash
python embedding.py -f models/irci -e 140 data/1k.txt output/test/embeddings.csv
```