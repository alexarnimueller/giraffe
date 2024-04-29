# Giraffe

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

<img src="data/logo.png" alt="giraffe" width="200"/>

This repository contains training, embedding and inference code for the "Graph Infused Representations Adapted For Future molecule Enhancements" (Giraffe) model used to create meaningful molecular representations for small molecules.

## Quick start
### Training
Training a new model on a file with SMILES strings can be achieved as follows:
```bash
python train.py -n chembl data/chembl24_10uM_20-100_proc.txt
```
To get all the options, call `python train.py --help`:
```
Usage: train.py [OPTIONS] FILENAME

Options:
  -n, --run_name TEXT         Name of the run for saving (filename if
                              omitted).
  -d, --delimiter TEXT        Column delimiter of input file.
  -c, --smls_col TEXT         Name of column that contains SMILES.
  -e, --epochs INTEGER        Nr. of epochs to train.
  -o, --dropout FLOAT         Dropout fraction.
  -b, --batch_size INTEGER    Number of molecules per batch.
  -r, --random                Randomly sample molecules in each training step.
  -es, --epoch_steps INTEGER  If random, number of steps per epoch.
  -v, --val FLOAT             Fraction of the data to use for validation.
  -l, --lr FLOAT              Learning rate.
  -f, --lr_factor FLOAT       Factor for learning rate decay.
  -s, --lr_step INTEGER       Step size for learning rate decay.
  -a, --after INTEGER         Epoch steps to save model.
  -p, --n_proc INTEGER        Number of CPU processes to use.
  --help                      Show this message and exit.
```

After training, a config file containing all the used options will be saved in the checkpoints folder. This file is used for later sampling and embedding tasks.

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

To get all available options, call `python sampling.py --help`:
```
Usage: sampling.py [OPTIONS] CHECKPOINTFOLDER

Options:
  -e, --epoch TEXT      Epoch of models to load.
  -s, --smiles TEXT     Reference SMILES to use as seed for sampling.
  -n, --num INTEGER     How many molecules to sample.
  -t, --temp FLOAT      Temperature to use for multinomial sampling.
  -l, --maxlen INTEGER  Maximum allowed SMILES string length.
  --help                Show this message and exit.
```

### Embedding
To embed SMILES strings using the pretrained GNN, proceed as follows:
```bash
python embedding.py -f models/irci -e 140 data/1k.txt output/test/embeddings.csv
```
To get all available options, call `python embedding.py --help`:
```
Usage: embedding.py [OPTIONS] INPUT_FILE OUTPUT_FILE

Options:
  -d, --delimiter TEXT      Column delimiter of input file.
  -c, --smls_col TEXT       Name of column that contains SMILES.
  -f, --folder TEXT         Checkpoint folder to load models from.
  -e, --epoch TEXT          Epoch of models to load.
  -b, --batch_size INTEGER  Batch size to use for embedding.
  -n, --n_jobs INTEGER      Number of cores to use for data loader.
  --help                    Show this message and exit.
```

## Benchmark
To benchmark the obtained representation, use `benchmark.py`. 
It relies on the [Chembench](https://github.com/shenwanxiang/ChemBench) and optionally the [CDDD](https://github.com/jrwnter/cddd) repositories. 
Please follow the installation instructions described in their READMEs.



### References

#### KLD Weight Annealing
https://github.com/haofuml/cyclical_annealing
