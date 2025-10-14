# Learning-To-Measure

Code for reproducing Learning-To-Measure: In-Context Active Feature Acquisition

## Install
We recommend using Conda with the provided `environment.yml` file.

```bash
conda env create -f environment.yml
```

## Quickstart

Begin by pretraining predictor on synthetic tasks:

```bash
python main.py --experiment sim --mode train
```

Once the predictor is trained, pretrain the policy on synthetic tasks:

```bash
python main.py --experiment sim --mode train_afa
```

Evaluate acquisition performance:

```bash
python main.py --experiment sim --mode test_afa
```
