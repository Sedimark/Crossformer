

# Crossformer

## What is it?

crossformer is a python package for multivariate time series data forecasting. The original idea is from the paper [Crossformer: A Transformer Model for Multivariate Time Series Forecasting](https://openreview.net/forum?id=vSVLM2j9eie). The package is designed to be easy to use and modular, so you can easily extend it to suit your needs. The package is implemented with the [lightning framework](https://pytorch-lightning.readthedocs.io/en/stable/) to reduce the boilerplate code.

## Key Features

- Transformer based model
- Data Processing
- Model Training & Evaluation
- Model Inference

## Installation

The source code can be found on GitHub at [https://github.com/Sedimark/Surrey_AI](https://github.com/Sedimark/Surrey_AI).

Binary installer for the latest released version is available on PyPI at [https://pypi.org/project/crossformer/](https://pypi.org/project/crossformer/).

To install the package, you can use pip:

```bash
pip install crossformer
```

## Project Structure
```text
.
├── crossformer (python package)
│   ├── data_tools
│   │   ├── __init__.py
│   │   └── data_interface.py
│   ├── model
│   │   ├── layers
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── decoder.py
│   │   │   ├── embedding.py
│   │   │   └── encoder.py
│   │   ├── __init__.py
│   │   └── crossformer.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── tools.py
│   └── __init__.py
├── scripts (wrap scripts)
│   ├── debugging.py
│   └── demo.csv
│   ├── demo.json
│   └── main.py
├── tests
│   ├── __init__.py
│   └── test_basic.py
├── LICENSE
├── MANIFEST.in
├── README.md
├── README_package.md
├── pyproject.toml
└── setup.py
```
The project structure is shown as above. If you want to continue develop on the existing project, you can refer to this package structure.

## Getting Staterd

This package is designed to be easy to use. Therefore, we implemented with lightning framework to reduce the boilerplate code. The package is designed to be modular, so you can easily extend it to suit your needs. However, there are three key sections that you need to be aware of when using the package. To get started with the package, you can follow the following sections:

### Configurations

Configuration files are used to set up the parameters for the model, data and experiment settings. A **dict** structure will be passed to the package. Therefore, you can select your prefer format for the configuration file. The basic structure of the configuration dict is as follows:

```python
cfg = {
    "data_dim": 8,              # number of features
    "in_len": 24,               # input time length
    "out_len": 24,              # output time length
    "seg_len": 2,               # segment length
    "window_size": 4,           # window size for segment merge
    "factor": 10,               # scaling factor (reduce the computation)
    "model_dim": 256,           # the hiden model dimension
    "feedforward_dim": 512,     # feedforward dimension
    "head_num": 4,              # number of attention heads
    "layer_num": 6,             # number of layers
    "dropout": 0.2,             # dropout rate
    "baseline": False,          # whether to use baseline
    "learning_rate": 0.1,       # learning rate
    "batch_size": 8,            # batch size
    "split": [
        0.7,
        0.2,
        0.1
    ],                          # split ratio for train, validation and test
    "seed": 2024,               # random seed
    "accelerator": "auto",      # accelerator for training (e.g. "gpu", "cpu", "tpu")
    "min_epochs": 1,            # minimum number of epochs
    "max_epochs": 200,          # maximum number of epochs
    "precision": 32,            # precision for training (e.g. 16, 32)
    "patience": 5,              # patience for early stopping
    "num_workers": 31,          # number of workers for data loading
}
```
You can modify and add the parameters according to your needs. 

### Data Preparation

Inside the package, we provide the data interface for loading data to the model (trainer). Therefore, you can easily pass your data (pandas.DataFrame) to the data interface. Currently, we only support the 2D data (pandas.DataFrame) for the model. And the data should be values-only, excluding the timestamps, column names and other information (metadata). 

Here is an example of randomly generated data:
```python
import pandas as pd
import numpy as np

sample_df = pd.DataFrame(np.random.rand(400, 8)) # randomly generated data 400 time steps and 8 features
```

Here is an example of how to load your data:
> We assume that the data is values-only and in the format of pandas.DataFrame. If you are not sure about your data format, please check the above generated data sample and follow it. Also, please be aware that your configurations should be compatible with your data. For example, the data_dim should be equal to the number of features in your data.

```python
from crossformer.data_tools import DataInterface

dm = data = DataInterface(df, **cfg) # df is the data (pandas.DataFrame) and cfg is the configuration dict
```

### Model Training & Evaluation
The package is implemented with lightning framework to reduce the boilerplate code. If you are not familiar with the lightning framework, please check the [lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/) for more information. We provide a very simple example of how to use the package for training and evaluation. 

> We assume that you have alrady installed the package and gone throught the above sections.

```python
from crossformer.data_tools import DataInterface
from crossformer.model.crossformer import CrossFormer
from lightning.pytorch import Trainer

# load the configuration file or use the sample above

# generate random data and initialize the data interface
sample_df = pd.DataFrame(np.random.rand(400, 8))
dm = DataInterface(sample_df, **cfg) 

# initialize the model
model = CrossFormer(**cfg)

# fit the model
trainer = Trainer(
            accelerator=cfg["accelerator"],
            precision=cfg["precision"],
            min_epochs=cfg["min_epochs"],
            max_epochs=cfg["max_epochs"],
            check_val_every_n_epoch=1,
            fast_dev_run=False,
        )
trainer.fit(model, datamodule=dm)

# evaluate the model
trainer.test(model, datamodule=dm)
```

## Additonal Information

We also provide some wrap scripts for the package usage. In this section, we provide the demo script for easily sage. Please note that the script can be modified and extended according to your needs. 

> Step 1: Install the package and additional dependencies for wrap scripts
```bash
pip install crossformer
pip install mlflow
```
We will use MLFlow for the experiment tracking. If you are not familiar with MLFlow, please check the [MLFlow documentation](https://www.mlflow.org/docs/latest/index.html) for more information. You can also use other tools for the experiment tracking, but you need to modify the script or pass the corresponding Logger to the trainer.

> Step 2: Prepare the data and configuration file

We put the data and configuration files for the demo under the path `./scripts/`. You can modify the data and configuration files according to your needs. Noted that the configuration file follows the above structure.

> Step 3: Run the script

First, we need to run the MLFlow server. You can run the following command in the terminal to start the MLFlow server:
```bash
python mlflow ui
```
Then, you can run the wrap script to start the training and evaluation on the demo settings. 
```bash
python scripts/main.py
```
You can also track the training via MLFlow UI. The MLFlow UI will be available at [http://localhost:5000](http://localhost:5000). You can check the training and evaluation results in the UI as well.

> Step 4: Extension

We also capsulate the training and inference features into a signle core function inside of the wrap script.
```python
core(
    df: pd.DataFrame,
    cfg: dict,
    flag: str = "fit",
):
    """Core function for the AI asset.

    Args:
        df (pd.DataFrame): The input data.
        cfg (dict): The configuration dictionary.
        flag (str): The flag for the function, either 'fit' or 'predict', which
            defaults to "fit".

    Returns:
       test_result (List): trained model's result on test set if flag 'fit'.
       predict_result (torch.tensor): predictions if flag 'predict'.
    """
```
If you want to use the core features for your own data without heavly modifications, you can follow this helpful interface.

## Acknowledgement


This software has been developed by the [University of Surrey](https://www.surrey.ac.uk/) under the [SEDIMARK(SEcure Decentralised Intelligent Data MARKetplace)](https://sedimark.eu/) project. SEDIMARK is funded by the European Union under the Horizon Europe framework programme [grant no. 101070074]. This project is also partly funded by UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant no. 10043699].
