"""
    Description: The demo script for the CrossFormer
    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""
import os
import json
import mlflow
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from datetime import datetime
from crossformer.data_tools.data_interface import DataInterface
from crossformer.model.crossformer import CrossFormer
from crossformer.utils.tools import model_ckpt, early_stop

#  cli functions
"""
    CLi functions is waiting for the implementation in the future
"""

# core_wheel
"""
    Core function for the AI asset
"""
def core():
    # config loading
    with open('cfg.json') as f:
        cfg = json.load(f)
    print(cfg)

    # model initialization
    model = CrossFormer(cfg)
    print(model)

    # data loading
    df = pd.read_csv(cfg['csv_path'])
    data = DataInterface(cfg)
    print(data)
    
if __name__ == "__main__":
    core()