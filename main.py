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
import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import MLFlowLogger
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

    # model initialization
    model = CrossFormer(cfg)
    
    # data loading
    df = pd.read_csv(cfg['csv_path'])
    data = DataInterface(df, **cfg)
    
    # mlflow
    mlflow_logger = MLFlowLogger(
                experiment_name='mlflow_logger_test',
                tracking_uri='http://127.0.0.1:8080'
            )
    
    trainer = pl.Trainer(
        accelerator=cfg['accelerator'],
        precision=cfg['precision'],
        min_epochs=cfg['min_epochs'],
        max_epochs=cfg['max_epochs'],
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        callbacks=[model_ckpt, early_stop],
        logger=mlflow_logger
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    core()