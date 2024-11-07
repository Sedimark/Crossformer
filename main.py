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
    Core wheel is the core function for the AI asset
"""
def core(df:pd.DataFrame, cfg: dict, flag: str = 'fit'):
    """
        Core function for the AI asset
        Args:
            df: pd.DataFrame, the input data
            cfg: dict, the config file
            flag: str, the flag for the function ('fit', or 'predict')
    """
    # mlflow
    mlflow_logger = MLFlowLogger(
                experiment_name='mlflow_logger_test',
                tracking_uri='http://127.0.0.1:8080',
            )
    
    # data loading
    if flag == 'predict':
        cfg['split'] = [0, 0, 1]
        # load the best model
        model = CrossFormer.load_from_checkpoint('mlruns/models/best_model.ckpt')
    else:
        model = CrossFormer(cfg)
    data = DataInterface(df, **cfg)

    # trainer
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

    if flag == 'fit':
        trainer.fit(model, data)
        test_result = trainer.test(model, data)
        return test_result
    elif flag == 'predict':
        return trainer.predict(model, data)
    else:
        print('Please specify the flag as "fit" or "predict"')
            
    
    

if __name__ == "__main__":
    with open('cfg.json') as f:
        cfg = json.load(f)
    result = core(pd.read_csv(cfg['csv_path']), cfg, 'predict')
    print(result)