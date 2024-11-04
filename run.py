"""
    Description: The demo script for the CrossFormer
    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""
import os
import json
import mlflow
from pytorch_lightning.loggers import MLFlowLogger
import pandas as pd
import pytorch_lightning as pl
from datetime import datetime
from crossformer.data_tools.data_interface import DataInterface
from crossformer.model.crossformer import CrossFormer
from crossformer.utils.tools import CustomCallback
import mlflow.pytorch

# config loading
with open('cfg.json') as f:
    cfg = json.load(f)
 
# mlflow setting


# data loading
df = pd.read_csv(cfg['csv_path'])
data = DataInterface(df, **cfg)

# model loading
model = CrossFormer(cfg)

# callbacks
unique_run_dir = os.path.join(cfg.get("save_path"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
mycallback = CustomCallback(path=unique_run_dir,**cfg)

# seed
pl.seed_everything(cfg['seed'])

# load pretrained model if applicable
if cfg.get("pretrained_model"):
    model = CrossFormer.load_from_checkpoint(cfg['pretrained_model'])

# trainer
trainer = pl.Trainer(
    accelerator=cfg['accelerator'],
    precision=cfg['precision'],
    min_epochs=cfg['min_epochs'],
    max_epochs=cfg['max_epochs'],
    check_val_every_n_epoch=1,
    callbacks=[mycallback],
    fast_dev_run=False,
)

# training
trainer.fit(model, data)

# testing
saved_path = unique_run_dir +'/' + cfg.get('filename') +'.ckpt'
trained_model = CrossFormer.load_from_checkpoint(saved_path,cfg=cfg)
test_result = trainer.test(trained_model, data)
print(test_result)

