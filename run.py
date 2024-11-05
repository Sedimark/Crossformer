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
from crossformer.utils.tools import CustomCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def cli_main(cfg):
    # data loading
    df = pd.read_csv(cfg['csv_path'])
    data = DataInterface(df, **cfg)
    
    # model loading
    model = CrossFormer(cfg)

    # callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=os.getcwd(), save_top_k=1, monitor='val_loss', mode='min')

    # cli object
    cli = LightningCLI(
        model_class=model,
        datamodule_class=data,
        run=False,
        trainer_defaults={
            "callbacks":[early_stopping, checkpoint_callback]
        }
    )

    if cli.trainer.global_rank==0:
        mlflow.pytorch.autolog()    
    cli.trainer.fit(cli.model,datamodule=cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)

if __name__ == "__main__":
    # config loading
    with open('cfg.json') as f:
        cfg = json.load(f)
    cli_main(cfg)



# callbacks
# unique_run_dir = os.path.join(cfg.get("save_path"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
# mycallback = CustomCallback(path=unique_run_dir,**cfg)

# # seed
# pl.seed_everything(cfg['seed'])

# # load pretrained model if applicable
# if cfg.get("pretrained_model"):
#     model = CrossFormer.load_from_checkpoint(cfg['pretrained_model'])

# # trainer
# trainer = pl.Trainer(
#     accelerator=cfg['accelerator'],
#     precision=cfg['precision'],
#     min_epochs=cfg['min_epochs'],
#     max_epochs=cfg['max_epochs'],
#     check_val_every_n_epoch=1,
#     callbacks=[mycallback],
#     fast_dev_run=False,
# )

# # training
# trainer.fit(model, data)

# # testing
# saved_path = unique_run_dir +'/' + cfg.get('filename') +'.ckpt'
# trained_model = CrossFormer.load_from_checkpoint(saved_path,cfg=cfg)
# test_result = trainer.test(trained_model, data)
# print(test_result)

