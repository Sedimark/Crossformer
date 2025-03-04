"""Main file.

The main file for training CrossFormer or predicting

Author: Peipei Wu (Paul) - Surrey
Maintainer: Peipei Wu (Paul) - Surrey
"""

import json
from crossformer.data_tools.data_interface import DataInterface
from crossformer.model.crossformer import CrossFormer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from lightning.pytorch import Trainer
import torch
import mlflow.pytorch
# from pytorch_lightning.cli import LightningCLI

#  cli functions
# TODO: waiting for implementation in future


# TODO: change to decorator later
def core(df: pd.DataFrame, cfg: dict, flag: str = "fit"):
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
    torch.set_float32_matmul_precision("medium")

    # data loading
    if flag == "predict":
        cfg["split"] = [0, 0, 1]
        # load the best model
        model = CrossFormer.load_from_checkpoint(
            "mlruns/models/best_model.ckpt"
        )
    else:
        model = CrossFormer(cfg)
    data = DataInterface(df, **cfg)

    # callbacks
    model_ckpt = ModelCheckpoint(
        monitor='val_SCORE',
        mode='min',
        save_top_k=1,
        save_weights_only=False,
    )

    early_stop = EarlyStopping(
        monitor='val_SCORE',
        patience=cfg["patience"],
        mode='min',
    )

    # trainer
    trainer = Trainer(
        accelerator=cfg["accelerator"],
        precision=cfg["precision"],
        min_epochs=cfg["min_epochs"],
        max_epochs=cfg["max_epochs"],
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        callbacks=[model_ckpt, early_stop],
    )

    if flag == "fit":
        mlflow.pytorch.autolog(
            checkpoint_monitor='val_SCORE'
        )  # the problem is here
        with mlflow.start_run() as run:
            trainer.fit(model, data)
        test_result = trainer.test(model, data, ckpt_path="best")
        return test_result
    elif flag == "predict":
        return trainer.predict(model, data)
    else:
        print('Please specify the flag as "fit" or "predict"')


def main(json_path, flag="fit"):
    """main function.

    Args:
        json_path (str): The path of cfg json file.
        flag (str): Flag indicates different purpose. Defaults to "fit".
    """

    mlflow.set_tracking_uri('http://localhost:5000')

    with open(json_path) as f:
        cfg = json.load(f)
    result = core(pd.read_csv(cfg["csv_path"]), cfg, flag=flag)
    if flag == "fit":
        print("Test Result:", result)
    elif flag == "predict":
        print("Prediction Result:", result)


if __name__ == "__main__":
    main(json_path="cfg_air.json", flag="fit")
