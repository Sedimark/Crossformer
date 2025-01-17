"""Main file.

The main file for training CrossFormer or predicting

Author: Peipei Wu (Paul) - Surrey
Maintainer: Peipei Wu (Paul) - Surrey
"""

import json
# import mlflow.cli
from crossformer.data_tools.data_interface import DataInterface
from crossformer.model.crossformer import CrossFormer
from crossformer.utils.tools import early_stop, model_ckpt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import torch
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

    # mlflow logger
    # mlflow.cli.server()  # start the mlflow server
    mlflow_logger = MLFlowLogger(
        experiment_name="mlflow_logger_test",
        tracking_uri="http://localhost:5000",
    )

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

    # trainer
    trainer = pl.Trainer(
        accelerator=cfg["accelerator"],
        precision=cfg["precision"],
        min_epochs=cfg["min_epochs"],
        max_epochs=cfg["max_epochs"],
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        callbacks=[model_ckpt, early_stop],
        logger=mlflow_logger,
    )

    if flag == "fit":
        trainer.fit(model, data)
        test_result = trainer.test(model, data)
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
    with open(json_path) as f:
        cfg = json.load(f)
    result = core(pd.read_csv(cfg["csv_path"]), cfg, "fit")
    print(result)


if __name__ == "__main__":
    main(json_path="cfg.json", flag="fit")
