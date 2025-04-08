"""CrossFormer: A Transformer for Time Series Multivariate Forecasting

The main entry of the library. It contains the core function for training and predicting.

Author: Peipei Wu (Paul) - Surrey
Maintainer: Peipei Wu (Paul) - Surrey
"""

import json
import os
from crossformer.data_tools.data_interface import DataInterface
from crossformer.model.crossformer import CrossFormer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer
import pandas as pd
import torch
import mlflow.pytorch
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

# from pytorch_lightning.cli import LightningCLI

#  cli functions
# TODO: waiting for implementation in future


def load_mlflow_env(config: dict):
    """
    Description: Load the mlflow env

    Args:
        config: dict, config file
    Returns:
        None
    """
    # config file is essential, need to handle exception later

    os.environ['MLFLOW_TRACKING_USERNAME'] = (
        config["MLFLOW"]["MLFLOW_TRACKING_USERNAME"].strip().replace("\n", "")
    )
    os.environ['MLFLOW_TRACKING_PASSWORD'] = (
        config["MLFLOW"]["MLFLOW_TRACKING_PASSWORD"].strip().replace("\n", "")
    )
    os.environ['AWS_ACCESS_KEY_ID'] = (
        config["MLFLOW"]["AWS_ACCESS_KEY_ID"].strip().replace("\n", "")
    )
    os.environ['AWS_SECRET_ACCESS_KEY'] = (
        config["MLFLOW"]["AWS_SECRET_ACCESS_KEY"].strip().replace("\n", "")
    )
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = (
        config["MLFLOW"]["MLFLOW_S3_ENDPOINT_URL"].strip().replace("\n", "")
    )
    os.environ['MLFLOW_EXPERIMENT'] = config["MLFLOW"]["MFLOW_EXPERIMENT_NAME"]
    os.environ['MLFLOW_TRACKING_URI'] = config["MLFLOW"]["MLFLOW_TRACKING_URI"]
    os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = "true"
    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = "1000"


def core(
    df: pd.DataFrame,
    config: dict,
    cfg: dict,
    flag: str = "fit",
):
    """Core function for the AI asset.

    Args:
        df (pd.DataFrame): The input data.
        config (dict): The configuration (System I/O) dictionary.
        cfg (dict): The configuration (AI model) dictionary.
        flag (str): The flag for the function, either 'fit' or 'predict', which
            defaults to "fit".

    Returns:
       test_result (List): trained model's result on test set if flag 'fit'.
       predict_result (torch.tensor): predictions if flag 'predict'.
    """
    torch.set_float32_matmul_precision("medium")

    # mlflow settings
    load_mlflow_env(config)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    if flag == 'fit':
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

        # init model & data
        model = CrossFormer(cfg=cfg)
        data = DataInterface(df, **cfg)

        # signature
        input_example = torch.randn(1, cfg['in_len'], cfg['data_dim'])
        output_example = model(input_example)
        signature = infer_signature(
            input_example.numpy(), output_example.detach().numpy()
        )
        mlflow.pytorch.autolog(checkpoint_monitor='val_SCORE', silent=True)
        with mlflow.start_run() as run:
            trainer.fit(model, data)
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                signature=signature,
            )
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, f"{cfg['experiment_name']}_best_model")
        test_result = trainer.test(model, data, ckpt_path="best")
        return test_result

    elif flag == 'predict':
        model = mlflow.pytorch.load_model(
            f"models:/{cfg['experiment_name']}_best_model/latest"
        )  # TODO: specify model later
        model.eval()
        input_tensor = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            predictions = model(input_tensor)
        df_predictions = pd.DataFrame(predictions.squeeze(0).numpy())
        return df_predictions
    else:
        raise TypeError("Please specify the flag as 'fit' or 'predict'")


def main(json_path, flag="fit"):
    """main function.

    Args:
        json_path (str): The path of cfg json file.
        flag (str): Flag indicates different purpose. Defaults to "fit".
        extensions: data augmentation | pruning (the future function)
    """

    with open(json_path) as f:
        cfg = json.load(f)
    if flag == "fit":
        result = core(pd.read_csv(cfg["csv_path"]), cfg, flag=flag)
        print("Test Result:", result)
    elif flag == "predict":
        result = core(
            pd.read_csv("data/air/sample_values.csv", header=None),
            cfg,
            flag=flag,
        )  # load test sample
        print("Prediction Result:", result)


if __name__ == "__main__":
    main(json_path="cfg_weather.json", flag="fit")
