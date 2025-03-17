"""Main file.

The main file for Channel pruning and training Pruned CrossFormer or predicting

Author: Sneha H - Surrey
Maintainer: Sneha H - Surrey
"""

import json
from crossformer.data_tools.data_interface import DataInterface
from crossformer.model.crossformer import CrossFormer
from crossformer.prune_model.channel_prune import channel_prune
from crossformer.utils.model_profiling import (
    get_model_size_parameters,
    profile_model,
    get_model_sparsity
)
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer
import pandas as pd
import torch
import mlflow.pytorch
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB
# from pytorch_lightning.cli import LightningCLI

#  cli functions
# TODO: waiting for implementation in future


def core(
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
    torch.set_float32_matmul_precision("medium")

    # mlflow settings
    # TODO: clean for automation
    mlflow.set_tracking_uri(cfg['tracking_uri'])
    mlflow.set_registry_uri(cfg['registry_uri'])
    mlflow.set_experiment(cfg['experiment_name'])

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

        print("Load Pruned Model and Start Training")
        model.load_state_dict(torch.load("mlruns/pruned_models/channel_pruned_model.pth"))

        model_parameters, model_size = get_model_size_parameters(model)
        print(f"model size: {model_size}, number of parameters: {model_parameters}")

        # Profile the model FLOPs
        input_shape = (1, 48, 5)  # Adjust based on your model's expected input
        profile_model(model, input_shape)

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
    elif flag == 'prune':
        model = mlflow.pytorch.load_model(
            f"models:/{cfg['experiment_name']}_best_model/latest"
        )  # TODO: specify model later

        state_dict = model.state_dict()

        model.load_state_dict(state_dict)

        model_parameters, model_size = get_model_size_parameters(model)
        print(f"model size: {model_size}, number of parameters: {model_parameters}")

        print("Start Channel Pruning")
        pruned_model = channel_prune(model, prune_ratio=0.3)
        sparse_model_parameters, sparse_model_size = get_model_size_parameters(pruned_model)
        model_sparsity = get_model_sparsity(model)
        print(f"sparse - model size: {sparse_model_size}, number of parameters: {sparse_model_parameters} \
            model_sparsity - {model_sparsity}")
        print(f"Sparse model has size={sparse_model_size} \
            MiB = {sparse_model_size / model_size * 100}% of dense model size")
        torch.save(pruned_model.state_dict(), 'mlruns/pruned_models/channel_pruned_model.pth')
    elif flag == 'predict':
        print(cfg['experiment_name'])
        print(f"models:/{cfg['experiment_name']}_best_model/latest")

        model = mlflow.pytorch.load_model(
            f"models:/{cfg['experiment_name']}_best_model/latest"
        )  # TODO: specify model later
        #print(f"model: {model}")
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
    elif flag == "prune":
        result = core(pd.read_csv(cfg["csv_path"]), cfg, flag=flag)
        print("Saved pruned model to train")
    elif flag == "predict":
        result = core(
            pd.read_csv("data/air/sample_values.csv", header=None),
            cfg,
            flag=flag,
        )  # load test sample
        print("Prediction Result:", result)


if __name__ == "__main__":
    #main(json_path="cfg_weather.json", flag="prune")
    main(json_path="cfg_weather_prune.json", flag="fit")
