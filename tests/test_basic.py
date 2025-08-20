import torch
import pandas as pd
import numpy as np

from crossformer.model.crossformer import CrossFormer
from crossformer.data_tools.data_interface import DataInterface


sample_cfg = {
    "data_dim": 8,
    "in_len": 12,
    "out_len": 12,
    "seg_len": 2,
    "window_size": 4,
    "factor": 10,
    "model_dim": 256,
    "feedforward_dim": 512,
    "head_num": 4,
    "layer_num": 2,
    "dropout": 0.2,
    "baseline": False,
    "learning_rate": 0.1,
    "batch_size": 32,
    "csv_path": "scripts/demo.csv",
    "split": [0.6, 0.2, 0.2],  # Increased test split for test data
    "seed": 2024,
    "accelerator": "auto",
    "min_epochs": 1,
    "max_epochs": 200,
    "precision": 32,
    "patience": 15,
    "num_workers": 1,  # Reduced workers to avoid the warning
    "tracking_uri": "http://localhost:5000",
    "registry_uri": "http://localhost:5000",
    "experiment_name": "air",
    "method": "minmax",
    "per_feature": True,
}

sample_df = pd.DataFrame(np.random.rand(1000, 8))  # Increased data size


def test_data():
    """
    Test the DataInterface class for data loading and preprocessing.
    """
    # check initialization
    data = DataInterface(sample_df, **sample_cfg)
    assert data is not None, "DataInterface initialization failed"
    data.setup()

    # check data preparation
    first_batches = [
        list(enumerate(data.train_dataloader()))[0][1],
        list(enumerate(data.val_dataloader()))[0][1],
        list(enumerate(data.test_dataloader()))[0][1],
    ]
    for first_batch in first_batches:
        assert first_batch is not None, "DataLoader returned empty batch"
        assert (
            len(first_batch) == 2
        ), "DataLoader batch should contain two elements"


def test_model():
    """
    Test the CrossFormer model for forward pass.
    """
    # check initialization
    model = CrossFormer(sample_cfg)
    assert model is not None, "CrossFormer initialization failed"

    # check forward pass with correct dimensions
    # input should be (batch_size, in_len, data_dim)
    x = torch.randn(2, sample_cfg["in_len"], sample_cfg["data_dim"])
    y = model(x)
    assert y is not None, "Forward pass failed"
    expected_shape = (2, sample_cfg["out_len"], sample_cfg["data_dim"])
    assert (
        y.shape == expected_shape
    ), f"Output shape mismatch: expected {expected_shape}, got {y.shape}"


def test_model_train():
    """
    Test the CrossFormer model for training step.
    """
    # check initialization
    model = CrossFormer(sample_cfg)
    assert model is not None, "CrossFormer initialization failed"

    # check training step with correct batch format
    x = torch.randn(2, sample_cfg["in_len"], sample_cfg["data_dim"])
    y = torch.randn(2, sample_cfg["out_len"], sample_cfg["data_dim"])
    batch = (x, y)  # Only x and y, no scale
    loss = model.training_step(batch, 0)
    assert loss is not None, "Training step failed"
    assert (
        "loss" in loss
    ), "Training step should return a dictionary with 'loss' key"


# test_data()
# test_model()
# test_model_train()
