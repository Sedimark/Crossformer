import torch
import pandas as pd
import numpy as np

from crossformer.model.crossformer import CrossFormer
from crossformer.data_tools.data_interface import DataInterface


def test_addition():
    assert 1 + 1 == 2, "1 + 1 should equal 2"


# test_cfg = {
#     "data_dim": 8,
#     "in_len": 24,
#     "out_len": 24,
#     "seg_len": 2,
#     "window_size": 4,
#     "factor": 10,
#     "model_dim": 256,
#     "feedforward_dim": 512,
#     "head_num": 4,
#     "layer_num": 6,
#     "dropout": 0.2,
#     "baseline": False,
#     "learning_rate": 0.1,
#     "batch_size": 8,
#     "csv_path": "data/air/values.csv",
#     "split": [0.7, 0.2, 0.1],
#     "seed": 2024,
#     "accelerator": "auto",
#     "min_epochs": 1,
#     "max_epochs": 200,
#     "precision": 32,
#     "patience": 5,
#     "num_workers": 31,
#     "tracking_uri": "http://localhost:5000",
#     "registry_uri": "http://localhost:5000",
#     "experiment_name": "air",
# }

# test_df = pd.DataFrame(np.random.rand(100, 8))


# def test_data():
#     data = DataInterface(test_df, **test_cfg)
#     data.setup()
#     train_loader = data.train_dataloader()

#     # Check if train_loader is not None
#     assert train_loader is not None, "Train dataloader is None"

#     # Check if train_loader is iterable
#     assert hasattr(train_loader, "__iter__"), "Train dataloader is not iterable"

#     # Check if train_loader returns batches of the correct size
#     for batch in train_loader:
#         assert isinstance(batch, tuple), "Batch is not a tuple"
#         assert (
#             len(batch) == 2
#         ), "Batch does not contain two elements (inputs and targets)"
#         inputs, targets = batch
#         assert (
#             inputs.shape[0] == test_cfg["batch_size"]
#         ), f"Batch size is incorrect: {inputs.shape[0]}"
#         assert (
#             targets.shape[0] == test_cfg["batch_size"]
#         ), f"Batch size is incorrect: {targets.shape[0]}"
#         break  # Test only the first batch to avoid iterating through the entire loader

# def test_model():
#     model = CrossFormer(**test_cfg)
#     model.train()

#     # Check if the model has the expected number of parameters
#     num_params = sum(p.numel() for p in model.parameters())
#     assert num_params > 0, "Model has no parameters"

#     # Check if the model can process a batch of data
#     input_tensor = torch.randn(
#         test_cfg["batch_size"], test_cfg["in_len"], test_cfg["data_dim"]
#     )
#     output_tensor = model(input_tensor)
#     assert (
#         output_tensor.shape == (test_cfg["batch_size"], test_cfg["out_len"], test_cfg["data_dim"])
#     ), f"Output shape is incorrect: {output_tensor.shape}"
