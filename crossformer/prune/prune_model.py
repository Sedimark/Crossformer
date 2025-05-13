import torch
import mlflow
from crossformer.model.crossformer import CrossFormer
from crossformer.prune.channel_prune import channel_prune
from crossformer.prune.unstructured_prune import FineGrainedPruner, get_pruning_ratios
from crossformer.utils.model_profiling import (
    get_model_size,
    get_num_parameters,
    get_model_size_parameters,
    profile_model,
    get_model_sparsity,
)
from mlflow.models.signature import infer_signature

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def prune_model(cfg: dict) -> tuple:
    """
    Load a model from path, apply pruning, and save the pruned model.
    Returns the new configuration and path to the pruned model.

    :param cfg: Configuration dictionary
    :return: Tuple containing the updated configuration and pruned model path
    """

    # Load main unpruned model from pth file path mentioned in cfg
    print(cfg["model_path"])
    model = CrossFormer(cfg=cfg)
    # Load the state dict from the .pth file
    model = torch.load(cfg["model_path"], map_location=torch.device("cpu"))

    # Compute initial model size and parameters
    model_size = get_model_size(model)
    model_parameters = get_num_parameters(model)
    print(
        f"Initial Model Size: {model_size / MiB:.2f} MiB, Parameters: {model_parameters}"
    )

    # Layer Pruning
    print("Starting Layer Pruning...")
    sparsity_dict = get_pruning_ratios(model)
    pruner = FineGrainedPruner(model, sparsity_dict)
    pruner.apply(model)

    # Channel Pruning
    print("Starting Channel Pruning...")
    pruned_model = channel_prune(model, prune_ratio=0.3)

    sparse_model_size = get_model_size(pruned_model, count_nonzero_only=True)
    sparse_model_parameters = get_num_parameters(
        pruned_model, count_nonzero_only=True
    )
    model_sparsity = get_model_sparsity(model)
    print(
        f"sparse - model size: {sparse_model_size / MiB:.2f}, number of parameters: {sparse_model_parameters} \
        model_sparsity - {model_sparsity}"
    )
    print(
        f"Sparse model has size={sparse_model_size / MiB:.2f} \
        MiB = {sparse_model_size / model_size * 100:.2f}% of dense model size"
    )

    print("with counting zeroes")
    sparse_model_size = get_model_size(pruned_model)
    sparse_model_parameters = get_num_parameters(pruned_model)
    model_sparsity = get_model_sparsity(model)
    print(
        f"sparse - model size: {sparse_model_size / MiB:.2f}, number of parameters: {sparse_model_parameters} \
        model_sparsity - {model_sparsity}"
    )
    print(
        f"Sparse model has size={sparse_model_size / MiB:.2f} \
        MiB = {sparse_model_size / model_size * 100:.2f}% of dense model size"
    )

    # Update pruned configuration
    pruned_cfg = (
        cfg.copy()
    )  # Create a copy to avoid modifying the original config
    pruned_cfg["model_dim"] = 176  # Update pruned model dimension
    pruned_cfg["feedforward_dim"] = 352  # Update pruned feedforward dimension

    # Signature
    input_example = torch.randn(1, cfg["in_len"], cfg["data_dim"])
    output_example = pruned_model(input_example)
    signature = infer_signature(
        input_example.numpy(), output_example.detach().numpy()
    )

    # Save using MLflow
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(
            pytorch_model=pruned_model,
            artifact_path="pruned_untrained",
            signature=signature,
        )

    # Register the pruned model
    model_uri = f"runs:/{run.info.run_id}/pruned_untrained"
    mlflow.register_model(
        model_uri, f"{cfg['experiment_name']}_pruned_untrained"
    )

    # Return updated configuration and model path
    return pruned_cfg, f"{cfg['experiment_name']}_pruned_untrained/latest"
