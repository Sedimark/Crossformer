from crossformer.model.crossformer import CrossFormer

from crossformer.prune.channel_prune import channel_prune
from crossformer.prune.unstructured_prune import (
    FineGrainedPruner,
    get_pruning_ratios,
)

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def prune(model: CrossFormer, prune_ratio: float = 0.3) -> CrossFormer:
    """Prune the CrossFormer model.

    Args:
        model (CrossFormer): The trained CrossFormer model.
        prune_ratio (float): The ratio of channels to prune.
            Default is 0.3 (30% of channels).

    Returns:
        CrossFormer: The pruned CrossFormer model (training needed).
    """

    # Layer Pruning
    sparsity_dict = get_pruning_ratios(model)
    pruner = FineGrainedPruner(model, sparsity_dict)
    pruner.apply(model)

    # Channel Pruning
    pruned_model = channel_prune(model, prune_ratio=prune_ratio)

    return pruned_model
