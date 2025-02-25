"""Training tools.

    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
import numpy as np

torch.optim.lr_scheduler


def scaler(data: np.array):
    """
    Description: Scale the data to [-1, 1]
    Args:
        data: np.array, data to be scaled
    Returns:
        scale: np.array, the maximum abs value
        base: np.array, scaled data
    """
    scale = np.max(np.absolute(data), axis=0)
    base = data / scale
    return np.nan_to_num(scale, copy=False), np.nan_to_num(base, copy=False)


model_ckpt = ModelCheckpoint(
    dirpath='mlruns/models',
    # filename='best_model',  # '{epoch}-{val_loss:.2f}',
    monitor='val_SCORE_epoch',
    mode='min',
    save_top_k=1,
    save_weights_only=False,
)

early_stop = EarlyStopping(
    monitor='val_SCORE_epoch',
    patience=20,
    mode='min',
)
