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
    dirpath="mlruns/models",
    # filename='best_model',  # '{epoch}-{val_loss:.2f}',
    monitor="val_SCORE_epoch",
    mode="min",
    save_top_k=1,
    save_weights_only=False,
)

early_stop = EarlyStopping(
    monitor="val_SCORE_epoch",
    patience=20,
    mode="min",
)


def diff_temporal(x):
    """
    Description: Calculate the difference between adjacent elements in a tensor.
    Args:
        x: torch.Tensor, input tensor
    Returns:
        torch.Tensor, tensor with differences
    """
    return x[:, 1:, :] - x[:, :-1, :]

class Preprocessor:
    def __init__(self, method="zscore", per_feature=True):
        self.method = method
        self.per_feature = per_feature
        self.fitted = False
        self.params = {}

    def fit(self, x: np.ndarray):
        """Fit the preprocessor to the data

        Args:
            x (np.ndarray): data (T, D)
        """
        if x.ndim not in [2]:
            raise ValueError("Input must be 2D (T,D)")
        axis = 0 if self.per_feature else None
        if self.method == "zscore":
            self.params["mean"] = np.mean(x, axis=axis, keepdims=True)
            self.params["std"] = np.std(x, axis=axis, keepdims=True) + 1e-8
        elif self.method == "minmax":
            self.params["min"] = np.min(x, axis=axis, keepdims=True)
            self.params["max"] = np.max(x, axis=axis, keepdims=True)
            self.params["std"] = self.params["max"] - self.params["min"] + 1e-8
        else:
            raise ValueError("Unsupported normalization method.")
        self.fitted = True

    def transform(self, x: np.ndarray):
        """Transform the data using the fitted preprocessor

        Args:
            x (np.ndarray): data (T, D)

        Returns:
            np.ndarray: transformed data
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transformation.")
        if self.method == "zscore":
            return (x - self.params["mean"]) / self.params["std"]
        elif self.method == "minmax":
            return (x - self.params["min"]) / self.params["std"]
        else:
            raise ValueError("Unsupported normalization method.")
        
    def export(self):
        """Export the preprocessor parameters

        Returns:
            dict: parameters for the preprocessor
        """
        return {
            "method": self.method,
            "per_feature": self.per_feature,
            "params": self.params,
        }
    
    @staticmethod
    def from_stats(stats: dict):
        """Create a Preprocessor from exported stats

        Args:
            stats (dict): exported parameters

        Returns:
            Preprocessor: instance of Preprocessor
        """
        obj = Preprocessor(method=stats["method"], per_feature=stats["per_feature"])
        obj.params = stats["params"]
        obj.fitted = True
        return obj
    
class Postprocessor:
    def __init__(self, stats):
        self.method = stats["method"]
        self.per_feature = stats["per_feature"]
        self.params = stats["params"]

    def inverse_transform(self, x:np.ndarray)-> np.ndarray:
        """Inverse transform the data using the fitted preprocessor

        Args:
            x (np.ndarray): transformed data (T, D)

        Returns:
            np.ndarray: original data
        """
        if self.method == "zscore":
            return x * self.params["std"] + self.params["mean"]
        elif self.method == "minmax":
            return x * self.params["std"] + self.params["min"]
        else:
            raise ValueError("Unsupported normalization method.")
