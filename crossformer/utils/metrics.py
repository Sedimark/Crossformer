"""Metrics.

In this script, the metrics and designed loss functions are provided.
Author: Peipei Wu (Paul) - Surrey
Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch
import math


def scaled_mse(pred, true, scale_factor, epsilon=1e-6):
    """Computes the scaled MSE loss.

    Args:
        pred (torch.Tensor): Predictions.
        true (torch.Tensor): Ground truth.
        scaled_factor (torch.Tensor): Scaled range.
        epsilon (float): Constant for avoid division by zero.
                         Defaults to 1e-6.

    Returns:
        torch.Tensor: Scaled MSE
    """
    diff = true - pred
    scaled_diff = diff**2 / (scale_factor**2 + epsilon)
    return torch.mean(scaled_diff)


def normalized_mse(pred, true, epsilon=1e-6):
    """Computes the normalized MSE loss.

    Args:
        pred (torch.Tensor): Predictions.
        true (torch.Tensor): Ground truths.
        epsilon (float): Constant for avoid division by zero.
                         Defaults to 1e-6.

     Returns:
        torch.Tensor: Normalized MSE
    """
    diff = true - pred
    norm_factors = torch.max(true, dim=-1) - torch.min(true, dim=-1)
    norm_diff = diff**2 / (norm_factors**2 + epsilon)
    return torch.mean(norm_diff)


def scaled_log_cosh(pred, true, scaled_factor, epsilon=1e-6):
    """_summary_

    Args:
        pred (torch.Tensor): Predictions.
        true (torch.Tensor): Ground truths.
        scaled_factor (torch.Tensor): Scaled range.
        epsilon (float): Constant for avoid division by zero.
                          Defaults to 1e-6.

     Returns:
        torch.Tensor: Scaled Log Cosh
    """
    diff = true - pred
    if diff.max() > 50:
        diff = torch.clamp(diff, max=50)
    scaled_log_cosh = torch.log(torch.cosh(diff)) / (scaled_factor + epsilon)
    return torch.mean(scaled_log_cosh)


def hybrid_loss(pred, true, alpha=0.5):
    """Computes hybrid loss.

    alpha controls the scale & (1 - alpha) controls outlier robustness
    Args:
        pred (torch.Tensor): Predictions.
        true (torch.Tensor): Ground truths.
        alpha (float): Scale sensitive. Defaults to 0.5.

     Returns:
        torch.Tensor: Hybrid loss
    """
    scaled_factor = torch.amax(true, dim=(0, 1)) - torch.amin(true, dim=(0, 1))
    scale_mse = scaled_mse(pred, true, scaled_factor)
    scale_log_cosh = scaled_log_cosh(pred, true, scaled_factor)
    # return alpha * scale_mse + (1 - alpha) * scale_log_cosh
    return (scale_mse * scale_mse / (scale_mse + scale_log_cosh)) + (
        scale_log_cosh * scale_log_cosh / (scale_mse + scale_log_cosh)
    )


def rse(pred, true):
    """Calculates the Root Relative Squared Error (RSE).

    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.

    Returns:
        torch.Tensor: RSE value.
    """
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(
        torch.sum((true - torch.mean(true)) ** 2)
    )


def corr(pred, true):
    """Calculates the correlation coefficient.

    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.

    Returns:
        torch.Tensor: Correlation coefficient.
    """
    u = torch.sum(
        (true - torch.mean(true, 0)) * (pred - torch.mean(pred, 0)), 0
    )
    d = torch.sqrt(
        torch.sum((true - torch.mean(true, 0)) ** 2, 0)
        * torch.sum((pred - torch.mean(pred, 0)) ** 2, 0)
    )
    return torch.mean(u / d)


def mae(pred, true):
    """Calculates the Mean Absolute Error (MAE).

    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.

    Returns:
        torch.Tensor: MAE value.
    """
    return torch.mean(torch.abs(pred - true))


def mse(pred, true):
    """Calculates the Mean Squared Error (MSE).

    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.

    Returns:
        torch.Tensor: MSE value.
    """
    return torch.mean((pred - true) ** 2)


def rmse(pred, true):
    """Calculates the Root Mean Squared Error (RMSE).

    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.

    Returns:
        torch.Tensor: RMSE value.
    """
    return torch.sqrt(mse(pred, true))


def mape(pred, true):
    """Calculates the Mean Absolute Percentage Error (MAPE).

    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.

    Returns:
        torch.Tensor: MAPE value.
    """
    return torch.mean(torch.abs((pred - true) / true))


def mspe(pred, true):
    """Calculates the Mean Squared Percentage Error (MSPE).

    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.

    Returns:
        torch.Tensor: MSPE value.
    """
    return torch.mean(torch.square((pred - true) / true))


def sanitize(x: torch.Tensor, default=100.0) -> float:
    """Ensure metric is a finite float, otherwise replace with default."""
    val = x.item() if isinstance(x, torch.Tensor) else x
    return val if math.isfinite(val) else default


def metric(pred, true):
    """Calculates various metrics, with inf/nan-safe outputs.

    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.

    Returns:
        dict: Dictionary containing safe MAE, MSE, RMSE, MAPE, MSPE, SCORE.
    """
    return {
        "MAE": sanitize(mae(pred, true)),
        "MSE": sanitize(mse(pred, true)),
        "RMSE": sanitize(rmse(pred, true)),
        "MAPE": sanitize(mape(pred, true)),
        "MSPE": sanitize(mspe(pred, true)),
        "SCORE": sanitize(hybrid_loss(pred, true)),
    }
