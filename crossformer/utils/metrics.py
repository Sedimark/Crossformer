"""Metrics.

Author: Peipei Wu (Paul) - Surrey
Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch


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


def metric(pred, true):
    """Calculates various metrics.

    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.

    Returns:
        dict: Dictionary containing MAE, MSE, RMSE, MAPE, and MSPE values.
    """
    mae_value = mae(pred, true)
    mse_value = mse(pred, true)
    rmse_value = rmse(pred, true)
    mape_value = mape(pred, true)
    mspe_value = mspe(pred, true)

    return {
        'MAE': mae_value,
        'MSE': mse_value,
        'RMSE': rmse_value,
        'MAPE': mape_value,
        'MSPE': mspe_value,
    }
