"""
    Description: Metrics
    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch

def RSE(pred, true):
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - torch.mean(true)) ** 2))

def CORR(pred, true):
    u = torch.sum((true - torch.mean(true, 0)) * (pred - torch.mean(pred, 0)), 0)
    d = torch.sqrt(torch.sum((true - torch.mean(true, 0)) ** 2, 0) * torch.sum((pred - torch.mean(pred, 0)) ** 2, 0))
    return torch.mean(u / d)

def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))

def MSE(pred, true):
    return torch.mean((pred - true) ** 2)

def RMSE(pred, true):
    return torch.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return torch.mean(torch.abs((pred - true) / true))

def MSPE(pred, true):
    return torch.mean(torch.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'MSPE':mspe}