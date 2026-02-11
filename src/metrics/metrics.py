import torch


def mse_loss(pred, true):
    return ((pred - true) ** 2).mean(-1).mean()


def n_mse_loss(pred, true):
    return (((pred - true) ** 2).sum(-1) / (true**2).sum(-1)).mean()


def mae_loss(pred, true):
    return (torch.abs(pred - true)).mean(-1).mean()


def mape_loss(pred, true):
    return (
        torch.mean(torch.abs((pred - true) / true))
        if torch.any(true != 0)
        else torch.tensor(0.0)
    )
