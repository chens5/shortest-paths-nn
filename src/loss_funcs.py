import torch

# MSE loss function
def msle_loss(pred, target):
    return mse_loss(torch.log(pred + 1),  torch.log(target + 1))

# scale loss function based on difference from L2
def weighted_mse_loss(pred, target, l2_diffs):
    errs = (target/l2_diffs) * torch.square(pred - target)
    return torch.mean(errs)

# NMSE loss function
def nmse_loss(pred, target):
    nz = torch.nonzero(target)
    errs = torch.square(pred - target)/torch.square(target+ 1e-4)
    return torch.mean(errs)

def nmae_loss(pred, target):
    nz = torch.nonzero(target)
    errs = torch.abs(pred[nz] - target[nz])/torch.abs(target[nz])
    return torch.mean(errs)

def mse_loss(pred, target):
    return torch.mean(torch.square(pred - target))

# MSE for sqrt distance
def sqrt_distance(pred, target):
    sqrt_distance = torch.sqrt(target)
    return torch.mean(torch.square(pred - sqrt_distance))

# MAE for sqrt distance
def sqrt_distance_MAE(pred, target):
    sqrt_distance = torch.sqrt(target)
    return torch.mean(torch.abs(pred - sqrt_distance))

def squared_distance_prediction_MSE(pred, target):
    return torch.mean(torch.square(torch.square(pred) - target))

def squared_distance_prediction_MAE(pred, target):
    return torch.mean(torch.abs(torch.square(pred) - target))

def square_distance_tar_MSE(pred, target):
    return torch.mean(torch.square(pred - torch.square(target)))

def square_distance_tar_MAE(pred, target):
    return torch.mean(torch.abs(pred - torch.square(target)))