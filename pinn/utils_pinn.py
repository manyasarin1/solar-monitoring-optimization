import os
import numpy as np
import torch

def load_npz(path):
    arr = np.load(path)["data"].astype(np.float32)
    arr = np.nan_to_num(arr)
    return arr

def make_time(n, dt_hours=1.0):
    # t = [0, dt, 2dt, ...] in hours (shape [n,1])
    t = np.arange(n, dtype=np.float32) * dt_hours
    return t.reshape(-1, 1)

def normalize_train(x):
    # z-score normalize columns, return (x_norm, (mean, std))
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    x_norm = (x - mean) / std
    return x_norm, (mean, std)

def apply_norm(x, stats):
    mean, std = stats
    return (x - mean) / (std + 1e-8)

def weak_label_tp(I, Ta, v, noct=45.0):
    """
    Simple NOCT-style weak label for panel temperature (°C):
    Tp ≈ Ta + k*I - k_wind*v
    k ≈ (NOCT-20)/800;  NOCT~45 => k~0.03125
    wind cooling heuristic k_wind~2.8 (°C per m/s)
    """
    k = (noct - 20.0) / 800.0
    Tp_weak = Ta + k * I - 2.8 * v
    return Tp_weak.astype(np.float32).reshape(-1, 1)

def to_tensor(*arrays, device="cpu"):
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in arrays]
