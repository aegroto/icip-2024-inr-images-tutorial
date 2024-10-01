import torch
import os


def load_device(force_cpu=False):
    if force_cpu:
        torch.device("cpu")
    else:
        return torch.device(os.environ.get("DEVICE") or "cuda")
