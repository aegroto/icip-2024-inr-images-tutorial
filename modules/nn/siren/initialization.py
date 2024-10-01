import torch
import math


def initialize_first_siren_layer(layer):
    w_std = 1.0 / layer.in_features
    torch.nn.init.uniform_(layer.weight, -w_std, w_std)


def initialize_siren_layer(layer, w0: float = 30.0, c: float = 6.0):
    w_std = math.sqrt(c / layer.in_features) / w0
    torch.nn.init.uniform_(layer.weight, -w_std, w_std)
