import torch
import torch.nn as nn

def make_layers(module_cls, num_layers, **kwargs):
    if num_layers < 0:
        raise ValueError("num_layers must be >= 0")

    return nn.Sequential(*[
        module_cls(**kwargs)
        for i in range(num_layers)
    ]) 