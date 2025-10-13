import torch
import torch.nn as nn
from typing import Dict, Any

_ACT_LAYERS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'ReLU6': nn.ReLU6,
    'ELU': nn.ELU,
    'SELU': nn.SELU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU,
    'Mish': nn.Mish,
    'SiLU': nn.SiLU,            # a.k.a. Swish
    'HSigmoid': nn.Hardsigmoid, # OpenMMLab alias
    'Swish': nn.SiLU
}

def build_activation_layer(act_cfg: Dict[str, Any]) -> nn.Module:
    """
    Build an activation layer from a tiny config dict like:
      dict(type='ReLU', inplace=True)
      dict(type='LeakyReLU', negative_slope=0.1, inplace=True)
      dict(type='Swish')  # alias of SiLU
      dict(type='HSigmoid')
    """
    assert isinstance(act_cfg, dict) and 'type' in act_cfg, "act_cfg must have a 'type' key"
    cfg = act_cfg.copy()
    act_type = cfg.pop('type')

    # print(cfg.get('inplace'))
    if act_type not in _ACT_LAYERS:
        raise ValueError(f"Unsupported activation type: {act_type}")

    Act = _ACT_LAYERS[act_type]
    return Act(**cfg)
