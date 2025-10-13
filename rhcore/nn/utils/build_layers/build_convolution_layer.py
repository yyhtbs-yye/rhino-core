from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

# Internal base classes used for isinstance checks / typing
from torch.nn.modules.conv import _ConvNd

_CONV_LAYERS = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    'ConvTranspose1d': nn.ConvTranspose1d,
    'ConvTranspose2d': nn.ConvTranspose2d,
    'ConvTranspose3d': nn.ConvTranspose3d,
}

def build_convolution_layer(conv_cfg: Optional[Dict[str, Any]],
                            in_channels: int,
                            out_channels: int,
                            kernel_size: Union[int, Tuple[int, int, int]],
                            stride: Union[int, Tuple[int, int, int]] = 1,
                            padding: Union[int, Tuple[int, int, int]] = 0,
                            dilation: Union[int, Tuple[int, int, int]] = 1,
                            groups: int = 1,
                            bias: bool = True) -> _ConvNd:
    """
    Build a convolution layer from a tiny config dict like:
        conv_cfg = dict(type='Conv2d')  # defaults to Conv2d if None
    """
    if conv_cfg is None:
        conv_type = 'Conv2d'
        extra = {}
    else:
        conv_type = conv_cfg.get('type', 'Conv2d')
        extra = {k: v for k, v in conv_cfg.items() if k != 'type'}

    if conv_type not in _CONV_LAYERS:
        raise ValueError(f"Unsupported conv type: {conv_type}")

    Conv = _CONV_LAYERS[conv_type]
    layer = Conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        **extra
    )
    return layer

