import torch
import torch.nn as nn
from typing import Dict


# Simple "registry" replacement
_UPSAMPLE_LAYERS = {
    'nearest': nn.Upsample,             # will set mode='nearest' below
    'bilinear': nn.Upsample,            # will set mode='bilinear' below
    'pixel_shuffle': nn.PixelShuffle,
}


def build_upsample_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build an upsample layer without mmengine.

    Args:
        cfg (dict): Must contain:
            - type (str): Layer type. One of {'nearest', 'bilinear', 'pixel_shuffle'}.
            - layer args: Args needed to instantiate the layer (e.g., in_channels, etc.)
                          For nn.Upsample types, provide `scale_factor` or `size` as usual.
        *args, **kwargs: Forwarded to the layer constructor.

    Returns:
        nn.Module: Instantiated upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'the cfg dict must contain the key "type", but got {cfg}')

    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')

    upsample_cls = _UPSAMPLE_LAYERS.get(layer_type, None)
    if upsample_cls is None:
        raise KeyError(f'Cannot find layer type "{layer_type}" in UPSAMPLE_LAYERS')

    # If using nn.Upsample, ensure the mode matches the requested type
    if upsample_cls is nn.Upsample:
        # Don't overwrite if user explicitly passed a different mode
        cfg_.setdefault('mode', layer_type)
        # Note: align_corners (for bilinear) can be passed via cfg_ if desired

    layer = upsample_cls(*args, **kwargs, **cfg_)
    return layer
