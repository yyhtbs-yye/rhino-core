import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from rhcore.nn.ops.pixel_norm import PixelNorm

_NORM_LAYERS = {
    # batch norms
    'BN1d': nn.BatchNorm1d,
    'BN2d': nn.BatchNorm2d,
    'BN': nn.BatchNorm2d,
    'BN3d': nn.BatchNorm3d,
    'SyncBN': nn.SyncBatchNorm,  # requires process group setup when used distributed
    # instance norms
    'IN1d': nn.InstanceNorm1d,
    'IN2d': nn.InstanceNorm2d,
    'IN': nn.InstanceNorm2d,
    'IN3d': nn.InstanceNorm3d,
    # group / layer norms
    'GN': nn.GroupNorm,
    'LN': nn.LayerNorm,
    'PN': PixelNorm,
}

def build_normalization_layer(norm_cfg: Dict[str, Any],
                              num_features: Optional[int] = None) -> Tuple[str, nn.Module]:
    """
    Build a norm layer. Returns (name, module) to match OpenMMLab's convention.

    Example norm_cfg:
      dict(type='BN', eps=1e-5, momentum=0.1)  # auto-resolves to BN2d unless explicitly BN1d/3d
      dict(type='SyncBN')
      dict(type='IN')         # InstanceNorm2d default
      dict(type='GN', num_groups=32)  # requires num_channels via num_features
      dict(type='LN')         # requires normalized_shape via num_features
    """
    assert isinstance(norm_cfg, dict) and 'type' in norm_cfg, "norm_cfg must have a 'type' key"
    cfg = norm_cfg.copy()
    norm_type = cfg.pop('type')

    if norm_type not in _NORM_LAYERS:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    Norm = _NORM_LAYERS[norm_type]

    # figure required args
    if Norm in (nn.GroupNorm,):
        assert num_features is not None, "GroupNorm requires num_features (channels)"
        num_groups = cfg.pop('num_groups', 32)
        layer = Norm(num_groups=num_groups, num_channels=num_features, **cfg)
        name = 'gn'
    elif Norm in (nn.LayerNorm,):
        assert num_features is not None, "LayerNorm requires normalized_shape (channels or shape)"
        layer = Norm(normalized_shape=num_features, **cfg)
        name = 'ln'
    elif Norm == PixelNorm:
        layer = Norm(**cfg)
        name = 'pn'
    else:
        # BatchNorm/SyncBN/InstanceNorm family: need num_features
        assert num_features is not None, f"{norm_type} requires num_features"
        layer = Norm(num_features, **cfg)
        # conventional names
        if 'Sync' in norm_type:
            name = 'syncbn'
        elif norm_type.startswith('BN'):
            name = 'bn'
        elif norm_type.startswith('IN'):
            name = 'in'
        else:
            name = 'norm'

    return name, layer
