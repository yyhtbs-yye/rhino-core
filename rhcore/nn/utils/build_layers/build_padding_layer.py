# ===== Helper types & builders for ConvModule (PyTorch >= 2.0) =====
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

def _expand_padding_for_dim(padding: Union[int, Tuple[int, ...]], dim: int) -> Tuple[int, ...]:
    """
    Expand common padding specs to PyTorch's explicit (left/right[/top/bottom...]) form.

    For dim == 1: (l, r)
    For dim == 2: (l, r, t, b)
    For dim == 3: (l, r, t, b, f, bk)  # width, height, depth order expected by PyTorch
    """
    if isinstance(padding, int):
        if dim == 1:
            return (padding, padding)
        elif dim == 2:
            return (padding, padding, padding, padding)
        elif dim == 3:
            return (padding, padding, padding, padding, padding, padding)

    # Tuple cases
    assert isinstance(padding, tuple), "padding must be int or tuple"
    if dim == 1:
        if len(padding) == 1:
            return (padding[0], padding[0])
        elif len(padding) == 2:
            return (padding[0], padding[1])
        else:
            raise ValueError("1D padding expects int or tuple of length 1 or 2.")
    elif dim == 2:
        if len(padding) == 2:
            # (pad_w, pad_h) -> (l, r, t, b)
            pw, ph = padding
            return (pw, pw, ph, ph)
        elif len(padding) == 4:
            # already (l, r, t, b)
            return padding
        else:
            raise ValueError("2D padding expects int, (w, h) or (l, r, t, b).")
    elif dim == 3:
        if len(padding) == 3:
            # (pw, ph, pd) -> (l, r, t, b, f, bk)
            pw, ph, pd = padding
            return (pw, pw, ph, ph, pd, pd)
        elif len(padding) == 6:
            return padding
        else:
            raise ValueError("3D padding expects int, (w, h, d) or (l, r, t, b, f, bk).")
    else:
        raise ValueError("dim must be 1, 2, or 3.")


def build_padding_layer(pad_cfg: Dict[str, Any],
                        padding: Union[int, Tuple[int, ...]]) -> nn.Module:
    """
    Build an explicit padding layer when nn.Conv*'s built-in padding isn't used.

    Supported:
      - type='reflect'  -> nn.ReflectionPad{1d,2d,3d}
      - type='replicate'-> nn.ReplicationPad{1d,2d,3d}
      - type='constant' -> nn.ConstantPad{1d,2d,3d} (requires value in pad_cfg['value'])
      - type='zeros' or 'circular' -> return nn.Identity() (use Conv's native padding)
    """
    pad_type = pad_cfg.get('type', 'zeros').lower()

    # If the conv will handle padding (zeros/circular), just no-op here
    if pad_type in ('zeros', 'circular'):
        return nn.Identity()

    # Infer dimensionality from the padding tuple length (fallback to 2D)
    dim: int
    if isinstance(padding, int):
        dim = 2
    else:
        dim = {2: 1, 4: 2, 6: 3}.get(len(padding), 2)

    expanded = _expand_padding_for_dim(padding, dim)

    if pad_type == 'reflect':
        if dim == 1:
            return nn.ReflectionPad1d(expanded)
        elif dim == 2:
            return nn.ReflectionPad2d(expanded)
        else:
            return nn.ReflectionPad3d(expanded)
    elif pad_type == 'replicate':
        if dim == 1:
            return nn.ReplicationPad1d(expanded)
        elif dim == 2:
            return nn.ReplicationPad2d(expanded)
        else:
            return nn.ReplicationPad3d(expanded)
    elif pad_type == 'constant':
        value = pad_cfg.get('value', 0.0)
        if dim == 1:
            return nn.ConstantPad1d(expanded, value)
        elif dim == 2:
            return nn.ConstantPad2d(expanded, value)
        else:
            return nn.ConstantPad3d(expanded, value)
    else:
        raise ValueError(f"Unsupported padding type: {pad_type}")
