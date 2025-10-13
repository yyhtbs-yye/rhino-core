import torch
import torch.nn as nn

def build_dropout_layer(drop_cfg):
    if drop_cfg is None:
        return nn.Identity()
    if isinstance(drop_cfg, (float, int)):
        p = float(drop_cfg)
        if p <= 0:
            return nn.Identity()
        return nn.Dropout(p=p)
    if isinstance(drop_cfg, dict):
        p = float(drop_cfg.get("p", 0.0))
        if p <= 0:
            return nn.Identity()
        # keep simple: only standard Dropout
        return nn.Dropout(p=p)
    # Fallback: disabled
    return nn.Identity()
