import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcore.utils.build_components import build_module

class Chain(nn.Module):

    def __init__(self, module_configs):
        super().__init__()

        assert isinstance(module_configs, (list, tuple)), "module_configs must be list/tuple"
        assert len(module_configs) > 0, "module_configs cannot be empty"

        nets = []
        for config in module_configs:
            if config is None:
                continue
            module = build_module(config)
            if isinstance(module, nn.Identity):
                continue
            nets.append(module)

        self.nets = nn.ModuleList(nets)

    def forward(self, x, **args):
        out = x
        for module in self.nets:
            out = module(out, **args)
        return out
