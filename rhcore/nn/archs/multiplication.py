import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcore.utils.build_components import build_module

class Repeat(nn.Module):

    def __init__(self, module_config, num_repeats=1, shared_params=False):
        super().__init__()

        self.module_config = module_config
        self.num_repeats = num_repeats
        self.shared_params = shared_params

        assert module_config is not None and not isinstance(module_config, nn.Identity), "the main path module must not be None or nn.Identity"
        assert num_repeats > 0, "num_repeats must be greater than 0"
        assert isinstance(shared_params, bool), "shared_params must be a boolean"
        if shared_params:
            # one module reused num_repeats times
            self.nets = nn.ModuleList([build_module(module_config)])
        else:
            # independent modules per repeat
            self.nets = nn.ModuleList([build_module(module_config) for _ in range(num_repeats)])

    def forward(self, x, **args):

        out = x

        if self.shared_params:
            module = self.nets[0]
            for _ in range(self.num_repeats):
                out = module(out, **args)
        else:
            for module in self.nets:
                out = module(out, **args)

        return out
