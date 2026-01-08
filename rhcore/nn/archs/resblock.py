import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcore.utils.build_components import build_module

class ResBlock(nn.Module):

    def __init__(self, module_config, learn_gate=False, gate_value=1.0):
        super().__init__()

        self.module_config = module_config
        self.learn_gate = learn_gate
        self.gate_value = gate_value

        assert self.module_config is not None, "the main path module must not be None or nn.Identity"

        self.net = build_module(module_config)

        if self.learn_gate:
            if not isinstance(gate_value, float):
                print(f"Got init Gate Value={gate_value}, Type={type(gate_value)}, Set to Default Value 1.0")
                gate_value = 1.0
            self.gate_value = nn.Parameter(torch.Tensor(gate_value), requires_grad=True)
        else:
            print(f"Using No Trainable Gate Value={self.gate_value}")

    def forward(self, x, **kwargs):

        out = self.net(x, **kwargs)

        if not isinstance(out, torch.Tensor):
            raise "the variable out must be a torch.Tensor"

        return out * self.gate_value + x