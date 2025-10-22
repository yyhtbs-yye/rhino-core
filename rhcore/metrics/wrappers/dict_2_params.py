import torch
import torch.nn as nn

class Dict2ListParams(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, results):
        metric_val = self.module(*list(results.values()))
        return metric_val
