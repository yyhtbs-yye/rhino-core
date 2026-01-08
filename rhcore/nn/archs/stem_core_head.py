import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcore.utils.build_components import build_module

class StemCoreHead(nn.Module):

    def __init__(self, stem_config=None, core_config=None, head_config=None):
        super().__init__()

        self.stem_config = stem_config
        self.core_config = core_config
        self.head_config = head_config
        if stem_config is None:
            self.stem = nn.Identity()
        else:
            self.stem = build_module(stem_config)

        if core_config is None:
            self.core = nn.Identity()
        else:
            self.core = build_module(core_config)

        if head_config is None:
            self.head = nn.Identity()
        else:
            self.head = build_module(head_config)

    def forward(self, x, **args):

        if self.stem_config is None:
            stem_out = self.stem(x)
        else:
            stem_out = self.stem(x, **args)
        if self.core_config is None:
            core_out = self.core(stem_out)
        else:
            core_out = self.core(stem_out, **args)

        if self.head_config is None:
            head_out = self.head(core_out)
        else:
            head_out = self.head(core_out, **args)

        return head_out
