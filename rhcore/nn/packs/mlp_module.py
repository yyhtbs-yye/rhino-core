from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from rhcore.nn.utils.build_layers import build_activation_layer, build_dropout_layer 

class MlpModule(nn.Module):
    """
    A 2-layer MLP with configurable activation, dropout, and execution order.

    Args:
        in_dim:  Input feature dimension.
        out_dim: Output feature dimension.
        h_dim:   Hidden layer dimension (between the two Linear layers).
        bias:    Whether to use bias in Linear layers. If a string is provided,
                 it is treated as truthy and bias is enabled.
        act_cfg: Activation layer config passed to `build_activation_layer`.
                 If None/False, Identity is used.
        drop_cfg: Dropout configuration. Accepts:
                  - None/0.0: no dropout (Identity)
                  - float/int: interpreted as p for nn.Dropout
                  - dict: must contain at least {"p": float}; any extra keys are ignored.
        order:   Execution order of sub-layers. Must contain *two* occurrences of "fc"
                 (each occurrence applies one Linear layer in sequence), and may include
                 "act" and "drop". Default: ('fc', 'act', 'drop', 'fc').
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        h_dim: int,
        bias: Union[bool, str] = True,
        act_cfg: Optional[Dict] = dict(type="ReLU"),
        drop_cfg=None,
        order: Tuple[str, ...] = ("fc", "act", "drop", "fc", "drop"),
        channel_first: bool = False,
    ):
        super().__init__()

        # Normalize bias
        bias_bool = bool(bias) if isinstance(bias, str) else bool(bias)

        self.channel_first = channel_first

        # Core layers: Linear (default) or 1x1 Conv2d when channel_first
        if self.channel_first:
            self.fc1 = nn.Conv2d(in_dim, h_dim, kernel_size=1, bias=bias_bool)
            self.fc2 = nn.Conv2d(h_dim, out_dim, kernel_size=1, bias=bias_bool)
        else:
            self.fc1 = nn.Linear(in_dim, h_dim, bias=bias_bool)
            self.fc2 = nn.Linear(h_dim, out_dim, bias=bias_bool)

        # Activation
        if act_cfg:
            self.act = build_activation_layer(act_cfg)
        else:
            self.act = nn.Identity()

        # Dropout
        if drop_cfg:
            self.drop = build_dropout_layer(drop_cfg)
        else:
            self.act = nn.Identity()

        # Validate & store order
        self.order = self._normalize_order(order)

        # Build a sequenced list for fast forward
        self._ops = self._make_ops(self.order)

    @staticmethod
    def _normalize_order(order: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Ensures there are exactly two 'fc' steps. If fewer, append; if more, trim.
        Unknown tokens are ignored.
        """
        valid = {"fc", "act", "drop"}
        filtered = [tok for tok in order if tok in valid]

        fc_count = filtered.count("fc")
        if fc_count < 2:
            # Append missing fc's at the end to make the module valid.
            filtered.extend(["fc"] * (2 - fc_count))
        elif fc_count > 2:
            # Keep only the first two 'fc' tokens.
            seen = 0
            new_filtered = []
            for tok in filtered:
                if tok == "fc":
                    if seen < 2:
                        new_filtered.append(tok)
                        seen += 1
                else:
                    new_filtered.append(tok)
            filtered = new_filtered

        return tuple(filtered)

    def _make_ops(self, order: Tuple[str, ...]):
        """
        Build the list of callables in the exact order.
        Each 'fc' occurrence maps to fc1 then fc2, in order of appearance.
        """
        ops = []
        fc_used = 0
        for tok in order:
            if tok == "fc":
                if fc_used == 0:
                    ops.append(("fc1", self.fc1))
                elif fc_used == 1:
                    ops.append(("fc2", self.fc2))
                # Ignore additional fc's (shouldn't happen after normalization)
                fc_used += 1
            elif tok == "act":
                ops.append(("act", self.act))
            elif tok == "drop":
                ops.append(("drop", self.drop))
            # Unknown tokens already filtered
        # If somehow fewer than 2 fc's made it through, append remaining
        if fc_used < 2:
            ops.append(("fc2" if fc_used == 1 else "fc1", self.fc1 if fc_used == 0 else self.fc2))
        return ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, layer in self._ops:
            x = layer(x)
        return x
