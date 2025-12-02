from typing import Literal
import math
import torch
import torch.nn as nn

def _fan_in(tensor: torch.Tensor) -> int:
    if tensor.ndim < 2:
        raise ValueError("Tensor must have at least 2 dims for fan_in")
    fan = tensor.size(1)
    for d in tensor.shape[2:]:
        fan *= d
    return fan

def _lecun_normal_(w: torch.Tensor) -> None:
    std = 1.0 / math.sqrt(_fan_in(w))
    nn.init.normal_(w, mean=0.0, std=std)

def normal_init(module: nn.Module, mean: float = 0.0, std: float = 0.02):
    """Replacement for timm.layers.normal_init."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean=mean, std=std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


def xavier_init(module: nn.Module, gain: float = 1.0):
    """Replacement for timm.layers.xavier_init (uniform version)."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.xavier_uniform_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0.0)

def init_weights(
    m: nn.Module,
    *,
    nonlinearity: Literal[
        "relu", "leaky_relu", "prelu",
        "gelu", "silu", "swish", "mish", "softplus",
        "selu", "tanh", "sigmoid", "linear", "elu", "celu", None
    ] = "relu",
    negative_slope: float = 0.01,
    skip_if_has_init: bool = True,
) -> None:
    """
    Initialize a single nn.Module (use with model.apply(init_weights)).

    Behavior (merged):
    - If a module defines its own `init_weights()`, skip it (configurable).
    - Convs (and ConvTranspose):
        * Default: He/Kaiming normal (fan_out). For LeakyReLU/PReLU, pass slope.
        * Bias -> zeros.
    - Linear: Xavier uniform with proper gain for the chosen nonlinearity; bias zeros.
    - Norms (Batch/Sync/Instance/Group/Layer): weight ones, bias zeros (if present).
    - Embedding: normal_(0, 0.02).
    - RNNs (RNN/GRU/LSTM): ih weights Kaiming uniform, hh weights orthogonal, bias zeros; LSTM forget gate bias = 1.
    - SELU path uses LeCun normal (fan_in) and pairs best with AlphaDropout.
    """
    # Respect custom per-module initialization logic
    if skip_if_has_init and hasattr(m, "init_weights") and callable(getattr(m, "init_weights")):
        return

    # --- Normalization layers ---
    if isinstance(
        m,
        (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
            nn.GroupNorm, nn.LayerNorm,
        ),
    ):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
        return

    # --- Embeddings ---
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        return

    # --- Recurrent layers ---
    if isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
        for name, p in m.named_parameters():
            if "weight_ih" in name:
                a = negative_slope if nonlinearity in {"leaky_relu", "prelu"} else 0.0
                nn.init.kaiming_uniform_(p, nonlinearity="leaky_relu" if a > 0 else "relu", a=a)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                if isinstance(m, nn.LSTM):
                    hs = m.hidden_size
                    p.data[hs:2*hs].fill_(1.0)  # forget gate bias
        return

    # --- Convs / ConvTranspose / Linear ---
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                      nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                      nn.Linear)):
        if nonlinearity is None:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        else:
            nl = nonlinearity.lower()

            # Choose init by nonlinearity
            if nl == "selu":
                # Self-Normalizing Networks recommendation
                _lecun_normal_(m.weight)
            elif nl in {"tanh", "sigmoid", "linear"}:
                gain = nn.init.calculate_gain(nl)  # tanh≈5/3, sigmoid/linear≈1
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                else:
                    # For convs, Xavier works too when followed by (near)linear squashing
                    nn.init.xavier_uniform_(m.weight, gain=gain)
            elif nl in {"leaky_relu", "prelu"}:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=negative_slope)
            else:
                # ReLU-like bucket: relu/elu/celu/gelu/silu/swish/mish/softplus
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
        return
