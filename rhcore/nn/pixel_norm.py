import torch
import torch.nn as nn
from rhcore.nn.ops.pixel_norm import pixel_norm

class PixelNorm(nn.Module):
    """Pixel Normalization.

    This module is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        eps (float, optional): Epsilon value. Defaults to 1e-6.
    """

    _abbr_ = 'pn'

    def __init__(self, *, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return pixel_norm(x, self.eps)