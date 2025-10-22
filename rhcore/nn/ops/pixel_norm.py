import torch
import torch.nn as nn

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
    
def pixel_norm(x, eps=1e-6):
    """Pixel Normalization.

    This normalization is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        x (torch.Tensor): Tensor to be normalized.
        eps (float, optional): Epsilon to avoid dividing zero.
            Defaults to 1e-6.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    if torch.__version__ >= '1.7.0':
        norm = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
    else:   # support older pytorch version
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
    norm = norm / torch.sqrt(torch.tensor(x.shape[1]).to(x))

    return x / (norm + eps)
