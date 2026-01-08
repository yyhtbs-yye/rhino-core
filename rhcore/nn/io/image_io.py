import torch
import torch.nn as nn

class FromRGB(nn.Module):
    """
    Map input RGB (or multi-channel) image to feature space.

    x: (B, C_in, H, W) -> (B, C_feat, H, W)
    """
    def __init__(self, in_channels=3, feat_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        return self.conv(x)

class ToRGB(nn.Module):
    """
    Map feature space back to RGB (or out_channels).

    x: (B, C_feat, H, W) -> (B, C_out, H, W)
    """
    def __init__(self, feat_channels=64, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(feat_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        return self.conv(x)

