# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from rhcore.nn.packs.conv_module import ConvModule

class UnetSkipConnectionBlock(nn.Module):

    def __init__(self,
                 outer_channels,
                 inner_channels,
                 in_channels=None,
                 submodule=None,
                 is_outermost=False,
                 is_innermost=False,
                 norm_cfg=dict(type='BN'),
                 use_dropout=False):
        super().__init__()
        # cannot be both outermost and innermost
        assert not (is_outermost and is_innermost), (
            "'is_outermost' and 'is_innermost' cannot be True"
            'at the same time.')
        self.is_outermost = is_outermost
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the unet skip connection block.
        # Only for IN, use bias since it does not have affine parameters.
        use_bias = norm_cfg['type'] == 'IN'

        kernel_size = 4
        stride = 2
        padding = 1
        if in_channels is None:
            in_channels = outer_channels
        down_conv_cfg = dict(type='Conv2d')
        down_norm_cfg = norm_cfg
        down_act_cfg = dict(type='LeakyReLU', negative_slope=0.2)
        up_conv_cfg = dict(type='ConvTranspose2d')
        up_norm_cfg = norm_cfg
        up_act_cfg = dict(type='ReLU')
        up_in_channels = inner_channels * 2
        up_bias = use_bias
        middle = [submodule]
        upper = []

        if is_outermost:
            down_act_cfg = None
            down_norm_cfg = None
            up_bias = True
            up_norm_cfg = None
            upper = [nn.Tanh()]
        elif is_innermost:
            down_norm_cfg = None
            up_in_channels = inner_channels
            middle = []
        else:
            upper = [nn.Dropout(0.5)] if use_dropout else []

        down = [
            ConvModule(
                in_channels=in_channels,
                out_channels=inner_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=use_bias,
                conv_cfg=down_conv_cfg,
                norm_cfg=down_norm_cfg,
                act_cfg=down_act_cfg,
                order=('act', 'conv', 'norm'))
        ]
        up = [
            ConvModule(
                in_channels=up_in_channels,
                out_channels=outer_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=up_bias,
                conv_cfg=up_conv_cfg,
                norm_cfg=up_norm_cfg,
                act_cfg=up_act_cfg,
                order=('act', 'conv', 'norm'))
        ]

        model = down + middle + up + upper

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.is_outermost:
            return self.model(x)

        # add skip connections
        return torch.cat([x, self.model(x)], 1)

class UNet(nn.Module):
    """U-Net generator composed of UnetSkipConnectionBlock modules.

    This follows the common pix2pix-style recursive construction:
    [outermost] down -> ... -> [innermost] -> ... -> up [outermost]

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_channels (int): Channels in the first encoder block. Default: 64.
        num_downs (int): Number of downsamplings (depth). Must be >= 5.
            For 256x256 inputs, 7-8 is typical. Every downsample halves H/W.
        norm_cfg (dict): Norm layer config passed to blocks. Default: dict(type='BN').
        use_dropout (bool): Use dropout in intermediate blocks. Default: False.
    Notes:
        - The input H and W should be divisible by 2**num_downs.
        - The outermost block ends with Tanh(), matching your block's behavior.
    """

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 base_channels: int = 64,
                 num_downs: int = 7,
                 norm_cfg: dict = dict(type='BN'),
                 use_dropout: bool = False):
        super().__init__()
        if num_downs < 5:
            raise ValueError(f'num_downs must be >= 5, but got {num_downs}')

        # innermost: 8x base -> 8x base (no submodule inside)
        unet_block = UnetSkipConnectionBlock(
            outer_channels=base_channels * 8,
            inner_channels=base_channels * 8,
            submodule=None,
            is_innermost=True,
            norm_cfg=norm_cfg,
            use_dropout=False)

        # stack (num_downs - 5) blocks at 8x base channels (deep middle)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                outer_channels=base_channels * 8,
                inner_channels=base_channels * 8,
                submodule=unet_block,
                norm_cfg=norm_cfg,
                use_dropout=use_dropout)

        # decoder/encoder steps reducing channel width
        unet_block = UnetSkipConnectionBlock(
            outer_channels=base_channels * 4,
            inner_channels=base_channels * 8,
            submodule=unet_block,
            norm_cfg=norm_cfg,
            use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(
            outer_channels=base_channels * 2,
            inner_channels=base_channels * 4,
            submodule=unet_block,
            norm_cfg=norm_cfg,
            use_dropout=use_dropout)

        if in_channels is None and out_channels is None:
            self.model = UnetSkipConnectionBlock(
                outer_channels=base_channels,
                inner_channels=base_channels * 2,
                submodule=unet_block,
                is_outermost=True,
                norm_cfg=norm_cfg,
                use_dropout=False)
        else:
            unet_block = UnetSkipConnectionBlock(
                outer_channels=base_channels,
                inner_channels=base_channels * 2,
                submodule=unet_block,
                norm_cfg=norm_cfg,
                use_dropout=use_dropout)

            self.model = UnetSkipConnectionBlock(
                outer_channels=out_channels,
                inner_channels=base_channels,
                in_channels=in_channels,
                submodule=unet_block,
                is_outermost=True,
                norm_cfg=norm_cfg,
                use_dropout=False)

    def forward(self, x):
        return self.model(x)

if __name__=="__main__":

    base_channels = 64
    c_channels = 3
    num_downs = 7
    model = UNet(3, 3, base_channels=base_channels, num_downs=num_downs)
    x = torch.randn(2, c_channels, 256, 256)
    y = model(x)

    print(y.shape)
    assert y.shape == (2, c_channels, 256, 256)
