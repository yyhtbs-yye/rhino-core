import torch
import torch.nn as nn
from rhcore.nn.packs.conv_module import ConvModule

class ResidueBlock(nn.Module):
    """Define a Residual Block with dropout layers.
    x -> conv -> drop -> conv -> h + x -> output
    
    Args:
        channels (int): Number of channels in the conv layer.
        padding_mode (str): The name of padding layer:
            'reflect' | 'replicate' | 'zeros'.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: True.
    """

    def __init__(self, channels, padding_mode='zero',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop=0.5):
        
        super().__init__()
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the residual block with dropout layers.
        # Only for IN, use bias to follow cyclegan's original implementation.
        use_bias = norm_cfg['type'] == 'IN'

        self.conv1 = ConvModule(in_channels=channels,
                                out_channels=channels,
                                kernel_size=3,
                                padding=1,
                                bias=use_bias,
                                act_cfg=act_cfg,
                                norm_cfg=norm_cfg,
                                padding_mode=padding_mode)

        self.drop = nn.Dropout(drop) 

        self.conv2 = ConvModule(in_channels=channels,
                                out_channels=channels,
                                kernel_size=3,
                                padding=1,
                                bias=use_bias,
                                act_cfg=None,
                                norm_cfg=norm_cfg,
                                padding_mode=padding_mode)

    def forward(self, x):
        out = x + self.conv2(self.drop(self.conv1(x)))
        return out