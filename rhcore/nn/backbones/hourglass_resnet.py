import torch
import torch.nn as nn
from rhcore.nn.packs.residue_blocks import ResidueBlock
from rhcore.nn.backbones.conv_decoder import ConvDecoder
from rhcore.nn.backbones.conv_encoder import ConvEncoder
from rhcore.nn.utils.make_layers import make_layers

from rhcore.nn.utils.init_weights.std_init import init_weights

class HourglassResNet(nn.Module):

    def __init__(self, 
                 in_channels=None,
                 out_channels=None,
                 base_channels=64, num_blocks=9, num_scales=3,
                 norm_cfg=dict(type='IN'), act_cfg=dict(type='ReLU'),
                 drop=0.0,  padding_mode='reflect'):
        
        super().__init__()

        self.encoder = ConvEncoder(base_channels=base_channels, 
                                   num_downsamples=num_scales, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Bottleneck Residue Blocks
        bottleneck_channels = base_channels * (2**num_scales)
        self.bottleneck = make_layers(ResidueBlock, num_blocks,
                                      channels=bottleneck_channels, padding_mode=padding_mode,
                                      norm_cfg=norm_cfg, act_cfg=act_cfg, drop=drop)

        self.decoder = ConvDecoder(base_channels=bottleneck_channels, 
                                   num_upsamples=num_scales, norm_cfg=norm_cfg, act_cfg=act_cfg)

        if in_channels is not None and out_channels is not None:
            self.stem = nn.Sequential(
                            nn.ReflectionPad2d(3),
                            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=0, bias=False),
                            nn.ReLU(inplace=True),
                        )
            self.head = nn.Sequential(
                            nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, padding=0),
                            nn.Tanh(),
                        )
        else:
            self.stem = nn.Identity()
            self.head = nn.Identity()

        self.apply(init_weights)
        
    def forward(self, x):

        h = self.stem(x)
        h = self.encoder(h)
        h = self.bottleneck(h)
        h = self.decoder(h)
        h = self.head(h)

        return h

if __name__ == "__main__":

    base_channels=64
    num_blocks=3
    num_scales=3

    batch=2
    height=64
    width=64

    device = 'cpu'
    in_channels = 3

    desc = 'Simple Test'

    print(f"\n[CASE] {desc}")

    model = HourglassResNetBackbone(**dict(base_channels=base_channels, num_blocks=8, num_scales=3)).to(device)
    
    x = torch.randn(batch, base_channels, height, width, device=device)
    y = model(x)

    # naive checks: tensor type/device, batch size, and spatial size preserved
    assert isinstance(y, torch.Tensor), "Output is not a tensor"
    assert y.device == x.device, "Output is on a different device"
    assert y.dtype == x.dtype, "Output dtype differs from input dtype"
    assert y.shape[0] == x.shape[0], "Batch dim changed"
    assert y.shape[2:] == x.shape[2:], f"Spatial size changed: {y.shape[2:]} vs {x.shape[2:]}"
    # channel check is intentionally skipped (decoder head may change channels)

    print(f"  input : {tuple(x.shape)}")
    print(f"  output: {tuple(y.shape)}")
    print("  ✅ shape OK")