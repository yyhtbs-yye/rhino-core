import torch
import torch.nn as nn

from rhcore.utils.build_components import build_module

class UNet(nn.Module):

    def __init__(self, down_config, up_config, num_levels):
        super().__init__()

        if not isinstance(num_levels, int) or num_levels < 2:
            raise ValueError(f"num_levels must be int >= 2, got {num_levels}")

        self.num_levels = num_levels

        down_configs = self._normalize_configs(down_config, num_levels, "down_config")
        up_configs = self._normalize_configs(up_config, num_levels - 1, "up_config")

        self.down_modules = nn.ModuleList([build_module(cfg) for cfg in down_configs])
        self.up_modules = nn.ModuleList([build_module(cfg) for cfg in up_configs])

    @staticmethod
    def _normalize_configs(config, expected_len, name):
        if expected_len == 0:
            return []

        if isinstance(config, (list, tuple)):
            if len(config) == expected_len:
                return list(config)
            if len(config) == 1:
                return list(config) * expected_len
            raise ValueError(f"{name} expects {expected_len} configs, got {len(config)}")

        if config is None:
            raise ValueError(f"{name} cannot be None")

        return [config for _ in range(expected_len)]

    def forward(self, x, **kwargs):
        skips = []
        out = x

        for idx, module in enumerate(self.down_modules):
            out = module(out, **kwargs) if kwargs else module(out)
            if not isinstance(out, torch.Tensor):
                raise TypeError("Down modules must return torch.Tensor")
            if idx != len(self.down_modules) - 1:
                skips.append(out)

        if len(skips) != len(self.up_modules):
            raise RuntimeError(
                f"Number of skip connections ({len(skips)}) does not match "
                f"decoder modules ({len(self.up_modules)})."
            )

        for skip, module in zip(reversed(skips), self.up_modules):
            out = torch.cat([out, skip], dim=1)
            out = module(out, **kwargs) if kwargs else module(out)
            if not isinstance(out, torch.Tensor):
                raise TypeError("Up modules must return torch.Tensor")

        return out

if __name__ == "__main__":
    torch.manual_seed(0)

    down_configs = [
        dict(
            path="torch.nn",
            name="Conv2d",
            params=dict(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
        ),
        dict(
            path="torch.nn",
            name="Conv2d",
            params=dict(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        ),
    ]

    up_configs = [
        dict(
            path="torch.nn",
            name="ConvTranspose2d",
            params=dict(in_channels=64 + 32, out_channels=32, kernel_size=4, stride=2, padding=1),
        ),
        dict(
            path="torch.nn",
            name="ConvTranspose2d",
            params=dict(in_channels=32 + 16, out_channels=16, kernel_size=4, stride=2, padding=1),
        ),
    ]

    model = UNet(down_configs, up_configs, num_levels=3)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
