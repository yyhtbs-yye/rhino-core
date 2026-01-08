import torch
import torch.nn as nn
from einops import rearrange

from functools import lru_cache

@lru_cache(maxsize=10)
def calculate_rpi_sa(window_size):
    """
    Calculate relative position index for self-attention when window_size is (Wh, Ww).
    Returns: [Wh*Ww, Wh*Ww] LongTensor of linearized 2D relative positions.
    """
    Wh, Ww = window_size  # height, width
    coords_h = torch.arange(Wh)
    coords_w = torch.arange(Ww)
    # 2, Wh, Ww  (use 'ij' to avoid default-change warning)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    # 2, N, N  where N = Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    # N, N, 2
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()

    # shift to start from 0
    relative_coords[:, :, 0] += Wh - 1  # delta_h in [0, 2*Wh-2]
    relative_coords[:, :, 1] += Ww - 1  # delta_w in [0, 2*Ww-2]

    # map 2D offsets -> 1D index
    # scale height offsets by width range
    relative_coords[:, :, 0] *= (2 * Ww - 1)
    relative_position_index = relative_coords.sum(-1)  # [N, N]
    return relative_position_index

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class DynamicPosEncoder(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.idx_cache = {}

    def forward(self, H, W, device):

        nT = H * W

        if (H, W) in self.idx_cache:
            relative_position_index, biases = self.idx_cache[(H, W)]

        else:
            relative_position_index = calculate_rpi_sa((H, W)).to(device)
            biases = torch.stack(torch.meshgrid(
                torch.arange(1 - H, H, device=device),
                torch.arange(1 - W, W, device=device),
                indexing='ij'
            ), dim=-1).reshape(-1, 2).float()
            self.idx_cache[(H, W)] = (relative_position_index, biases)

        rpi_table = self.pos(biases)  # 2H-1 * 2W-1, heads

        relative_position_bias = rpi_table[relative_position_index.view(-1)].view(nT, nT, -1)  # H*W, H*W, nH
        relative_position_bias = rearrange(relative_position_bias, 'nT1 nT2 nH -> 1 nH nT1 nT2')

        return relative_position_bias