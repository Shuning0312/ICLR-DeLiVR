import torch
import torch.nn as nn
import torch.nn.functional as F

from .so3 import SO3Head

def unit_embed_xy(H, W, device, dtype):
    """Map 2D pixel coordinates to unit 3D vectors: p(x,y) ~ normalize([x, y, 1]).
    Returns: (H*W, 3)
    """
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing='ij',
    )
    ones = torch.ones_like(xs)
    P = torch.stack([xs, ys, ones], dim=-1)  # (H,W,3)
    P = P / torch.linalg.norm(P, dim=-1, keepdim=True).clamp(min=1e-9)
    return P.view(-1, 3)

class LiePositionalBias(nn.Module):
    """Builds a relative attention bias matrix from rotated 3D unit coordinates.
    Given rotation R (B,3,3) and base coords P (HW,3), we compute
      B_ij = dot( (R P_i), P_j ).
    The bias is projected to `n_heads` and added to attention logits.
    """
    def __init__(self, n_heads, scale=1.0):
        super().__init__()
        self.n_heads = n_heads
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, R, P):
        # R: (B,3,3), P: (HW,3)
        B, _, _ = R.shape
        HW = P.shape[0]
        # Rotate coordinates: (B,HW,3)
        RP = (R @ P.t().unsqueeze(0)).transpose(1,2).contiguous()
        # Dot with base coords: (B,HW,HW)
        bias = torch.einsum('bik,jk->bij', RP, P)
        # Expand to heads: (B, n_heads, HW, HW)
        bias = bias.unsqueeze(1).expand(B, self.n_heads, HW, HW) * self.scale
        return bias

class LieMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pos_bias = LiePositionalBias(n_heads=num_heads, scale=1.0)

    def forward(self, x, R, P):
        # x: (B, HW, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        bias = self.pos_bias(R, P)  # (B, heads, N, N)
        assert bias.dim() == 4 and bias.shape[0] == B and bias.shape[2] == N and bias.shape[3] == N, \
            f"pos bias must be (B,heads,N,N), got {tuple(bias.shape)} while x is {(B,N,C)}"
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MLP(nn.Module):
    def __init__(self, dim, hidden_mult=4, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * hidden_mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * hidden_mult, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LieTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LieMultiHeadAttention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_mult=int(mlp_ratio), drop=drop)

    def forward(self, x, R, P):
        x = x + self.attn(self.norm1(x), R, P)
        x = x + self.mlp(self.norm2(x))
        return x

class SO3AwarePositionalEncoder(nn.Module):
    """Compute base 3D unit positional encodings P (HW,3) at runtime and cache by shape."""
    def __init__(self):
        super().__init__()
        self.cache = {}

    def get_P(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if key not in self.cache:
            P = unit_embed_xy(H, W, device, dtype)
            self.cache[key] = P
        return self.cache[key]
