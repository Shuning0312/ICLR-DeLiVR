import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .so3 import SO3Head, log_so3



class GDFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.66, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dim = dim
        self.hidden = hidden
        
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, 
                                padding=1, groups=hidden * 2, bias=True)
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:

        B, TN, C = x.shape
        N = H * W
        
        # (B, T*N, C) -> (B*T, C, H, W)
        x = x.view(B, T, N, C).permute(0, 1, 3, 2).reshape(B * T, C, H, W)
        
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2  # Gated Linear Unit
        x = self.project_out(x)
        x = self.drop(x)
        
        # (B*T, C, H, W) -> (B, T*N, C)
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, C)
        x = x.view(B, T * N, C)
        return x


class DeepPatchEmbed(nn.Module):

    def __init__(self, in_ch: int, embed_dim: int, patch: int):
        super().__init__()
        self.patch = patch
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
        )
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], torch.Tensor]:
        shallow_feat = self.stem(x)       # (B, C, H, W)
        feat = self.proj(shallow_feat)    # (B, C, H', W')
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, C)
        return tokens, (H, W), shallow_feat


class NonLinearLieBias(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_heads)
        )
    
    def forward(self, geometry_sim: torch.Tensor) -> torch.Tensor:
        # (B, N, N) -> (B, N, N, 1) -> MLP -> (B, N, N, num_heads)
        bias = self.mlp(geometry_sim.unsqueeze(-1))
        # (B, N, N, num_heads) -> (B, num_heads, N, N)
        return bias.permute(0, 3, 1, 2)


class ResidualGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, delta: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:

        gate = self.gate(torch.cat([delta, x_in], dim=1))
        return gate * delta + (1 - gate) * x_in


class CAB(nn.Module):

    def __init__(self, n_feat: int, kernel_size: int = 3, reduction: int = 8, bias: bool = True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias),
            nn.GELU(),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, n_feat // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res = self.ca(res) * res
        return res + x



class SmoothUpsample(nn.Module):

    def __init__(self, in_dim: int, out_ch: int, scale: int):
        super().__init__()
        self.scale = scale

        self.conv = nn.Conv2d(in_dim, out_ch * scale * scale, kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(scale)
        
        self.smooth = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.upsample(x)

        x = self.smooth(x)
        return x


def build_temporal_decay_bias(T: int, N: int, device, dtype, tau = 2.0):
    t = torch.arange(T, device=device, dtype=torch.float32)   
    dt = (t[:, None] - t[None, :]).abs()                      # (T,T)
    
    if isinstance(tau, torch.Tensor):
        tau_val = tau.float()  # 确保 fp32
    else:
        tau_val = float(tau)
    
    decay_ts = torch.exp(-dt / tau_val)                       # (T,T) fp32

    decay = decay_ts[:, None, :, None].expand(T, N, T, N).reshape(T * N, T * N)
    return decay.to(dtype)


class PatchEmbed(nn.Module):
    """
    Conv2d-based patch embedding:
        x: (B, C_in, H, W) -> feat: (B, C, H', W') -> tokens: (B, N=H'*W', C)
    """
    def __init__(self, in_ch: int, embed_dim: int, patch: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        feat = self.proj(x)                             # (B, C, H', W')
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)        # (B, N, C)
        return tokens, (H, W)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class LieRelativeBias(nn.Module):
    def __init__(self):
        super().__init__()
        self._p_cache = {}  # key: (H, W, device, dtype) -> (N, 3)

    def _get_p(self, H: int, W: int, device, dtype) -> torch.Tensor:
        key = (H, W, device, dtype)
        if key in self._p_cache:
            return self._p_cache[key]

        ys = torch.arange(H, device=device, dtype=dtype)
        xs = torch.arange(W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)

        cx = (W - 1) * 0.5
        cy = (H - 1) * 0.5
        xn = (grid_x - cx) / max(W - 1, 1)
        yn = (grid_y - cy) / max(H - 1, 1)
        ones = torch.ones_like(xn)
        p = torch.stack([xn, yn, ones], dim=-1)                 # (H,W,3)
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-9)           # L2 normalize
        p = p.view(H * W, 3).contiguous()                        # (N,3)

        self._p_cache[key] = p
        return p

    def forward(self, R: torch.Tensor, H: int, W: int, dtype=None) -> torch.Tensor:
        """
        R: (B,3,3)
        return: bias (B, N, N)
        """
        B = R.shape[0]
        device = R.device
        if dtype is None:
            dtype = R.dtype

        p = self._get_p(H, W, device, dtype)              # (N,3)
        # 旋转 p：p_rot[b] = p @ R[b]^T   (等价于 (R @ p^T)^T)
        # (N,3) @ (3,3) -> (N,3), 对 B 做批量
        p_rot = torch.einsum('nc,bcg->bng', p, R.transpose(1, 2))  # (B,N,3)

        # bias[b] = p_rot[b] @ p^T -> (N,N)
        pT = p.t().unsqueeze(0).expand(B, 3, H * W)        # (B,3,N)
        bias = torch.bmm(p_rot, pT)                        # (B,N,N)
        return bias


class LieMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.nonlinear_bias = NonLinearLieBias(num_heads)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        assert bias.dim() == 3 and bias.shape[0] == B and bias.shape[1] == N and bias.shape[2] == N, \
            f"bias must be (B,N,N), got {tuple(bias.shape)} while x is {(B,N,C)}"
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale      # (B, heads, N, N)
        
        lie_bias = self.nonlinear_bias(bias)  # (B, num_heads, N, N)
        attn = attn + lie_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


class LieTransformerBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.66, drop: float = 0.0,
                 attn_drop: float = 0.0, init_values: float = 1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LieMultiHeadAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.gdfn = GDFN(dim, mlp_ratio=mlp_ratio, drop=drop)
        
        self.gamma1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor, bias: torch.Tensor, 
                T: int, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:

        res, attn = self.attn(self.norm1(x), bias)
        x = x + self.gamma1.unsqueeze(0).unsqueeze(0) * res

        x = x + self.gamma2.unsqueeze(0).unsqueeze(0) * self.gdfn(self.norm2(x), T, H, W)
        return x, attn

class LieTransformerNet(nn.Module):

    def __init__(self,
                 in_ch: int = 3,
                 embed_dim: int = 96,
                 depth: int = 6,
                 heads: int = 4,
                 mlp_ratio: float = 2.66, 
                 patch: int = 4,
                 so3_mode: str = 'so2',
                 angle_max_deg: float = 45.0,
                 so3_stochastic: bool = True,

                 temporal_tau: float = 2.0,
                 temporal_band: int = 1,
                 
                 temporal_time_weight: float = 1.0,
                 temporal_time_kappa: float = 1.0,
                
                 enable_spatial_bias: bool = True,
                 enable_temporal_bias: bool = True,
                 enable_decay: bool = True,
                 enable_band_mask: bool = True,
               
                 learnable_tau: bool = True,
                
                 num_cab: int = 2,
                 
                 return_aux: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.patch = patch
        self.temporal_band = temporal_band
        self.temporal_time_weight = temporal_time_weight
        self.temporal_time_kappa = temporal_time_kappa
        self.return_aux = return_aux
        self.learnable_tau = learnable_tau
        
        self.enable_spatial_bias = enable_spatial_bias
        self.enable_temporal_bias = enable_temporal_bias
        self.enable_decay = enable_decay
        self.enable_band_mask = enable_band_mask

       
        if learnable_tau:
            # softplus(x) + 0.5 = tau => x = log(exp(tau - 0.5) - 1)
            init_val = math.log(math.exp(temporal_tau - 0.5) - 1 + 1e-6)
            self.tau_raw = nn.Parameter(torch.tensor(init_val))
        else:
            self.temporal_tau = temporal_tau

        self.embed = DeepPatchEmbed(in_ch, embed_dim, patch)
        self.depatch = SmoothUpsample(embed_dim, in_ch, patch)

        # bounded rotation head
        self.so3_head = SO3Head(in_dim=embed_dim,
                                angle_max_deg=angle_max_deg,
                                mode=so3_mode,
                                stochastic=so3_stochastic)

        # relative bias
        self.rel_bias = LieRelativeBias()


        self.blocks = nn.ModuleList([
            LieTransformerBlock(embed_dim, heads, mlp_ratio=mlp_ratio, drop=0.0, attn_drop=0.0)
            for _ in range(depth)
        ])

        self.refinement = nn.Sequential(*[CAB(embed_dim) for _ in range(num_cab)])

        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

        self._x_in = None
    
    def get_tau(self) -> torch.Tensor:
        if self.learnable_tau:
            return F.softplus(self.tau_raw) + 0.5  # 确保 tau >= 0.5
        else:
            return torch.tensor(self.temporal_tau, device=self.tau_raw.device if hasattr(self, 'tau_raw') else 'cpu')

    def pixel_head(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = tokens.shape
        feat = tokens.transpose(1, 2).contiguous().view(B, C, H, W)  # (B, embed_dim, H', W')
        delta = self.depatch(feat)                                   # (B, in_ch, H, W)
        if self._x_in is not None:
            return self._x_in + delta
        return delta

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B, C_in, H, W = x.shape
        assert C_in % self.in_ch == 0, "Channel not divisible by in_ch"
        T = C_in // self.in_ch
        assert T >= 1 and T % 2 == 1, "Expect odd T, e.g., 3/5/7"

        self._x_in = x

        x_bt = x.view(B, T, self.in_ch, H, W).reshape(B * T, self.in_ch, H, W)  # (B*T,3,H,W)
        tokens, (H_p, W_p), shallow_feat = self.embed(x_bt)    # tokens: (B*T, N, C), shallow_feat: (B*T, C, H, W)
        N = H_p * W_p
        feat = self.embed.proj(shallow_feat)             # (B*T, C, H_p, W_p)

        # 2) 每帧旋转 R_t
        R_bt, mu, log_sigma, omega = self.so3_head(feat)  # (B*T,3,3)
        R = R_bt.view(B, T, 3, 3)                         # (B,T,3,3)

        # Adjacent-frame Lie velocity sequence v_t = || log( R_{t-1}^T R_t ) ||
        if R.dim() == 4 and R.size(1) >= 2:
            R_prev = R[:, :-1]                               # (B, T-1, 3,3)
            R_curr = R[:,  1:]                               # (B, T-1, 3,3)
            R_rel_adj = torch.matmul(R_prev.transpose(-1, -2), R_curr)   # (B,T-1,3,3)
            omega_adj = log_so3(R_rel_adj)                   # (B, T-1, 3)
            v_seq = omega_adj.norm(dim=-1)                   # (B, T-1)
        else:
            v_seq = None

        device, dtype = tokens.device, tokens.dtype
        P = self.rel_bias._get_p(H_p, W_p, device, dtype)  # (N,3)

        RP = torch.einsum('btcd,dn->btcn', R, P.t()).transpose(-1, -2).contiguous()

        bias_space = torch.einsum('btic,bsjc->btisj', RP, RP).reshape(B, T * N, T * N)
        if not getattr(self, 'enable_spatial_bias', True):
            bias_space = bias_space.new_zeros(bias_space.shape)

        with torch.no_grad():
            # relative rotation R_rel[b,t,s] = R_t^T R_s
            R_rel = torch.einsum('btij,bsjk->btsik', R.transpose(-1, -2), R)  # (B,T,T,3,3)
            # Δω = log(R_rel) -> (B,T,T,3)
            domega = log_so3(R_rel)                                              # (B,T,T,3)
            theta_rel = domega.norm(dim=-1).clamp(min=0.0)                       # (B,T,T)
            # map to bias; negative distance in logit space
            b_time_ts = - theta_rel / max(1e-6, self.temporal_time_kappa)        # (B,T,T)
            # expand to token-level (B,T*N,T*N)
            b_time = b_time_ts[:, :, :, None, None].expand(B, T, T, N, N).reshape(B, T*N, T*N)
            if not getattr(self, 'enable_temporal_bias', True):
                b_time = b_time.new_zeros(b_time.shape)

        tau = self.get_tau() if self.learnable_tau else self.temporal_tau
        decay = build_temporal_decay_bias(T, N, device, dtype, tau=tau)  # (T*N,T*N)
        if not getattr(self, 'enable_decay', True):
            decay = decay.new_ones(decay.shape)
        
        # final spatiotemporal bias: spatial + (time_weight * time_bias), then apply decay and band mask
        bias = bias_space + self.temporal_time_weight * b_time
        bias = bias * decay.unsqueeze(0)  # (B,T*N,T*N)
        band = int(getattr(self, "temporal_band", 1))
        if getattr(self, 'enable_band_mask', True) and band >= 0:
            with torch.no_grad():
                ts = torch.arange(T, device=device)
                dt_ok = (ts[:, None] - ts[None, :]).abs() <= band   # (T,T) bool
                mask = dt_ok[:, None, :, None].expand(T, N, T, N).reshape(T * N, T * N)  # (T*N,T*N)
            bias = bias.masked_fill(~mask.unsqueeze(0), -1e4)

        X = tokens.view(B, T, N, self.embed_dim).reshape(B, T * N, self.embed_dim)  # (B,T*N,C)
        
        shortcut = feat.flatten(2).transpose(1, 2).view(B, T * N, self.embed_dim)
        
        attns = []
        for blk in self.blocks:
            X, attn = blk(X, bias, T, H_p, W_p) 
            attns.append(attn)
        
        X = X + shortcut

        X_map = X.view(B * T, N, self.embed_dim).transpose(1, 2).reshape(B * T, self.embed_dim, H_p, W_p)
        
        X_map = self.refinement(X_map)
        
        y_bt = self.depatch(X_map)  # (B*T, 3, H, W)
        y = y_bt.reshape(B, T * self.in_ch, H, W)
        
        y_frames = y.view(B * T, self.in_ch, H, W)
        x_frames = self._x_in.view(B * T, self.in_ch, H, W)
        out_frames = x_frames + self.res_scale * y_frames
        out = out_frames.view(B, T * self.in_ch, H, W)

        tau_val = self.get_tau() if self.learnable_tau else self.temporal_tau
        aux = {
            'R': R_bt.view(B, T, 3, 3), 
            'mu': mu.view(B, T, -1), 
            'log_sigma': log_sigma.view(B, T, -1), 
            'omega': omega.view(B, T, -1), 
            'lie_vel_seq': v_seq, 
            'attns': attns,
            'tau': tau_val.item() if isinstance(tau_val, torch.Tensor) else tau_val,
            'res_scale': self.res_scale.item()
        }
        return out, aux
