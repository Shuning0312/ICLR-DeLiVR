# src/models/so3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def hat(v):
    vx, vy, vz = v.unbind(dim=-1)
    O = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([O, -vz,  vy], dim=-1),
        torch.stack([vz,  O, -vx], dim=-1),
        torch.stack([-vy, vx,  O], dim=-1),
    ], dim=-2)

def exp_so3(omega):
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp(min=1e-9)
    k = omega / theta
    K = hat(k)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega.shape[:-1] + (3,3))
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    KK = K @ K
    return I + sin_t * K + (1. - cos_t) * KK

def vee(M):
    # M: (...,3,3) skew-symmetric
    return torch.stack([M[..., 2,1] - M[..., 1,2],
                        M[..., 0,2] - M[..., 2,0],
                        M[..., 1,0] - M[..., 0,1]], dim=-1) * 0.5

def log_so3(R, eps: float = 1e-6):
    """
    Batched log map from SO(3) to so(3) (axis-angle vector).
    R: (...,3,3) rotation matrices
    return: (...,3) axis-angle vector omega, with ||omega|| in [0, pi]
    """
    # numerical safety
    trace = (R[..., 0,0] + R[..., 1,1] + R[..., 2,2]).clamp(-1.0, 3.0)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    # generic case: 0 < theta < pi
    K = (R - R.transpose(-1, -2)) * 0.5
    # avoid division by zero
    sin_theta = torch.sin(theta).clamp(min=eps)
    k = vee(K) / sin_theta.unsqueeze(-1)

    # near 0: use first-order approx
    small = (theta < 1e-3)
    if small.any():
        # omega ≈ vee(R - I)
        omega_small = vee(R - torch.eye(3, device=R.device, dtype=R.dtype))
        k = torch.where(small.unsqueeze(-1), F.normalize(omega_small + eps, dim=-1), k)

    # near pi: use alternative from diagonal
    near_pi = (theta > (math.pi - 1e-3))
    if near_pi.any():
        # Find axis from R+I; take the largest diagonal component
        RpI = (R + torch.eye(3, device=R.device, dtype=R.dtype))
        axis = torch.stack([RpI[...,0,0], RpI[...,1,1], RpI[...,2,2]], dim=-1)
        # pick the max component to avoid degeneracy
        _, idx = axis.max(dim=-1, keepdim=True)
        e = torch.zeros_like(axis)
        e.scatter_(-1, idx, 1.0)
        # normalize chosen column of RpI
        v = (RpI @ e.unsqueeze(-1)).squeeze(-1)
        v = F.normalize(v + eps, dim=-1)
        k = torch.where(near_pi.unsqueeze(-1), v, k)

    omega = k * theta.unsqueeze(-1)
    return omega

def _bound_axis_angle(omega_raw, angle_max_rad: float, mode: str):
    """
    omega_raw: (...,3)
    mode: 'so3'  -> bound magnitude only
          'so2'  -> restrict to z-axis planar rotation
    """
    if mode == 'so2':
        # 仅在相机z轴旋转：theta由原预测的z分量给出，并做tanh界定
        theta_raw = omega_raw[..., 2:3]                  # (...,1)
        theta = angle_max_rad * torch.tanh(theta_raw)    # (...,1)
        k = torch.zeros_like(omega_raw)
        k[..., 2] = 1.0                                  # (...,3) , z-axis
        return k * theta                                 # (...,3)
    else:
        # 保持方向，限制模长
        theta_raw = torch.linalg.norm(omega_raw, dim=-1, keepdim=True)  # (...,1)
        theta = angle_max_rad * torch.tanh(theta_raw)                   # (...,1)
        k = omega_raw / (theta_raw + 1e-9)
        return k * theta

class SO3Head(nn.Module):

    def __init__(self, in_dim, hidden=128, angle_max_deg: float = 90.0,
                 mode: str = 'so3', stochastic: bool = True):
        super().__init__()
        
        # 预处理卷积块：在池化前过滤雨痕干扰
        # 雨痕是高频噪声，直接平均会干扰全局运动估计
        # 两层卷积让网络有机会在聚合全局运动前"清洗"特征
        self.pre_process = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.BatchNorm2d(in_dim),  # 运动估计对幅度不敏感，BN有助于收敛
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.GELU()
        )
        
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),  # mu(3) + log_sigma(3)
        )
        self.angle_max_rad = math.radians(angle_max_deg)
        assert mode in ('so3', 'so2')
        self.mode = mode
        self.stochastic = stochastic
        
        # 零初始化最后一层，使初始omega≈0，即R≈I
        self._init_weights()

    def _init_weights(self):
        """零初始化最后一层，使训练初期旋转接近单位阵"""
        # 最后一层是 nn.Linear(hidden, 6)
        last_linear = self.net[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, feat):
        # feat: (B, C, H, W)
        # 先过预处理，过滤雨痕干扰
        feat_refined = self.pre_process(feat)
        
        vec = self.net(feat_refined)        # (B,6)
        mu, log_sigma = vec[:, :3], vec[:, 3:]
        sigma = torch.exp(log_sigma).clamp(1e-6, 10.0)

        if self.stochastic and self.training:
            eps = torch.randn_like(mu)
        else:
            eps = torch.zeros_like(mu)

        omega_raw = mu + sigma * eps        # (B,3)
        omega = _bound_axis_angle(omega_raw, self.angle_max_rad, self.mode)  # (B,3)
        R = exp_so3(omega)                  # (B,3,3)
        return R, mu, log_sigma, omega