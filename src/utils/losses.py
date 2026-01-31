
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTLoss(nn.Module):

    def __init__(self, high_freq_weight: float = 1.5):
        super().__init__()
        self.high_freq_weight = high_freq_weight
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: (B, C, H, W) 预测和目标图像
        """
        # 2D FFT
        fft_x = torch.fft.rfft2(x, norm='backward')
        fft_y = torch.fft.rfft2(y, norm='backward')
        
        # 振幅损失 (主要约束)
        loss_amp = F.l1_loss(fft_x.abs(), fft_y.abs())
        
        # 相位损失 (辅助约束，权重较低)
        loss_pha = F.l1_loss(fft_x.angle(), fft_y.angle())
        
        return loss_amp + 0.3 * loss_pha


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return F.l1_loss(x, y)

class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return F.mse_loss(x, y)

class CharbonnierLoss(nn.Module):
    """Smooth L1: sqrt((x-y)^2 + eps^2)"""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

class EdgeLoss(nn.Module):
    """Gradient (Sobel-like) loss to preserve edges."""
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('gx', gx)
        self.register_buffer('gy', gy)
    def _grad(self, x):
        # 确保卷积核在正确的设备和 dtype 上
        gx_kernel = self.gx.to(x.device, x.dtype).expand(x.size(1), 1, 3, 3)
        gy_kernel = self.gy.to(x.device, x.dtype).expand(x.size(1), 1, 3, 3)
        gx = F.conv2d(x, gx_kernel, padding=1, groups=x.size(1))
        gy = F.conv2d(x, gy_kernel, padding=1, groups=x.size(1))
        return torch.sqrt(gx*gx + gy*gy + 1e-6)
    def forward(self, x, y):
        return F.l1_loss(self._grad(x), self._grad(y))

def ssim_naive(x, y, eps=1e-6):
    mu_x = x.mean([2,3], keepdim=True)
    mu_y = y.mean([2,3], keepdim=True)
    sigma_x = ((x - mu_x)**2).mean([2,3], keepdim=True)
    sigma_y = ((y - mu_y)**2).mean([2,3], keepdim=True)
    sigma_xy = ((x - mu_x)*(y - mu_y)).mean([2,3], keepdim=True)
    C1, C2 = 0.01**2, 0.03**2
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2 + eps)
    ssim = num / den
    return ssim.mean()

class SSIMLossNaive(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return 1.0 - ssim_naive(x, y)

class RestorationLoss(nn.Module):
    """Weighted combo: L1 + L2 + Charbonnier + Edge + SSIM + FFT."""
    def __init__(self, w_l1=1.0, w_l2=0.0, w_charb=0.0, w_edge=0.0, w_ssim=0.0, w_fft=0.0):
        super().__init__()
        self.w_l1 = w_l1
        self.w_l2 = w_l2
        self.w_charb = w_charb
        self.w_edge = w_edge
        self.w_ssim = w_ssim
        self.w_fft = w_fft
        self.l1 = L1Loss()
        self.l2 = L2Loss()
        self.charb = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.ssim = SSIMLossNaive()
        self.fft = FFTLoss()

    def forward(self, pred, target):
        loss = 0.0
        if self.w_l1: loss = loss + self.w_l1 * self.l1(pred, target)
        if self.w_l2: loss = loss + self.w_l2 * self.l2(pred, target)
        if self.w_charb: loss = loss + self.w_charb * self.charb(pred, target)
        if self.w_edge: loss = loss + self.w_edge * self.edge(pred, target)
        if self.w_ssim: loss = loss + self.w_ssim * self.ssim(pred, target)
        if self.w_fft: loss = loss + self.w_fft * self.fft(pred, target)
        return loss
