import torch
import math
import torch.nn.functional as F

def psnr(x, y):
    # x,y: (B,C,H,W) in [0,1]
    mse = F.mse_loss(x, y, reduction='mean').item()
    if mse == 0:
        return 99.0
    return 10 * math.log10(1.0 / mse)

# Lightweight SSIM (optional)
# To keep dependencies simple, we provide a naive version (not windowed like official).
def ssim_naive(x, y, eps=1e-6):
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    C1, C2 = 0.01**2, 0.03**2
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2 + eps)
    return (num / den).item()
