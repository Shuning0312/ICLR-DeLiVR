"""
Test/Inference script for video derain model.
Evaluates on test set and computes PSNR, SSIM metrics.

Usage:
    # 使用配置文件
    python test_video.py --config src/config/derain_ntu.yaml --checkpoint /path/to/last.pth
    
    # 直接指定参数
    python test_video.py \
        --data_root /home/exp/dataset/test/derain \
        --checkpoint /home/exp/lie-transformer-camera/runs_NTU_video/last.pth \
        --input_dirname rainy --target_dirname gt \
        --save_images 1 --output_dir results/test_output
"""

import argparse
import math
import os
from pathlib import Path
from collections import defaultdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np

# Metrics
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from src.data.video_derain import RainVideoWindowDataset
from src.models.lie_transformer_net import LieTransformerNet

# Try to import lpips for perceptual metric
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. LPIPS metric will be skipped.")


# -----------------------
# Config loading
# -----------------------
def load_config(config_path):
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# -----------------------
# Args
# -----------------------
def get_args():
    ap = argparse.ArgumentParser(description='Test video derain model and compute metrics')
    
    # config file
    ap.add_argument('--config', type=str, default='', 
                    help='path to YAML config file (e.g., src/config/derain_ntu.yaml)')
    
    # data
    ap.add_argument('--data_root', type=str, default='',
                    help='test data root (e.g., /home/exp/dataset/test/derain)')
    ap.add_argument('--input_dirname', type=str, default='rainy',
                    help='subfolder name for input frames')
    ap.add_argument('--target_dirname', type=str, default='gt',
                    help='subfolder name for target/GT frames')
    
    # temporal settings
    ap.add_argument('--T', type=int, default=5, help='number of frames in temporal window')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--pad_mode', type=str, default='replicate')
    ap.add_argument('--stack_dim', type=str, default='TC')
    ap.add_argument('--crop_size', type=int, default=0, 
                    help='crop size (0 = no crop, use full resolution)')
    
    # model
    ap.add_argument('--checkpoint', type=str, required=True,
                    help='path to model checkpoint (e.g., last.pth)')
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--depth', type=int, default=12)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--mlp_ratio', type=float, default=4.0)
    ap.add_argument('--patch', type=int, default=8)
    ap.add_argument('--so3_mode', type=str, default='so2')
    ap.add_argument('--angle_max_deg', type=float, default=35.0)
    ap.add_argument('--so3_stochastic', type=int, default=0)
    ap.add_argument('--time_weight', type=float, default=0.5)
    ap.add_argument('--time_kappa', type=float, default=1.0)
    
    # output
    ap.add_argument('--save_images', type=int, default=0,
                    help='1: save output images; 0: only compute metrics')
    ap.add_argument('--output_dir', type=str, default='results/test_output',
                    help='directory to save output images')
    ap.add_argument('--save_comparison', type=int, default=1,
                    help='1: save comparison grid (input|output|gt); 0: save output only')
    
    # performance
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    # TTA (Test-Time Augmentation)
    ap.add_argument('--tta', type=int, default=0,
                    help='1: enable TTA (self-ensemble with flips); 0: disable (faster)')
    ap.add_argument('--tta_mode', type=str, default='flip', choices=['flip', 'flip_rot'],
                    help='TTA mode: flip (h/v flips), flip_rot (flips + 90/180/270 rotations)')
    
    # First parse to get config path
    args, remaining = ap.parse_known_args()
    
    # Load config file if specified
    if args.config:
        config = load_config(args.config)
        # Use val_root or data_root from config as test data if not specified
        if not args.data_root:
            args.data_root = config.get('val_root', config.get('data_root', ''))
        # Set other defaults from config
        for key in ['input_dirname', 'target_dirname', 'T', 'stride', 'pad_mode', 
                    'stack_dim', 'embed_dim', 'depth', 'heads', 'mlp_ratio', 'patch',
                    'so3_mode', 'angle_max_deg', 'so3_stochastic', 'time_weight', 'time_kappa']:
            if key in config and getattr(args, key) == ap.get_default(key):
                setattr(args, key, config[key])
    
    # Re-parse to allow CLI override
    args = ap.parse_args()
    
    # Validate
    if not args.data_root:
        raise ValueError("--data_root is required")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    
    return args


def merge_to_channels(x, stack_dim, T):
    """Convert (B,T,C,H,W) to (B,C*T,H,W) if needed."""
    if stack_dim == 'TC':
        B, T0, C, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W).contiguous()
    return x


def extract_center_input(x_tc_or_ct, stack_dim, center):
    """Extract center frame (B,3,H,W) for visualization."""
    if stack_dim == 'TC':
        return x_tc_or_ct[:, center, :, :, :]
    else:
        return x_tc_or_ct[:, 3 * center:3 * (center + 1), :, :]


def inference_with_tta(model, x, mode='flip'):

    outputs = []
    
    out1, _ = model(x)
    outputs.append(out1)
    
    out2, _ = model(x.flip(-1))  # flip width
    outputs.append(out2.flip(-1))
    
    out3, _ = model(x.flip(-2))  # flip height
    outputs.append(out3.flip(-2))
    
    if mode == 'flip_rot':
        out4, _ = model(x.flip(-1).flip(-2))
        outputs.append(out4.flip(-2).flip(-1))
        
        x_rot90 = torch.rot90(x, k=1, dims=[-2, -1])
        out5, _ = model(x_rot90)
        outputs.append(torch.rot90(out5, k=-1, dims=[-2, -1]))
        
        x_rot90_hflip = x_rot90.flip(-1)
        out6, _ = model(x_rot90_hflip)
        outputs.append(torch.rot90(out6.flip(-1), k=-1, dims=[-2, -1]))
        
        x_rot270 = torch.rot90(x, k=3, dims=[-2, -1])
        out7, _ = model(x_rot270)
        outputs.append(torch.rot90(out7, k=-3, dims=[-2, -1]))
        
        x_rot270_hflip = x_rot270.flip(-1)
        out8, _ = model(x_rot270_hflip)
        outputs.append(torch.rot90(out8.flip(-1), k=-3, dims=[-2, -1]))
    
    out = torch.stack(outputs, dim=0).mean(dim=0)
    return out


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for metric computation."""
    # (C,H,W) or (B,C,H,W) -> (H,W,C) numpy
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def compute_metrics(pred, gt, lpips_fn=None):
    """Compute PSNR, SSIM, and optionally LPIPS."""
    pred_np = tensor_to_numpy(pred)
    gt_np = tensor_to_numpy(gt)
    
    # PSNR
    psnr = compute_psnr(gt_np, pred_np, data_range=255)
    
    # SSIM
    ssim = compute_ssim(gt_np, pred_np, data_range=255, channel_axis=2)
    
    metrics = {'psnr': psnr, 'ssim': ssim}
    
    # LPIPS (if available)
    if lpips_fn is not None:
        with torch.no_grad():
            # LPIPS expects [-1, 1] range
            pred_lpips = pred * 2 - 1
            gt_lpips = gt * 2 - 1
            if pred_lpips.dim() == 3:
                pred_lpips = pred_lpips.unsqueeze(0)
                gt_lpips = gt_lpips.unsqueeze(0)
            lpips_val = lpips_fn(pred_lpips, gt_lpips).item()
            metrics['lpips'] = lpips_val
    
    return metrics


def main():
    args = get_args()
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Output directory
    output_dir = Path(args.output_dir)
    if args.save_images:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f'Output directory: {output_dir}')
    
    # Dataset
    print(f'Loading test data from: {args.data_root}')
    test_set = RainVideoWindowDataset(
        root=args.data_root,
        split='val',  # Use val mode for test
        rainy_dirname=args.input_dirname,
        gt_dirname=args.target_dirname,
        T=args.T,
        stride=args.stride,
        pad_mode=args.pad_mode,
        crop_size=args.crop_size if args.crop_size > 0 else None,
        stack_dim=args.stack_dim,
        val_ratio=1.0  # Use all data
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    print(f'Test set size: {len(test_set)} samples')
    
    print('Loading model...')
    model = LieTransformerNet(
        in_ch=3 * args.T,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        patch=args.patch,
        so3_mode=args.so3_mode,
        angle_max_deg=args.angle_max_deg,
        so3_stochastic=bool(args.so3_stochastic),
        temporal_time_weight=args.time_weight,
        temporal_time_kappa=args.time_kappa,
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel/DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    if 'epoch' in checkpoint:
        print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    print('Model loaded successfully!')
    
    # TTA info
    if getattr(args, 'tta', 0) and args.tta:
        tta_mode = getattr(args, 'tta_mode', 'flip')
        n_aug = 3 if tta_mode == 'flip' else 8
        print(f'TTA enabled: mode={tta_mode}, {n_aug}x self-ensemble')
    
    lpips_fn = None
    if LPIPS_AVAILABLE:
        try:
            lpips_fn = lpips.LPIPS(net='alex').to(device)
            lpips_fn.eval()
            print('LPIPS metric enabled')
        except Exception as e:
            print(f'Warning: Could not initialize LPIPS: {e}')
    
    center = args.T // 2
    
    all_metrics = defaultdict(list)
    
    try:
        from tqdm import tqdm
        loader = tqdm(test_loader, desc='Testing')
    except ImportError:
        loader = test_loader
    
    # Inference
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            x_tc = batch['input'].to(device)
            y = batch['target'].to(device)
            
            # Extract center input for comparison
            rainy_center = extract_center_input(x_tc, args.stack_dim, center)
            
            # Merge to channels
            x = merge_to_channels(x_tc, args.stack_dim, args.T)
            
            _, _, H_orig, W_orig = x.shape
            patch_size = args.patch
            pad_h = (patch_size - H_orig % patch_size) % patch_size
            pad_w = (patch_size - W_orig % patch_size) % patch_size
            
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
            # Forward pass (with optional TTA)
            if getattr(args, 'tta', 0) and args.tta:
                out = inference_with_tta(model, x, mode=getattr(args, 'tta_mode', 'flip'))
            else:
                out, aux = model(x)
            
            if pad_h > 0 or pad_w > 0:
                out = out[:, :, :H_orig, :W_orig]
            
            out_center = out[:, 3 * center:3 * (center + 1), :, :]
            out_center = out_center.clamp(0, 1)
            
            # Compute metrics
            for b in range(out_center.size(0)):
                metrics = compute_metrics(
                    out_center[b:b+1], 
                    y[b:b+1], 
                    lpips_fn
                )
                for k, v in metrics.items():
                    all_metrics[k].append(v)
            
            # Save images
            if args.save_images:
                for b in range(out_center.size(0)):
                    sample_idx = idx * args.batch_size + b
                    
                    if args.save_comparison:
                        # Save comparison grid: input | output | gt
                        from torchvision.utils import make_grid
                        grid = make_grid([
                            rainy_center[b].cpu(),
                            out_center[b].cpu(),
                            y[b].cpu()
                        ], nrow=3, padding=2)
                        save_image(grid, output_dir / f'{sample_idx:06d}_compare.png')
                    else:
                        # Save output only
                        save_image(out_center[b], output_dir / f'{sample_idx:06d}_output.png')
    
    # Compute average metrics
    print('\n' + '=' * 50)
    print('Test Results:')
    print('=' * 50)
    
    results = {}
    for metric_name, values in all_metrics.items():
        avg = np.mean(values)
        std = np.std(values)
        results[metric_name] = {'mean': float(avg), 'std': float(std)}
        print(f'{metric_name.upper():>8}: {avg:.4f} ± {std:.4f}')
    
    print('=' * 50)
    print(f'Total samples: {len(all_metrics["psnr"])}')
    
    # Save results to JSON
    results_file = output_dir / 'metrics.json' if args.save_images else Path('metrics.json')
    results['num_samples'] = len(all_metrics['psnr'])
    results['checkpoint'] = str(args.checkpoint)
    results['data_root'] = str(args.data_root)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {results_file}')
    
    return results


if __name__ == '__main__':
    main()
