"""
Train a model that takes a T-frame rainy stack and predicts the clean center frame.

Highlights:
- Input: (B,T,C,H,W) or (B,3*T,H,W). Model expects (B,3*T,H,W); we merge when needed.
- Supervision: only the center frame (3ch).
- Rotation: bounded SO(3/SO(2)) via so3_head (--so3_mode, --angle_max_deg, --so3_stochastic).
- Logging: TensorBoard scalars + (optional) image quads:
      [Rainy center | Pred (bounded) | Pred (no-rotation) | GT]
- AMP: optional mixed precision (--amp 1).
- Optional validation via --val_root; safe skip if absent.
- Multi-GPU: DistributedDataParallel for balanced memory usage.
"""
from torch.amp import autocast, GradScaler
import argparse
import math
import os
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class ModelEMA:

    def __init__(self, model, decay=0.999, device=None):
        self.module = deepcopy(model.module if hasattr(model, 'module') else model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            model_module = model.module if hasattr(model, 'module') else model
            for ema_v, model_v in zip(self.module.state_dict().values(), 
                                       model_module.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)
        
    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
    
    def state_dict(self):
        return self.module.state_dict()
    
    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)

# TensorBoard (graceful fallback)
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    class SummaryWriter:
        def __init__(self, *a, **k): pass
        def add_scalar(self, *a, **k): pass
        def add_image(self, *a, **k): pass
        def add_histogram(self, *a, **k): pass
        def close(self): pass

from torchvision.utils import make_grid, save_image

from src.data.video_derain import RainVideoWindowDataset
from src.models.lie_transformer_net import LieTransformerNet
from src.utils.losses import RestorationLoss


def load_config(config_path):
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(config, args):
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def get_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('--config', type=str, default='', 
                    help='path to YAML config file (e.g., src/config/derain_ntu.yaml)')
    
    # data
    ap.add_argument('--data_root', type=str, default='', help='train root with rainy/ and gt/')
    ap.add_argument('--val_root', type=str, default='', help='optional val root with rainy/ and gt/')
    ap.add_argument('--T', type=int, default=5)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--pad_mode', type=str, default='replicate', choices=['replicate', 'valid'])
    ap.add_argument('--crop_size', type=int, default=256)
    ap.add_argument('--stack_dim', type=str, default='TC', choices=['TC', 'CT'],
                    help='TC: dataset gives (B,T,C,H,W) then we merge; CT: dataset gives (B,3*T,H,W)')

    # dataset folder names
    ap.add_argument('--input_dirname', type=str, default='rainy',
                    help='subfolder name for input frames (e.g., rainy, hazy, blurry, input)')
    ap.add_argument('--target_dirname', type=str, default='gt',
                    help='subfolder name for target/ground-truth frames (e.g., gt, clean, target)')

    # train
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--embed_dim', type=int, default=96)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--mlp_ratio', type=float, default=4.0)
    ap.add_argument('--patch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--save_dir', type=str, default='runs_video5')

    # losses (weights)
    ap.add_argument('--w_l1', type=float, default=1.0)
    ap.add_argument('--w_l2', type=float, default=0.0)
    ap.add_argument('--w_charb', type=float, default=0.0)
    ap.add_argument('--w_edge', type=float, default=0.0)
    ap.add_argument('--w_ssim', type=float, default=0.0)
    ap.add_argument('--w_fft', type=float, default=0.0,
                    help='weight for FFT frequency domain loss (recommended: 0.05)')
    
    # training strategy
    ap.add_argument('--warmup_epochs', type=int, default=10,
                    help='number of warmup epochs for learning rate scheduler')
    ap.add_argument('--grad_clip', type=float, default=1.0,
                    help='gradient clipping norm (use 0.5 for more stable training)')

    # rotation control
    ap.add_argument('--so3_mode', type=str, default='so2', choices=['so3', 'so2'],
                    help='so2: planar rotation (recommended for rain tilt); so3: full 3D')
    ap.add_argument('--angle_max_deg', type=float, default=45.0,
                    help='upper bound of rotation angle (degrees)')
    ap.add_argument('--so3_stochastic', type=int, default=0,
                    help='1: sample with sigma when training; 0: use mean (deterministic)')

    # rotation regularization
    ap.add_argument('--angle_reg', type=float, default=0.0,
                    help='weight for angle magnitude regularization (radians)')
    
    # temporal time bias
    ap.add_argument('--time_weight', type=float, default=1.0,
                    help='weight for Lie-velocity time bias added to logits')
    ap.add_argument('--time_kappa', type=float, default=1.0,
                    help='temperature to convert ||Δω|| into time bias; smaller=sharper')
    
    # Lie-velocity regularization
    ap.add_argument('--w_lievel', type=float, default=0.02,
                    help='weight for Lie-velocity regularization (small, e.g., 0.02)')
    ap.add_argument('--lievel_smooth', type=float, default=0.5,
                    help='balance between magnitude(1-beta) and smoothness(beta) for Lie-velocity loss')

    # logging / visualization
    ap.add_argument('--log_dir', type=str, default='', help='defaults to save_dir if empty')
    ap.add_argument('--vis_every', type=int, default=1000, help='steps between train image visualizations')
    ap.add_argument('--val_vis_n', type=int, default=4, help='num of val samples visualized per epoch')
    ap.add_argument('--vis_compare_no_rot', type=int, default=1,
                    help='1: add extra visualization with no-rotation (R=I) output')

    # performance
    ap.add_argument('--amp', type=int, default=1, help='1: enable mixed precision')
    ap.add_argument('--gpus', type=str, default='', 
                    help='comma-separated GPU IDs to use (e.g., "0,1,2" or "1,2,3,6,7"). If empty, use all available GPUs.')
    
    # resume training
    ap.add_argument('--resume', type=str, default='', help='path to checkpoint to resume from')
    ap.add_argument('--resume_epoch', type=int, default=None, help='manually specify epoch to resume from (for old format checkpoints)')

    # DDP settings
    ap.add_argument('--use_ddp', type=int, default=1, help='1: use DistributedDataParallel (balanced memory); 0: use DataParallel')
    ap.add_argument('--local_rank', type=int, default=-1, help='local rank for DDP (set by torchrun)')

    # EMA settings
    ap.add_argument('--use_ema', type=int, default=1, help='1: enable EMA for better test performance')
    ap.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate (0.999 recommended)')

    # First parse to get config path
    args, remaining = ap.parse_known_args()
    
    # Load config file if specified
    if args.config:
        config = load_config(args.config)
        # Set defaults from config
        ap.set_defaults(**config)
    
    args = ap.parse_args()
    
    # Validate required fields
    if not args.data_root:
        raise ValueError("--data_root is required (via config file or CLI)")
    
    return args



def aux_dict(aux):
    # DataParallel may wrap as list[dict]
    if isinstance(aux, dict):
        return aux
    if isinstance(aux, (list, tuple)) and len(aux) > 0 and isinstance(aux[0], dict):
        return aux[0]
    return None


def merge_to_channels(x, stack_dim, T):
    # x: (B,T,C,H,W) or (B,3*T,H,W)
    if stack_dim == 'TC':
        B, T0, C, H, W = x.shape
        assert T0 == T, f"T mismatch: got {T0} from data, expected {T}"
        return x.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W).contiguous()
    return x


def extract_center_input(x_tc_or_ct, stack_dim, center):
    # (B,3,H,W) rainy center for visualization
    if stack_dim == 'TC':
        return x_tc_or_ct[:, center, :, :, :]
    else:
        return x_tc_or_ct[:, 3 * center:3 * (center + 1), :, :]


from torchvision.utils import make_grid, save_image

def _to_img3(x: torch.Tensor) -> torch.Tensor:

    if x.dim() == 4:
        x = x[0]
    elif x.dim() == 2:
        x = x.unsqueeze(0)  # 1,H,W

    if x.dim() == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
        # 可能是 HWC
        x = x.permute(2, 0, 1)

    if x.shape[0] == 1:
        x = x.expand(3, -1, -1)

    return x.detach().float().cpu().clamp(0, 1)

def make_and_save_image(writer, log_dir, imgs, tag, step, nrow):
    imgs3 = [_to_img3(im) for im in imgs]               # 确保都是 (3,H,W)
    grid = make_grid(imgs3, nrow=nrow, padding=2)
    writer.add_image(tag, grid, step)
    save_image(grid, (log_dir / 'images' / f'{tag.replace("/", "_")}_{step:08d}.png').as_posix())


@torch.no_grad()
def forward_with_angle_cap(model, x, cap_deg: float):
    """
    Temporarily set so3_head.angle_max_rad = cap_deg (degrees) for a forward pass (R=I if 0).
    Works with DataParallel (access model.module if present).
    """
    m = getattr(model, 'module', model)
    if not hasattr(m, 'so3_head'):
        return model(x)  # graceful fallback
    old = float(getattr(m.so3_head, 'angle_max_rad', 0.0))
    m.so3_head.angle_max_rad = math.radians(cap_deg)
    try:
        out, aux = model(x)
    finally:
        m.so3_head.angle_max_rad = old
    return out, aux

def setup_ddp(args):
    """Initialize DDP environment."""
    # Check if launched with torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif args.local_rank >= 0:
        # Fallback for older torch.distributed.launch
        rank = args.local_rank
        world_size = torch.cuda.device_count()
        local_rank = args.local_rank
    else:
        return None, None, None  # Not DDP mode
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process (rank 0 or non-DDP)."""
    return rank is None or rank == 0


def main():
    args = get_args()
    
    # DDP setup (must be done before other CUDA operations)
    use_ddp = bool(args.use_ddp) and torch.cuda.is_available()
    rank, world_size, local_rank = None, None, None
    
    if use_ddp:
        rank, world_size, local_rank = setup_ddp(args)
        if rank is not None:
            device = torch.device(f'cuda:{local_rank}')
            if rank == 0:
                print(f'Using DistributedDataParallel with {world_size} GPUs')
                print(f'Each GPU has balanced memory usage!')
        else:
            # DDP not initialized (single GPU or no torchrun)
            use_ddp = False
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print('DDP not initialized. Using single GPU or DataParallel.')
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Parse GPU IDs (only for non-DDP mode)
    gpu_ids = None
    if not use_ddp and args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]
        if gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
            if is_main_process(rank):
                print(f'Using GPUs: {gpu_ids} (visible as 0-{len(gpu_ids)-1} in PyTorch)')

    # dirs (only main process creates directories and writer)
    save_dir = Path(args.save_dir or 'runs_video5').expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else save_dir
    
    if is_main_process(rank):
        save_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / 'images').mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir.as_posix())
    else:
        writer = None  # Non-main processes don't log

    # datasets
    train_set = RainVideoWindowDataset(
        root=args.data_root, split='train',
        rainy_dirname=args.input_dirname, gt_dirname=args.target_dirname,
        T=args.T, stride=args.stride, pad_mode=args.pad_mode,
        crop_size=args.crop_size, stack_dim=args.stack_dim
    )
    
    # Use DistributedSampler for DDP
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp and rank is not None else None
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=True,
        persistent_workers=bool(args.workers > 0)
    )

    val_loader = None
    if args.val_root:
        val_set = RainVideoWindowDataset(
            root=args.val_root, split='val',
            rainy_dirname=args.input_dirname, gt_dirname=args.target_dirname,
            T=args.T, stride=args.stride, pad_mode=args.pad_mode,
            crop_size=args.crop_size, stack_dim=args.stack_dim, val_ratio=1.0
        )
        if len(val_set) > 0:
            val_loader = DataLoader(
                val_set, batch_size=1, shuffle=False,
                num_workers=max(1, args.workers // 2), pin_memory=True,
                persistent_workers=bool(args.workers > 0)
            )

    # model
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

    # multi-GPU setup
    if use_ddp and rank is not None:
        # DistributedDataParallel - balanced memory across all GPUs
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if rank == 0:
            print(f'Model wrapped with DistributedDataParallel on GPU {local_rank}')
    elif torch.cuda.device_count() > 1:
        # Fallback to DataParallel (unbalanced memory)
        if gpu_ids and len(gpu_ids) > 1:
            device_ids = list(range(len(gpu_ids)))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            print(f'Using DataParallel on {len(gpu_ids)} GPUs (device_ids={device_ids})')
            print('WARNING: DataParallel has unbalanced memory. Consider using DDP with torchrun.')
        else:
            model = torch.nn.DataParallel(model)
            print(f'Using DataParallel on all {torch.cuda.device_count()} available GPUs')
            print('WARNING: DataParallel has unbalanced memory. Consider using DDP with torchrun.')

    use_ema = bool(getattr(args, 'use_ema', 0)) and is_main_process(rank)
    ema_model = None
    if use_ema:
        ema_decay = getattr(args, 'ema_decay', 0.999)
        ema_model = ModelEMA(model, decay=ema_decay, device=device)
        print(f'EMA enabled with decay={ema_decay}')

    # optimizer / scheduler / losses
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingWarmRestarts
    warmup_epochs = getattr(args, 'warmup_epochs', 10)
    
    T_0 = 500 
    T_mult = 2 
    main_scheduler = CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=T_mult, eta_min=1e-6)
    
    if warmup_epochs > 0:
        # Warmup: 从 0.1*lr 线性增加到 lr
        warmup_iters = warmup_epochs * len(train_loader)
        warmup_scheduler = LinearLR(opt, start_factor=0.1, total_iters=warmup_iters)
        sched = SequentialLR(opt, [warmup_scheduler, main_scheduler], milestones=[warmup_iters])
        if is_main_process(rank):
            print(f'Using LR scheduler: Warmup({warmup_epochs} epochs) + CosineWarmRestarts(T_0={T_0}, T_mult={T_mult})')
    else:
        sched = main_scheduler
        if is_main_process(rank):
            print(f'Using LR scheduler: CosineWarmRestarts(T_0={T_0}, T_mult={T_mult})')
    
    loss_fn = RestorationLoss(
        args.w_l1, args.w_l2, args.w_charb, args.w_edge, args.w_ssim,
        w_fft=getattr(args, 'w_fft', 0.0)
    )
    l1 = nn.L1Loss()
    center = args.T // 2
    global_step = 0

    use_amp = bool(args.amp and torch.cuda.is_available())
    scaler = GradScaler(device='cuda', enabled=use_amp)

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        if Path(args.resume).exists():
            if is_main_process(rank):
                print(f'Resuming from checkpoint: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model'])
            
            # Load optimizer state if available
            if 'optimizer' in checkpoint:
                opt.load_state_dict(checkpoint['optimizer'])
                if is_main_process(rank):
                    print('Loaded optimizer state')
            
            # Load scheduler state if available
            if 'scheduler' in checkpoint:
                sched.load_state_dict(checkpoint['scheduler'])
                if is_main_process(rank):
                    print('Loaded scheduler state')
            
            # Load scaler state if available
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
                if is_main_process(rank):
                    print('Loaded scaler state')
            
            # Load training state
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                if is_main_process(rank):
                    print(f'Resuming from epoch {start_epoch}')
            else:
                # For old format checkpoints, try to infer epoch from filename or use manual specification
                if args.resume_epoch is not None:
                    start_epoch = args.resume_epoch + 1
                    if is_main_process(rank):
                        print(f'Using manually specified epoch: resuming from epoch {start_epoch}')
                else:
                    checkpoint_name = Path(args.resume).name
                    if 'epoch' in checkpoint_name.lower():
                        # Try to extract epoch number from filename like "checkpoint_epoch_100.pth"
                        import re
                        epoch_match = re.search(r'epoch[_-]?(\d+)', checkpoint_name.lower())
                        if epoch_match:
                            start_epoch = int(epoch_match.group(1)) + 1
                            if is_main_process(rank):
                                print(f'Inferred epoch from filename: resuming from epoch {start_epoch}')
                        elif is_main_process(rank):
                            print('Warning: Cannot determine epoch from checkpoint. Starting from epoch 1.')
                    elif is_main_process(rank):
                        print('Warning: Old format checkpoint without epoch info. Starting from epoch 1.')
                        print('Note: Use --resume_epoch N to manually specify the epoch to resume from.')
                        print('Note: Future checkpoints will save epoch information for proper resume.')
            
            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
                if is_main_process(rank):
                    print(f'Resuming from global step {global_step}')
            
            # Load EMA state if available
            if use_ema and ema_model is not None and 'model_ema' in checkpoint:
                ema_model.load_state_dict(checkpoint['model_ema'])
                if is_main_process(rank):
                    print('Loaded EMA state')
            
            if is_main_process(rank):
                print('Checkpoint loaded successfully!')
        elif is_main_process(rank):
            print(f'Warning: Checkpoint file {args.resume} not found. Starting from scratch.')

    # Debug: print head settings (only main process)
    if is_main_process(rank):
        m = getattr(model, 'module', model)
        print('SO3Head mode:', getattr(m.so3_head, 'mode', None))
        try:
            print('SO3Head angle_max_deg:', math.degrees(getattr(m.so3_head, 'angle_max_rad', 0.0)))
        except Exception:
            pass

    # progress wrapper
    try:
        from tqdm import tqdm
        def tqdm_wrap(it, **kw): return tqdm(it, **kw)
    except Exception:
        def tqdm_wrap(it, **kw): return it

    # -------- train epochs --------
    for epoch in range(start_epoch, args.epochs + 1):
        # Set epoch for distributed sampler (important for proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        loop = tqdm_wrap(train_loader, desc=f'Epoch {epoch}/{args.epochs}') if is_main_process(rank) else train_loader

        for batch in loop:
            x_tc = batch['input'].to(device)  # (B,T,C,H,W) or (B,3*T,H,W)
            y = batch['target'].to(device)    # (B,3,H,W)

            rainy_center = extract_center_input(x_tc, args.stack_dim, center)
            x = merge_to_channels(x_tc, args.stack_dim, args.T)  # (B,3*T,H,W)

            # forward + loss (AMP)
            with autocast(device_type='cuda', enabled=use_amp):
            # with torch.cuda.amp.autocast(enabled=use_amp):
                out, aux = model(x)                                # (B,3*T,H,W)
                out_center = out[:, 3 * center:3 * (center + 1), :, :]
                loss = loss_fn(out_center, y)

                # angle regularization
                ad = aux_dict(aux)
                if args.angle_reg > 0 and ad is not None and 'omega' in ad:
                    theta_mean = ad['omega'].norm(dim=1).mean()
                    loss = loss + args.angle_reg * theta_mean

                L_lievel = torch.tensor(0.0, device=device)
                if args.w_lievel > 0.0 and isinstance(ad, dict) and ad.get("lie_vel_seq", None) is not None:
                    v = ad["lie_vel_seq"]                          # (B, T-1)
                    if v is not None:
                        # magnitude penalty: encourage small inter-frame pose change
                        L_mag = v.mean()
                        # smoothness penalty: encourage temporal smoothness of the velocity
                        if v.size(1) > 1:
                            dv = (v[:, 1:] - v[:, :-1]).abs().mean()
                        else:
                            dv = torch.tensor(0.0, device=v.device)
                        beta = float(args.lievel_smooth)
                        L_lievel = (1.0 - beta) * L_mag + beta * dv
                        loss = loss + args.w_lievel * L_lievel

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_clip = getattr(args, 'grad_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            
            if use_ema and ema_model is not None:
                ema_model.update(model)

            # logging scalars (only main process)
            if is_main_process(rank) and writer is not None:
                writer.add_scalar('train/loss', float(loss.item()), global_step)
                writer.add_scalar('train/lr', float(opt.param_groups[0]['lr']), global_step)
                if ad is not None and 'omega' in ad:
                    deg = (ad['omega'].norm(dim=1) * (180.0 / math.pi)).detach()
                    writer.add_scalar('so3/angle_deg_mean', float(deg.mean().item()), global_step)
                    writer.add_scalar('so3/angle_deg_max', float(deg.max().item()), global_step)
                
                # log Lie-velocity loss
                writer.add_scalar('train/loss_lievel', L_lievel.item() if torch.is_tensor(L_lievel) else 0.0, global_step)

                if ad is not None:
                    if 'tau' in ad:
                        writer.add_scalar('model/tau', float(ad['tau']), global_step)
                    if 'res_scale' in ad:
                        writer.add_scalar('model/res_scale', float(ad['res_scale']), global_step)

                # visualization
                if args.vis_every > 0 and (global_step % args.vis_every) == 0:
                    if args.vis_compare_no_rot:
                        out0, _ = forward_with_angle_cap(model, x, 0.0)          # no-rotation
                        out0_center = out0[:, 3 * center:3 * (center + 1), :, :]
                        make_and_save_image(writer, log_dir,
                                            [rainy_center, out_center, out0_center, y],
                                            'train/vis_compare', global_step, nrow=4)
                    else:
                        make_and_save_image(writer, log_dir,
                                            [rainy_center, out_center, y],
                                            'train/vis_triplet', global_step, nrow=3)

            global_step += 1
            if hasattr(loop, 'set_postfix'):
                loop.set_postfix(loss=f'{loss.item():.4f}')

        # ----- validate (only main process) -----
        if is_main_process(rank):
            if val_loader is None:
                print('Val: skipped (no val_root provided)')
            elif (epoch + 1) % 50 == 0:  # 每50个epoch验证一次
                eval_model = ema_model.module if (use_ema and ema_model is not None) else model
                eval_model.eval()
                
                s, n = 0.0, 0
                vis_count = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x_tc = batch['input'].to(device)
                        y = batch['target'].to(device)

                        rainy_center = extract_center_input(x_tc, args.stack_dim, center)
                        x = merge_to_channels(x_tc, args.stack_dim, args.T)

                        out, _ = eval_model(x)
                        out_center = out[:, 3 * center:3 * (center + 1), :, :]

                        s += l1(out_center.clamp(0, 1), y).item()
                        n += 1

                        if vis_count < max(0, args.val_vis_n) and writer is not None:
                            if args.vis_compare_no_rot:
                                out0, _ = forward_with_angle_cap(eval_model, x, 0.0)
                                out0_center = out0[:, 3 * center:3 * (center + 1), :, :]
                                make_and_save_image(writer, log_dir,
                                                    [rainy_center, out_center, out0_center, y],
                                                    'val/vis_compare', global_step + vis_count, nrow=4)
                            else:
                                make_and_save_image(writer, log_dir,
                                                    [rainy_center, out_center, y],
                                                    'val/vis_triplet', global_step + vis_count, nrow=3)
                            vis_count += 1

                val_l1 = s / max(1, n)
                if writer is not None:
                    writer.add_scalar('val/l1', float(val_l1), global_step)
                ema_tag = ' (EMA)' if (use_ema and ema_model is not None) else ''
                print(f'Val L1{ema_tag}: {val_l1:.4f}')
            else:
                print(f'Val: skipped (epoch {epoch + 1}, next validation at epoch {((epoch + 1) // 50 + 1) * 50})')

        # sched & checkpoint (all processes update scheduler, but only main saves)
        sched.step()
        
        if is_main_process(rank):
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'args': vars(args)
            }
            if use_ema and ema_model is not None:
                checkpoint['model_ema'] = ema_model.state_dict()
            torch.save(checkpoint, save_dir / 'last.pth')
        
        # Synchronize all processes at end of epoch
        if use_ddp and rank is not None:
            dist.barrier()

    if is_main_process(rank) and writer is not None:
        writer.close()
    
    # Clean up DDP
    cleanup_ddp()


if __name__ == '__main__':
    main()