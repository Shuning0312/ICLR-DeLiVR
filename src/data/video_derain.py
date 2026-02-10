
import os, glob, random
from typing import List, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

import os, glob, random
from typing import List, Optional
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import random

def _rotate_small_np(img: np.ndarray, angle_deg: float) -> np.ndarray:
    H, W = img.shape[:2]
    diag = int(((H**2 + W**2) ** 0.5) + 0.5)
    pad_h = max(0, (diag - H) // 2 + 2)
    pad_w = max(0, (diag - W) // 2 + 2)
    if pad_h or pad_w:
        img_pad = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
    else:
        img_pad = img
    pil = Image.fromarray(img_pad)
    rot = pil.rotate(angle_deg, resample=Image.BILINEAR, expand=False)
    rot = np.array(rot)
    y0, x0 = pad_h, pad_w
    return rot[y0:y0+H, x0:x0+W, :]



IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

def is_image_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS

def list_images(folder: str):
    return sorted([p for p in glob.glob(os.path.join(folder, '*')) if is_image_file(p)])

def _read_img(path):
    try:
        with Image.open(path) as im:
            return np.array(im.convert('RGB'))
    except Exception as e:
        print(f"Warning: Failed to read image: {path}")
        print(f"File exists: {os.path.exists(path)}")
        print(f"File size: {os.path.getsize(path) if os.path.exists(path) else 'N/A'} bytes")
        print(f"Error: {e}")
        print("Using fallback black image...")

        return np.zeros((256, 256, 3), dtype=np.uint8)

# def _read_img(path):
#     return np.array(Image.open(path).convert('RGB'))

def _resize_keep_ratio_min_side(img: np.ndarray, min_side: int) -> np.ndarray:
    H, W = img.shape[:2]
    short_side = min(H, W)
    if short_side >= min_side:
        return img
    scale = float(min_side) / float(short_side)
    new_H = max(1, int(round(H * scale)))
    new_W = max(1, int(round(W * scale)))
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_W, new_H), Image.BILINEAR)
    return np.array(pil_img)

def _pad_to_at_least(img: np.ndarray, target_h: int, target_w: int, mode: str = 'reflect') -> np.ndarray:
    H, W = img.shape[:2]
    pad_top = max(0, (target_h - H) // 2)
    pad_bottom = max(0, target_h - H - pad_top)
    pad_left = max(0, (target_w - W) // 2)
    pad_right = max(0, target_w - W - pad_left)
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return img
    return np.pad(img,
                  ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                  mode=mode)

def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    H, W = img.shape[:2]
    if H == size and W == size:
        return img
    y = max(0, (H - size) // 2)
    x = max(0, (W - size) // 2)
    return img[y:y+size, x:x+size]

def _random_crop(img: np.ndarray, size: int) -> np.ndarray:
    H, W = img.shape[:2]
    if H == size and W == size:
        return img
    y = 0 if H == size else random.randint(0, H - size)
    x = 0 if W == size else random.randint(0, W - size)
    return img[y:y+size, x:x+size]

def _to_tensor(img: np.ndarray) -> torch.Tensor:
    # HWC uint8 -> CHW float32 [0,1]
    return torch.from_numpy(img).permute(2,0,1).float() / 255.0

def _random_crop_pair_stack(frames, target, crop: Optional[int], is_train: bool):
    if crop is None or crop <= 0:
        return frames, target

    frames = [_resize_keep_ratio_min_side(f, crop) for f in frames]
    target = _resize_keep_ratio_min_side(target, crop)

    frames = [_pad_to_at_least(f, crop, crop, mode='reflect') for f in frames]
    target = _pad_to_at_least(target, crop, crop, mode='reflect')

    if is_train:

        H, W = target.shape[:2]
        y = 0 if H == crop else random.randint(0, H - crop)
        x = 0 if W == crop else random.randint(0, W - crop)
        frames = [f[y:y+crop, x:x+crop] for f in frames]
        target = target[y:y+crop, x:x+crop]
    else:

        frames = [_center_crop(f, crop) for f in frames]
        target = _center_crop(target, crop)
    return frames, target


def _hflip(img): 
    return img[:, ::-1].copy()

def _augment(frames, target, rot_prob: float = 0.9, deg_max: float = 8.0, hflip_prob: float = 0.5):

    if random.random() < hflip_prob:
        frames = [_hflip(f) for f in frames]
        target = _hflip(target)
    
    if deg_max > 0 and random.random() < rot_prob:
        angle = random.uniform(-deg_max, deg_max)
        frames = [_rotate_small_np(f, angle) for f in frames]
        target = _rotate_small_np(target, angle)
    
    return frames, target

def _generate_windows(n: int, T: int, stride: int, pad_mode: str):
    half = T // 2
    idxs = []
    if pad_mode == 'replicate':
        for c in range(n):
            left = max(0, c - half)
            right = min(n - 1, c + half)
            window = list(range(left, right + 1))
            while len(window) < T:
                if window[0] > 0:
                    window = [window[0]-1] + window
                else:
                    window = window + [window[-1]]
            while len(window) > T:
                window.pop()
            if (c % stride) == 0:
                idxs.append(window)
    elif pad_mode == 'valid':
        for c in range(half, n - half, stride):
            idxs.append(list(range(c - half, c + half + 1)))
    else:
        raise ValueError("pad_mode must be 'replicate' or 'valid'")
    return idxs

class RainVideoWindowDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 rainy_dirname: str = 'rainy',
                 gt_dirname: str = 'gt',
                 T: int = 5,
                 stride: int = 1,
                 pad_mode: str = 'replicate',  # 'replicate' or 'valid'
                 crop_size: Optional[int] = 256,
                 stack_dim: str = 'TC',  # 'TC' -> (T,C,H,W), 'CT' -> (C*T,H,W)
                 list_file: Optional[str] = None,
                 val_ratio: float = 0.1,
                 seed: int = 42):
        super().__init__()
        assert T % 2 == 1, "T should be odd, e.g., 5"
        assert stack_dim in ('TC', 'CT')
        self.root = root
        self.T = T
        self.half = T // 2
        self.stride = stride
        self.pad_mode = pad_mode
        self.crop = crop_size
        self.stack_dim = stack_dim
        self.split = split

        rainy_root = os.path.join(root, rainy_dirname)
        gt_root = os.path.join(root, gt_dirname)
        assert os.path.isdir(rainy_root) and os.path.isdir(gt_root), "rainy/gt directory not found."

        subdirs = sorted([d for d in os.listdir(rainy_root) if os.path.isdir(os.path.join(rainy_root, d))])
        if len(subdirs) > 0:
            seqs = subdirs
        else:
            seqs = ['all_images']
        assert len(seqs) > 0, f"No sequences in {rainy_root}"

        if list_file is not None and os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                listed = [x.strip() for x in f if x.strip()]
            if split == 'train':
                seqs = [s for s in listed if s in seqs]
            else:
                seqs = [s for s in listed if s in seqs]
        else:
            random.Random(seed).shuffle(seqs)
            n_val = max(1, int(len(seqs) * val_ratio))
            if split == 'train':
                seqs = seqs[n_val:] if n_val < len(seqs) else seqs
            else:
                seqs = seqs[:n_val]

        self.index = []
        for s in seqs:
            if s == 'all_images':
                rdir = rainy_root
                gdir = gt_root
            else:
                rdir = os.path.join(rainy_root, s)
                gdir = os.path.join(gt_root, s)
            

            if not os.path.isdir(rdir):
                continue
            if not os.path.isdir(gdir):
                continue

            rpaths = list_images(rdir)                                  
            gpaths_list = list_images(gdir)                            
            gpaths = {os.path.basename(p): p for p in gpaths_list} 
            n = len(rpaths)
            if n == 0:
                continue
            windows = _generate_windows(n, T, stride, pad_mode)
            
            for w in windows:
                names = [os.path.basename(rpaths[i]) for i in w]
                center_name = names[self.half]
                
                if center_name.startswith('rfc-'):
                    gt_name = center_name.replace('rfc-', 'gtc-')
                else:
                    gt_name = center_name
                
                if gt_name not in gpaths:
                    continue
                self.index.append({
                    'seq': s,
                    'rframes': [rpaths[i] for i in w],
                    'gframe': gpaths[gt_name],
                    'center_name': center_name,
                })

        assert len(self.index) > 0, "No windows assembled. Check folder names and filenames."

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        rec = self.index[i]
        rimgs = [_read_img(p) for p in rec['rframes']]
        gt = _read_img(rec['gframe'])

        if self.split == 'train':
            rimgs, gt = _random_crop_pair_stack(rimgs, gt, self.crop, is_train=True)
            rimgs, gt = _augment(rimgs, gt)
        else:
            if self.crop:
                rimgs, gt = _random_crop_pair_stack(rimgs, gt, self.crop, is_train=False)
            else:
                if len(rimgs) > 0:
                    min_h = min([img.shape[0] for img in rimgs] + [gt.shape[0]])
                    min_w = min([img.shape[1] for img in rimgs] + [gt.shape[1]])
                    size = min(min_h, min_w)
                    rimgs = [_center_crop(_pad_to_at_least(img, size, size), size) for img in rimgs]
                    gt = _center_crop(_pad_to_at_least(gt, size, size), size)

        r_t = torch.stack([_to_tensor(im) for im in rimgs], dim=0)  # (T,C,H,W)
        gt_t = _to_tensor(gt)  # (C,H,W)

        if self.stack_dim == 'CT':
            T, C, H, W = r_t.shape
            r_t = r_t.permute(1,0,2,3).reshape(C*T, H, W)

        return {
            'input': r_t,
            'target': gt_t,
            'seq': rec['seq'],
            'name': rec['center_name'],
        }