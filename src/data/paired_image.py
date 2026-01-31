import os, glob, random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

def _imread(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)

def random_crop_pair(a, b, crop):
    H, W = a.shape[:2]
    if crop is None or crop <= 0 or (H < crop or W < crop):
        return a, b
    y = random.randint(0, H - crop)
    x = random.randint(0, W - crop)
    return a[y:y+crop, x:x+crop], b[y:y+crop, x:x+crop]

def to_tensor(img):
    # HWC uint8 -> CHW float32 in [0,1]
    t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return t

class PairedFolder(Dataset):
    def __init__(self, root, split='train', input_dirname='input', target_dirname='target',
                 crop_size=256):
        self.root = root
        self.split = split
        self.input_dir = os.path.join(root, split, input_dirname)
        self.target_dir = os.path.join(root, split, target_dirname)
        self.crop = crop_size

        self.inputs = sorted(glob.glob(os.path.join(self.input_dir, '*')))
        assert len(self.inputs) > 0, f"No files in {self.input_dir}"
        # Assume matching filenames exist in target_dir
        self.targets = [os.path.join(self.target_dir, os.path.basename(p)) for p in self.inputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        a = _imread(self.inputs[idx])
        b = _imread(self.targets[idx])
        # random horizontal flip for train
        # if self.split == 'train' and random.random() < 0.5:
        #     a = a[:, ::-1].copy()
        #     b = b[:, ::-1].copy()
        a, b = random_crop_pair(a, b, self.crop if self.split=='train' else None)
        a, b = to_tensor(a), to_tensor(b)
        return {'input': a, 'target': b, 'name': os.path.basename(self.inputs[idx])}
