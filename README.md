# DeLiVR: Differential Spatiotemporal Lie Bias for Efficient Video Deraining

This repository provides a PyTorch implementation of **DeLiVR**, a Differential Spatiotemporal Lie Bias for Efficient video deraining tasks.


## 🔔 News

**DeLiVR** has been **accepted to ICLR 2026** 🎉  

- Paper: [Arxiv](https://arxiv.org/abs/2509.21719)
- OpenReview discussion: [OpenReview](https://openreview.net/forum?id=W2eNfLmCHY)
- Pretrained Weights: [Google Drive](https://drive.google.com/file/d/14DEAb2Ch0AjOoQeLHVZ365QDhfI4mLz1/view?usp=sharing)

---
## 1. Data Layout

For paired video deraining (rainy → clean), organize your dataset as:

```
/home/exp/dataset/
├── train/
│   ├── derain/                    # NTU dataset
│   │   ├── rainy/                 # input_dirname
│   │   │   ├── seq001/
│   │   │   │   ├── 0001.jpg
│   │   │   │   ├── 0002.jpg
│   │   │   │   └── ...
│   │   │   └── seq002/
│   │   └── gt/                    # target_dirname
│   │       ├── seq001/
│   │       │   ├── 0001.jpg
│   │       │   └── ...
│   │       └── seq002/
│   ├── heavy/                     # Heavy rain dataset
│   │   ├── input/
│   │   └── gt/
│   ├── light/                     # Light rain dataset
│   │   ├── input/
│   │   └── gt/
│   └── weatherbench/              # WeatherBench dataset
│       ├── input/
│       └── target/
│
└── test/
    ├── derain/
    │   ├── rainy/
    │   └── gt/
    ├── heavy/
    │   ├── input/
    │   └── gt/
    ├── light/
    │   ├── input/
    │   └── gt/
    └── weatherbench/
        ├── input/
        └── target/
```

### Supported Datasets

| Dataset | Train Path | Test Path | Input Dir | Target Dir |
|---------|------------|-----------|-----------|------------|
| [NTU (Derain)](https://github.com/hotndy/SPAC-SupplementaryMaterials) | `/home/exp/dataset/train/derain` | `/home/exp/dataset/test/derain` | `rainy` | `gt` |
| [Heavy Rain (RainSynComplex25)](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018) | `/home/exp/dataset/train/heavy` | `/home/exp/dataset/test/heavy` | `input` | `gt` |
| [Light Rain (RainSynLight25)](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018) | `/home/exp/dataset/train/light` | `/home/exp/dataset/test/light` | `input` | `gt` |
| [WeatherBench](https://github.com/guanqiyuan/WeatherBench) | `/home/exp/dataset/train/weatherbench` | `/home/exp/dataset/test/weatherbench` | `input` | `target` |

---

## 2. Installation

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch >= 1.12
- torchvision
- numpy
- opencv-python
- pyyaml
- tensorboard
- tqdm

---

## 3. Configuration

All training parameters are specified in YAML config files under `src/config/`:

```
src/config/
  ├── derain_ntu.yaml         # NTU Derain dataset
  ├── derain_heavy.yaml       # Heavy rain dataset
  ├── derain_light.yaml       # Light rain dataset
  └── derain_weatherbench.yaml # WeatherBench dataset
```

### Key Configuration Parameters

```yaml
# Data
data_root: /path/to/train/data
val_root: /path/to/test/data
input_dirname: rainy          # Input folder name
target_dirname: gt            # Ground truth folder name
T: 5                          # Number of temporal frames
crop_size: 256                # Training crop size

# Model
embed_dim: 128                # Embedding dimension
depth: 12                     # Number of transformer blocks
heads: 8                      # Number of attention heads
patch: 8                      # Patch size

# Lie Group
so3_mode: so2                 # so2 (planar) or so3 (full 3D)
angle_max_deg: 35.0           # Maximum rotation angle (degrees)

# Training
batch_size: 8
epochs: 8000
lr: 0.0002
warmup_epochs: 10
grad_clip: 0.5

# Losses
w_charb: 1.0                  # Charbonnier loss
w_ssim: 0.05                  # SSIM loss
w_fft: 0.05                   # FFT frequency loss
w_edge: 0.05                  # Edge loss
w_lievel: 0.01                # Lie velocity regularization

# EMA
use_ema: 1
ema_decay: 0.999
```

---

## 4. Training

### Single Dataset Training (Multi-GPU DDP)

```bash
cd /home/exp/ICLR-DeLiVR/lie-transformer

# NTU Derain
bash run_ddp.sh 8 --config src/config/derain_ntu.yaml

# Heavy Rain
bash run_ddp.sh 8 --config src/config/derain_heavy.yaml

# Light Rain
bash run_ddp.sh 8 --config src/config/derain_light.yaml

# WeatherBench
bash run_ddp.sh 8 --config src/config/derain_weatherbench.yaml
```

### Resume Training from Checkpoint

```bash
bash run_ddp.sh 8 \
    --config src/config/derain_ntu.yaml \
    --resume /home/exp/lie-transformer/runs_NTU_video/last.pth
```

---

## 5. Testing / Evaluation

```bash
python test_video.py \
    --config src/config/derain_ntu.yaml  \
    --checkpoint /home/exp/lie-transformer/runs_NTU_video/last.pth \
    --data_root /home/exp/dataset/test/derain
```


---

## 6. Project Structure

```
lie-transformer/
├── src/
│   ├── config/
│   │   ├── derain_ntu.yaml
│   │   ├── derain_heavy.yaml
│   │   ├── derain_light.yaml
│   │   └── derain_weatherbench.yaml
│   ├── data/
│   │   ├── video_derain.py         
│   │   └── paired_image.py         
│   ├── models/
│   │   ├── so3.py                 
│   │   ├── lie_attention.py      
│   │   └── lie_transformer_net.py  
│   └── utils/
│       └── metrics.py             
├── train_video5.py                 # Main training script
├── test_video.py                   # Evaluation script
├── run_ddp.sh                      # DDP launch script
├── requirements.txt
└── README.md
```

---


## 7. Citation

If you find this code useful, please consider citing:

```bibtex
@article{sun2025delivr,
  title={DeLiVR: Differential Spatiotemporal Lie Bias for Efficient Video Deraining},
  author={Sun, Shuning and Lu, Jialang and Chen, Xiang and Wang, Jichao and Lu, Dianjie and Zhang, Guijuan and Gao, Guangwei and Zheng, Zhuoran},
  journal={arXiv preprint arXiv:2509.21719},
  year={2025}
}
```

---

## 8. License

This project is released under the MIT License.
