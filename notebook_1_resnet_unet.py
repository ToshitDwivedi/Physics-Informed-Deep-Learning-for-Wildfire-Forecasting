"""
notebook_1_resnet_unet.py
─────────────────────────────────────────────────────────────────────────────
MODEL 1 BASELINE — ResNet-50 UNet
Run as a Kaggle Notebook cell block.

Architecture:
    Encoder : torchvision ResNet-50 (pretrained ImageNet)
               First conv adapted: 3→24 channels via weight averaging
    Decoder : 4 bilinear-upsample blocks with skip connections
    Output  : Conv2d(64,1,1) → Sigmoid
Expected params: ~25M
Expected T4 training time: ~45 min / 100 epochs at batch=12, 128×128
"""

# ── Cell 1: Install and imports ───────────────────────────────────────────────
import subprocess, sys

# All packages pre-installed on Kaggle except wandb and opencv
subprocess.run([sys.executable, "-m", "pip", "install",
                "wandb", "opencv-python-headless", "scikit-learn", "-q"],
               check=False)

import os, time, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, "/kaggle/working")
from trainer import TrainConfig, WildfireLoss, WildfireMetrics, CurriculumScheduler, Trainer
from wildfire_dataset import make_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Cell 2: ResNet-UNet architecture ─────────────────────────────────────────

class DoubleConv(nn.Module):
    """Conv2d→BN→ReLU→Conv2d→BN→ReLU block used in decoder."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """
    Bilinear upsample + skip connection concatenation + DoubleConv.
    Uses bilinear interpolation — avoids checkerboard artefacts from
    transposed convolutions.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    upsampled feature map
            skip: encoder skip connection feature map
        """
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """
    ResNet-50 encoder + 4-stage bilinear-upsample decoder.

    Input:  (B, 24, H, W)  — 24-channel physics features from pipeline
    Output: (B, 1,  H, W)  — fire probability ∈ [0, 1]

    Channel adaptation: ImageNet pretrained weights (3-channel) are averaged
    across the channel dimension and tiled to 24 channels, preserving the
    learned low-level feature structure.
    """

    IN_CHANNELS: int = 24

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        # ── Load pretrained ResNet-50 backbone ───────────────────────────────
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        # ── Adapt first conv: 3→24 channels ──────────────────────────────────
        old_conv   = backbone.conv1
        old_weight = old_conv.weight.data                    # (64, 3, 7, 7)
        # Average across input channels → (64, 1, 7, 7), then repeat 24 times
        new_weight = old_weight.mean(dim=1, keepdim=True).repeat(1, self.IN_CHANNELS, 1, 1)
        # Scale so the initial output variance matches the 3-channel case
        new_weight = new_weight * (3.0 / self.IN_CHANNELS)

        new_conv = nn.Conv2d(
            self.IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        new_conv.weight.data = new_weight
        backbone.conv1 = new_conv

        # ── Extract encoder stages ────────────────────────────────────────────
        # Stage outputs at 64, 128, 256, 512 channels
        # Spatial: 128→64→32→16→8 (for 128×128 input)
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                   backbone.relu, backbone.maxpool)  # (B,64,32,32)
        self.enc1 = backbone.layer1   # (B, 256, 32, 32)
        self.enc2 = backbone.layer2   # (B, 512, 16, 16)
        self.enc3 = backbone.layer3   # (B,1024,  8,  8)
        self.enc4 = backbone.layer4   # (B,2048,  4,  4)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(2048, 1024)

        # ── Decoder: 4 upsample blocks ────────────────────────────────────────
        # Each block: upsample → concat skip → DoubleConv
        self.dec4 = UpsampleBlock(1024, 1024, 512)   # 4→8
        self.dec3 = UpsampleBlock(512,  512,  256)   # 8→16
        self.dec2 = UpsampleBlock(256,  256,  128)   # 16→32
        self.dec1 = UpsampleBlock(128,  64,   64)    # 32→64

        # ── Final upsample to input resolution + output head ─────────────────
        self.output_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 24, H, W) physics-processed input
        Returns:
            (B, 1, H, W) fire probability
        """
        # Encoder
        e0 = self.enc0(x)    # (B, 64,   H/4,  W/4)
        e1 = self.enc1(e0)   # (B, 256,  H/4,  W/4)
        e2 = self.enc2(e1)   # (B, 512,  H/8,  W/8)
        e3 = self.enc3(e2)   # (B, 1024, H/16, W/16)
        e4 = self.enc4(e3)   # (B, 2048, H/32, W/32)

        # Bottleneck
        b  = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(b,  e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)

        # Upsample back to input resolution
        d0 = F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.output_head(d0)

    def count_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Cell 3: Verify model ──────────────────────────────────────────────────────

def verify_model() -> None:
    """Quick shape and parameter check before training."""
    model = ResNetUNet(pretrained=False).to(DEVICE)
    x     = torch.randn(2, 24, 128, 128).to(DEVICE)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 128, 128), f"Wrong output shape: {y.shape}"
    assert y.min() >= 0.0 and y.max() <= 1.0, "Output not in [0,1]"
    params = model.count_params()
    print(f"✅ ResNetUNet output shape: {y.shape}")
    print(f"   Trainable params: {params/1e6:.2f}M")
    del model, x, y
    torch.cuda.empty_cache()

verify_model()


# ── Cell 4: Build dataloaders ─────────────────────────────────────────────────
# ── Rebuild HDF5 if missing ───────────────────────────────────────────────────
import h5py, tensorflow as tf
from tqdm.auto import tqdm

HDF5_PATH    = "/kaggle/working/wildfire_data.h5"
KAGGLE_INPUT = "/kaggle/input/datasets/fantineh/next-day-wildfire-spread"

if os.path.exists(HDF5_PATH):
    print(f"✅ HDF5 exists ({os.path.getsize(HDF5_PATH)/1e9:.2f} GB)")
else:
    print("Rebuilding HDF5...")
    CHANNEL_NAMES = ["elevation","th","vs","tmmn","tmmx","sph",
                     "pr","pdsi","NDVI","population","erc","PrevFireMask"]
    TARGET = "FireMask"
    PATCH_SIZE = 64
    FEATURE_DESC = {
        n: tf.io.FixedLenFeature([PATCH_SIZE, PATCH_SIZE], tf.float32)
        for n in CHANNEL_NAMES + [TARGET]
    }
    def parse_ex(s):
        p = tf.io.parse_single_example(s, FEATURE_DESC)
        return tf.stack([p[c] for c in CHANNEL_NAMES], axis=-1), p[TARGET]

    all_files = sorted([os.path.join(KAGGLE_INPUT, f)
                        for f in os.listdir(KAGGLE_INPUT) if "tfrecord" in f])
    splits = {s: [f for f in all_files if f"_{s}_" in os.path.basename(f)]
              for s in ("train","eval","test")}

    with h5py.File(HDF5_PATH, "w") as hf:
        for split, files in splits.items():
            with open(files[0],"rb") as fh:
                comp = "GZIP" if fh.read(2)==b"\x1f\x8b" else ""
            grp = hf.require_group(split)
            di  = grp.create_dataset("inputs",  shape=(0,64,64,12),
                      maxshape=(None,64,64,12), dtype="float32",
                      chunks=(32,64,64,12), compression="gzip", compression_opts=4)
            dt  = grp.create_dataset("targets", shape=(0,64,64),
                      maxshape=(None,64,64), dtype="float32",
                      chunks=(32,64,64), compression="gzip", compression_opts=4)
            ds  = (tf.data.TFRecordDataset(files, compression_type=comp)
                   .map(parse_ex, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(128).prefetch(2))
            idx = 0
            for ib, tb in tqdm(ds, desc=f"  {split}"):
                n = ib.shape[0]
                di.resize(idx+n, axis=0); dt.resize(idx+n, axis=0)
                di[idx:idx+n] = ib.numpy(); dt[idx:idx+n] = tb.numpy()
                idx += n
            print(f"  {split}: {idx:,} samples")
    print(f"✅ HDF5 saved: {os.path.getsize(HDF5_PATH)/1e9:.2f} GB")
assert os.path.exists(HDF5_PATH), f"HDF5 not found: {HDF5_PATH}"

BATCH_SIZE = 12   # safe for ResNet-50 at 128×128 on T4

train_dl, val_dl, test_dl = make_dataloaders(
    hdf5_path   = HDF5_PATH,
    batch_size  = BATCH_SIZE,
    resolution  = 128,
    num_workers = 2,
)

# Smoke test one batch
batch = next(iter(train_dl))
print(f"\nBatch shapes:")
print(f"  inputs   : {batch['inputs'].shape}")
print(f"  targets  : {batch['targets'].shape}")
print(f"  prev_fire: {batch['prev_fire'].shape}")


# ── Cell 5: Training setup ────────────────────────────────────────────────────

CKPT_DIR = "/kaggle/working/outputs/checkpoints/resnet_unet"
os.makedirs(CKPT_DIR, exist_ok=True)

config = TrainConfig(
    model_name      = "resnet_unet",
    resolution      = 128,
    batch_size      = BATCH_SIZE,
    lr              = 1e-4,
    weight_decay    = 1e-2,
    n_epochs        = 50,
    lambda_max      = 1.0,
    lambda_eik      = 0.0,
    lambda_reg      = 0.0,
    alpha_focal     = 0.85,
    gamma_focal     = 3.0,
    checkpoint_dir  = CKPT_DIR,
    mixed_precision = True,
    grad_clip       = 0.5,
    wandb_project   = "pi-vm",
    wandb_run_name  = "resnet_unet_baseline",
    save_every      = 5,
)

model = ResNetUNet(pretrained=True).to(DEVICE)

# ── Freeze encoder — train decoder only ──────────────────────────────────────
for name, param in model.named_parameters():
    if name.startswith(("enc", "bottleneck")):
        param.requires_grad_(True)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params (decoder only): {trainable/1e6:.2f}M")

loss_fn   = WildfireLoss(lambda_pde=0.0, lambda_eik=0.0, lambda_reg=0.0)
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config.lr, weight_decay=config.weight_decay
)

# 5-epoch linear warmup → cosine annealing
warmup_sched  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
cosine_sched  = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-6)
scheduler     = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[5])

curriculum    = CurriculumScheduler(lambda_max=0.0)

trainer = Trainer(
    model          = model,
    loss_fn        = loss_fn,
    optimizer      = optimizer,
    scheduler      = scheduler,
    curriculum     = curriculum,
    config         = config,
    checkpoint_dir = CKPT_DIR,
)

# Auto-resume if checkpoint exists
trainer.load_checkpoint()


# ── Cell 6: Train ─────────────────────────────────────────────────────────────

t_start  = time.time()
history  = trainer.fit(train_dl, val_dl, n_epochs        = 50)
t_total  = time.time() - t_start

print(f"\nTotal training time: {t_total/3600:.2f} hours")


# ── Cell 7: Evaluate on test set ─────────────────────────────────────────────

def evaluate_on_test(trainer: Trainer, test_dl, model_name: str) -> dict:
    """Run test set evaluation and print formatted results."""
    print(f"\n{'='*55}")
    print(f"TEST SET EVALUATION — {model_name}")
    print(f"{'='*55}")

    metrics = trainer.val_epoch(test_dl)

    rows = [
        ("CSI",          metrics.get("CSI",          float("nan"))),
        ("IoU",          metrics.get("IoU",          float("nan"))),
        ("Precision",    metrics.get("Precision",    float("nan"))),
        ("Recall",       metrics.get("Recall",       float("nan"))),
        ("F1",           metrics.get("F1",           float("nan"))),
        ("PR-AUC",       metrics.get("PR_AUC",       float("nan"))),
        ("PCR",          metrics.get("PCR",          float("nan"))),
        ("ECE",          metrics.get("ECE",          float("nan"))),
        ("Fréchet dist", metrics.get("Frechet_dist", float("nan"))),
    ]
    for name, val in rows:
        print(f"  {name:20s}: {val:.4f}")

    params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\n  Params         : {params/1e6:.2f}M")
    print(f"  Train time     : {t_total/3600:.2f}h")
    print(f"{'='*55}")
    return metrics

test_metrics = evaluate_on_test(trainer, test_dl, "ResNet-UNet")

# Save test metrics to disk for the comparison table
import json
results = {
    "model":       "resnet_unet",
    "params_M":    sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    "train_time_h": t_total / 3600,
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    "history":     {k: [float(x) for x in v] for k, v in history.items()},
}
with open("/kaggle/working/results_resnet_unet.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✅ Results saved to /kaggle/working/results_resnet_unet.json")


# ── Cell 8: Learning curve plot ───────────────────────────────────────────────

def plot_learning_curve(history: dict, model_name: str, save_path: str) -> None:
    """Plot CSI and loss vs epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Training Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    val_csi = [x for x in history["val_csi"] if not (isinstance(x, float) and x != x)]
    if val_csi:
        ax2.plot(range(1, len(val_csi)+1), val_csi,
                 label="Val CSI", color="darkorange")
        ax2.axhline(max(val_csi), linestyle="--", color="red",
                    alpha=0.5, label=f"Best={max(val_csi):.4f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("CSI")
    ax2.set_title(f"{model_name} — Validation CSI")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {save_path}")

plot_learning_curve(
    history,
    model_name = "ResNet-UNet",
    save_path  = "/kaggle/working/curve_resnet_unet.png",
)
