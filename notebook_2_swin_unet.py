"""
notebook_2_swin_unet.py
─────────────────────────────────────────────────────────────────────────────
MODEL 2 BASELINE — Swin-UNet
Run as a Kaggle Notebook cell block.

Architecture:
    Encoder : timm Swin-Tiny (pretrained ImageNet)
               patch_embed adapted: 3→24 channels via weight averaging
    Decoder : 4 PatchExpand blocks (PixelShuffle) with skip connections
    Output  : Conv2d(64,1,1) → Sigmoid
Expected params: ~28M
VRAM note: batch=8 at 128×128 uses ~11GB on T4
"""

# ── Cell 1: Install and imports ───────────────────────────────────────────────
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install",
                "timm", "wandb", "opencv-python-headless", "scikit-learn", "-q"],
               check=False)

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import timm

sys.path.insert(0, "/kaggle/working")
from trainer import TrainConfig, WildfireLoss, CurriculumScheduler, Trainer
from wildfire_dataset import make_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Cell 2: Swin-UNet architecture ────────────────────────────────────────────

class PatchExpandBlock(nn.Module):
    """
    Patch Expanding: PixelShuffle(2) → Conv2d → LayerNorm.
    Doubles spatial resolution, halves channels.

    PixelShuffle is preferred over transposed convolutions because it avoids
    checkerboard artefacts and is parameter-efficient.

    Args:
        in_ch:   input channels (must be divisible by 4 for PixelShuffle(2))
        skip_ch: skip connection channels (concatenated before conv)
        out_ch:  output channels
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        # PixelShuffle(2) requires in_ch divisible by 4
        # We project first to ensure divisibility
        ps_in = in_ch
        if ps_in % 4 != 0:
            ps_in = (in_ch // 4 + 1) * 4
            self.pre_proj = nn.Conv2d(in_ch, ps_in, 1, bias=False)
        else:
            self.pre_proj = nn.Identity()

        self.pixel_shuffle = nn.PixelShuffle(2)                 # ch: ps_in → ps_in//4
        ps_out = ps_in // 4

        self.conv = nn.Sequential(
            nn.Conv2d(ps_out + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, C, H, W) lower-resolution feature map
            skip: (B, skip_ch, 2H, 2W) encoder skip connection
        """
        x = self.pre_proj(x)
        x = self.pixel_shuffle(x)                               # (B, C//4, 2H, 2W)

        # Align spatial dims if mismatch (edge case for non-power-of-2 sizes)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)                                        # (B, out_ch, 2H, 2W)

        # LayerNorm over channel dim: permute → norm → permute back
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)                              # (B,H,W,C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)                              # (B,C,H,W)
        return x


class SwinUNet(nn.Module):
    """
    Swin-Tiny encoder + 4-stage PixelShuffle decoder.

    Input:  (B, 24, H, W)
    Output: (B, 1,  H, W) fire probability ∈ [0, 1]

    The Swin encoder is loaded with pretrained ImageNet weights.
    patch_embed.proj weights are adapted from 3→24 input channels
    using the same channel-averaging trick as ResNet-UNet.
    """

    IN_CHANNELS: int = 24

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        # ── Load Swin-Tiny with feature extraction ────────────────────────────
        # features_only=True returns intermediate feature maps at each stage
        # out_indices=(0,1,2,3) gives features after each of the 4 Swin stages
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained    = pretrained,
            features_only = True,
            out_indices   = (0, 1, 2, 3),
            img_size      = 128,
            in_chans      = 3,              # adapt after loading
        )

        # ── Adapt patch_embed: 3→24 channels ─────────────────────────────────
        old_proj   = self.encoder.patch_embed.proj
        old_weight = old_proj.weight.data                        # (96, 3, 4, 4)
        new_weight = old_weight.mean(dim=1, keepdim=True).repeat(1, self.IN_CHANNELS, 1, 1)
        new_weight = new_weight * (3.0 / self.IN_CHANNELS)      # preserve output scale

        new_proj = nn.Conv2d(
            self.IN_CHANNELS,
            old_proj.out_channels,
            kernel_size = old_proj.kernel_size,
            stride      = old_proj.stride,
            padding     = old_proj.padding,
            bias        = old_proj.bias is not None,
        )
        new_proj.weight.data = new_weight
        if old_proj.bias is not None:
            new_proj.bias.data = old_proj.bias.data.clone()
        self.encoder.patch_embed.proj = new_proj

        # ── Get encoder output channel sizes ─────────────────────────────────
        # Swin-Tiny: stage outputs are 96, 192, 384, 768 channels
        # Spatial for 128×128: 32×32, 16×16, 8×8, 4×4
        enc_chs = self.encoder.feature_info.channels()           # [96, 192, 384, 768]

        # ── Stem projection: raw input → first skip (before patch embed) ─────
        # We need a skip at full resolution (128×128) — create from raw input
        self.stem_skip = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        # Stage 4→3: 768 → 384, skip=384
        self.dec3 = PatchExpandBlock(enc_chs[3], enc_chs[2], 256)
        # Stage 3→2: 256 → 192, skip=192
        self.dec2 = PatchExpandBlock(256, enc_chs[1], 128)
        # Stage 2→1: 128 → 96, skip=96
        self.dec1 = PatchExpandBlock(128, enc_chs[0], 64)
        # Stage 1→0: 64 → 64, skip=stem (64)
        self.dec0 = PatchExpandBlock(64, 64, 64)

        # ── Output head ───────────────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 24, H, W)
        Returns:
            (B, 1, H, W)
        """
        B, C, H, W = x.shape

        # Stem skip at full resolution
        stem = self.stem_skip(x)                                # (B,64,H,W)

        # Swin encoder: returns list of feature maps
        # Each feature map is (B, H', W', C') — note channel-last from Swin
        feats = self.encoder(x)

        # Convert channel-last → channel-first (Swin returns NHWC)
        e0 = feats[0].permute(0, 3, 1, 2).contiguous()        # (B,96,  H/4, W/4)
        e1 = feats[1].permute(0, 3, 1, 2).contiguous()        # (B,192, H/8, W/8)
        e2 = feats[2].permute(0, 3, 1, 2).contiguous()        # (B,384, H/16,W/16)
        e3 = feats[3].permute(0, 3, 1, 2).contiguous()        # (B,768, H/32,W/32)

        # Decode
        d3 = self.dec3(e3, e2)                                 # (B,256, H/16,W/16)
        d2 = self.dec2(d3, e1)                                 # (B,128, H/8, W/8)
        d1 = self.dec1(d2, e0)                                 # (B,64,  H/4, W/4)
        d0 = self.dec0(d1, stem)                               # (B,64,  H,   W)

        # Final upsample to input resolution if needed
        if d0.shape[2:] != (H, W):
            d0 = F.interpolate(d0, size=(H, W), mode="bilinear", align_corners=False)

        return self.output_head(d0)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Cell 3: Verify model ──────────────────────────────────────────────────────

def verify_model() -> None:
    model = SwinUNet(pretrained=False).to(DEVICE)
    x = torch.randn(2, 24, 128, 128).to(DEVICE)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 128, 128), f"Wrong output shape: {y.shape}"
    assert y.min() >= 0.0 and y.max() <= 1.0
    params = model.count_params()
    print(f"✅ SwinUNet output shape: {y.shape}")
    print(f"   Trainable params: {params/1e6:.2f}M")
    del model, x, y
    torch.cuda.empty_cache()

verify_model()


# ── Cell 4: Dataloaders ───────────────────────────────────────────────────────

HDF5_PATH  = "/kaggle/working/wildfire_data.h5"
BATCH_SIZE = 8    # Swin at 128×128 uses ~11GB — must be 8 not 12

train_dl, val_dl, test_dl = make_dataloaders(
    hdf5_path   = HDF5_PATH,
    batch_size  = BATCH_SIZE,
    resolution  = 128,
    num_workers = 2,
)

batch = next(iter(train_dl))
print(f"Batch shapes — inputs: {batch['inputs'].shape}  targets: {batch['targets'].shape}")


# ── Cell 5: Training setup ────────────────────────────────────────────────────

CKPT_DIR = "/kaggle/working/outputs/checkpoints/swin_unet"
os.makedirs(CKPT_DIR, exist_ok=True)

config = TrainConfig(
    model_name      = "swin_unet",
    resolution      = 128,
    batch_size      = BATCH_SIZE,
    lr              = 1e-4,
    weight_decay    = 1e-2,
    n_epochs        = 50,
    lambda_max      = 0.0,
    lambda_eik      = 0.0,
    lambda_reg      = 0.0,
    alpha_focal     = 0.85,
    gamma_focal     = 3.0,
    checkpoint_dir  = CKPT_DIR,
    mixed_precision = True,
    grad_clip       = 0.5,
    wandb_project   = "pi-vm",
    wandb_run_name  = "swin_unet_baseline",
    save_every      = 5,
)

model     = SwinUNet(pretrained=True).to(DEVICE)
loss_fn   = WildfireLoss(lambda_pde=0.0, lambda_eik=0.0, lambda_reg=0.0)
optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
cosine_sched = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-6)
scheduler    = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[5])

curriculum = CurriculumScheduler(lambda_max=0.0)

trainer = Trainer(
    model          = model,
    loss_fn        = loss_fn,
    optimizer      = optimizer,
    scheduler      = scheduler,
    curriculum     = curriculum,
    config         = config,
    checkpoint_dir = CKPT_DIR,
)
trainer.load_checkpoint()


# ── Cell 6: Train ─────────────────────────────────────────────────────────────

t_start = time.time()
history = trainer.fit(train_dl, val_dl, n_epochs        = 50)
t_total = time.time() - t_start
print(f"\nTotal training time: {t_total/3600:.2f} hours")


# ── Cell 7: Evaluate and save ────────────────────────────────────────────────

test_metrics = trainer.val_epoch(test_dl)

print(f"\n{'='*55}")
print("TEST SET EVALUATION — Swin-UNet")
print(f"{'='*55}")
for k, v in test_metrics.items():
    print(f"  {k:20s}: {float(v):.4f}")

results = {
    "model":        "swin_unet",
    "params_M":     model.count_params() / 1e6,
    "train_time_h": t_total / 3600,
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    "history":      {k: [float(x) for x in v] for k, v in history.items()},
}
with open("/kaggle/working/results_swin_unet.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Results saved to /kaggle/working/results_swin_unet.json")
