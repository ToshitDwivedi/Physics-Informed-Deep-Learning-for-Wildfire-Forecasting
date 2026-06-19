"""
notebook_3_vm_unet.py
─────────────────────────────────────────────────────────────────────────────
MODEL 3 BASELINE — VM-UNet (standard VMamba, NO physics gating)
Run as a Kaggle Notebook cell block.

This is the DIRECT baseline that PI-VMUNet will be compared against.
The only difference between VM-UNet and PI-VMUNet is:
  VM-UNet : Δ = Softplus(Linear(x))           ← standard SSM
  PI-VMUNet: Δ = Softplus(Linear(x)) * gate   ← PGSS with Rothermel gate

Architecture:
    Encoder : 4 stages of [VSSBlock + PatchMerge]
    Decoder : 4 stages of [PatchExpand + VSSBlock + skip]
    Output  : Conv2d(64,1,1) → Sigmoid
Expected params: ~32M
VRAM note: batch=8 at 128×128
"""

# ── Cell 1: Install and imports ───────────────────────────────────────────────
import subprocess, sys

# mamba-ssm requires CUDA — must run on GPU notebook
subprocess.run([sys.executable, "-m", "pip", "install",
                "wandb", "opencv-python-headless", "scikit-learn",
                "einops", "-q"], check=False)
import os as _os
_os.system("pip install causal-conv1d --no-build-isolation -q 2>&1 | tail -2")
_os.system("rm -rf /tmp/mamba_build && cp -r /kaggle/input/datasets/mbrosseau/mamba-ssm/mamba-main/mamba-main /tmp/mamba_build && pip install /tmp/mamba_build --no-build-isolation -q 2>&1 | tail -3")
_os.system("pip install transformers==4.44.2 -q 2>&1 | tail -2")
for _m in list(sys.modules.keys()):
    if 'mamba' in _m or 'transformers' in _m: del sys.modules[_m]

import os, time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from einops import rearrange

sys.path.insert(0, "/kaggle/working")
from trainer import TrainConfig, WildfireLoss, CurriculumScheduler, Trainer
from wildfire_dataset import make_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    assert torch.cuda.is_available(), "mamba-ssm requires CUDA — enable GPU in notebook settings"


# ── Cell 2: SS1D selective scan (mamba-ssm CUDA kernel) ─────────────────────
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _MAMBA_AVAILABLE = True
    print("✅ mamba-ssm CUDA kernel available")
except ImportError as exc:
    raise RuntimeError("mamba-ssm CUDA kernel is required but not available.") from exc


# ── Cell 3: SS2D — 4-direction spatial selective scan ────────────────────────

class SS2D(nn.Module):
    """
    2D Selective State Space scan with 4 directions:
        Direction 0: top-left  → bottom-right (row major)
        Direction 1: bottom-right → top-left  (reverse)
        Direction 2: top-right → bottom-left  (transposed)
        Direction 3: bottom-left → top-right  (transposed reverse)

    Merge strategy: element-wise sum of all 4 direction outputs.
    This is the standard VMamba SS2D formulation (Liu et al. 2024).

    Args:
        d_model:  input/output channels
        d_state:  SSM state dimension (default 16)
        d_inner:  inner projection dimension (default d_model * 2)
        dt_rank:  rank of Δ projection (default ceil(d_model/16))
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_inner: int = None,
        dt_rank: int = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner or d_model * 2
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.K       = 4    # number of scan directions

        # Input projections
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # SSM parameters — one set per direction
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            for _ in range(self.K)
        ])
        self.dt_proj = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True)
            for _ in range(self.K)
        ])

        # A: log-parameterised state matrix, shape (d_inner, d_state)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(self.d_inner, 1)
        self.A_log = nn.ParameterList([
            nn.Parameter(torch.log(A.clone()))
            for _ in range(self.K)
        ])

        # D: skip connection
        self.D = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner))
            for _ in range(self.K)
        ])

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.out_norm = nn.LayerNorm(d_model)

    def _scan_direction(
        self,
        x_flat: torch.Tensor,
        direction: int,
    ) -> torch.Tensor:
        """
        Run SSM scan on a flattened sequence for one direction.

        Args:
            x_flat: (B, L, d_inner) — sequence to scan
            direction: 0–3

        Returns:
            y_flat: (B, L, d_inner)
        """
        # Project to Δ, B, C
        xBC = self.x_proj[direction](x_flat)                   # (B,L,dt_rank+2*d_state)
        dt  = xBC[..., :self.dt_rank]
        B   = xBC[..., self.dt_rank:self.dt_rank + self.d_state]
        C   = xBC[..., self.dt_rank + self.d_state:]

        # Δ = Softplus(dt_proj(dt)) — standard, NO Rothermel gating here
        delta = F.softplus(self.dt_proj[direction](dt))        # (B,L,d_inner)

        A = -torch.exp(self.A_log[direction].float())          # (d_inner,d_state)
        D =  self.D[direction].float()

        if not _MAMBA_AVAILABLE:
            raise RuntimeError("selective_scan_fn is required but not available.")

        with torch.amp.autocast('cuda', enabled=False):
            y = selective_scan_fn(
                x_flat.float().transpose(1, 2).contiguous(),
                delta.float().transpose(1, 2).contiguous(),
                A.float().contiguous(),
                B.float().transpose(1, 2).contiguous(),
                C.float().transpose(1, 2).contiguous(),
                D.float().contiguous() if D is not None else None,
                delta_softplus=False,
            )

        y = y.transpose(1, 2).contiguous().to(x_flat.dtype)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, d_model) — channel-last spatial features
        Returns:
            y: (B, H, W, d_model)
        """
        B, H, W, C = x.shape
        L = H * W

        # Input gate projection
        xz = self.in_proj(x)                                   # (B,H,W,2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                         # each (B,H,W,d_inner)
        x_in = F.silu(x_in)
        z    = F.silu(z)

        # Flatten for scanning
        x_flat = x_in.view(B, L, self.d_inner)                # (B,L,d_inner)

        # 4 directions via flipping
        seqs = [
            x_flat,                                            # TL→BR
            x_flat.flip(1),                                   # BR→TL
            x_flat.view(B, H, W, self.d_inner)
                   .transpose(1, 2).contiguous()
                   .view(B, L, self.d_inner),                 # TR→BL (transposed)
            x_flat.view(B, H, W, self.d_inner)
                   .transpose(1, 2).contiguous()
                   .view(B, L, self.d_inner).flip(1),         # BL→TR
        ]

        ys = []
        for k, seq in enumerate(seqs):
            y = self._scan_direction(seq, k)
            # Reverse the direction to align outputs
            if k == 1:
                y = y.flip(1)
            elif k == 2:
                y = y.view(B, W, H, self.d_inner).transpose(1, 2).contiguous().view(B, L, self.d_inner)
            elif k == 3:
                y = y.flip(1).view(B, W, H, self.d_inner).transpose(1, 2).contiguous().view(B, L, self.d_inner)
            ys.append(y)

        # Merge: sum all directions
        y_merged = sum(ys)                                     # (B,L,d_inner)
        y_merged = y_merged * z.view(B, L, self.d_inner)      # gate

        y_out = self.out_proj(y_merged)                        # (B,L,d_model)
        y_out = self.out_norm(y_out)
        return y_out.view(B, H, W, C)


# ── Cell 4: VSSBlock ─────────────────────────────────────────────────────────

class VSSBlock(nn.Module):
    """
    Vision State Space Block: LayerNorm → SS2D → residual + LayerNorm.

    Standard VMamba formulation — NO physics gating (that's PI-VMUNet's
    contribution). Δ is purely learned: Δ = Softplus(Linear(x)).

    Args:
        d_model:      channel dimension
        d_state:      SSM state size (default 16)
        mlp_ratio:    MLP expansion ratio after SS2D (default 0 = no MLP)
        drop_path:    stochastic depth rate (default 0.0)
    """

    def __init__(
        self,
        d_model:   int,
        d_state:   int   = 16,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ss2d  = SS2D(d_model=d_model, d_state=d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = nn.Identity()   # DropPath omitted for simplicity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, d_model) channel-last
        Returns:
            x: (B, H, W, d_model)
        """
        # SS2D residual
        x = x + self.drop_path(self.ss2d(self.norm1(x)))
        return x


# ── Cell 5: PatchMerge and PatchExpand ───────────────────────────────────────

class PatchMerge(nn.Module):
    """
    2× spatial downsampling + 2× channel expansion.
    Takes 2×2 patches and projects to 2×channel_dim.

    Input:  (B, H,   W,   C)
    Output: (B, H/2, W/2, 2C)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        # Pad if H or W is odd
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
        if W % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        # Downsample: take every other pixel in 4 positions
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1)              # (B,H/2,W/2,4C)
        x  = self.norm(x)
        x  = self.proj(x)                                      # (B,H/2,W/2,2C)
        return x


class PatchExpand(nn.Module):
    """
    2× spatial upsampling + 0.5× channel reduction.
    Inverse of PatchMerge.

    Input:  (B, H,   W,   C)
    Output: (B, 2H,  2W,  C//2)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Project to 4× channels, then pixel-shuffle in channel-last
        self.proj = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        x = self.proj(x)                                       # (B,H,W,2C)
        # Rearrange to (B, 2H, 2W, C//1) — channel-last pixel shuffle
        x = x.view(B, H, W, 2, C)
        x = x.permute(0, 1, 3, 2, 4).contiguous()            # (B,H,2,W,C)
        x = x.view(B, H * 2, W, C)
        # Now expand W dimension
        x = x.unsqueeze(3).repeat(1, 1, 1, 2, 1)             # (B,2H,W,2,C)
        x = x.view(B, H * 2, W * 2, C)
        x = self.norm(x)
        return x


# ── Cell 6: VM-UNet ──────────────────────────────────────────────────────────

class VMUNet(nn.Module):
    """
    VM-UNet: VMamba encoder-decoder without physics gating.

    Encoder: stem → [VSSBlock + PatchMerge] × 4
    Decoder: [PatchExpand + VSSBlock + skip] × 4 → head

    This is the direct baseline. PI-VMUNet adds PGSS to VSSBlock.

    Input:  (B, 24, H, W)
    Output: (B, 1,  H, W) ∈ [0,1]
    """

    IN_CHANNELS: int = 24

    def __init__(
        self,
        dims:    tuple = (96, 192, 384, 768),
        d_state: int   = 16,
    ) -> None:
        super().__init__()
        self.dims = dims

        # ── Stem: project 24 channels → dims[0] ─────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, dims[0], kernel_size=4, stride=4, bias=False),
            # Output: (B, dims[0], H/4, W/4)
        )
        self.stem_norm = nn.LayerNorm(dims[0])

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.merges     = nn.ModuleList()
        for i in range(4):
            in_dim = dims[i]
            self.enc_blocks.append(VSSBlock(d_model=in_dim, d_state=d_state))
            if i < 3:
                self.merges.append(PatchMerge(in_dim))
            else:
                self.merges.append(nn.Identity())

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = VSSBlock(d_model=dims[3], d_state=d_state)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec_expands = nn.ModuleList()
        self.dec_norms   = nn.ModuleList()
        self.dec_blocks  = nn.ModuleList()
        self.skip_projs  = nn.ModuleList()

        for i in range(3, -1, -1):
            in_dim   = dims[i]
            skip_dim = dims[i]
            out_dim  = dims[i - 1] if i > 0 else dims[0] // 2

            self.dec_expands.append(PatchExpand(in_dim))
            # After expand + skip concat, channels = in_dim + skip_dim
            # Project down to out_dim
            self.skip_projs.append(
                nn.Linear(in_dim + skip_dim, out_dim, bias=False)
            )
            self.dec_norms.append(nn.LayerNorm(out_dim))
            self.dec_blocks.append(VSSBlock(d_model=out_dim, d_state=d_state))

        # ── Head ─────────────────────────────────────────────────────────────
        head_dim = dims[0] // 2
        self.head = nn.Sequential(
            nn.Conv2d(head_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def _to_channel_last(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) → (B,H,W,C)"""
        return x.permute(0, 2, 3, 1).contiguous()

    def _to_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        """(B,H,W,C) → (B,C,H,W)"""
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 24, H, W)
        Returns:
            (B, 1, H, W)
        """
        B, C, H, W = x.shape

        # Stem: (B,24,H,W) → (B,dims[0],H/4,W/4) → channel-last
        s = self.stem(x)                                        # (B,d0,H/4,W/4)
        s = self._to_channel_last(s)                           # (B,H/4,W/4,d0)
        s = self.stem_norm(s)

        # Encoder: save skip connections (channel-last)
        skips = []
        feat  = s
        for i, (block, merge) in enumerate(zip(self.enc_blocks, self.merges)):
            feat = block(feat)
            skips.append(feat)
            if i < 3:
                feat = merge(feat)

        # Bottleneck
        feat = self.bottleneck(feat)

        # Decoder
        for i, (expand, skip_proj, norm, block) in enumerate(
            zip(self.dec_expands, self.skip_projs, self.dec_norms, self.dec_blocks)
        ):
            skip = skips[3 - i]                                # reverse order

            feat = expand(feat)                                # upsample

            # Align spatial size if needed
            if feat.shape[1:3] != skip.shape[1:3]:
                feat_cf = self._to_channel_first(feat)
                feat_cf = F.interpolate(feat_cf, size=skip.shape[1:3],
                                        mode="bilinear", align_corners=False)
                feat = self._to_channel_last(feat_cf)

            # Concatenate skip and project
            feat = torch.cat([feat, skip], dim=-1)             # (B,H,W,in+skip)
            feat = skip_proj(feat)
            feat = norm(feat)
            feat = block(feat)

        # Back to channel-first, upsample to input resolution, head
        feat = self._to_channel_first(feat)                    # (B,d0//2,H',W')
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(feat)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Cell 7: Verify model ──────────────────────────────────────────────────────

def verify_model() -> None:
    model = VMUNet().to(DEVICE)
    x = torch.randn(2, 24, 128, 128).to(DEVICE)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 128, 128), f"Wrong output shape: {y.shape}"
    assert y.min() >= 0.0 and y.max() <= 1.0
    params = model.count_params()
    print(f"✅ VMUNet output shape: {y.shape}")
    print(f"   Trainable params: {params/1e6:.2f}M")
    del model, x, y
    torch.cuda.empty_cache()

verify_model()


# ── Cell 8: Dataloaders ───────────────────────────────────────────────────────

HDF5_PATH  = "/kaggle/working/wildfire_data.h5"
BATCH_SIZE = 16  # doubled — halves batches per epoch, fits in 12h

# Runtime stability toggles (do not change scientific behavior)
STABILITY_MODE     = True
NUM_WORKERS        = 0 if STABILITY_MODE else 2
PIN_MEMORY         = False if STABILITY_MODE else True
PERSISTENT_WORKERS = False if STABILITY_MODE else True
MIXED_PRECISION    = False if STABILITY_MODE else True
FAIL_ON_NONFINITE  = False   # skip NaN batches, don't crash
LOG_CUDA_MEMORY    = False

train_dl, val_dl, test_dl = make_dataloaders(
    hdf5_path   = HDF5_PATH,
    batch_size  = BATCH_SIZE,
    resolution  = 128,
    num_workers = NUM_WORKERS,
    pin_memory  = PIN_MEMORY,
    persistent_workers = PERSISTENT_WORKERS,
)

batch = next(iter(train_dl))
print(f"Batch shapes — inputs: {batch['inputs'].shape}  targets: {batch['targets'].shape}")


# ── Cell 9: Training setup ────────────────────────────────────────────────────

CKPT_DIR = "/kaggle/working/outputs/checkpoints/vm_unet"
os.makedirs(CKPT_DIR, exist_ok=True)

config = TrainConfig(
    model_name      = "vm_unet",
    resolution      = 128,
    batch_size      = BATCH_SIZE,
    lr              = 1e-4,
    weight_decay    = 1e-2,
    n_epochs        = 30,
    lambda_max      = 0.0,
    lambda_eik      = 0.0,
    lambda_reg      = 0.0,
    alpha_focal     = 0.85,
    gamma_focal     = 3.0,
    checkpoint_dir  = CKPT_DIR,
    mixed_precision = MIXED_PRECISION,
    grad_clip       = 0.5,
    wandb_project   = "pi-vm",
    wandb_run_name  = "vm_unet_baseline",
    save_every      = 5,
    fail_on_nonfinite = FAIL_ON_NONFINITE,
    log_cuda_memory   = LOG_CUDA_MEMORY,
)

model     = VMUNet().to(DEVICE)
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


# ── Cell 10: Train ────────────────────────────────────────────────────────────

t_start = time.time()
history = trainer.fit(train_dl, val_dl, n_epochs=config.n_epochs)
t_total = time.time() - t_start
print(f"\nTotal training time: {t_total/3600:.2f} hours")


# ── Cell 11: Evaluate and save ────────────────────────────────────────────────

test_metrics = trainer.val_epoch(test_dl)

print(f"\n{'='*55}")
print("TEST SET EVALUATION — VM-UNet")
print(f"{'='*55}")
for k, v in test_metrics.items():
    print(f"  {k:20s}: {float(v):.4f}")

results = {
    "model":        "vm_unet",
    "params_M":     model.count_params() / 1e6,
    "train_time_h": t_total / 3600,
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    "history":      {k: [float(x) for x in v] for k, v in history.items()},
}
with open("/kaggle/working/results_vm_unet.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Results saved to /kaggle/working/results_vm_unet.json")
