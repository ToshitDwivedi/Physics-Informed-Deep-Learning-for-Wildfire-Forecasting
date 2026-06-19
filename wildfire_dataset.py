"""
wildfire_dataset.py
─────────────────────────────────────────────────────────────────────────────
PyTorch Dataset for the Next Day Wildfire Spread (Huot et al. 2022) HDF5 file.
Used by all three baseline notebooks: ResNet-UNet, Swin-UNet, VM-UNet.

Batch format (matches Trainer expected format):
    {
        'inputs'   : (B, 24, H, W)  — pipeline-processed 24-channel tensor
        'targets'  : (B, 1,  H, W)  — binary fire mask at T+24h
        'prev_fire': (B, 1,  H, W)  — previous fire mask (for L_PDE)
        'raw'      : (B, 12, H, W)  — raw 12-channel input (for debug)
    }

Resolution: 128×128 (bilinear upsample from native 64×64).
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import our pipeline
import sys
sys.path.insert(0, "/kaggle/working")
from transforms import FullPreprocessingPipeline


class WildfireDataset(Dataset):
    """
    HDF5-backed dataset with on-the-fly physics feature engineering.

    Args:
        hdf5_path:   path to wildfire_data.h5
        split:       'train', 'eval', or 'test'
        resolution:  output spatial resolution (default 128)
        compute_sdf: whether to compute SDF channel (slow — disable for speed)
        max_samples: cap dataset size (useful for dry-run / debugging)
    """

    def __init__(
        self,
        hdf5_path:   str,
        split:       str  = "train",
        resolution:  int  = 128,
        compute_sdf: bool = False,      # SDF is slow; disable during training
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hdf5_path  = hdf5_path
        self.split      = split
        self.resolution = resolution

        # Open HDF5 to get length only — keep closed between __getitem__ calls
        # (h5py is not fork-safe with num_workers > 0 if file is kept open)
        with h5py.File(hdf5_path, "r") as f:
            self.n_samples = f[split]["inputs"].shape[0]

        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)

        # Pipeline runs on CPU — GPU version would require moving it to device
        self.pipeline = FullPreprocessingPipeline(
            pixel_size_m = 1000.0,
            compute_sdf  = compute_sdf,
        )
        self.pipeline.eval()   # no dropout / BN tracking needed

        # Keep HDF5 file handle as None — open lazily in __getitem__
        self._hdf5_file: Optional[h5py.File] = None

    def _get_file(self) -> h5py.File:
        """Lazy open HDF5 — required for DataLoader num_workers > 0."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self._hdf5_file

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns one sample dict.

        The raw 64×64 patch is upsampled to self.resolution before pipeline.
        Target is upsampled with nearest-neighbour to preserve binary values.
        """
        f = self._get_file()

        # Raw 12-channel input (64,64,12) → (12,64,64)
        raw_np  = f[self.split]["inputs"][idx].astype(np.float32)   # (H,W,12)
        raw_np  = raw_np.transpose(2, 0, 1)                         # (12,H,W)

        # Target fire mask (64,64) with values {-1=no-data, 0=no-fire, 1=fire}
        tgt_np   = f[self.split]["targets"][idx].astype(np.float32)  # (H,W)
        mask_np  = (tgt_np >= 0).astype(np.float32)                  # 1=valid, 0=no-data
        tgt_np   = np.clip(tgt_np, 0.0, 1.0)                        # -1→0, 0→0, 1→1

        # Convert to tensors
        raw  = torch.from_numpy(raw_np).unsqueeze(0)                 # (1,12,H,W)
        tgt  = torch.from_numpy(tgt_np).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # Upsample to target resolution
        if self.resolution != 64:
            raw  = F.interpolate(
                raw, size=(self.resolution, self.resolution),
                mode="bilinear", align_corners=False
            )
            tgt  = F.interpolate(
                tgt, size=(self.resolution, self.resolution),
                mode="nearest"
            )
            mask = F.interpolate(
                mask, size=(self.resolution, self.resolution),
                mode="nearest"
            )

        raw  = raw.squeeze(0)   # (12,H,W)
        tgt  = tgt.squeeze(0)   # (1,H,W)
        mask = mask.squeeze(0)  # (1,H,W)

        # Run physics pipeline
        with torch.no_grad():
            physics = self.pipeline(raw.unsqueeze(0)).squeeze(0)   # (24,H,W)

        # Previous fire mask = channel 9 of physics output (pass-through PrevFireMask)
        # Also available directly as raw channel 11
        prev_fire = (raw[11:12] > 0.5).float()                     # (1,H,W)

        return {
            "inputs":    physics,    # (24,H,W)
            "targets":   tgt,        # (1,H,W)  values in {0,1}
            "valid_mask":mask,       # (1,H,W)  1=valid pixel, 0=no-data
            "prev_fire": prev_fire,  # (1,H,W)
            "raw":       raw,        # (12,H,W) — for debug only
        }

    def __del__(self):
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass


def make_dataloaders(
    hdf5_path:   str,
    batch_size:  int = 12,
    resolution:  int = 128,
    num_workers: int = 2,
    pin_memory:  bool = True,
    persistent_workers: bool = False,
    max_train:   Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test dataloaders.

    Args:
        hdf5_path:   path to wildfire_data.h5
        batch_size:  batch size (use 8 for Swin/VM-UNet on T4)
        resolution:  128 for all models (T4 VRAM constraint)
        num_workers: 2 is safe on Kaggle T4
        pin_memory:  enable fast host-to-device transfer
        persistent_workers: keep workers alive between epochs
        max_train:   cap training samples (None = use all)

    Returns:
        (train_dl, val_dl, test_dl)
    """
    train_ds = WildfireDataset(hdf5_path, "train", resolution, max_samples=max_train)
    val_ds   = WildfireDataset(hdf5_path, "eval",  resolution)
    test_ds  = WildfireDataset(hdf5_path, "test",  resolution)

    use_persistent = persistent_workers and num_workers > 0

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=use_persistent, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )

    print(f"Dataset split sizes:")
    print(f"  train : {len(train_ds):,} samples  ({len(train_dl)} batches)")
    print(f"  val   : {len(val_ds):,} samples  ({len(val_dl)} batches)")
    print(f"  test  : {len(test_ds):,} samples  ({len(test_dl)} batches)")

    return train_dl, val_dl, test_dl
