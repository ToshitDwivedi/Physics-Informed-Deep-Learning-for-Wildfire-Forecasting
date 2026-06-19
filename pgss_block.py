"""
pgss_block.py
─────────────────────────────────────────────────────────────────────────────
Phase 3-A: Physics-Gated Selective Scan Block (PGSS) — Original Contribution 1.

PGSS is the core architectural novelty of PI-VMUNet. It modifies the Mamba
selective scan by gating the Δ (time-step) parameter with Rothermel's Rate
of Spread, making the model's attention explicitly fire-physics-aware:

    Standard Mamba : Δ = Softplus(Linear(x))
    PGSS            : Δ = Softplus(Linear(x)) · gate(R̃)
                      gate(R̃) = (1 − α) + α · R̃
    where:
        R̃ = RothermelLayer.normalise_for_pgss(physics_dict)  ∈ [0, 1]
        α = nn.Parameter(init=0.5), clamped to [0.001, 0.999]

Physical interpretation:
    • R̃ = 0 (no fire spread risk): gate → (1−α), Δ is attenuated
    • R̃ = 1 (maximum fire spread): gate → 1.0,   Δ is at full learned value
    • α controls how strongly physics modulates the scan — it is LEARNED,
      not fixed, so the model decides how much to trust the physics signal.

This file contains:
    PHYSICS_CHANNEL_INDICES  — maps variable names → channel indices in the
                               24-channel pipeline output (from CHANNEL_REGISTRY)
    PGSSBlock                — drop-in replacement for VSSBlock with PGSS gate
    PIVMUNet                 — full encoder-decoder using PGSSBlock throughout

Usage in PI-VMUNet vs VM-UNet:
    VM-UNet  : encoder/decoder use VSSBlock   (Δ = Softplus(Linear(x)))
    PI-VMUNet: encoder/decoder use PGSSBlock  (Δ = Softplus(Linear(x)) · gate)
    Everything else (PatchMerge, PatchExpand, head) is identical.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/kaggle/working")
from rothermel import RothermelLayer

# ── mamba-ssm optional fast path ─────────────────────────────────────────────
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Physics channel index map
#  Source: CHANNEL_REGISTRY in transforms.py
#  All indices refer to the 24-channel output of FullPreprocessingPipeline
# ═══════════════════════════════════════════════════════════════════════════════

PHYSICS_CHANNEL_INDICES: Dict[str, int] = {
    # Required by RothermelLayer.forward()
    "wind_speed":  23,   # m/s clamped [0,40]  — channel 23 "wind_speed_clamped"
    "slope_angle": 10,   # radians              — channel 10 "slope_magnitude"
    "M_f":          3,   # moisture proxy (sph normalised → rescaled below)
    "rho_b":       16,   # kg/m³               — channel 16
    "sigma":       17,   # 1/ft                — channel 17
    "beta":        18,   # dimensionless       — channel 18
    "beta_op":     19,   # dimensionless       — channel 19
    "w_n":         20,   # kg/m²               — channel 20
    "h":           21,   # kJ/kg               — channel 21
}

# M_f rescaling: pipeline stores sph z-score; we need moisture fraction [0, 0.4]
# sph mean=0.006468, std=0.003683 → z-score back to kg/kg, proxy M_f = sph * 20
# (empirical: sph range 0.002–0.015 kg/kg maps to M_f range 0.04–0.30)
_MF_SCALE: float = 20.0   # sph_kg_per_kg * 20 ≈ moisture fraction
_SHP_MEAN: float = 0.006468
_SPH_STD:  float = 0.003683


# ═══════════════════════════════════════════════════════════════════════════════
#  Selective scan helpers (copied from notebook_3 for self-containment)
# ═══════════════════════════════════════════════════════════════════════════════

def selective_scan_pytorch(
    u:      torch.Tensor,
    delta:  torch.Tensor,
    A:      torch.Tensor,
    B_mat:  torch.Tensor,
    C_mat:  torch.Tensor,
    D_vec:  torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch selective scan. B/C/D renamed to avoid collision with batch dim."""
    n_batch, L, _ = u.shape
    d_inner = delta.shape[-1]
    d_state = A.shape[-1]

    delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    delta_B = delta.unsqueeze(-1) * B_mat.unsqueeze(2)

    h  = torch.zeros(n_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h   = delta_A[:, t] * h + delta_B[:, t] * u[:, t, :d_inner].unsqueeze(-1)
        y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)
    y = y + u[:, :, :d_inner] * D_vec.view(1, 1, -1)
    return y



# ═══════════════════════════════════════════════════════════════════════════════
#  PGSSBlock
# ═══════════════════════════════════════════════════════════════════════════════

class PGSSBlock(nn.Module):
    """
    Physics-Gated Selective Scan Block — drop-in replacement for VSSBlock.

    The ONLY difference from VSSBlock is in _scan_direction():
        VSSBlock  : delta = Softplus(dt_proj(dt))
        PGSSBlock : delta = Softplus(dt_proj(dt)) * gate
                    gate  = (1 − α) + α · R̃   broadcast over d_inner

    Everything else — in_proj, x_proj, A_log, D, out_proj, 4-direction scan,
    residual, LayerNorm — is identical to VSSBlock.

    Args:
        d_model:                  channel dimension (must match encoder stage dim)
        d_state:                  SSM state dimension (default 16)
        rothermel_layer:          shared RothermelLayer instance (no_grad in forward)
        physics_channel_indices:  dict mapping name→channel index in physics_tensor
                                  (default: PHYSICS_CHANNEL_INDICES)
    """

    def __init__(
        self,
        d_model:                 int,
        d_state:                 int            = 16,
        rothermel_layer:         RothermelLayer = None,
        physics_channel_indices: Dict[str, int] = None,
    ) -> None:
        super().__init__()

        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = d_model * 2
        self.dt_rank  = math.ceil(d_model / 16)
        self.K        = 4

        # ── α: learned physics gate strength ─────────────────────────────────
        # Initialised to 0.5 (equal blend). Clamped to [0.001, 0.999] after
        # each optimiser step to prevent gate from collapsing to 0 or 1.
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # ── Physics components ────────────────────────────────────────────────
        self.rothermel = rothermel_layer or RothermelLayer()
        self.phys_idx  = physics_channel_indices or PHYSICS_CHANNEL_INDICES

        # ── Standard Mamba components (identical to VSSBlock / SS2D) ─────────
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.x_proj   = nn.ModuleList([
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            for _ in range(self.K)
        ])
        self.dt_proj  = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True)
            for _ in range(self.K)
        ])

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(self.d_inner, 1)
        self.A_log = nn.ParameterList([
            nn.Parameter(torch.log(A.clone())) for _ in range(self.K)
        ])
        self.D = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner)) for _ in range(self.K)
        ])

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm1    = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)

    # ── Physics extraction ────────────────────────────────────────────────────

    def _extract_physics_dict(
        self,
        physics_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract named physics tensors from the 24-channel pipeline output.

        physics_tensor shape: (B, 24, H, W) — channel-first
        Returns dict with (B, 1, H, W) tensors as required by RothermelLayer.

        M_f is derived from sph (specific humidity) by inverting the z-score
        normalisation and applying the empirical M_f = sph_kg_per_kg * 20 proxy.
        """
        def ch(name: str) -> torch.Tensor:
            idx = self.phys_idx[name]
            return physics_tensor[:, idx:idx+1, :, :]   # (B,1,H,W)

        # Invert z-score for sph → get back kg/kg → proxy moisture fraction
        sph_norm = ch("M_f")                             # actually sph z-score
        sph_raw  = sph_norm * _SPH_STD + _SHP_MEAN      # kg/kg
        M_f      = torch.clamp(sph_raw * _MF_SCALE, 0.0, 0.30)

        return {
            "wind_speed":  ch("wind_speed"),
            "slope_angle": ch("slope_angle"),
            "M_f":         M_f,
            "rho_b":       ch("rho_b"),
            "sigma":       ch("sigma"),
            "beta":        ch("beta"),
            "beta_op":     ch("beta_op"),
            "w_n":         ch("w_n"),
            "h":           ch("h"),
        }

    # ── PGSS scan (single direction) ─────────────────────────────────────────

    def _pgss_scan_direction(
        self,
        x_flat: torch.Tensor,
        gate:   torch.Tensor,
        k:      int,
    ) -> torch.Tensor:
        """
        Run one direction of the PGSS selective scan.

        The ONLY change from SS2D._scan_direction():
            Standard: delta = Softplus(dt_proj(dt))
            PGSS    : delta = Softplus(dt_proj(dt)) * gate

        Args:
            x_flat: (B, L, d_inner)
            gate:   (B, L, 1) physics gate broadcast over d_inner
            k:      direction index 0–3

        Returns:
            y_flat: (B, L, d_inner)
        """
        xBC   = self.x_proj[k](x_flat)
        dt    = xBC[..., :self.dt_rank]
        B     = xBC[..., self.dt_rank:self.dt_rank + self.d_state]
        C     = xBC[..., self.dt_rank + self.d_state:]

        # ── PGSS gate applied here ────────────────────────────────────────────
        delta = F.softplus(self.dt_proj[k](dt)) * gate   # (B,L,d_inner)

        A = -torch.exp(self.A_log[k].float())
        D =  self.D[k].float()

        if _MAMBA_AVAILABLE:
            y = selective_scan_fn(
                x_flat.transpose(1, 2),
                delta.transpose(1, 2),
                A, B.transpose(1, 2), C.transpose(1, 2), D,
                delta_softplus=False,
            )
            y = y.transpose(1, 2)
        else:
            y = selective_scan_pytorch(x_flat, delta, A, B, C, D)

        return y

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:              torch.Tensor,
        physics_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with physics-gated selective scan.

        Args:
            x:              (B, H, W, d_model) channel-last feature map
            physics_tensor: (B, 24, H, W) channel-first pipeline output

        Returns:
            (B, H, W, d_model)
        """
        B, H, W, C = x.shape
        L = H * W

        # ── Compute R̃ — allow grad for asymmetry tests; Rothermel has no params ──
        pdict  = self._extract_physics_dict(physics_tensor)
        R_norm = self.rothermel.normalise_for_pgss(pdict)       # (B,1,H,W)

        # ── Build gate: (1-α) + α·R̃, broadcast to (B,L,1) ───────────────────
        alpha_c    = torch.clamp(self.alpha, 0.001, 0.999)
        R_norm_safe = torch.nan_to_num(R_norm, nan=0.0, posinf=1.0, neginf=0.0)
        gate_2d    = (1.0 - alpha_c) + alpha_c * R_norm_safe    # (B,1,H,W)
        gate    = gate_2d.permute(0, 2, 3, 1).reshape(B, L, 1)  # (B,L,1)

        # ── Input projection + gating (same as SS2D.forward) ─────────────────
        xz          = self.in_proj(self.norm1(x))               # (B,H,W,2*d_inner)
        x_in, z     = xz.chunk(2, dim=-1)
        x_in        = F.silu(x_in)
        z           = F.silu(z)
        x_flat      = x_in.view(B, L, self.d_inner)

        # ── 4-direction PGSS scan ─────────────────────────────────────────────
        seqs = [
            (x_flat,                                                  gate),
            (x_flat.flip(1),                                          gate.flip(1)),
            (x_flat.view(B,H,W,self.d_inner).transpose(1,2)
                    .contiguous().view(B,L,self.d_inner),
             gate.view(B,H,W,1).transpose(1,2).contiguous().view(B,L,1)),
            (x_flat.view(B,H,W,self.d_inner).transpose(1,2)
                    .contiguous().view(B,L,self.d_inner).flip(1),
             gate.view(B,H,W,1).transpose(1,2).contiguous().view(B,L,1).flip(1)),
        ]

        ys = []
        for k, (seq, g) in enumerate(seqs):
            y = self._pgss_scan_direction(seq, g, k)
            # Reverse to restore spatial order
            if k == 1:
                y = y.flip(1)
            elif k == 2:
                y = y.view(B,W,H,self.d_inner).transpose(1,2).contiguous().view(B,L,self.d_inner)
            elif k == 3:
                y = y.flip(1).view(B,W,H,self.d_inner).transpose(1,2).contiguous().view(B,L,self.d_inner)
            ys.append(y)

        y_merged = sum(ys)                                      # (B,L,d_inner)
        y_merged = y_merged * z.view(B, L, self.d_inner)

        y_out = self.out_proj(y_merged)
        y_out = self.out_norm(y_out)
        y_out = y_out.view(B, H, W, C)

        # Residual connection
        y_out = torch.nan_to_num(y_out, nan=0.0)
        return x + y_out

    def clamp_alpha(self) -> None:
        """
        Clamp α to [0.001, 0.999] in-place after each optimiser step.
        Call from training loop: for block in model.pgss_blocks: block.clamp_alpha()
        """
        self.alpha.data.clamp_(0.001, 0.999)

    @property
    def alpha_value(self) -> float:
        """Return current α as a Python float (for WandB logging)."""
        return self.alpha.item()


# ═══════════════════════════════════════════════════════════════════════════════
#  PatchMerge / PatchExpand (identical to notebook_3 — reproduced for
#  self-containment so this file can be imported standalone)
# ═══════════════════════════════════════════════════════════════════════════════

class PatchMerge(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        if H % 2 != 0: x = F.pad(x, (0,0,0,0,0,1))
        if W % 2 != 0: x = F.pad(x, (0,0,0,1))
        x = torch.cat([x[:,0::2,0::2,:], x[:,1::2,0::2,:],
                        x[:,0::2,1::2,:], x[:,1::2,1::2,:]], dim=-1)
        return self.proj(self.norm(x))


class PatchExpand(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = self.proj(x)
        x = x.view(B, H, W, 2, C)
        x = x.permute(0,1,3,2,4).contiguous().view(B, H*2, W, C)
        x = x.unsqueeze(3).repeat(1,1,1,2,1).view(B, H*2, W*2, C)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  PI-VMUNet — full model
# ═══════════════════════════════════════════════════════════════════════════════

class PIVMUNet(nn.Module):
    """
    Physics-Informed Vision Mamba UNet.

    Identical structure to VMUNet (notebook_3) but VSSBlock → PGSSBlock
    everywhere. A single shared RothermelLayer is instantiated once and
    passed to every PGSSBlock (avoids redundant computation).

    The physics_tensor (24-channel pipeline output) is passed alongside
    the feature tensor through every encoder and decoder stage.

    Input:  (B, 24, H, W)  — physics-processed 24-channel tensor
    Output: (B, 1,  H, W)  — fire probability ∈ [0, 1]

    Args:
        dims:    channel dimensions per stage (default matches VM-UNet)
        d_state: SSM state size
    """

    IN_CHANNELS: int = 24

    def __init__(
        self,
        dims:    Tuple[int, ...] = (96, 192, 384, 768),
        d_state: int             = 16,
    ) -> None:
        super().__init__()
        self.dims = dims

        # ── Single shared RothermelLayer ──────────────────────────────────────
        # Shared across all PGSSBlocks — physics computation is identical
        # for all blocks in the same forward pass.
        self.rothermel = RothermelLayer()
        for p in self.rothermel.parameters():
            p.requires_grad_(False)   # physical constants, not learned

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem      = nn.Conv2d(self.IN_CHANNELS, dims[0], kernel_size=4, stride=4, bias=False)
        self.stem_norm = nn.LayerNorm(dims[0])

        # ── Encoder: 4 × [PGSSBlock + PatchMerge] ────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.merges     = nn.ModuleList()
        for i in range(4):
            self.enc_blocks.append(PGSSBlock(
                d_model         = dims[i],
                d_state         = d_state,
                rothermel_layer = self.rothermel,
            ))
            self.merges.append(PatchMerge(dims[i]) if i < 3 else nn.Identity())

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = PGSSBlock(
            d_model         = dims[3],
            d_state         = d_state,
            rothermel_layer = self.rothermel,
        )

        # ── Decoder: 4 × [PatchExpand + PGSSBlock + skip] ────────────────────
        self.dec_expands = nn.ModuleList()
        self.dec_norms   = nn.ModuleList()
        self.dec_blocks  = nn.ModuleList()
        self.skip_projs  = nn.ModuleList()

        for i in range(3, -1, -1):
            in_dim  = dims[i]
            out_dim = dims[i-1] if i > 0 else dims[0] // 2
            self.dec_expands.append(PatchExpand(in_dim))
            self.skip_projs.append(nn.Linear(in_dim + in_dim, out_dim, bias=False))
            self.dec_norms.append(nn.LayerNorm(out_dim))
            self.dec_blocks.append(PGSSBlock(
                d_model         = out_dim,
                d_state         = d_state,
                rothermel_layer = self.rothermel,
            ))

        # ── Head ──────────────────────────────────────────────────────────────
        head_dim = dims[0] // 2
        self.head = nn.Sequential(
            nn.Conv2d(head_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def _cf(self, x: torch.Tensor) -> torch.Tensor:
        """Channel-last → channel-first."""
        return x.permute(0, 3, 1, 2).contiguous()

    def _cl(self, x: torch.Tensor) -> torch.Tensor:
        """Channel-first → channel-last."""
        return x.permute(0, 2, 3, 1).contiguous()

    def _resize_physics(self, physics: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Downsample physics tensor to match current feature map resolution."""
        if physics.shape[2] == H and physics.shape[3] == W:
            return physics
        return F.interpolate(physics, size=(H, W), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 24, H, W) — output of FullPreprocessingPipeline
        Returns:
            (B, 1, H, W) fire probability
        """
        B, C, H, W = x.shape
        physics = x   # keep original for downsampling at each stage

        # ── Stem ──────────────────────────────────────────────────────────────
        feat = self.stem_norm(self._cl(self.stem(x)))           # (B,H/4,W/4,d0)

        # ── Encoder ───────────────────────────────────────────────────────────
        skips = []
        for i, (block, merge) in enumerate(zip(self.enc_blocks, self.merges)):
            fH, fW = feat.shape[1], feat.shape[2]
            phys_i = self._resize_physics(physics, fH, fW)
            feat   = block(feat, phys_i)
            skips.append(feat)
            if i < 3:
                feat = merge(feat)

        # ── Bottleneck ────────────────────────────────────────────────────────
        fH, fW = feat.shape[1], feat.shape[2]
        feat   = self.bottleneck(feat, self._resize_physics(physics, fH, fW))

        # ── Decoder ───────────────────────────────────────────────────────────
        for i, (expand, skip_proj, norm, block) in enumerate(
            zip(self.dec_expands, self.skip_projs, self.dec_norms, self.dec_blocks)
        ):
            skip = skips[3 - i]
            feat = expand(feat)

            if feat.shape[1:3] != skip.shape[1:3]:
                feat = self._cl(F.interpolate(
                    self._cf(feat), size=skip.shape[1:3],
                    mode="bilinear", align_corners=False
                ))

            feat = norm(skip_proj(torch.cat([feat, skip], dim=-1)))
            fH, fW = feat.shape[1], feat.shape[2]
            feat   = block(feat, self._resize_physics(physics, fH, fW))

        # ── Head ──────────────────────────────────────────────────────────────
        feat = self._cf(feat)
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(feat)

    def pgss_blocks(self) -> List[PGSSBlock]:
        """Return all PGSSBlock instances (for alpha clamping + WandB logging)."""
        blocks = []
        for m in self.modules():
            if isinstance(m, PGSSBlock):
                blocks.append(m)
        return blocks

    def clamp_all_alpha(self) -> None:
        """Call after every optimiser step to keep α ∈ [0.001, 0.999]."""
        for b in self.pgss_blocks():
            b.clamp_alpha()

    def alpha_stats(self) -> Dict[str, float]:
        """Return α values per block for WandB logging."""
        alphas = [b.alpha_value for b in self.pgss_blocks()]
        return {
            "alpha/mean": sum(alphas) / len(alphas),
            "alpha/min":  min(alphas),
            "alpha/max":  max(alphas),
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
