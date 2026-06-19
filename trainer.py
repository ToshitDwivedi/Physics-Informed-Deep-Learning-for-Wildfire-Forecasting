"""
trainer.py
─────────────────────────────────────────────────────────────────────────────
Complete training framework for PI-VM (Physics-Informed Vision Mamba).
Used by ALL models: ResNet-UNet, Swin-UNet, VM-UNet, PI-VMUNet.

Components:
    TrainConfig          — dataclass for all hyperparameters
    WildfireLoss         — L_total = L_Seg + λ_PDE·L_PDE + λ_Eik·L_Eik + λ_Reg
    WildfireMetrics      — CSI, IoU, Precision, Recall, F1, PR-AUC, PCR, ECE, FPD
    CurriculumScheduler  — 3-phase λ_PDE ramp
    Trainer              — full train/val loop with checkpointing + WandB

Session-disconnect safety:
    • Checkpoints saved every 5 epochs to checkpoint_dir
    • Best model saved separately by CSI
    • KeyboardInterrupt caught → emergency checkpoint saved before exit
    • load_checkpoint() auto-detects latest epoch to resume from
"""

from __future__ import annotations

import glob
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Optional imports — fail gracefully
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    warnings.warn("cv2 not found — Fréchet Perimeter Distance will return nan")

try:
    from scipy.spatial.distance import directed_hausdorff
    from sklearn.metrics import average_precision_score
    _SCIPY_SKLEARN_AVAILABLE = True
except ImportError:
    _SCIPY_SKLEARN_AVAILABLE = False
    warnings.warn("scipy/sklearn not found — PR-AUC and FPD will return nan")

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    warnings.warn("wandb not found — logging disabled")


# ═══════════════════════════════════════════════════════════════════════════════
#  TrainConfig
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """
    All hyperparameters for a single training run.
    Pass this to Trainer.__init__().
    """
    model_name:       str   = "resnet_unet"
    resolution:       int   = 128          # T4 VRAM constraint — do NOT set to 256
    batch_size:       int   = 12
    lr:               float = 1e-4
    weight_decay:     float = 1e-2
    n_epochs:         int   = 100
    lambda_max:       float = 1.0          # max λ_PDE reached at epoch 81
    lambda_eik:       float = 0.1          # λ_Eikonal (fixed)
    lambda_reg:       float = 1e-4         # L2 regularisation weight
    alpha_focal:      float = 0.75         # focal loss α (class balance)
    gamma_focal:      float = 2.0          # focal loss γ
    checkpoint_dir:   str   = "/kaggle/working/checkpoints"
    mixed_precision:  bool  = True         # fp16 via torch.cuda.amp
    fail_on_nonfinite: bool = True         # raise on NaN/Inf loss
    log_cuda_memory:  bool  = False        # print CUDA memory summary per epoch
    grad_clip:        float = 1.0          # gradient clipping max norm
    wandb_project:    str   = "pi-vm"
    wandb_run_name:   str   = ""           # auto-set to model_name if empty
    pixel_size_m:     float = 1000.0       # 1 km resolution
    save_every:       int   = 5            # checkpoint every N epochs
    val_every:        int   = 1            # validate every N epochs


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 1 — WildfireLoss
# ═══════════════════════════════════════════════════════════════════════════════

class WildfireLoss(nn.Module):
    """
    Combined segmentation + physics loss for wildfire spread prediction.

    L_total = L_Seg + λ_PDE · L_PDE + λ_Eik · L_Eikonal + λ_Reg · L_Reg
    L_Seg   = L_Dice + L_Focal

    Args:
        alpha_focal: focal loss α weight for the positive (fire) class.
                     Should be high (0.75) to compensate for <2% fire pixels.
        gamma_focal: focal loss γ focusing parameter.
        lambda_pde:  weight for PDE level-set loss. Set by CurriculumScheduler.
        lambda_eik:  weight for Eikonal regularisation.
        lambda_reg:  weight for L2 parameter regularisation.
    """

    def __init__(
        self,
        alpha_focal:  float = 0.75,
        gamma_focal:  float = 2.0,
        lambda_pde:   float = 0.0,
        lambda_eik:   float = 0.0,
        lambda_reg:   float = 1e-4,
    ) -> None:
        super().__init__()
        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal
        self.lambda_pde  = lambda_pde
        self.lambda_eik  = lambda_eik
        self.lambda_reg  = lambda_reg

    # ── Soft Dice Loss ────────────────────────────────────────────────────────

    def _dice_loss(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Soft Dice loss — mask-aware, excludes no-data pixels.

        Args:
            y_hat:      (B, 1, H, W) predicted probabilities ∈ [0, 1]
            y_true:     (B, 1, H, W) binary ground truth {0, 1}
            valid_mask: (B, 1, H, W) float mask — 1=valid, 0=exclude
            eps:        stability constant
        """
        p = y_hat.float().view(y_hat.shape[0], -1)
        g = y_true.float().view(y_true.shape[0], -1)

        if valid_mask is not None:
            m = valid_mask.float().view(y_hat.shape[0], -1)
            p = p * m
            g = g * m
            denom = m.sum(dim=1).clamp(min=1.0)
            intersection = (p * g).sum(dim=1)
            dice = (2.0 * intersection + eps) / (
                (p * m).sum(dim=1) + (g * m).sum(dim=1) + eps
            )
        else:
            intersection = (p * g).sum(dim=1)
            dice = (2.0 * intersection + eps) / (p.sum(dim=1) + g.sum(dim=1) + eps)

        return 1.0 - dice.mean()

    # ── Focal Loss ────────────────────────────────────────────────────────────

    def _focal_loss(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Focal loss — mask-aware, always in float32 to prevent NaN in fp16.

        Args:
            y_hat:      (B, 1, H, W) predicted probabilities
            y_true:     (B, 1, H, W) binary labels
            valid_mask: (B, 1, H, W) float mask — 1=valid, 0=exclude
        """
        # Force float32 — (1-p)^gamma underflows to 0 in float16 → NaN
        p = torch.clamp(y_hat.float(), 1e-7, 1.0 - 1e-7)
        g = y_true.float()

        p_t          = p * g + (1.0 - p) * (1.0 - g)
        alpha        = self.alpha_focal * g + (1.0 - self.alpha_focal) * (1.0 - g)
        focal_weight = alpha * (1.0 - p_t) ** self.gamma_focal
        bce          = -(g * torch.log(p) + (1.0 - g) * torch.log(1.0 - p))
        pixel_loss   = focal_weight * bce

        if valid_mask is not None:
            m    = valid_mask.float()
            n    = m.sum().clamp(min=1.0)
            loss = (pixel_loss * m).sum() / n
        else:
            loss = pixel_loss.mean()

        return loss

    # ── PDE Level-Set Loss ────────────────────────────────────────────────────

    def _pde_loss(
        self,
        y_hat: torch.Tensor,
        y_prev: torch.Tensor,
        r_map: torch.Tensor,
        dx: float = 1000.0,
        dt: float = 86400.0,
    ) -> torch.Tensor:
        """
        Discrete upwind Hamilton-Jacobi level-set PDE loss.

        Enforces: ∂Ψ/∂t + R · ||∇Ψ||_upwind = 0
        near the fire boundary where ∇Ψ is large.

        Logit transform: Ψ̂ = log(ŷ / (1-ŷ)) converts probability to pseudo-SDF.
        Upwind gradient prevents numerical diffusion for hyperbolic PDEs.

        Args:
            y_hat:  (B, 1, H, W) predicted fire probability at T+24h
            y_prev: (B, 1, H, W) previous fire mask (binary) at T
            r_map:  (B, 1, H, W) Rothermel ROS in m/s
            dx:     spatial resolution in metres (1000 m)
            dt:     time step in seconds (86400 s = 24 h)

        Returns:
            scalar PDE residual loss
        """
        # Convert probability to pseudo-SDF via logit
        psi_hat  = torch.log(
            torch.clamp(y_hat,  1e-6, 1.0 - 1e-6) /
            torch.clamp(1.0 - y_hat, 1e-6, 1.0 - 1e-6)
        )                                                         # (B,1,H,W)

        psi_prev = torch.log(
            torch.clamp(y_prev.float() + 1e-6, 1e-6, 1.0 - 1e-6) /
            torch.clamp(1.0 - y_prev.float() - 1e-6, 1e-6, 1.0 - 1e-6)
        )

        # Upwind finite-difference gradient of Ψ̂
        # Backward differences (upwind for outward-propagating front)
        Dm_x = (psi_hat[:, :, :, 1:]  - psi_hat[:, :, :, :-1]) / dx
        Dp_x = (psi_hat[:, :, :, :-1] - psi_hat[:, :, :, 1:])  / dx
        Dm_y = (psi_hat[:, :, 1:, :]  - psi_hat[:, :, :-1, :]) / dx
        Dp_y = (psi_hat[:, :, :-1, :] - psi_hat[:, :, 1:, :])  / dx

        # Pad to restore spatial dims
        Dm_x = F.pad(Dm_x, (1, 0))
        Dp_x = F.pad(Dp_x, (0, 1))
        Dm_y = F.pad(Dm_y, (0, 0, 1, 0))
        Dp_y = F.pad(Dp_y, (0, 0, 0, 1))

        # Godunov upwind scheme
        grad_norm = torch.sqrt(
            torch.clamp(Dm_x, min=0.0) ** 2 +
            torch.clamp(Dp_x, max=0.0) ** 2 +
            torch.clamp(Dm_y, min=0.0) ** 2 +
            torch.clamp(Dp_y, max=0.0) ** 2 +
            1e-8
        )

        # Temporal derivative: (Ψ_t+1 - Ψ_t) / dt
        dpsi_dt = (psi_hat - psi_prev) / dt

        # PDE residual: ∂Ψ/∂t + R·||∇Ψ|| = 0
        R_mps = torch.clamp(r_map, 0.0, 200.0 / 60.0)           # cap at 200 m/min in m/s
        residual = dpsi_dt + R_mps * grad_norm

        # Apply only near fire boundary (where gradient is meaningful)
        boundary_mask = (grad_norm > grad_norm.mean() * 0.1).float()
        n_boundary    = boundary_mask.sum().clamp(min=1.0)
        loss = (residual ** 2 * boundary_mask).sum() / n_boundary
        return loss

    # ── Eikonal Regularisation ────────────────────────────────────────────────

    def _eikonal_loss(self, y_hat: torch.Tensor, dx: float = 1000.0) -> torch.Tensor:
        """
        Eikonal regularisation: enforce ||∇Ψ̂|| ≈ 1 (valid SDF property).

        L_Eik = mean((||∇Ψ̂|| - 1)²)

        Args:
            y_hat: (B, 1, H, W) predicted fire probability
            dx:    spatial resolution in metres

        Returns:
            scalar Eikonal loss
        """
        psi = torch.log(
            torch.clamp(y_hat, 1e-6, 1.0 - 1e-6) /
            torch.clamp(1.0 - y_hat, 1e-6, 1.0 - 1e-6)
        )

        # Central differences for gradient magnitude
        grad_x = (psi[:, :, :, 2:] - psi[:, :, :, :-2]) / (2.0 * dx)
        grad_y = (psi[:, :, 2:, :] - psi[:, :, :-2, :]) / (2.0 * dx)

        # Match spatial dims by cropping
        grad_x = grad_x[:, :, 1:-1, :]
        grad_y = grad_y[:, :, :, 1:-1]

        grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return ((grad_norm - 1.0) ** 2).mean()

    # ── L2 Regularisation (skip Rothermel buffers) ────────────────────────────

    def _reg_loss(self, model: Optional[nn.Module]) -> torch.Tensor:
        """
        L2 regularisation on non-physics parameters.
        Skips RothermelLayer buffers (physical constants should not be penalised).

        Args:
            model: the neural network being trained

        Returns:
            scalar L2 loss, or zero tensor if model is None
        """
        if model is None:
            return torch.tensor(0.0)

        l2 = torch.tensor(0.0, device=next(model.parameters(), torch.tensor(0.0)).device
                          if len(list(model.parameters())) > 0 else torch.device("cpu"))

        for name, param in model.named_parameters():
            # Skip Rothermel buffers and the PGSS alpha (physical gate, not pure ML)
            if "rothermel" not in name.lower() and param.requires_grad:
                l2 = l2 + param.norm(2) ** 2

        return l2

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        y_hat:  torch.Tensor,
        y_true: torch.Tensor,
        y_prev: Optional[torch.Tensor] = None,
        r_map:  Optional[torch.Tensor] = None,
        model:  Optional[nn.Module]    = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Args:
            y_hat:  (B, 1, H, W) predicted fire probability ∈ [0, 1]
            y_true: (B, 1, H, W) ground truth fire mask {0, 1}
            y_prev: (B, 1, H, W) previous fire mask (needed for L_PDE)
            r_map:  (B, 1, H, W) Rothermel ROS in m/s (needed for L_PDE)
            model:  nn.Module (needed for L_Reg)

        Returns:
            dict with keys: total, seg, dice, focal, pde, eikonal, reg
        """
        # Ensure y_true is binary float and same shape
        y_true_bin = (y_true > 0.5).float()

        # valid_mask passed via kwargs — excludes no-data pixels from loss
        valid_mask = kwargs.get("valid_mask", None)

        # Segmentation losses (always computed)
        L_dice  = self._dice_loss(y_hat, y_true_bin, valid_mask)
        L_focal = self._focal_loss(y_hat, y_true_bin, valid_mask)
        L_seg   = L_dice + L_focal

        # PDE losses (skip if lambda=0 — saves compute during warm-up)
        device = y_hat.device
        zero   = torch.tensor(0.0, device=device)

        if self.lambda_pde > 0.0 and y_prev is not None and r_map is not None:
            L_pde = self._pde_loss(y_hat, y_prev, r_map)
            L_eik = self._eikonal_loss(y_hat)
        else:
            L_pde = zero
            L_eik = zero

        # Regularisation
        if self.lambda_reg > 0.0 and model is not None:
            L_reg = self._reg_loss(model)
        else:
            L_reg = zero

        L_total = (L_seg
                   + self.lambda_pde * L_pde
                   + self.lambda_eik * L_eik
                   + self.lambda_reg * L_reg)

        return {
            "total":   L_total,
            "seg":     L_seg,
            "dice":    L_dice,
            "focal":   L_focal,
            "pde":     L_pde,
            "eikonal": L_eik,
            "reg":     L_reg,
        }

    def set_lambda_pde(self, value: float) -> None:
        """Update λ_PDE — called by CurriculumScheduler each epoch."""
        self.lambda_pde = value


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 2 — WildfireMetrics
# ═══════════════════════════════════════════════════════════════════════════════

class WildfireMetrics:
    """
    Evaluation metrics for wildfire spread prediction.

    All inputs must be numpy arrays (call .detach().cpu().numpy() first).
    Works on flattened or spatial arrays.

    Metrics:
        CSI          — Critical Success Index (primary operational metric)
        IoU          — Intersection over Union (same formula as CSI, kept separate)
        Precision    — TP / (TP + FP)
        Recall       — TP / (TP + FN)
        F1           — Harmonic mean of Precision and Recall
        PR_AUC       — Area under Precision-Recall curve
        PCR          — Physical Consistency Rate
        ECE          — Expected Calibration Error (15 bins)
        Frechet_dist — Fréchet Perimeter Distance (directed Hausdorff on contours)
    """

    PIXEL_SIZE_M: float = 1000.0   # 1 km resolution
    N_ECE_BINS:   int   = 15

    def compute(
        self,
        y_hat_prob: np.ndarray,
        y_true:     np.ndarray,
        r_map:      Optional[np.ndarray] = None,
        threshold:  float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute all metrics for one batch or epoch-aggregated predictions.

        Args:
            y_hat_prob: predicted probabilities, any shape, values ∈ [0, 1]
            y_true:     binary ground truth, same shape as y_hat_prob
            r_map:      Rothermel ROS in m/s, same shape (optional)
            threshold:  binarisation threshold for predicted mask

        Returns:
            dict of scalar metric values
        """
        # Flatten to 1D
        prob  = y_hat_prob.flatten().astype(np.float32)
        true  = (y_true.flatten() > 0.5).astype(np.float32)
        pred  = (prob >= threshold).astype(np.float32)

        # Confusion matrix components
        TP = float((pred * true).sum())
        FP = float((pred * (1.0 - true)).sum())
        FN = float(((1.0 - pred) * true).sum())
        TN = float(((1.0 - pred) * (1.0 - true)).sum())

        eps = 1e-8

        csi       = TP / (TP + FP + FN + eps)
        iou       = csi                                   # identical formula
        precision = TP / (TP + FP + eps)
        recall    = TP / (TP + FN + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)

        pr_auc    = self._pr_auc(prob, true)
        pcr       = self._pcr(pred, r_map)
        ece       = self._ece(prob, true)
        frechet   = self._frechet_dist(pred, true, y_hat_prob.shape)

        return {
            "CSI":          csi,
            "IoU":          iou,
            "Precision":    precision,
            "Recall":       recall,
            "F1":           f1,
            "PR_AUC":       pr_auc,
            "PCR":          pcr,
            "ECE":          ece,
            "Frechet_dist": frechet,
        }

    def _pr_auc(self, prob: np.ndarray, true: np.ndarray) -> float:
        """Area under Precision-Recall curve."""
        if not _SCIPY_SKLEARN_AVAILABLE:
            return float("nan")
        if true.sum() == 0:
            return float("nan")
        try:
            return float(average_precision_score(true, prob))
        except Exception:
            return float("nan")

    def _pcr(
        self,
        pred_binary: np.ndarray,
        r_map: Optional[np.ndarray],
    ) -> float:
        """
        Physical Consistency Rate: fraction of pixels where the predicted
        spread distance ≤ R_Rothermel · Δt.

        A pixel 'spreads' if pred=1 and prev=0 (newly ignited).
        Since we don't have per-pixel spread distance in this simplified form,
        we check: R_map_mps * dt / pixel_size ∈ [0, 1] gives max spread
        fraction per pixel.

        Returns 1.0 if r_map is None (no physics available).
        """
        if r_map is None:
            return 1.0

        dt = 86400.0   # seconds in 24 hours
        max_spread_px = (r_map.flatten() * dt / self.PIXEL_SIZE_M)
        predicted_spread = pred_binary.flatten()

        # PCR: predicted spread ≤ physical maximum
        consistent = (predicted_spread <= np.clip(max_spread_px, 0.0, 1.0))
        return float(consistent.mean())

    def _ece(self, prob: np.ndarray, true: np.ndarray) -> float:
        """
        Expected Calibration Error with 15 equal-width confidence bins.

        ECE = Σ_b (|B_b| / N) · |acc(b) - conf(b)|
        """
        n     = len(prob)
        bins  = np.linspace(0.0, 1.0, self.N_ECE_BINS + 1)
        ece   = 0.0

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (prob >= lo) & (prob < hi)
            if mask.sum() == 0:
                continue
            acc  = true[mask].mean()
            conf = prob[mask].mean()
            ece += (mask.sum() / n) * abs(conf - acc)

        return float(ece)

    def _frechet_dist(
        self,
        pred_binary: np.ndarray,
        true_binary: np.ndarray,
        original_shape: tuple,
    ) -> float:
        """
        Fréchet Perimeter Distance: directed Hausdorff distance between
        predicted and ground-truth fire perimeters (contours).

        Operates on 2D spatial arrays. If input is flattened, uses last
        two dims of original_shape to reshape.

        Returns np.nan if either mask has no fire pixels or cv2/scipy unavailable.
        """
        if not _CV2_AVAILABLE or not _SCIPY_SKLEARN_AVAILABLE:
            return float("nan")

        # Reshape to 2D for contour extraction
        try:
            if len(original_shape) == 4:
                H, W = original_shape[-2], original_shape[-1]
            elif len(original_shape) == 2:
                H, W = original_shape
            else:
                H = W = int(math.sqrt(len(pred_binary)))

            pred_2d = pred_binary.reshape(-1, H, W)[0].astype(np.uint8)
            true_2d = true_binary.reshape(-1, H, W)[0].astype(np.uint8)
        except Exception:
            return float("nan")

        if pred_2d.sum() == 0 or true_2d.sum() == 0:
            return float("nan")

        try:
            pred_contours, _ = cv2.findContours(
                pred_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            true_contours, _ = cv2.findContours(
                true_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not pred_contours or not true_contours:
                return float("nan")

            # Stack all contour points
            pred_pts = np.vstack([c.reshape(-1, 2) for c in pred_contours])
            true_pts = np.vstack([c.reshape(-1, 2) for c in true_contours])

            # Directed Hausdorff (symmetric: take max of both directions)
            d1 = directed_hausdorff(pred_pts, true_pts)[0]
            d2 = directed_hausdorff(true_pts, pred_pts)[0]
            return float(max(d1, d2))
        except Exception:
            return float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 3 — CurriculumScheduler
# ═══════════════════════════════════════════════════════════════════════════════

class CurriculumScheduler:
    """
    3-phase curriculum for λ_PDE:

    Phase 1 (epochs 1–30):   λ_PDE = 0          (learn pure segmentation first)
    Phase 2 (epochs 31–80):  λ_PDE = linear ramp 0 → λ_max
    Phase 3 (epochs 81–100): λ_PDE = λ_max      (full physics constraint)

    Rationale: abruptly enabling the PDE loss at epoch 1 causes gradient
    conflict with the segmentation loss before the model has learned basic
    fire shape. The warm-up lets the model first learn plausible outputs,
    then physics refines the fire front geometry.

    Args:
        lambda_max: maximum λ_PDE value
        phase1_end: last epoch of pure segmentation (default 30)
        phase2_end: last epoch of ramp phase (default 80)
    """

    def __init__(
        self,
        lambda_max:  float = 1.0,
        phase1_end:  int   = 30,
        phase2_end:  int   = 80,
    ) -> None:
        self.lambda_max  = lambda_max
        self.phase1_end  = phase1_end
        self.phase2_end  = phase2_end
        self.current_lambda = 0.0

    def step(self, epoch: int) -> float:
        """
        Compute λ_PDE for the given epoch and store it internally.

        Args:
            epoch: current epoch (1-indexed)

        Returns:
            current λ_PDE value
        """
        if epoch <= self.phase1_end:
            lam = 0.0
        elif epoch <= self.phase2_end:
            # Linear ramp from 0 to lambda_max
            progress = (epoch - self.phase1_end) / (self.phase2_end - self.phase1_end)
            lam = self.lambda_max * progress
        else:
            lam = self.lambda_max

        self.current_lambda = lam
        return lam

    def state_dict(self) -> Dict:
        return {
            "lambda_max":      self.lambda_max,
            "phase1_end":      self.phase1_end,
            "phase2_end":      self.phase2_end,
            "current_lambda":  self.current_lambda,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.lambda_max     = state["lambda_max"]
        self.phase1_end     = state["phase1_end"]
        self.phase2_end     = state["phase2_end"]
        self.current_lambda = state["current_lambda"]


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 4 — Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Full training loop with checkpoint safety and WandB logging.

    Handles:
        • Mixed precision (fp16) via torch.cuda.amp
        • Gradient clipping
        • Curriculum scheduler (λ_PDE ramp)
        • Checkpoint every N epochs + best model by CSI
        • Auto-resume from latest checkpoint on restart
        • KeyboardInterrupt → emergency checkpoint

    Expected dataloader batch format:
        {'inputs': (B,12,H,W), 'targets': (B,1,H,W), 'prev_fire': (B,1,H,W)}
        'prev_fire' and 'r_map' are optional — if absent, L_PDE = 0.

    Args:
        model:          nn.Module to train
        loss_fn:        WildfireLoss instance
        optimizer:      torch.optim.Optimizer
        scheduler:      torch.optim.lr_scheduler (or None)
        curriculum:     CurriculumScheduler instance
        config:         TrainConfig dataclass
        checkpoint_dir: directory for saving checkpoints
    """

    def __init__(
        self,
        model:          nn.Module,
        loss_fn:        WildfireLoss,
        optimizer:      torch.optim.Optimizer,
        scheduler,                              # LR scheduler, any type
        curriculum:     CurriculumScheduler,
        config:         TrainConfig,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        self.model          = model
        self.loss_fn        = loss_fn
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.curriculum     = curriculum
        self.config         = config
        self.checkpoint_dir = checkpoint_dir or config.checkpoint_dir

        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model          = self.model.to(self.device)
        self.scaler         = torch.amp.GradScaler('cuda', enabled=config.mixed_precision)
        self.metrics_fn     = WildfireMetrics()

        # Training state
        self.start_epoch    = 1
        self.best_csi       = 0.0
        self.history: Dict[str, List] = {
            "train_loss": [], "val_csi": [], "val_iou": [],
            "val_ece": [], "lambda_pde": [], "lr": [],
        }

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # WandB initialisation
        self._init_wandb()

    # ── WandB ─────────────────────────────────────────────────────────────────

    def _init_wandb(self) -> None:
        """Initialise WandB with online/offline fallback."""
        self._wandb_enabled = False
        if not _WANDB_AVAILABLE:
            return
        try:
            run_name = self.config.wandb_run_name or self.config.model_name
            wandb.init(
                project = self.config.wandb_project,
                name    = run_name,
                config  = self.config.__dict__,
                mode    = "online",
                resume  = "allow",
            )
            self._wandb_enabled = True
            print("✅ WandB online logging enabled")
        except Exception:
            try:
                wandb.init(
                    project = self.config.wandb_project,
                    name    = self.config.wandb_run_name or self.config.model_name,
                    config  = self.config.__dict__,
                    mode    = "offline",
                    dir     = "./wandb",
                )
                self._wandb_enabled = True
                print("⚠️  WandB offline mode — logs saved to ./wandb/")
            except Exception as e:
                print(f"⚠️  WandB disabled: {e}")

    def _log(self, metrics: Dict, step: int) -> None:
        """Log metrics to WandB if enabled."""
        if self._wandb_enabled:
            try:
                wandb.log(metrics, step=step)
            except Exception:
                pass

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, tag: str = "") -> str:
        """
        Save full training state to disk.

        Args:
            epoch: current epoch number
            tag:   optional tag appended to filename (e.g. "best", "emergency")

        Returns:
            path of saved checkpoint
        """
        fname = f"checkpoint_epoch_{epoch}{('_' + tag) if tag else ''}.pt"
        path  = os.path.join(self.checkpoint_dir, fname)

        torch.save({
            "epoch":        epoch,
            "model_state":  self.model.state_dict(),
            "optim_state":  self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_csi":     self.best_csi,
            "history":      self.history,
            "curriculum":   self.curriculum.state_dict(),
            "config":       self.config.__dict__,
        }, path)
        return path

    def load_checkpoint(self, path: Optional[str] = None) -> int:
        """
        Load checkpoint and resume training state.
        If path is None, auto-detects latest checkpoint in checkpoint_dir.

        Args:
            path: explicit checkpoint path, or None for auto-detect

        Returns:
            epoch to resume from (start_epoch + 1)
        """
        if path is None:
            # Find latest epoch checkpoint
            pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt")
            files   = glob.glob(pattern)
            # Filter out tagged files (best, emergency) for epoch detection
            epoch_files = [
                f for f in files
                if f.replace(".pt", "").split("_")[-1].isdigit()
            ]
            if not epoch_files:
                print("No checkpoint found — starting from epoch 1")
                return 1
            # Sort by epoch number
            epoch_files.sort(key=lambda f: int(
                os.path.basename(f).replace("checkpoint_epoch_", "").replace(".pt", "")
            ))
            path = epoch_files[-1]

        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_csi   = ckpt.get("best_csi", 0.0)
        self.history    = ckpt.get("history", self.history)
        self.curriculum.load_state_dict(ckpt.get("curriculum", self.curriculum.state_dict()))

        epoch = ckpt["epoch"]
        print(f"  Resumed from epoch {epoch}, best CSI = {self.best_csi:.4f}")
        self.start_epoch = epoch + 1
        return self.start_epoch

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run one full training epoch.

        Args:
            dataloader: yields dicts with keys 'inputs', 'targets',
                        optionally 'prev_fire' and 'r_map'

        Returns:
            dict of mean loss components for this epoch
        """
        self.model.train()
        totals: Dict[str, float] = {
            "total": 0.0, "seg": 0.0, "dice": 0.0,
            "focal": 0.0, "pde": 0.0, "eikonal": 0.0, "reg": 0.0,
        }
        grad_norms: List[float] = []
        n_batches = 0

        for batch in dataloader:
            inputs     = batch["inputs"].to(self.device)
            targets    = batch["targets"].to(self.device)
            valid_mask = batch.get("valid_mask")
            y_prev     = batch.get("prev_fire")
            r_map      = batch.get("r_map")

            if valid_mask is not None: valid_mask = valid_mask.to(self.device)
            if y_prev is not None:     y_prev     = y_prev.to(self.device)
            if r_map is not None:      r_map      = r_map.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.config.mixed_precision):
                y_hat  = self.model(inputs)
                losses = self.loss_fn(
                    y_hat, targets,
                    y_prev=y_prev, r_map=r_map, model=self.model,
                    valid_mask=valid_mask,
                )

            # NaN/Inf guard — fail fast unless explicitly allowed
            if not torch.isfinite(losses["total"]):
                if self.config.fail_on_nonfinite:
                    raise ValueError("Non-finite loss detected during training.")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(losses["total"]).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            grad_norms.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in losses.items():
                totals[k] += v.item()
            n_batches += 1

        # Compute means
        means = {k: v / max(n_batches, 1) for k, v in totals.items()}
        means["grad_norm"] = float(np.mean(grad_norms)) if grad_norms else 0.0
        return means

    # ── Validation epoch ──────────────────────────────────────────────────────

    def val_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run one full validation epoch and compute all metrics.

        Args:
            dataloader: same format as train dataloader

        Returns:
            dict of mean metric values for this epoch
        """
        self.model.eval()

        all_probs:  List[np.ndarray] = []
        all_trues:  List[np.ndarray] = []
        all_rmaps:  List[np.ndarray] = []
        val_losses: Dict[str, float] = {
            "total": 0.0, "seg": 0.0, "dice": 0.0, "focal": 0.0,
        }
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs     = batch["inputs"].to(self.device)
                targets    = batch["targets"].to(self.device)
                valid_mask = batch.get("valid_mask")
                y_prev     = batch.get("prev_fire")
                r_map      = batch.get("r_map")

                if valid_mask is not None: valid_mask = valid_mask.to(self.device)
                if y_prev is not None:     y_prev     = y_prev.to(self.device)
                if r_map is not None:      r_map      = r_map.to(self.device)

                with autocast(enabled=self.config.mixed_precision):
                    y_hat  = self.model(inputs)
                    losses = self.loss_fn(
                        y_hat, targets,
                        y_prev=y_prev, r_map=r_map,
                        valid_mask=valid_mask,
                    )

                for k in val_losses:
                    val_losses[k] += losses[k].item()

                # Collect for metric computation — only valid pixels
                vm = valid_mask.cpu().float().numpy() if valid_mask is not None else None
                probs_np = y_hat.cpu().float().numpy()
                tgt_np   = targets.cpu().float().numpy()
                if vm is not None:
                    # Mask out no-data pixels by setting them to 0
                    probs_np = probs_np * vm
                    tgt_np   = tgt_np * vm
                all_probs.append(probs_np)
                all_trues.append(tgt_np)
                if r_map is not None:
                    all_rmaps.append(r_map.cpu().float().numpy())
                n_batches += 1

        # Concatenate all batches
        probs = np.concatenate(all_probs, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        rmaps = np.concatenate(all_rmaps, axis=0) if all_rmaps else None

        metrics = self.metrics_fn.compute(probs, trues, r_map=rmaps)
        metrics.update({f"val_{k}": v / max(n_batches, 1)
                        for k, v in val_losses.items()})
        return metrics

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def fit(
        self,
        train_dl: DataLoader,
        val_dl:   DataLoader,
        n_epochs: Optional[int] = None,
    ) -> Dict[str, List]:
        """
        Full training loop with curriculum, checkpointing, and logging.

        Args:
            train_dl: training dataloader
            val_dl:   validation dataloader
            n_epochs: number of epochs (defaults to config.n_epochs)

        Returns:
            training history dict
        """
        n_epochs = n_epochs or self.config.n_epochs

        print(f"\n{'='*60}")
        print(f"Training: {self.config.model_name}")
        print(f"Device  : {self.device}")
        print(f"Epochs  : {self.start_epoch} → {n_epochs}")
        print(f"Ckpt dir: {self.checkpoint_dir}")
        print(f"{'='*60}\n")

        try:
            for epoch in range(self.start_epoch, n_epochs + 1):
                t0 = time.time()

                # ── Curriculum: update λ_PDE ──────────────────────────────
                lam_pde = self.curriculum.step(epoch)
                self.loss_fn.set_lambda_pde(lam_pde)

                # ── Train ─────────────────────────────────────────────────
                train_metrics = self.train_epoch(train_dl)

                # ── LR scheduler step ─────────────────────────────────────
                if self.scheduler is not None:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                # ── Validate ──────────────────────────────────────────────
                val_metrics = {}
                if epoch % self.config.val_every == 0:
                    val_metrics = self.val_epoch(val_dl)

                epoch_time = time.time() - t0

                # ── History ───────────────────────────────────────────────
                self.history["train_loss"].append(train_metrics["total"])
                self.history["val_csi"].append(val_metrics.get("CSI", float("nan")))
                self.history["val_iou"].append(val_metrics.get("IoU", float("nan")))
                self.history["val_ece"].append(val_metrics.get("ECE", float("nan")))
                self.history["lambda_pde"].append(lam_pde)
                self.history["lr"].append(current_lr)

                # ── Print ─────────────────────────────────────────────────
                csi_str = f"{val_metrics.get('CSI', float('nan')):.4f}" \
                          if val_metrics else "—"
                print(
                    f"Epoch {epoch:3d}/{n_epochs} | "
                    f"Loss={train_metrics['total']:.4f} "
                    f"(dice={train_metrics['dice']:.3f} "
                    f"focal={train_metrics['focal']:.3f} "
                    f"pde={train_metrics['pde']:.3f}) | "
                    f"CSI={csi_str} | "
                    f"λ_pde={lam_pde:.3f} | "
                    f"LR={current_lr:.1e} | "
                    f"t={epoch_time:.1f}s"
                )

                # ── WandB logging ─────────────────────────────────────────
                log_dict = {
                    "epoch":       epoch,
                    "lambda_pde":  lam_pde,
                    "lr":          current_lr,
                    "epoch_time":  epoch_time,
                }
                log_dict.update({f"train/{k}": v for k, v in train_metrics.items()})
                log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                self._log(log_dict, step=epoch)

                # ── Save best model ───────────────────────────────────────
                current_csi = val_metrics.get("CSI", 0.0)
                if current_csi > self.best_csi and not math.isnan(current_csi):
                    self.best_csi = current_csi
                    path = self.save_checkpoint(epoch, tag="best")
                    # Also save a fixed-name "best_model.pt" for easy access
                    best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                    import shutil
                    shutil.copy(path, best_path)
                    print(f"  ★ New best CSI={self.best_csi:.4f} → {best_path}")

                # ── Save periodic checkpoint ──────────────────────────────
                if epoch % self.config.save_every == 0:
                    path = self.save_checkpoint(epoch)
                    print(f"  ✓ Checkpoint saved: {path}")

                if self.config.log_cuda_memory and torch.cuda.is_available():
                    print(torch.cuda.memory_summary(abbreviated=True))

        except KeyboardInterrupt:
            print("\n⚠️  KeyboardInterrupt — saving emergency checkpoint...")
            path = self.save_checkpoint(epoch, tag="emergency")
            print(f"  Emergency checkpoint saved: {path}")
            print("  Resume by calling trainer.load_checkpoint() and trainer.fit()")

        finally:
            if self._wandb_enabled:
                try:
                    wandb.finish()
                except Exception:
                    pass

        print(f"\n✅ Training complete. Best CSI = {self.best_csi:.4f}")
        return self.history
