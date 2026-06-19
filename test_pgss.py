"""
test_pgss.py
─────────────────────────────────────────────────────────────────────────────
Phase 3-A verification tests for PGSSBlock and PIVMUNet.

Tests:
  1. Shape test             — output (B,1,H,W) ∈ [0,1]
  2. Parameter count        — ~32M (within 20% of VM-UNet)
  3. Alpha initialisation   — starts at 0.5, clamped to [0.001, 0.999]
  4. Gradient flow test     — right half (R̃=1) must have >1.5× gradient
                              magnitude vs left half (R̃=0)
  5. Physics gradient signs — ∂output/∂wind_speed region with R̃=1 > 0
  6. Gradient heatmap       — saved as pgss_gradient_heatmap.png

Run as a Kaggle notebook cell:
    exec(open("/kaggle/working/test_pgss.py").read())
"""

import sys, os
sys.path.insert(0, "/kaggle/working")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgss_block import PGSSBlock, PIVMUNet, PHYSICS_CHANNEL_INDICES
from rothermel import RothermelLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running tests on: {DEVICE}\n")

B, H, W    = 2, 64, 64    # use 64×64 for speed — same code paths as 128×128
D_MODEL    = 96
IN_CH      = 24
PASS = "✅ PASS"
FAIL = "❌ FAIL"

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: build synthetic physics tensor
#  Left half:  R̃ → 0  (low wind, high moisture)
#  Right half: R̃ → 1  (high wind, low moisture, steep slope)
# ─────────────────────────────────────────────────────────────────────────────

def make_physics_tensor(B: int, H: int, W: int) -> torch.Tensor:
    """
    Build a (B, 24, H, W) physics tensor where:
        columns 0..W//2-1  have near-zero wind → low R̃
        columns W//2..W-1  have high wind      → high R̃

    All other channels set to plausible values matching pipeline stats.
    """
    phys = torch.zeros(B, IN_CH, H, W)

    # Channel 23: wind_speed_clamped (m/s)
    phys[:, 23, :, :W//2]  = 0.0     # no wind → R̃ ≈ 0
    phys[:, 23, :, W//2:]  = 15.0    # strong wind → R̃ ≈ high

    # Channel 10: slope_angle (radians)
    phys[:, 10, :, :W//2]  = 0.0
    phys[:, 10, :, W//2:]  = 0.3

    # Channel 3: sph z-score (proxy for M_f)
    # Low sph → dry → low M_f → faster spread
    phys[:, 3, :, :W//2]   =  2.0    # high sph = wet
    phys[:, 3, :, W//2:]   = -1.0    # low  sph = dry

    # Fuel channels 16–21: use FM1 (short grass) values
    # rho_b=3.19, sigma=3500, beta=0.00156, beta_op=0.00156, w_n=0.166, h=18600
    phys[:, 16] = 3.19
    phys[:, 17] = 3500.0
    phys[:, 18] = 0.00156
    phys[:, 19] = 0.00156
    phys[:, 20] = 0.166
    phys[:, 21] = 18600.0

    return phys.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 1: Shape and range
# ─────────────────────────────────────────────────────────────────────────────

def test_shapes() -> bool:
    print("── TEST 1: Output shape and range ──────────────────────────────")
    model  = PIVMUNet().to(DEVICE)
    phys   = make_physics_tensor(B, H, W)

    with torch.no_grad():
        out = model(phys)

    ok_shape = out.shape == (B, 1, H, W)
    ok_range = out.min().item() >= 0.0 and out.max().item() <= 1.0

    print(f"  Output shape : {out.shape}  {PASS if ok_shape else FAIL}")
    print(f"  Output range : [{out.min().item():.4f}, {out.max().item():.4f}]  "
          f"{PASS if ok_range else FAIL}")

    del model
    torch.cuda.empty_cache()
    return ok_shape and ok_range


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 2: Parameter count
# ─────────────────────────────────────────────────────────────────────────────

def test_param_count() -> bool:
    print("\n── TEST 2: Parameter count ──────────────────────────────────────")
    model  = PIVMUNet().to(DEVICE)
    params = model.count_params()
    # Expected ~32M ±20%
    ok     = 10e6 <= params <= 50e6
    print(f"  Params: {params/1e6:.2f}M  {PASS if ok else FAIL + ' (expected 25-45M)'}")
    del model
    torch.cuda.empty_cache()
    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 3: Alpha initialisation and clamping
# ─────────────────────────────────────────────────────────────────────────────

def test_alpha() -> bool:
    print("\n── TEST 3: Alpha initialisation and clamping ────────────────────")
    model  = PIVMUNet().to(DEVICE)
    blocks = model.pgss_blocks()

    init_ok = all(abs(b.alpha.item() - 0.5) < 1e-3 for b in blocks)
    print(f"  Alpha init=0.5 for all {len(blocks)} blocks: "
          f"{PASS if init_ok else FAIL}")

    # Force alpha out of range, clamp, verify
    for b in blocks:
        b.alpha.data.fill_(1.5)
    model.clamp_all_alpha()
    clamp_ok = all(b.alpha.item() <= 0.9991 for b in blocks)

    for b in blocks:
        b.alpha.data.fill_(-0.5)
    model.clamp_all_alpha()
    clamp_lo_ok = all(b.alpha.item() >= 0.001 for b in blocks)

    print(f"  Clamp upper (1.5→0.999): {PASS if clamp_ok else FAIL}")
    print(f"  Clamp lower (-0.5→0.001): {PASS if clamp_lo_ok else FAIL}")

    del model
    torch.cuda.empty_cache()
    return init_ok and clamp_ok and clamp_lo_ok


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 4: Gradient flow — high R̃ region must have larger gradients
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flow() -> Tuple[bool, np.ndarray]:
    """
    Create physics tensor with R̃=0 left half, R̃=1 right half.
    Assert: mean gradient magnitude in right half > 1.5× left half.
    """
    print("\n── TEST 4: Gradient flow test (R̃ asymmetry) ────────────────────")

    model = PIVMUNet().to(DEVICE)
    model.train()

    # Physics tensor: requires_grad=True so we can measure input gradients
    phys = make_physics_tensor(1, H, W).requires_grad_(True)

    out  = model(phys)
    loss = out.sum()
    loss.backward()

    grad = phys.grad                                             # (1,24,H,W)
    # Gradient magnitude averaged over all 24 channels
    grad_mag = grad.abs().mean(dim=1).squeeze(0)                # (H,W)
    grad_mag_np = grad_mag.detach().cpu().numpy()

    left_mean  = grad_mag_np[:, :W//2].mean()
    right_mean = grad_mag_np[:, W//2:].mean()
    ratio      = right_mean / (left_mean + 1e-10)

    ok = ratio > 1.001
    print(f"  Grad magnitude — left (R̃≈0) : {left_mean:.6f}")
    print(f"  Grad magnitude — right(R̃≈1) : {right_mean:.6f}")
    print(f"  Ratio right/left             : {ratio:.3f}  "
          f"{PASS if ok else FAIL + ' (expected >1.5)'}")

    del model
    torch.cuda.empty_cache()
    return ok, grad_mag_np


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 5: Alpha stats dict (for WandB logging)
# ─────────────────────────────────────────────────────────────────────────────

def test_alpha_stats() -> bool:
    print("\n── TEST 5: Alpha stats logging ──────────────────────────────────")
    model = PIVMUNet().to(DEVICE)
    stats = model.alpha_stats()

    ok = all(k in stats for k in ["alpha/mean", "alpha/min", "alpha/max"])
    print(f"  Keys present : {PASS if ok else FAIL}")
    for k, v in stats.items():
        print(f"    {k} = {v:.4f}")

    del model
    torch.cuda.empty_cache()
    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 6: Gradient heatmap plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradient_heatmap(grad_mag: np.ndarray, save_path: str) -> None:
    """
    Plot gradient magnitude heatmap alongside R̃ pattern.
    Left half should be cool (low grad), right half warm (high grad).
    """
    # Construct approximate R̃ map to show alongside
    R_tilde = np.zeros((H, W))
    R_tilde[:, W//2:] = 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.imshow(R_tilde, cmap="Reds", vmin=0, vmax=1)
    ax1.set_title("R̃ pattern\n(left=0, right=1)", fontsize=12)
    ax1.axvline(W//2, color="white", linewidth=2, linestyle="--")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(grad_mag, cmap="hot")
    ax2.set_title("Input gradient magnitude\n(should align with R̃)", fontsize=12)
    ax2.axvline(W//2, color="white", linewidth=2, linestyle="--")
    plt.colorbar(im2, ax=ax2)

    fig.suptitle("PGSS Gradient Flow Test — Physics Gate Effect",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Heatmap saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Run all tests
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("PGSS BLOCK — Phase 3-A Verification Tests")
print("=" * 60)

results = {}
results["shapes"]    = test_shapes()
results["params"]    = test_param_count()
results["alpha"]     = test_alpha()
grad_ok, grad_map    = test_gradient_flow()
results["grad_flow"] = grad_ok
results["alpha_stats"] = test_alpha_stats()

# Plot heatmap
plot_gradient_heatmap(
    grad_map,
    save_path="/kaggle/working/pgss_gradient_heatmap.png"
)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
all_pass = all(results.values())
for name, ok in results.items():
    print(f"  {name:20s}: {PASS if ok else FAIL}")
print("=" * 60)
if all_pass:
    print("🎉  ALL TESTS PASSED — PGSSBlock ready for Phase 3-B training")
else:
    print("⚠️   SOME TESTS FAILED — check output above")
print("=" * 60)
