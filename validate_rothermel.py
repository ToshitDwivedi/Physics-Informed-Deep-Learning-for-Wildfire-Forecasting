"""
validate_rothermel.py
─────────────────────────────────────────────────────────────────────────────
Validation of RothermelLayer against FARSITE reference values and
gradient sign checks.

Run as a Kaggle notebook cell block — all outputs printed to stdout.

FARSITE reference values (Andrews 2018, RMRS-GTR-371):
  FM1 (short grass):  wind=5mph,  slope=0%,  M_f=8%  → R ≈ 2–5   m/min
  FM2 (timber grass): wind=5mph,  slope=0%,  M_f=8%  → R ≈ 3–8   m/min
  FM4 (chaparral):    wind=10mph, slope=20%, M_f=6%  → R ≈ 15–35 m/min
Tolerance: ±20% of the reference midpoint.
"""

import sys, os
sys.path.insert(0, "/kaggle/working")

import torch
import math

from rothermel import RothermelLayer
from transforms import FuelModelEncoder

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mph_to_mps(mph: float) -> float:
    """Miles per hour → metres per second."""
    return mph * 0.44704

def pct_slope_to_rad(pct: float) -> float:
    """Percent slope → radians.  20% slope = arctan(0.20)."""
    return math.atan(pct / 100.0)

def moisture_frac(pct: float) -> float:
    """Moisture percentage → fraction."""
    return pct / 100.0

def make_physics_dict(
    fuel_model_code: int,
    wind_mps: float,
    slope_rad: float,
    M_f: float,
    H: int = 4,
    W: int = 4,
) -> dict[str, torch.Tensor]:
    """
    Build a minimal physics_dict for a uniform (1, 1, H, W) patch.
    Fuel constants come from FuelModelEncoder lookup table.
    All tensors require_grad=True for gradient checks.
    """
    encoder = FuelModelEncoder()

    code = torch.ones(1, 1, H, W, dtype=torch.long) * fuel_model_code
    fuel = encoder(code)   # (1, 6, H, W): rho_b, sigma, beta, beta_op, w_n, h

    def scalar_field(val: float, grad: bool = False) -> torch.Tensor:
        t = torch.ones(1, 1, H, W) * val
        if grad:
            t = t.requires_grad_(True)
        return t

    U           = scalar_field(wind_mps,  grad=True)
    slope_angle = scalar_field(slope_rad, grad=True)
    M_f_tensor  = scalar_field(M_f,       grad=True)

    return {
        "wind_speed":   U,
        "slope_angle":  slope_angle,
        "M_f":          M_f_tensor,
        "rho_b":        fuel[:, 0:1],
        "sigma":        fuel[:, 1:2],
        "beta":         fuel[:, 2:3],
        "beta_op":      fuel[:, 3:4],
        "w_n":          fuel[:, 4:5],
        "h":            fuel[:, 5:6],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FARSITE validation
# ─────────────────────────────────────────────────────────────────────────────

FARSITE_CASES = [
    {
        "name":         "FM1 — Short Grass",
        "fuel_code":    1,
        "wind_mph":     5.0,
        "slope_pct":    0.0,
        "moisture_pct": 8.0,
        "R_min":        2.0,   # m/min lower bound
        "R_max":        5.0,   # m/min upper bound
    },
    {
        "name":         "FM2 — Timber Grass/Understory",
        "fuel_code":    2,
        "wind_mph":     5.0,
        "slope_pct":    0.0,
        "moisture_pct": 8.0,
        "R_min":        3.0,
        "R_max":        8.0,
    },
    {
        "name":         "FM4 — Chaparral",
        "fuel_code":    4,
        "wind_mph":     10.0,
        "slope_pct":    20.0,
        "moisture_pct": 6.0,
        "R_min":        15.0,
        "R_max":        35.0,
    },
]

def run_farsite_validation() -> bool:
    """
    Run all FARSITE test cases. Return True if all pass within ±20% tolerance.
    """
    model = RothermelLayer()
    model.eval()

    print("=" * 65)
    print("FARSITE VALIDATION — RothermelLayer vs. Andrews (2018)")
    print("=" * 65)

    all_pass = True
    for case in FARSITE_CASES:
        pdict = make_physics_dict(
            fuel_model_code = case["fuel_code"],
            wind_mps        = mph_to_mps(case["wind_mph"]),
            slope_rad       = pct_slope_to_rad(case["slope_pct"]),
            M_f             = moisture_frac(case["moisture_pct"]),
        )

        with torch.no_grad():
            out = model(pdict)

        R = out["R_mpm"].mean().item()
        lo, hi = case["R_min"], case["R_max"]
        # Apply ±20% tolerance window around the reference range
        lo_tol = lo * 0.80
        hi_tol = hi * 1.20
        passed = lo_tol <= R <= hi_tol

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n  {case['name']}")
        print(f"    Input : wind={case['wind_mph']} mph, "
              f"slope={case['slope_pct']}%, "
              f"moisture={case['moisture_pct']}%")
        print(f"    Expected: {lo}–{hi} m/min  (±20% → {lo_tol:.1f}–{hi_tol:.1f})")
        print(f"    Got     : {R:.3f} m/min")
        print(f"    {status}")

        if not passed:
            all_pass = False

        # Print intermediates for diagnostics
        print(f"    Intermediates:")
        print(f"      I_R     = {out['_I_R'].mean().item():.4f}")
        print(f"      xi      = {out['_xi'].mean().item():.6f}")
        print(f"      phi_w   = {out['_phi_w'].mean().item():.4f}")
        print(f"      phi_s   = {out['_phi_s'].mean().item():.4f}")
        print(f"      eta_M   = {out['_eta_M'].mean().item():.4f}")
        print(f"      eta_s   = {out['_eta_s'].mean().item():.4f}")
        print(f"      epsilon = {out['_epsilon'].mean().item():.6f}")
        print(f"      Q_ig    = {out['_Q_ig'].mean().item():.2f} kJ/kg")

    print("\n" + "=" * 65)
    if all_pass:
        print("✅  ALL FARSITE CASES PASSED")
    else:
        print("⚠️   SOME CASES OUTSIDE TOLERANCE — review intermediates above")
    print("=" * 65)
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  Gradient sign checks
# ─────────────────────────────────────────────────────────────────────────────

def run_gradient_checks() -> bool:
    """
    Verify gradient signs match physical expectations:
      ∂R/∂U     > 0  (more wind → faster spread)
      ∂R/∂M_f   < 0  (more moisture → slower spread)
      ∂R/∂slope > 0  (steeper slope → faster uphill spread)
    """
    model = RothermelLayer()

    print("\n" + "=" * 65)
    print("GRADIENT SIGN CHECKS")
    print("=" * 65)

    # Use FM1 at moderate conditions for gradient check
    pdict = make_physics_dict(
        fuel_model_code = 1,
        wind_mps        = mph_to_mps(5.0),
        slope_rad       = pct_slope_to_rad(10.0),
        M_f             = moisture_frac(8.0),
    )

    out   = model(pdict)
    R_sum = out["R_mpm"].sum()
    R_sum.backward()

    U_grad     = pdict["wind_speed"].grad.mean().item()
    Mf_grad    = pdict["M_f"].grad.mean().item()
    slope_grad = pdict["slope_angle"].grad.mean().item()

    checks = [
        ("∂R/∂U     > 0  (wind speeds spread)",    U_grad     > 0, U_grad),
        ("∂R/∂M_f   < 0  (moisture slows spread)", Mf_grad    < 0, Mf_grad),
        ("∂R/∂slope > 0  (slope speeds spread)",   slope_grad > 0, slope_grad),
    ]

    all_pass = True
    for desc, passed, val in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {desc}   (grad = {val:.6f})")
        if not passed:
            all_pass = False

    print("=" * 65)
    if all_pass:
        print("✅  ALL GRADIENT SIGNS CORRECT")
    else:
        print("❌  GRADIENT SIGN FAILURE — physics coupling is broken")
    print("=" * 65)
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  PGSS normalisation check
# ─────────────────────────────────────────────────────────────────────────────

def run_pgss_check() -> bool:
    """
    Verify R_norm ∈ [0, 1] and that it changes with wind (not constant).
    """
    model = RothermelLayer()
    print("\n" + "=" * 65)
    print("PGSS GATE CHECK  (R_norm for Mamba Δ modulation)")
    print("=" * 65)

    results = []
    for wind_mph in [0.0, 5.0, 15.0, 30.0]:
        pdict = make_physics_dict(
            fuel_model_code = 1,
            wind_mps        = mph_to_mps(wind_mph),
            slope_rad       = 0.0,
            M_f             = moisture_frac(8.0),
        )
        with torch.no_grad():
            R_norm = model.normalise_for_pgss(pdict).mean().item()
        results.append(R_norm)
        print(f"  wind={wind_mph:5.1f} mph  →  R_norm = {R_norm:.4f}")

    in_range = all(0.0 <= r <= 1.0 for r in results)
    monotone = all(results[i] <= results[i+1] for i in range(len(results)-1))

    print(f"\n  Range check  [0,1]: {'✅ PASS' if in_range else '❌ FAIL'}")
    print(f"  Monotone in wind:   {'✅ PASS' if monotone else '❌ FAIL'}")
    print("=" * 65)
    return in_range and monotone


# ─────────────────────────────────────────────────────────────────────────────
#  Run all checks
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__" or True:   # runs in both script and notebook cell
    farsite_ok  = run_farsite_validation()
    grad_ok     = run_gradient_checks()
    pgss_ok     = run_pgss_check()

    print("\n" + "═" * 65)
    overall = farsite_ok and grad_ok and pgss_ok
    if overall:
        print("🎉  ALL CHECKS PASSED — RothermelLayer ready for PGSS + L_PDE")
    else:
        print("⚠️   SOME CHECKS FAILED — see details above")
    print("═" * 65)
