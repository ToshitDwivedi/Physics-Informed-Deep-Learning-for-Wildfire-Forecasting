"""
rothermel.py
─────────────────────────────────────────────────────────────────────────────
Differentiable Rothermel (1972) surface fire spread model as a PyTorch
nn.Module. Used as the physics engine in:
  • PGSS  — Physics-Gated Selective Scan (gates Mamba's Δ parameter)
  • L_PDE — Level Set PDE loss (provides R for the Hamilton-Jacobi term)

All operations are pure PyTorch — no NumPy in forward() — so gradients
flow through every equation back to wind speed, moisture, slope, and
fuel constants.

Reference:
  Rothermel, R.C. (1972). A mathematical model for predicting fire spread
  in wildland fuels. USDA Forest Service Research Paper INT-115.

Input expected from FullPreprocessingPipeline.get_physics_dict():
  physics_dict keys (all tensors (B, 1, H, W)):
    wind_speed   — m/s  (clamped to [0, 40] by pipeline)
    slope_angle  — radians
    M_f          — fuel moisture fraction [0, 0.4]
    rho_b        — bulk density kg/m³
    sigma        — surface-area-to-volume ratio 1/ft
    beta         — packing ratio
    beta_op      — optimum packing ratio
    w_n          — net fuel load kg/m²
    h            — heat content kJ/kg

Physical unit notes:
  • sigma is kept in 1/ft throughout (Rothermel's original unit system)
  • U (wind) is internally converted m/s → ft/min for Rothermel equations
  • R is output in m/min, m/s, and normalised [0,1]
  • Moisture of extinction M_x = 0.30 (fixed, typical grass/brush fuels)
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class RothermelLayer(nn.Module):
    """
    Rothermel (1972) surface fire Rate of Spread — fully differentiable.

    No learnable parameters. All coefficients are physical constants
    registered as scalar buffers so they move with .to(device) automatically.

    Design decisions:
      • All intermediate tensors keep their (B, 1, H, W) shape throughout,
        so the module is spatially aware and vectorised over the batch.
      • Clamping is applied to physically impossible inputs BEFORE any
        nonlinear operation to prevent NaN/Inf in forward AND in gradients.
      • epsilon (fine-particle fraction) uses the Rothermel formula
        epsilon = exp(-138/sigma) which is more numerically stable than
        the polynomial form for large sigma.
    """

    # ── Physical constants ────────────────────────────────────────────────────
    M_X: float = 0.30          # moisture of extinction (fraction) — grass/brush
    R_MAX: float = 200.0       # physical upper bound on ROS (m/min)

    # Unit conversion: m/s → ft/min  (1 m = 3.281 ft, 1 min = 60 s)
    MS_TO_FTMIN: float = 3.281 * 60.0   # = 196.86

    # Wind speed clamp (m/s) — above 30 m/s is instrument error or hurricane
    U_MAX_MS: float = 30.0

    # Slope angle clamp (radians) — tan(1.3 rad) ≈ 3.6 → 360% grade
    SLOPE_MAX_RAD: float = 1.3

    def __init__(self) -> None:
        super().__init__()
        # Register as buffers so .to(device) works automatically
        self.register_buffer("m_x",        torch.tensor(self.M_X))
        self.register_buffer("r_max",      torch.tensor(self.R_MAX))
        self.register_buffer("ms_to_ftmin",torch.tensor(self.MS_TO_FTMIN))

    # ─────────────────────────────────────────────────────────────────────────
    #  Forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, physics_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute Rothermel Rate of Spread for every pixel in the batch.

        Args:
            physics_dict: output of FullPreprocessingPipeline.get_physics_dict()
                          All tensors are (B, 1, H, W).

        Returns:
            dict with keys:
                'R_mpm'  — ROS in m/min,       (B, 1, H, W), clamped [0, 200]
                'R_mps'  — ROS in m/s,         (B, 1, H, W)
                'R_norm' — ROS / 200.0 ∈ [0,1],(B, 1, H, W)  ← used by PGSS
                All intermediate tensors are also returned for debugging /
                loss inspection (keys prefixed with '_').
        """
        # ── Unpack inputs ─────────────────────────────────────────────────────
        U           = physics_dict["wind_speed"]   # m/s   (B,1,H,W)
        slope_angle = physics_dict["slope_angle"]  # rad   (B,1,H,W)
        M_f         = physics_dict["M_f"]          # frac  (B,1,H,W)
        rho_b       = physics_dict["rho_b"]        # kg/m³ (B,1,H,W)
        sigma       = physics_dict["sigma"]        # 1/ft  (B,1,H,W)
        beta        = physics_dict["beta"]         # -     (B,1,H,W)
        beta_op     = physics_dict["beta_op"]      # -     (B,1,H,W)
        w_n         = physics_dict["w_n"]          # kg/m² (B,1,H,W)
        h           = physics_dict["h"]            # kJ/kg (B,1,H,W)

        # ── Clamp physically impossible values ───────────────────────────────
        U_clamped   = torch.clamp(U,           0.0, self.U_MAX_MS)
        slope_c     = torch.clamp(slope_angle, 0.0, self.SLOPE_MAX_RAD)
        M_f_c       = torch.clamp(M_f,         0.0, self.M_X)        # can't exceed M_x
        # Prevent division by zero in beta/beta_op ratio
        beta_op_c   = torch.clamp(beta_op,     1e-6, None)
        sigma_c     = torch.clamp(sigma,       1.0,  None)           # sigma=0 → degenerate

        # ── Convert wind: m/s → ft/min (Rothermel's unit system) ─────────────
        U_ftmin = U_clamped * self.ms_to_ftmin   # ft/min

        # ── Step 2: Reaction intensity I_R ───────────────────────────────────
        # Maximum reaction velocity Γ'_max  (1/min)
        sigma15    = sigma_c ** 1.5
        Gamma_max  = sigma15 / (495.0 + 0.594 * sigma15)

        # Optimum reaction velocity exponent A
        A          = 133.0 / (sigma_c ** 0.7913)

        # Actual reaction velocity Γ'  (1/min)
        ratio      = beta / beta_op_c                                # β/β_op
        Gamma_prime = Gamma_max * (ratio ** A) * torch.exp(A * (1.0 - ratio))

        # Moisture damping coefficient η_M  ∈ [0, 1]
        r_M        = M_f_c / self.m_x                                # M_f / M_x
        eta_M      = (1.0
                      - 2.59  * r_M
                      + 5.11  * r_M ** 2
                      - 3.52  * r_M ** 3)
        eta_M      = torch.clamp(eta_M, 0.0, 1.0)                   # physical bound

        # Mineral damping coefficient η_s (simplified, no-correction form)
        eta_s      = 0.174 * sigma_c ** (-0.19)
        eta_s      = torch.clamp(eta_s, 0.0, 1.0)

        # Reaction intensity  I_R  (kJ/m²/min — using kJ/kg for h)
        I_R        = Gamma_prime * w_n * h * eta_M * eta_s

        # ── Step 3: Propagating flux ratio ξ ─────────────────────────────────
        xi = (torch.exp((0.792 + 0.681 * sigma_c ** 0.5) * (beta + 0.1))
              / (192.0 + 0.2595 * sigma_c))

        # ── Step 4: Wind coefficient φ_w ─────────────────────────────────────
        C      = 7.47  * torch.exp(-0.133  * sigma_c ** 0.55)
        B_coef = 0.02526 * sigma_c ** 0.54
        E      = 0.715  * torch.exp(-3.59e-4 * sigma_c)

        # (β/β_op)^(-E): clamp base away from zero
        ratio_E    = torch.clamp(ratio, 1e-6, None) ** (-E)
        phi_w      = C * (3.281 * U_clamped) ** B_coef * ratio_E
        # Note: 3.281 * U_clamped converts m/s → ft/s (Rothermel wind in ft/min
        # is divided by 60 implicitly through B_coef calibration).
        # The standard form uses U in ft/min / 60 = ft/s. We use ft/s directly
        # which matches published implementations (Andrews 2018).

        # ── Step 5: Slope coefficient φ_s ────────────────────────────────────
        phi_s = 5.275 * torch.tan(slope_c) ** 2

        # ── Step 6: Heat sink denominator ────────────────────────────────────
        epsilon    = torch.exp(-138.0 / sigma_c)                     # fine-particle fraction
        Q_ig       = 581.0 + 2594.0 * M_f_c                         # kJ/kg

        # ── Step 7: Rate of Spread R (m/min) ─────────────────────────────────
        R_numerator   = I_R * xi * (1.0 + phi_w + phi_s)
        R_denominator = rho_b * epsilon * Q_ig + 1e-8
        R_mpm         = R_numerator / R_denominator
        R_mpm         = torch.clamp(R_mpm, 0.0, self.R_MAX)

        R_mps  = R_mpm / 60.0                                        # m/s
        R_norm = R_mpm / self.R_MAX                                  # [0, 1]

        return {
            "R_mpm":    R_mpm,
            "R_mps":    R_mps,
            "R_norm":   R_norm,
            # Intermediates exposed for debugging and PDE loss
            "_I_R":     I_R,
            "_xi":      xi,
            "_phi_w":   phi_w,
            "_phi_s":   phi_s,
            "_eta_M":   eta_M,
            "_eta_s":   eta_s,
            "_epsilon": epsilon,
            "_Q_ig":    Q_ig,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  PGSS interface
    # ─────────────────────────────────────────────────────────────────────────

    def normalise_for_pgss(self, physics_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convenience wrapper: compute R and return R_norm for use as the
        PGSS gate R̃ in:
            Δ_{i,j} = Softplus(Linear(x)) · ((1−α) + α · R̃_{i,j})

        Args:
            physics_dict: output of FullPreprocessingPipeline.get_physics_dict()

        Returns:
            R_norm: (B, 1, H, W) tensor in [0, 1]
        """
        return self.forward(physics_dict)["R_norm"]
