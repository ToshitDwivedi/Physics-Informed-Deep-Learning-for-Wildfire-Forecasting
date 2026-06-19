"""
transforms.py
─────────────────────────────────────────────────────────────────────────────
Physics-aware feature engineering for PI-VM (Physics-Informed Vision Mamba).

All transforms are differentiable PyTorch nn.Module subclasses unless noted.
Designed for the Next Day Wildfire Spread dataset (Huot et al. 2022).

Input tensor channel layout (12 channels, indices 0-11):
    [0]  elevation    – Digital elevation model          (m)
    [1]  th           – Wind direction                   (degrees, RAW: clamp to [0,360])
    [2]  vs           – Wind speed                       (m/s,    RAW: clamp to [0,40])
    [3]  tmmn         – Min surface temperature          (K)
    [4]  tmmx         – Max surface temperature          (K)
    [5]  sph          – Specific humidity                (kg/kg)
    [6]  pr           – Precipitation                    (mm,     RAW: clamp ≥ 0)
    [7]  pdsi         – Palmer Drought Severity Index    (unitless)
    [8]  NDVI         – Norm. Diff. Veg. Index           (RAW ×10000, rescale to [-1,1])
    [9]  population   – Population density               (people/km²)
    [10] erc          – Energy Release Component         (index)
    [11] PrevFireMask – Previous-day fire mask           {-1, 0, 1}

EDA anomalies addressed:
    • th:   raw min=-505900, std=3163 → clamp to [0, 360] before use
    • vs:   raw min=-82.65           → clamp to [0, 40]  before use
    • NDVI: raw mean=5297            → divide by 10000   to get [-1, 1]
    • pr:   raw min=-167.4           → clamp to [0, ∞)   before use
    • erc:  raw min=-1196            → clamp to [0, ∞)   before use
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


# ═══════════════════════════════════════════════════════════════════════════════
#  CHANNEL REGISTRY  — documents all 24 output channels of the pipeline
# ═══════════════════════════════════════════════════════════════════════════════

CHANNEL_REGISTRY: Dict[int, Dict[str, str]] = {
    # ── Pass-through channels (normalised) ──────────────────────────────────
    0:  {"name": "elevation_norm",    "unit": "z-score",    "source": "elevation",
         "description": "Z-score normalised elevation"},
    1:  {"name": "tmmn_norm",         "unit": "z-score",    "source": "tmmn",
         "description": "Z-score normalised min daily temperature"},
    2:  {"name": "tmmx_norm",         "unit": "z-score",    "source": "tmmx",
         "description": "Z-score normalised max daily temperature"},
    3:  {"name": "sph_norm",          "unit": "z-score",    "source": "sph",
         "description": "Z-score normalised specific humidity"},
    4:  {"name": "pr_log",            "unit": "log(mm+1)",  "source": "pr",
         "description": "Log1p-transformed precipitation (right-skewed)"},
    5:  {"name": "pdsi_norm",         "unit": "z-score",    "source": "pdsi",
         "description": "Z-score normalised Palmer Drought Severity Index"},
    6:  {"name": "ndvi_scaled",       "unit": "[-1,1]",     "source": "NDVI",
         "description": "NDVI rescaled from raw ×10000 integers to [-1, 1]"},
    7:  {"name": "population_log",    "unit": "log(p+1)",   "source": "population",
         "description": "Log1p-transformed population density"},
    8:  {"name": "erc_norm",          "unit": "z-score",    "source": "erc",
         "description": "Z-score normalised Energy Release Component"},
    9:  {"name": "prev_fire_mask",    "unit": "{-1,0,1}",   "source": "PrevFireMask",
         "description": "Previous-day fire mask (pass-through)"},
    # ── SlopeAspectTransform outputs ────────────────────────────────────────
    10: {"name": "slope_magnitude",   "unit": "radians",    "source": "elevation",
         "description": "Terrain slope magnitude φ ∈ [0, π/2]"},
    11: {"name": "slope_aspect",      "unit": "radians",    "source": "elevation",
         "description": "Terrain aspect (downslope direction) θ_asp ∈ [-π, π]"},
    # ── WindVectorTransform outputs ─────────────────────────────────────────
    12: {"name": "wind_east",         "unit": "m/s",        "source": "vs, th",
         "description": "Eastward wind component U·cos(θ)"},
    13: {"name": "wind_north",        "unit": "m/s",        "source": "vs, th",
         "description": "Northward wind component U·sin(θ)"},
    14: {"name": "wind_slope_align",  "unit": "[-1,1]",     "source": "vs, th, elevation",
         "description": "cos(θ_wind - θ_asp): alignment of wind with downslope direction"},
    # ── HeatSinkTransform outputs ────────────────────────────────────────────
    15: {"name": "Q_ig",              "unit": "kJ/kg",      "source": "sph (proxy for M_f)",
         "description": "Heat of pre-ignition Q_ig = 581 + 2594·M_f (Rothermel 1972)"},
    # ── FuelModelEncoder outputs ─────────────────────────────────────────────
    16: {"name": "rho_b",             "unit": "kg/m³",      "source": "erc (proxy fuel model)",
         "description": "Fuel bulk density (Anderson 13 lookup)"},
    17: {"name": "sigma",             "unit": "1/ft",       "source": "erc (proxy fuel model)",
         "description": "Surface-area-to-volume ratio (Anderson 13 lookup)"},
    18: {"name": "beta",              "unit": "dimensionless","source": "erc (proxy fuel model)",
         "description": "Packing ratio (Anderson 13 lookup)"},
    19: {"name": "beta_op",           "unit": "dimensionless","source": "erc (proxy fuel model)",
         "description": "Optimum packing ratio (Anderson 13 lookup)"},
    20: {"name": "w_n",               "unit": "kg/m²",      "source": "erc (proxy fuel model)",
         "description": "Net fuel load (Anderson 13 lookup)"},
    21: {"name": "h",                 "unit": "kJ/kg",      "source": "erc (proxy fuel model)",
         "description": "Fuel heat content (Anderson 13 lookup)"},
    # ── SignedDistanceTransform output ───────────────────────────────────────
    22: {"name": "psi_prev",          "unit": "[-1,1]",     "source": "PrevFireMask",
         "description": "Signed distance function of previous fire front (Eikonal init)"},
    # ── Derived scalar ───────────────────────────────────────────────────────
    23: {"name": "wind_speed_clamped","unit": "m/s",        "source": "vs",
         "description": "Wind speed clamped to [0, 40] m/s for Rothermel use"},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  NORMALISATION STATISTICS  (computed from EDA, Huot et al. 2022 dataset)
# ═══════════════════════════════════════════════════════════════════════════════

# mean, std per channel index (for Z-score normalisation)
_CHANNEL_STATS: Dict[str, Tuple[float, float]] = {
    "elevation": (904.6,   846.5),
    "tmmn":      (281.9,   18.1),
    "tmmx":      (297.7,   19.08),
    "sph":       (0.006468, 0.003683),
    "pdsi":      (-0.74,   2.477),
    "erc":       (53.63,   25.26),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASS 1 — SlopeAspectTransform
# ═══════════════════════════════════════════════════════════════════════════════

class SlopeAspectTransform(nn.Module):
    """
    Compute terrain slope magnitude and aspect from a Digital Elevation Model.

    Uses 3×3 Sobel kernels registered as non-learnable buffers.
    Gradients flow through the convolution (differentiable w.r.t. elevation input).

    Args:
        pixel_size_m: spatial resolution of the DEM in metres (default 1000 m for
                      the 1 km Next Day Wildfire Spread dataset).

    Input:
        Z : (B, 1, H, W)  elevation in metres

    Output:
        (B, 2, H, W)
            channel 0 — slope_magnitude φ  in radians  ∈ [0, π/2]
            channel 1 — slope_aspect    θ  in radians  ∈ [-π, π]
                        convention: direction of steepest ASCENT measured
                        clockwise from North (standard cartographic aspect)
    """

    def __init__(self, pixel_size_m: float = 1000.0) -> None:
        super().__init__()
        self.pixel_size_m = pixel_size_m

        # Sobel kernels — registered as buffers so they move with .to(device)
        # but are NOT updated by the optimiser
        sobel_x = torch.tensor(
            [[-1.0,  0.0,  1.0],
             [-2.0,  0.0,  2.0],
             [-1.0,  0.0,  1.0]], dtype=torch.float32
        ).reshape(1, 1, 3, 3) / (8.0 * pixel_size_m)

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]], dtype=torch.float32
        ).reshape(1, 1, 3, 3) / (8.0 * pixel_size_m)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: (B, 1, H, W) elevation tensor in metres
        Returns:
            (B, 2, H, W) — [slope_magnitude, slope_aspect]
        """
        # Reflect-pad to preserve spatial dimensions
        Z_pad = F.pad(Z, (1, 1, 1, 1), mode="reflect")

        dz_dx = F.conv2d(Z_pad, self.sobel_x)   # ∂Z/∂x  (B,1,H,W)
        dz_dy = F.conv2d(Z_pad, self.sobel_y)   # ∂Z/∂y  (B,1,H,W)

        # Slope magnitude: φ = arctan(||∇Z||)
        grad_magnitude = torch.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1e-8)
        slope = torch.atan(grad_magnitude)                    # (B,1,H,W) radians

        # Aspect: θ = atan2(∂Z/∂x, ∂Z/∂y)  — direction of steepest ascent
        aspect = torch.atan2(dz_dx, dz_dy)                   # (B,1,H,W) radians

        return torch.cat([slope, aspect], dim=1)              # (B,2,H,W)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASS 2 — WindVectorTransform
# ═══════════════════════════════════════════════════════════════════════════════

class WindVectorTransform(nn.Module):
    """
    Decompose wind speed + direction into vector components and compute
    alignment with terrain slope.

    EDA note: raw 'th' contains extreme outliers (min=-505900).
    We clamp to [0, 360] before conversion.
    Raw 'vs' contains negative values (min=-82.65): clamp to [0, 40] m/s.

    Input:
        wind_speed  U     : (B, 1, H, W)  m/s  (raw — will be clamped)
        wind_dir    theta : (B, 1, H, W)  degrees meteorological convention
                            (0°=North, 90°=East, clockwise)
        slope_aspect      : (B, 1, H, W)  radians (output of SlopeAspectTransform)

    Output:
        (B, 3, H, W)
            channel 0 — U_east            = U · cos(θ_rad)
            channel 1 — U_north           = U · sin(θ_rad)
            channel 2 — wind_slope_align  = cos(θ_wind_math - θ_asp)
                        1.0 → wind perfectly aligned with downslope (max ROS)
                        0.0 → perpendicular
                       -1.0 → directly upslope (minimum wind contribution)
    """

    # Physical clamp bounds
    WIND_SPEED_MIN: float = 0.0
    WIND_SPEED_MAX: float = 40.0   # m/s  (~144 km/h — above this is rare/instrument error)
    WIND_DIR_MIN:   float = 0.0
    WIND_DIR_MAX:   float = 360.0

    def forward(
        self,
        wind_speed: torch.Tensor,
        wind_dir:   torch.Tensor,
        slope_aspect: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            (B, 3, H, W) — [U_east, U_north, wind_slope_alignment]
        """
        # Clamp outliers identified in EDA
        U     = torch.clamp(wind_speed, self.WIND_SPEED_MIN, self.WIND_SPEED_MAX)
        theta = torch.clamp(wind_dir,   self.WIND_DIR_MIN,   self.WIND_DIR_MAX)

        # Meteorological → mathematical angle convention
        # Met: 0°=North, clockwise.  Math: 0°=East, counter-clockwise.
        # θ_math = 90° - θ_met  →  wind from North (0°) blows South → -y direction
        theta_rad = (90.0 - theta) * (math.pi / 180.0)

        U_east  = U * torch.cos(theta_rad)               # (B,1,H,W)
        U_north = U * torch.sin(theta_rad)               # (B,1,H,W)

        # Wind-slope alignment: how much wind blows downslope
        # aspect is direction of steepest ascent; downslope = aspect + π
        downslope_angle = slope_aspect + math.pi
        alignment = torch.cos(theta_rad - downslope_angle)  # (B,1,H,W)  ∈ [-1,1]

        return torch.cat([U_east, U_north, alignment], dim=1)  # (B,3,H,W)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASS 3 — HeatSinkTransform
# ═══════════════════════════════════════════════════════════════════════════════

class HeatSinkTransform(nn.Module):
    """
    Compute the heat of pre-ignition Q_ig from fuel moisture content.

    Rothermel (1972) equation:
        Q_ig = 581.0 + 2594.0 · M_f    (kJ/kg)

    where M_f is fuel moisture fraction [0, M_x] with M_x ≈ 0.30–0.40
    (moisture of extinction, above which fire cannot sustain).

    Dataset note: the Next Day Wildfire Spread dataset does not include a
    direct fuel moisture channel. We use 'sph' (specific humidity, kg/kg)
    as a proxy after rescaling to [0, 0.4] range, which is physically
    consistent with field fuel moisture observations.

    Input:
        M_f : (B, 1, H, W)  fuel moisture fraction [0, 0.4]
              (pass specific humidity scaled to this range)

    Output:
        (B, 1, H, W)  Q_ig in kJ/kg
    """

    # Rothermel 1972 coefficients
    Q_BASE:  float = 581.0
    Q_SLOPE: float = 2594.0

    # Physical bounds for moisture fraction
    M_F_MIN: float = 0.0
    M_F_MAX: float = 0.40   # moisture of extinction upper bound

    def forward(self, M_f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M_f: (B, 1, H, W) fuel moisture fraction, clamped to [0, 0.4]
        Returns:
            (B, 1, H, W) Q_ig in kJ/kg
        """
        M_f_clamped = torch.clamp(M_f, self.M_F_MIN, self.M_F_MAX)
        Q_ig = self.Q_BASE + self.Q_SLOPE * M_f_clamped
        return Q_ig


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASS 4 — FuelModelEncoder
# ═══════════════════════════════════════════════════════════════════════════════

class FuelModelEncoder(nn.Module):
    """
    Map Anderson 13 fuel model codes (integers 1–13) to physical constants
    used in Rothermel's Rate of Spread equation.

    The lookup table is registered as a frozen embedding (not updated by the
    optimiser). Code 0 is reserved for "no fuel" and maps to all-zeros.

    Physical constants per fuel model (Anderson 1982 / Scott & Burgan 2005):
        rho_b   — bulk density          (kg/m³)
        sigma   — surface-area-to-volume ratio  (1/ft)
        beta    — packing ratio         (dimensionless)
        beta_op — optimum packing ratio (dimensionless)
        w_n     — net fuel load         (kg/m²)
        h       — heat content          (kJ/kg)

    Input:
        fuel_code : (B, 1, H, W)  integer fuel model code ∈ {0, 1, …, 13}
                    In practice, the dataset provides 'erc' (Energy Release
                    Component) as the closest available proxy. We discretise
                    erc into 13 bins as an approximation; the RothermelLayer
                    uses these constants downstream.

    Output:
        (B, 6, H, W)  [rho_b, sigma, beta, beta_op, w_n, h]
    """

    # ── Anderson 13 fuel model constants ─────────────────────────────────────
    # Source: Andrews (2018) "The Rothermel Surface Fire Spread Model and
    #         Associated Developments" USDA Gen. Tech. Rep. RMRS-GTR-371
    #
    # Row index = fuel model code (0 = no-fuel pad, 1–13 = Anderson models)
    # Columns:  rho_b   sigma    beta     beta_op   w_n     h
    #           kg/m³   1/ft   (dimless) (dimless) kg/m²  kJ/kg
    FUEL_TABLE: list[list[float]] = [
        # 0  — no fuel (padding row)
        [0.000,    0.0,  0.0000,  0.0000,  0.000,     0.0],
        # 1  — short grass (1 ft)
        [0.166,  3500.0, 0.00122, 0.00281, 0.166,  18622.0],
        # 2  — timber (grass/understory)
        [0.897,  2784.0, 0.00322, 0.00337, 0.897,  18622.0],
        # 3  — tall grass (2.5 ft)
        [0.230,  1500.0, 0.00153, 0.00447, 0.675,  18622.0],
        # 4  — chaparral (6 ft)
        [0.897,  1739.0, 0.00516, 0.00337, 2.468,  19259.0],
        # 5  — brush (2 ft)
        [0.448,  1683.0, 0.00267, 0.00431, 0.448,  18622.0],
        # 6  — dormant brush / hardwood slash
        [0.448,  1564.0, 0.00287, 0.00450, 0.787,  18622.0],
        # 7  — southern rough
        [0.448,  1739.0, 0.00257, 0.00431, 0.562,  18622.0],
        # 8  — compact timber litter
        [2.242,  2000.0, 0.01122, 0.00200, 0.674,  18622.0],
        # 9  — hardwood litter
        [0.448,  2500.0, 0.00179, 0.00282, 0.337,  18622.0],
        # 10 — timber (litter + understory)
        [1.121,  2000.0, 0.00561, 0.00282, 1.348,  18622.0],
        # 11 — slash: light logging
        [1.345,  1500.0, 0.00896, 0.00447, 1.571,  18622.0],
        # 12 — slash: medium logging
        [2.242,  1500.0, 0.01493, 0.00447, 3.591,  18622.0],
        # 13 — slash: heavy logging
        [3.587,  1500.0, 0.02389, 0.00447, 6.730,  18622.0],
    ]
    N_FUEL_MODELS: int = 14   # 0 (pad) + 13 Anderson models
    N_CONSTANTS:   int = 6

    def __init__(self) -> None:
        super().__init__()
        # Build embedding table: shape (14, 6)
        table = torch.tensor(self.FUEL_TABLE, dtype=torch.float32)  # (14, 6)
        # Register as buffer — moves to device automatically, never trained
        self.register_buffer("table", table)

    def forward(self, fuel_code: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fuel_code: (B, 1, H, W) integer codes ∈ [0, 13]
        Returns:
            (B, 6, H, W) physical constants
        """
        B, _, H, W = fuel_code.shape
        # Clamp codes to valid range and cast to long for indexing
        codes = torch.clamp(fuel_code.squeeze(1).long(), 0, self.N_FUEL_MODELS - 1)
        # (B, H, W) → lookup → (B, H, W, 6)
        constants = self.table[codes]                         # (B, H, W, 6)
        # Rearrange to (B, 6, H, W)
        return constants.permute(0, 3, 1, 2).contiguous()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASS 5 — SignedDistanceTransform
# ═══════════════════════════════════════════════════════════════════════════════

class SignedDistanceTransform(nn.Module):
    """
    Convert a binary fire mask into a normalised Signed Distance Function (SDF)
    for Level Set initialisation.

    Convention:
        ψ < 0  →  inside fire (burned)
        ψ = 0  →  fire front (interface)
        ψ > 0  →  outside fire (unburned)

    Normalisation: divide by max absolute distance in each sample so ψ ∈ [-1, 1].
    If the entire patch is fire or no-fire, ψ is set to -1.0 or +1.0 uniformly.

    ⚠ NOT DIFFERENTIABLE — uses scipy for EDT. Call in preprocessing, not in
    the training forward pass.

    Input:
        fire_mask : (B, 1, H, W)  binary {0, 1} or {-1, 0, 1} (no-data → 0)

    Output:
        (B, 1, H, W)  ψ_prev normalised to [-1, 1]
    """

    def forward(self, fire_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fire_mask: (B, 1, H, W) binary fire mask
        Returns:
            (B, 1, H, W) normalised SDF
        """
        B = fire_mask.shape[0]
        device = fire_mask.device

        # Work in numpy — EDT is not differentiable so no autograd needed
        mask_np = fire_mask.detach().cpu().squeeze(1).numpy()   # (B, H, W)
        # Treat -1 (no-data) as unburned
        mask_np = (mask_np > 0.5).astype(np.float32)            # binary {0.0, 1.0}

        psi_np = np.zeros_like(mask_np)

        for b in range(B):
            m = mask_np[b]   # (H, W)

            fire_pixels    = m.sum()
            nonfire_pixels = (1.0 - m).sum()

            if fire_pixels == 0:
                # Entire patch unburned → uniform +1
                psi_np[b] = 1.0
                continue
            if nonfire_pixels == 0:
                # Entire patch burned → uniform -1
                psi_np[b] = -1.0
                continue

            # Distance from unburned pixels to nearest fire pixel
            dist_outside = distance_transform_edt(1.0 - m)   # positive outside
            # Distance from fire pixels to nearest unburned pixel
            dist_inside  = distance_transform_edt(m)          # positive inside

            # Signed distance: negative inside fire, positive outside
            sdf = dist_outside - dist_inside                   # (H, W)

            # Normalise to [-1, 1]
            max_dist = np.abs(sdf).max()
            if max_dist > 0:
                sdf = sdf / max_dist
            psi_np[b] = sdf

        psi_tensor = torch.from_numpy(psi_np).unsqueeze(1).to(device)  # (B,1,H,W)
        return psi_tensor


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASS 6 — FullPreprocessingPipeline
# ═══════════════════════════════════════════════════════════════════════════════

class FullPreprocessingPipeline(nn.Module):
    """
    End-to-end preprocessing pipeline: raw 12-channel tensor → 24-channel
    physics-ready tensor for PI-VM.

    Differentiable channels (0–22, excluding SDF):
        All normalisation, Sobel, wind decomposition, and heat sink transforms
        propagate gradients. The pipeline can therefore be used inside the
        model's forward pass if needed (e.g., for physics-informed losses that
        backprop through input features).

    Non-differentiable channel:
        psi_prev (ch 22) uses SignedDistanceTransform (scipy EDT).
        Detach from the autograd graph before computing SDF in training.

    Input channel indices (raw 12-channel):
        0=elevation, 1=th, 2=vs, 3=tmmn, 4=tmmx, 5=sph,
        6=pr, 7=pdsi, 8=NDVI, 9=population, 10=erc, 11=PrevFireMask

    Output:
        (B, 24, H, W) — see CHANNEL_REGISTRY for full documentation.

    Args:
        pixel_size_m: spatial resolution in metres (default 1000 for 1 km data)
        compute_sdf:  if True, compute SDF channel (disable in training for speed)
    """

    # Z-score statistics from EDA (Huot et al. 2022 dataset)
    _NORM_STATS: Dict[str, Tuple[float, float]] = {
        "elevation": (904.6,   846.5),
        "tmmn":      (281.9,   18.1),
        "tmmx":      (297.7,   19.08),
        "sph":       (0.006468, 0.003683),
        "pdsi":      (-0.74,   2.477),
        "erc":       (53.63,   25.26),
    }

    def __init__(
        self,
        pixel_size_m: float = 1000.0,
        compute_sdf: bool = True,
    ) -> None:
        super().__init__()
        self.pixel_size_m = pixel_size_m
        self.compute_sdf  = compute_sdf

        self.slope_aspect  = SlopeAspectTransform(pixel_size_m=pixel_size_m)
        self.wind_vector   = WindVectorTransform()
        self.heat_sink     = HeatSinkTransform()
        self.fuel_encoder  = FuelModelEncoder()
        self.sdf           = SignedDistanceTransform()

        # Register normalisation stats as buffers
        for name, (mean, std) in self._NORM_STATS.items():
            self.register_buffer(f"mean_{name}", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer(f"std_{name}",  torch.tensor(std,  dtype=torch.float32))

    def _zscore(self, x: torch.Tensor, name: str) -> torch.Tensor:
        mean = getattr(self, f"mean_{name}")
        std  = getattr(self, f"std_{name}")
        return (x - mean) / (std + 1e-8)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw: (B, 12, H, W) raw input tensor (channel order per schema above)
        Returns:
            (B, 24, H, W) physics-ready feature tensor
        """
        assert raw.shape[1] == 12, f"Expected 12 input channels, got {raw.shape[1]}"

        # ── Extract raw channels ─────────────────────────────────────────────
        elev  = raw[:, 0:1]    # elevation (m)
        th    = raw[:, 1:2]    # wind direction (°)
        vs    = raw[:, 2:3]    # wind speed (m/s)
        tmmn  = raw[:, 3:4]    # min temp (K)
        tmmx  = raw[:, 4:5]    # max temp (K)
        sph   = raw[:, 5:6]    # specific humidity (kg/kg)
        pr    = raw[:, 6:7]    # precipitation (mm)
        pdsi  = raw[:, 7:8]    # drought index
        ndvi  = raw[:, 8:9]    # NDVI (raw ×10000)
        pop   = raw[:, 9:10]   # population density
        erc   = raw[:, 10:11]  # energy release component
        prev  = raw[:, 11:12]  # previous fire mask

        # ── Clamp EDA-identified outliers ────────────────────────────────────
        vs_clean  = torch.clamp(vs,  0.0,  40.0)
        th_clean  = torch.clamp(th,  0.0, 360.0)
        pr_clean  = torch.clamp(pr,  0.0, None)
        erc_clean = torch.clamp(erc, 0.0, None)

        # ── Pass-through channels (normalised, indices 0–9) ──────────────────
        ch_elev = self._zscore(elev, "elevation")
        ch_tmmn = self._zscore(tmmn, "tmmn")
        ch_tmmx = self._zscore(tmmx, "tmmx")
        ch_sph  = self._zscore(sph,  "sph")
        ch_pr   = torch.log1p(pr_clean)                      # log(mm+1)
        ch_pdsi = self._zscore(pdsi, "pdsi")
        ch_ndvi = ndvi / 10000.0                              # rescale to [-1,1]
        ch_pop  = torch.log1p(torch.clamp(pop, 0.0, None))   # log(p+1)
        ch_erc  = self._zscore(erc_clean, "erc")
        ch_prev = prev                                        # pass-through {-1,0,1}

        # ── SlopeAspectTransform (indices 10–11) ─────────────────────────────
        slope_asp = self.slope_aspect(elev)                   # (B,2,H,W)

        # ── WindVectorTransform (indices 12–14) ──────────────────────────────
        slope_aspect_ch = slope_asp[:, 1:2]                   # aspect only
        wind_vec = self.wind_vector(vs_clean, th_clean, slope_aspect_ch)  # (B,3,H,W)

        # ── HeatSinkTransform (index 15) ─────────────────────────────────────
        # Use sph as proxy for fuel moisture, scaled to [0, 0.4]
        # sph range ~[0, 0.086] → scale by 4.65 to reach [0, 0.4]
        M_f_proxy = torch.clamp(sph * 4.65, 0.0, 0.4)
        Q_ig = self.heat_sink(M_f_proxy)                      # (B,1,H,W)

        # ── FuelModelEncoder (indices 16–21) ─────────────────────────────────
        # Discretise ERC [0, ∞) into 13 bins as fuel model proxy
        erc_norm_01 = torch.clamp(erc_clean / 100.0, 0.0, 1.0)
        fuel_code = torch.clamp(
            (erc_norm_01 * 13).long(), 0, 13
        ).float()
        fuel_constants = self.fuel_encoder(fuel_code.long())   # (B,6,H,W)

        # ── SignedDistanceTransform (index 22) ───────────────────────────────
        if self.compute_sdf:
            psi_prev = self.sdf(prev)                          # (B,1,H,W), not differentiable
        else:
            psi_prev = torch.zeros_like(prev)

        # ── Wind speed clean for Rothermel direct use (index 23) ─────────────
        ch_vs_clean = vs_clean                                 # (B,1,H,W)

        # ── Concatenate all 24 channels ──────────────────────────────────────
        output = torch.cat([
            ch_elev,        # 0
            ch_tmmn,        # 1
            ch_tmmx,        # 2
            ch_sph,         # 3
            ch_pr,          # 4
            ch_pdsi,        # 5
            ch_ndvi,        # 6
            ch_pop,         # 7
            ch_erc,         # 8
            ch_prev,        # 9
            slope_asp,      # 10–11
            wind_vec,       # 12–14
            Q_ig,           # 15
            fuel_constants, # 16–21
            psi_prev,       # 22
            ch_vs_clean,    # 23
        ], dim=1)

        assert output.shape[1] == 24, f"Expected 24 output channels, got {output.shape[1]}"
        return output

    def get_physics_dict(self, physics_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract named physics variables from a 24-channel physics tensor.
        Used by RothermelLayer and PGSSBlock.

        Args:
            physics_tensor: (B, 24, H, W) output of this pipeline
        Returns:
            dict with keys: wind_speed, wind_dir_rad, slope_angle, slope_aspect,
                            M_f, rho_b, sigma, beta, beta_op, w_n, h, psi_prev
        """
        return {
            "wind_speed":   physics_tensor[:, 23:24],    # (B,1,H,W) m/s
            "wind_east":    physics_tensor[:, 12:13],    # (B,1,H,W)
            "wind_north":   physics_tensor[:, 13:14],    # (B,1,H,W)
            "slope_angle":  physics_tensor[:, 10:11],    # (B,1,H,W) radians
            "slope_aspect": physics_tensor[:, 11:12],    # (B,1,H,W) radians
            "M_f":          torch.clamp(physics_tensor[:, 3:4] * 0.003683 + 0.006468,
                                        0.0, 0.4),       # reverse z-score → kg/kg → proxy M_f
            "rho_b":        physics_tensor[:, 16:17],
            "sigma":        physics_tensor[:, 17:18],
            "beta":         physics_tensor[:, 18:19],
            "beta_op":      physics_tensor[:, 19:20],
            "w_n":          physics_tensor[:, 20:21],
            "h":            physics_tensor[:, 21:22],
            "Q_ig":         physics_tensor[:, 15:16],
            "psi_prev":     physics_tensor[:, 22:23],
        }
