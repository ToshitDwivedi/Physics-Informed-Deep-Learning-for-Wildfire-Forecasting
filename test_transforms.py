"""
tests/test_transforms.py
─────────────────────────────────────────────────────────────────────────────
Pytest test suite for transforms.py

Run with:
    pytest tests/test_transforms.py -v

All tests run on CPU. No GPU required.
"""

import math
import sys
import os

# Fix: __file__ is not defined in Jupyter/Kaggle notebook kernels
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, "/kaggle/working")  # explicit Kaggle fallback

import numpy as np
import pytest
import torch

from transforms import (
    SlopeAspectTransform,
    WindVectorTransform,
    HeatSinkTransform,
    FuelModelEncoder,
    SignedDistanceTransform,
    FullPreprocessingPipeline,
    CHANNEL_REGISTRY,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def batch_size() -> int:
    return 4

@pytest.fixture
def hw() -> tuple[int, int]:
    return (64, 64)

@pytest.fixture
def raw_batch(batch_size, hw) -> torch.Tensor:
    """Synthetic raw 12-channel batch with physically plausible values."""
    B, H, W = batch_size, *hw
    t = torch.zeros(B, 12, H, W)
    t[:, 0]  = 500.0 + torch.randn(B, H, W) * 100   # elevation (m)
    t[:, 1]  = 180.0 + torch.randn(B, H, W) * 30    # wind dir (degrees)
    t[:, 2]  = 5.0   + torch.rand(B, H, W)  * 3     # wind speed (m/s)
    t[:, 3]  = 280.0 + torch.randn(B, H, W) * 10    # tmmn (K)
    t[:, 4]  = 295.0 + torch.randn(B, H, W) * 10    # tmmx (K)
    t[:, 5]  = 0.006 + torch.rand(B, H, W)  * 0.003 # sph (kg/kg)
    t[:, 6]  = torch.rand(B, H, W) * 2              # pr (mm) — non-negative
    t[:, 7]  = torch.randn(B, H, W)                 # pdsi
    t[:, 8]  = (torch.rand(B, H, W) - 0.5) * 10000 # NDVI raw ×10000
    t[:, 9]  = torch.rand(B, H, W) * 50             # population
    t[:, 10] = 30.0 + torch.rand(B, H, W) * 40      # erc (index)
    t[:, 11] = (torch.rand(B, H, W) > 0.95).float() # PrevFireMask — sparse fire
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS 1 — SlopeAspectTransform
# ─────────────────────────────────────────────────────────────────────────────

class TestSlopeAspectTransform:

    def test_output_shape(self, batch_size, hw):
        """Output must be (B, 2, H, W)."""
        transform = SlopeAspectTransform()
        Z = torch.randn(batch_size, 1, *hw)
        out = transform(Z)
        assert out.shape == (batch_size, 2, *hw), \
            f"Expected {(batch_size, 2, *hw)}, got {out.shape}"

    def test_flat_surface_zero_slope(self, hw):
        """On a perfectly flat DEM, slope magnitude must be 0.0 everywhere."""
        transform = SlopeAspectTransform()
        Z_flat = torch.ones(1, 1, *hw) * 500.0   # constant elevation
        out = transform(Z_flat)
        slope = out[:, 0]   # channel 0 = slope magnitude

        # Allow small numerical tolerance from Sobel on constant input
        assert slope.abs().max().item() < 1e-5, \
            f"Flat surface should give zero slope, got max={slope.abs().max().item():.2e}"

    def test_slope_range(self, batch_size, hw):
        """Slope must be in [0, π/2]."""
        transform = SlopeAspectTransform()
        Z = torch.randn(batch_size, 1, *hw) * 500
        out = transform(Z)
        slope = out[:, 0]
        assert slope.min().item() >= -1e-6, \
            f"Slope below 0: {slope.min().item()}"
        assert slope.max().item() <= math.pi / 2 + 1e-4, \
            f"Slope above π/2: {slope.max().item()}"

    def test_aspect_range(self, batch_size, hw):
        """Aspect must be in [-π, π]."""
        transform = SlopeAspectTransform()
        Z = torch.randn(batch_size, 1, *hw) * 500
        out = transform(Z)
        aspect = out[:, 1]
        assert aspect.min().item() >= -math.pi - 1e-4
        assert aspect.max().item() <= math.pi  + 1e-4

    def test_steeper_hill_larger_slope(self, hw):
        """A steeper gradient should produce a larger slope value."""
        transform = SlopeAspectTransform(pixel_size_m=1000.0)
        # Gentle slope: 1m rise per pixel
        Z_gentle = torch.zeros(1, 1, *hw)
        for i in range(hw[1]):
            Z_gentle[0, 0, :, i] = float(i) * 1.0
        # Steep slope: 100m rise per pixel
        Z_steep = torch.zeros(1, 1, *hw)
        for i in range(hw[1]):
            Z_steep[0, 0, :, i] = float(i) * 100.0

        slope_gentle = transform(Z_gentle)[:, 0].mean().item()
        slope_steep  = transform(Z_steep)[:, 0].mean().item()
        assert slope_steep > slope_gentle, \
            "Steeper terrain should give larger slope magnitude"

    def test_gradient_flows(self, hw):
        """Gradients must flow through SlopeAspectTransform."""
        transform = SlopeAspectTransform()
        Z = torch.randn(1, 1, *hw, requires_grad=True)
        out = transform(Z)
        out.sum().backward()
        assert Z.grad is not None and Z.grad.abs().sum().item() > 0, \
            "No gradient from SlopeAspectTransform"


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS 2 — WindVectorTransform
# ─────────────────────────────────────────────────────────────────────────────

class TestWindVectorTransform:

    def test_output_shape(self, batch_size, hw):
        transform = WindVectorTransform()
        U     = torch.rand(batch_size, 1, *hw) * 10
        theta = torch.rand(batch_size, 1, *hw) * 360
        asp   = torch.randn(batch_size, 1, *hw)
        out   = transform(U, theta, asp)
        assert out.shape == (batch_size, 3, *hw), \
            f"Expected {(batch_size, 3, *hw)}, got {out.shape}"

    def test_wind_speed_zero_gives_zero_components(self, hw):
        """Zero wind speed → zero east and north components."""
        transform = WindVectorTransform()
        U     = torch.zeros(1, 1, *hw)
        theta = torch.ones(1, 1, *hw) * 90.0   # East wind
        asp   = torch.zeros(1, 1, *hw)
        out   = transform(U, theta, asp)
        U_east  = out[:, 0]
        U_north = out[:, 1]
        assert U_east.abs().max().item()  < 1e-6
        assert U_north.abs().max().item() < 1e-6

    def test_north_wind_decomposition(self, hw):
        """
        Met wind direction 0° = wind FROM North = wind blowing SOUTH.
        θ_math = 90° - 0° = 90° → U_east = cos(90°) ≈ 0, U_north = sin(90°) = 1
        But wind blows South, so U_north should be negative in standard coords.
        We verify the magnitude is correct: sqrt(east² + north²) ≈ U.
        """
        transform = WindVectorTransform()
        spd = 10.0
        U     = torch.ones(1, 1, *hw) * spd
        theta = torch.zeros(1, 1, *hw)           # 0° = from North
        asp   = torch.zeros(1, 1, *hw)
        out   = transform(U, theta, asp)
        U_east  = out[:, 0]
        U_north = out[:, 1]
        magnitude = torch.sqrt(U_east**2 + U_north**2)
        assert torch.allclose(magnitude, torch.ones_like(magnitude) * spd, atol=1e-4), \
            "Wind vector magnitude should equal wind speed"

    def test_alignment_range(self, batch_size, hw):
        """Wind-slope alignment must be in [-1, 1]."""
        transform = WindVectorTransform()
        U     = torch.rand(batch_size, 1, *hw) * 15
        theta = torch.rand(batch_size, 1, *hw) * 360
        asp   = torch.randn(batch_size, 1, *hw)
        out   = transform(U, theta, asp)
        align = out[:, 2]
        assert align.min().item() >= -1.0 - 1e-5
        assert align.max().item() <=  1.0 + 1e-5

    def test_outlier_clamping(self, hw):
        """Extreme outlier wind values from EDA should be clamped."""
        transform = WindVectorTransform()
        U_extreme     = torch.ones(1, 1, *hw) * -82.65   # raw dataset min
        theta_extreme = torch.ones(1, 1, *hw) * -505900  # raw dataset min
        asp = torch.zeros(1, 1, *hw)
        # Should not raise and should produce finite output
        out = transform(U_extreme, theta_extreme, asp)
        assert torch.isfinite(out).all(), "Outlier clamping failed — non-finite output"

    def test_gradient_flows(self, hw):
        transform = WindVectorTransform()
        U     = torch.rand(1, 1, *hw, requires_grad=True) * 10
        theta = torch.ones(1, 1, *hw) * 180.0
        asp   = torch.zeros(1, 1, *hw)
        out   = transform(U, theta, asp)
        out.sum().backward()
        assert U.grad is not None and U.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS 3 — HeatSinkTransform
# ─────────────────────────────────────────────────────────────────────────────

class TestHeatSinkTransform:

    def test_output_shape(self, batch_size, hw):
        transform = HeatSinkTransform()
        M_f = torch.rand(batch_size, 1, *hw) * 0.4
        out = transform(M_f)
        assert out.shape == (batch_size, 1, *hw)

    def test_dry_fuel(self, hw):
        """At M_f=0: Q_ig = 581.0 kJ/kg (Rothermel 1972)."""
        transform = HeatSinkTransform()
        M_f = torch.zeros(1, 1, *hw)
        out = transform(M_f)
        assert torch.allclose(out, torch.ones_like(out) * 581.0, atol=0.1), \
            f"At M_f=0, Q_ig should be 581.0, got {out.mean().item():.2f}"

    def test_wet_fuel(self, hw):
        """At M_f=0.3: Q_ig = 581 + 2594*0.3 = 1359.2 kJ/kg."""
        transform = HeatSinkTransform()
        M_f = torch.ones(1, 1, *hw) * 0.3
        out = transform(M_f)
        expected = 581.0 + 2594.0 * 0.3   # = 1359.2
        assert torch.allclose(out, torch.ones_like(out) * expected, atol=0.1), \
            f"At M_f=0.3, Q_ig should be {expected:.1f}, got {out.mean().item():.2f}"

    def test_linearity(self, hw):
        """Q_ig should increase linearly with M_f."""
        transform = HeatSinkTransform()
        m1 = torch.ones(1, 1, *hw) * 0.1
        m2 = torch.ones(1, 1, *hw) * 0.2
        q1 = transform(m1).mean().item()
        q2 = transform(m2).mean().item()
        # Expected increase: 2594 * 0.1 = 259.4
        assert abs((q2 - q1) - 259.4) < 0.5, \
            f"Q_ig increment should be 259.4, got {q2-q1:.2f}"

    def test_moisture_increases_Q_ig(self, hw):
        """Wetter fuel requires more energy to ignite."""
        transform = HeatSinkTransform()
        M_dry = torch.zeros(1, 1, *hw)
        M_wet = torch.ones(1, 1, *hw) * 0.4
        assert transform(M_wet).mean() > transform(M_dry).mean()

    def test_gradient_flows(self, hw):
        transform = HeatSinkTransform()
        M_f = torch.rand(1, 1, *hw, requires_grad=True) * 0.4
        out = transform(M_f)
        out.sum().backward()
        assert M_f.grad is not None and M_f.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS 4 — FuelModelEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TestFuelModelEncoder:

    def test_output_shape(self, batch_size, hw):
        encoder = FuelModelEncoder()
        codes = torch.randint(0, 14, (batch_size, 1, *hw))
        out = encoder(codes)
        assert out.shape == (batch_size, 6, *hw), \
            f"Expected {(batch_size, 6, *hw)}, got {out.shape}"

    def test_no_fuel_code_zero(self, hw):
        """Fuel code 0 (no fuel) → all physical constants = 0."""
        encoder = FuelModelEncoder()
        codes = torch.zeros(1, 1, *hw, dtype=torch.long)
        out = encoder(codes)
        assert out.abs().max().item() < 1e-6, \
            "Code 0 (no-fuel) should give all-zero constants"

    def test_known_fuel_model_1(self, hw):
        """
        Fuel model 1 (short grass): rho_b=0.166, sigma=3500, beta=0.00122.
        Check first 3 constants.
        """
        encoder = FuelModelEncoder()
        codes = torch.ones(1, 1, *hw, dtype=torch.long)   # code = 1
        out = encoder(codes)   # (1, 6, H, W)
        rho_b = out[0, 0, 0, 0].item()
        sigma = out[0, 1, 0, 0].item()
        beta  = out[0, 2, 0, 0].item()
        assert abs(rho_b - 0.166) < 1e-4,   f"rho_b mismatch: {rho_b}"
        assert abs(sigma - 3500.0) < 0.1,   f"sigma mismatch: {sigma}"
        assert abs(beta  - 0.00122) < 1e-5, f"beta mismatch: {beta}"

    def test_all_codes_finite(self, hw):
        """All valid codes (0–13) should produce finite outputs."""
        encoder = FuelModelEncoder()
        for code in range(14):
            codes = torch.ones(1, 1, *hw, dtype=torch.long) * code
            out = encoder(codes)
            assert torch.isfinite(out).all(), f"Non-finite output for fuel code {code}"

    def test_out_of_range_codes_clamped(self, hw):
        """Codes outside [0,13] should not crash (clamped to valid range)."""
        encoder = FuelModelEncoder()
        codes_high = torch.ones(1, 1, *hw, dtype=torch.long) * 99
        codes_low  = torch.ones(1, 1, *hw, dtype=torch.long) * (-5)
        out_high = encoder(codes_high)
        out_low  = encoder(codes_low)
        assert torch.isfinite(out_high).all()
        assert torch.isfinite(out_low).all()

    def test_different_codes_give_different_outputs(self, hw):
        """Different fuel models should produce different physical constants."""
        encoder = FuelModelEncoder()
        out1 = encoder(torch.ones(1, 1, *hw, dtype=torch.long) * 1)
        out8 = encoder(torch.ones(1, 1, *hw, dtype=torch.long) * 8)
        assert not torch.allclose(out1, out8), \
            "Fuel models 1 and 8 should have different constants"

    def test_table_not_trainable(self):
        """The lookup table must not be a trainable parameter."""
        encoder = FuelModelEncoder()
        param_names = [name for name, _ in encoder.named_parameters()]
        assert "table" not in param_names, \
            "FuelModelEncoder.table should be a buffer, not a parameter"


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS 5 — SignedDistanceTransform
# ─────────────────────────────────────────────────────────────────────────────

class TestSignedDistanceTransform:

    def test_output_shape(self, batch_size, hw):
        sdf = SignedDistanceTransform()
        mask = (torch.rand(batch_size, 1, *hw) > 0.9).float()
        out = sdf(mask)
        assert out.shape == (batch_size, 1, *hw)

    def test_output_range(self, batch_size, hw):
        """Output must be in [-1, 1]."""
        sdf = SignedDistanceTransform()
        mask = (torch.rand(batch_size, 1, *hw) > 0.9).float()
        out = sdf(mask)
        assert out.min().item() >= -1.0 - 1e-5
        assert out.max().item() <=  1.0 + 1e-5

    def test_all_unburned_gives_positive(self, hw):
        """If no fire, SDF should be uniformly +1."""
        sdf = SignedDistanceTransform()
        mask = torch.zeros(1, 1, *hw)
        out = sdf(mask)
        assert torch.allclose(out, torch.ones_like(out), atol=1e-5), \
            "All-unburned mask should give uniform +1 SDF"

    def test_all_burned_gives_negative(self, hw):
        """If fully burned, SDF should be uniformly -1."""
        sdf = SignedDistanceTransform()
        mask = torch.ones(1, 1, *hw)
        out = sdf(mask)
        assert torch.allclose(out, -torch.ones_like(out), atol=1e-5), \
            "All-burned mask should give uniform -1 SDF"

    def test_fire_pixels_negative_unburned_positive(self, hw):
        """
        Fire pixels should have negative SDF, unburned positive.
        Use a 10×10 fire square in a 64×64 patch.
        """
        sdf = SignedDistanceTransform()
        mask = torch.zeros(1, 1, *hw)
        mask[0, 0, 27:37, 27:37] = 1.0   # 10×10 fire square at centre

        out = sdf(mask)
        psi = out[0, 0]   # (H, W)

        # Deep interior of fire (centre pixel) should be negative
        centre_val = psi[32, 32].item()
        assert centre_val < 0, \
            f"Centre of fire square should be negative, got {centre_val:.4f}"

        # Far outside (corner) should be positive
        corner_val = psi[0, 0].item()
        assert corner_val > 0, \
            f"Corner (far from fire) should be positive, got {corner_val:.4f}"

    def test_sdf_tent_shape_for_circle(self, hw):
        """
        For a circular fire region, the SDF exterior should increase
        monotonically away from the fire boundary.
        """
        sdf = SignedDistanceTransform()
        mask = torch.zeros(1, 1, *hw)
        H, W = hw
        cy, cx = H // 2, W // 2
        radius = 15
        for i in range(H):
            for j in range(W):
                if (i - cy)**2 + (j - cx)**2 <= radius**2:
                    mask[0, 0, i, j] = 1.0

        out = sdf(mask)[0, 0]   # (H, W)

        # Sample points at increasing distance from edge along the x-axis
        # Edge is around cx + radius
        edge = cx + radius
        v_near = out[cy, min(edge + 2, W - 1)].item()
        v_far  = out[cy, min(edge + 10, W - 1)].item()
        assert v_far > v_near, \
            f"SDF should increase away from fire boundary: near={v_near:.3f}, far={v_far:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS 6 — FullPreprocessingPipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPreprocessingPipeline:

    def test_output_shape(self, raw_batch, hw):
        """Output must be (B, 24, H, W)."""
        pipeline = FullPreprocessingPipeline()
        out = pipeline(raw_batch)
        B = raw_batch.shape[0]
        assert out.shape == (B, 24, *hw), \
            f"Expected {(B, 24, *hw)}, got {out.shape}"

    def test_no_nan_in_output(self, raw_batch):
        """No NaN values in output for clean input."""
        pipeline = FullPreprocessingPipeline()
        out = pipeline(raw_batch)
        assert not torch.isnan(out).any(), "NaN found in pipeline output"

    def test_no_inf_in_output(self, raw_batch):
        """No Inf values in output."""
        pipeline = FullPreprocessingPipeline()
        out = pipeline(raw_batch)
        assert torch.isfinite(out).all(), "Inf found in pipeline output"

    def test_ndvi_rescaled(self, hw):
        """NDVI output channel should be in [-1, 1] (rescaled from raw ×10000)."""
        pipeline = FullPreprocessingPipeline(compute_sdf=False)
        raw = torch.zeros(1, 12, *hw)
        raw[:, 8] = 5000.0   # raw NDVI = 5000 → should become 0.5
        out = pipeline(raw)
        ndvi_out = out[:, 6]   # channel 6 = ndvi_scaled
        expected = 0.5
        assert torch.allclose(ndvi_out, torch.ones_like(ndvi_out) * expected, atol=1e-4), \
            f"NDVI 5000 raw should rescale to 0.5, got {ndvi_out.mean().item():.4f}"

    def test_psi_prev_range(self, raw_batch):
        """SDF channel (22) must be in [-1, 1]."""
        pipeline = FullPreprocessingPipeline(compute_sdf=True)
        out = pipeline(raw_batch)
        psi = out[:, 22]
        assert psi.min().item() >= -1.0 - 1e-5
        assert psi.max().item() <=  1.0 + 1e-5

    def test_wind_speed_channel_non_negative(self, raw_batch):
        """Channel 23 (wind_speed_clamped) must be ≥ 0."""
        pipeline = FullPreprocessingPipeline(compute_sdf=False)
        out = pipeline(raw_batch)
        vs_out = out[:, 23]
        assert vs_out.min().item() >= 0.0, \
            f"Wind speed channel has negative values: {vs_out.min().item()}"

    def test_outlier_input_produces_finite_output(self, hw):
        """Extreme outlier inputs (as seen in EDA) should produce finite outputs."""
        pipeline = FullPreprocessingPipeline(compute_sdf=False)
        raw = torch.zeros(1, 12, *hw)
        raw[:, 1] = -505900.0   # th extreme
        raw[:, 2] = -82.65      # vs extreme negative
        raw[:, 6] = -167.4      # pr extreme negative
        raw[:, 10] = -1196.0    # erc extreme negative
        raw[:, 8]  = 9966.0     # NDVI max raw
        out = pipeline(raw)
        assert torch.isfinite(out).all(), \
            "Pipeline should handle EDA outliers gracefully"

    def test_gradient_flows_through_differentiable_channels(self, raw_batch):
        """
        Gradients should flow through all differentiable transforms.
        Test by requiring grad on elevation and checking grad is non-zero
        after summing the output.
        """
        pipeline = FullPreprocessingPipeline(compute_sdf=False)
        raw = raw_batch.clone().requires_grad_(False)
        elev = raw[:, 0:1].clone().requires_grad_(True)
        raw_with_grad = torch.cat([elev, raw[:, 1:]], dim=1)

        out = pipeline(raw_with_grad)
        # Sum only the differentiable slope channel (10)
        out[:, 10].sum().backward()
        assert elev.grad is not None and elev.grad.abs().sum() > 0, \
            "Gradient should flow from slope channel back to elevation input"

    def test_get_physics_dict_keys(self, raw_batch):
        """get_physics_dict should return all required keys for RothermelLayer."""
        pipeline = FullPreprocessingPipeline(compute_sdf=False)
        out = pipeline(raw_batch)
        pdict = pipeline.get_physics_dict(out)
        required_keys = {
            "wind_speed", "wind_east", "wind_north",
            "slope_angle", "slope_aspect",
            "M_f", "rho_b", "sigma", "beta", "beta_op", "w_n", "h",
            "Q_ig", "psi_prev"
        }
        missing = required_keys - set(pdict.keys())
        assert not missing, f"Missing physics dict keys: {missing}"

    def test_get_physics_dict_shapes(self, raw_batch, hw):
        """All physics dict tensors should be (B, 1, H, W)."""
        pipeline = FullPreprocessingPipeline(compute_sdf=False)
        out = pipeline(raw_batch)
        pdict = pipeline.get_physics_dict(out)
        B = raw_batch.shape[0]
        for key, tensor in pdict.items():
            assert tensor.shape == (B, 1, *hw), \
                f"physics_dict['{key}'] has wrong shape: {tensor.shape}"

    def test_M_f_proxy_range(self, raw_batch):
        """M_f proxy in physics dict should be in [0, 0.4]."""
        pipeline = FullPreprocessingPipeline(compute_sdf=False)
        out = pipeline(raw_batch)
        pdict = pipeline.get_physics_dict(out)
        M_f = pdict["M_f"]
        assert M_f.min().item() >= -1e-5
        assert M_f.max().item() <= 0.4 + 1e-5


# ─────────────────────────────────────────────────────────────────────────────
#  CHANNEL REGISTRY integrity check
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelRegistry:

    def test_all_24_channels_documented(self):
        """CHANNEL_REGISTRY must document exactly 24 channels (indices 0–23)."""
        assert len(CHANNEL_REGISTRY) == 24, \
            f"Expected 24 entries in CHANNEL_REGISTRY, got {len(CHANNEL_REGISTRY)}"

    def test_indices_are_contiguous(self):
        """Channel indices must be exactly 0, 1, 2, ..., 23."""
        indices = sorted(CHANNEL_REGISTRY.keys())
        assert indices == list(range(24)), \
            f"CHANNEL_REGISTRY indices are not contiguous 0–23: {indices}"

    def test_each_entry_has_required_fields(self):
        """Each entry must have 'name', 'unit', 'source', 'description'."""
        required = {"name", "unit", "source", "description"}
        for idx, entry in CHANNEL_REGISTRY.items():
            missing = required - set(entry.keys())
            assert not missing, \
                f"Channel {idx} missing fields: {missing}"
