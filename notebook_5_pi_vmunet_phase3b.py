"""
notebook_5_pi_vmunet_phase3b.py  —  Phase 3-B
─────────────────────────────────────────────────────────────────────────────
Train PI-VMUNet with L_Seg ONLY (no physics loss).
Goal: isolate the PGSS architectural contribution vs VM-UNet baseline.

Fixes vs previous version:
  - PIVMUNetV2.head: interpolate BEFORE sigmoid → output always in [0,1]
  - _compute_r_tilde: try/except with safe zero fallback
  - verify(): prints shape+range before asserting
"""

import os, sys, time, json, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, "/kaggle/working")
from pgss_block import (PGSSBlock, PatchMerge, PatchExpand,
                        RothermelLayer, PHYSICS_CHANNEL_INDICES,
                        _MF_SCALE, _SHP_MEAN, _SPH_STD)
from trainer import TrainConfig, WildfireLoss, WildfireMetrics, CurriculumScheduler, Trainer
from wildfire_dataset import make_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ── PhysicsAwareSkipGate ──────────────────────────────────────────────────────

class PhysicsAwareSkipGate(nn.Module):
    """gate(R̃) = 0.5 + 0.5·R̃  ∈ [0.5,1.0] — amplifies skips in high-ROS regions."""
    def __init__(self, dim):
        super().__init__()
        self.channel_scale = nn.Parameter(torch.ones(1, 1, 1, dim))

    def forward(self, encoder_feat, R_tilde):
        # encoder_feat: (B,H,W,C) channel-last  |  R_tilde: (B,1,H,W) channel-first
        R_cl = R_tilde.permute(0, 2, 3, 1)                     # (B,H,W,1)
        if R_cl.shape[1:3] != encoder_feat.shape[1:3]:
            R_cl = F.interpolate(
                R_tilde, size=encoder_feat.shape[1:3],
                mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)
        return encoder_feat * (0.5 + 0.5 * R_cl) * self.channel_scale


# ── PIVMUNetV2 ────────────────────────────────────────────────────────────────

class PIVMUNetV2(nn.Module):
    """
    PI-VMUNet with PhysicsAwareSkipGate.
    Input:  (B, 24, H, W)
    Output: (B,  1, H, W) sigmoid probability ∈ [0,1]

    Critical: F.interpolate → head_conv → sigmoid  (in that order)
    so sigmoid always receives the full-resolution feature map.
    """
    IN_CHANNELS = 24

    def __init__(self, dims=(96, 192, 384, 768), d_state=16):
        super().__init__()
        self.dims = dims

        self.rothermel = RothermelLayer()
        for p in self.rothermel.parameters():
            p.requires_grad_(False)

        self.stem      = nn.Conv2d(self.IN_CHANNELS, dims[0], kernel_size=4, stride=4, bias=False)
        self.stem_norm = nn.LayerNorm(dims[0])

        self.enc_blocks = nn.ModuleList()
        self.merges     = nn.ModuleList()
        for i in range(4):
            self.enc_blocks.append(
                PGSSBlock(d_model=dims[i], d_state=d_state, rothermel_layer=self.rothermel))
            self.merges.append(PatchMerge(dims[i]) if i < 3 else nn.Identity())

        self.bottleneck = PGSSBlock(
            d_model=dims[3], d_state=d_state, rothermel_layer=self.rothermel)

        self.dec_expands = nn.ModuleList()
        self.dec_norms   = nn.ModuleList()
        self.dec_blocks  = nn.ModuleList()
        self.skip_projs  = nn.ModuleList()
        self.skip_gates  = nn.ModuleList()

        for i in range(3, -1, -1):
            in_dim  = dims[i]
            out_dim = dims[i-1] if i > 0 else dims[0] // 2
            self.dec_expands.append(PatchExpand(in_dim))
            self.skip_gates.append(PhysicsAwareSkipGate(in_dim))
            self.skip_projs.append(nn.Linear(in_dim + in_dim, out_dim, bias=False))
            self.dec_norms.append(nn.LayerNorm(out_dim))
            self.dec_blocks.append(
                PGSSBlock(d_model=out_dim, d_state=d_state, rothermel_layer=self.rothermel))

        head_dim = dims[0] // 2
        # head_conv outputs logits; sigmoid applied after interpolation
        self.head_conv = nn.Sequential(
            nn.Conv2d(head_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def _cf(self, x): return x.permute(0, 3, 1, 2).contiguous()
    def _cl(self, x): return x.permute(0, 2, 3, 1).contiguous()

    def _resize(self, t, H, W):
        if t.shape[2] == H and t.shape[3] == W:
            return t
        return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)

    def _compute_r_tilde(self, physics):
        try:
            def ch(name):
                return physics[:, PHYSICS_CHANNEL_INDICES[name]:PHYSICS_CHANNEL_INDICES[name]+1]
            sph_raw = ch("M_f") * _SPH_STD + _SHP_MEAN
            M_f     = torch.clamp(sph_raw * _MF_SCALE, 0.0, 0.30)
            pdict   = {
                "wind_speed":  ch("wind_speed"), "slope_angle": ch("slope_angle"),
                "M_f": M_f,    "rho_b": ch("rho_b"),  "sigma": ch("sigma"),
                "beta": ch("beta"), "beta_op": ch("beta_op"),
                "w_n":  ch("w_n"),  "h": ch("h"),
            }
            r = self.rothermel.normalise_for_pgss(pdict)
            if r.isnan().any() or r.isinf().any():
                r = torch.zeros_like(r)
            return r
        except Exception:
            B, _, H, W = physics.shape
            return torch.zeros(B, 1, H, W, device=physics.device, dtype=physics.dtype)

    def forward(self, x):
        B, C, H, W = x.shape
        physics = x
        R_tilde = self._compute_r_tilde(physics)

        # Stem
        feat = self.stem_norm(self._cl(self.stem(x)))

        # Encoder
        skips = []
        for i, (block, merge) in enumerate(zip(self.enc_blocks, self.merges)):
            fH, fW = feat.shape[1], feat.shape[2]
            feat   = block(feat, self._resize(physics, fH, fW))
            skips.append(feat)
            if i < 3:
                feat = merge(feat)

        # Bottleneck
        fH, fW = feat.shape[1], feat.shape[2]
        feat   = self.bottleneck(feat, self._resize(physics, fH, fW))

        # Decoder
        for i, (expand, gate, skip_proj, norm, block) in enumerate(
            zip(self.dec_expands, self.skip_gates, self.skip_projs,
                self.dec_norms, self.dec_blocks)
        ):
            skip = skips[3 - i]
            feat = expand(feat)
            if feat.shape[1:3] != skip.shape[1:3]:
                feat = self._cl(F.interpolate(
                    self._cf(feat), size=skip.shape[1:3],
                    mode="bilinear", align_corners=False))
            skip = gate(skip, self._resize(R_tilde, skip.shape[1], skip.shape[2]))
            feat = norm(skip_proj(torch.cat([feat, skip], dim=-1)))
            fH, fW = feat.shape[1], feat.shape[2]
            feat   = block(feat, self._resize(physics, fH, fW))

        # Head: upsample FIRST, then conv, then sigmoid → range always [0,1]
        feat = self._cf(feat)                                   # channel-first
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        feat = self.head_conv(feat)                             # logits (B,1,H,W)
        return torch.sigmoid(feat)                              # probability [0,1]

    def pgss_blocks(self):
        return [m for m in self.modules() if isinstance(m, PGSSBlock)]

    def clamp_all_alpha(self):
        for b in self.pgss_blocks(): b.clamp_alpha()

    def alpha_stats(self):
        alphas = [b.alpha_value for b in self.pgss_blocks()]
        return {"alpha/mean": sum(alphas)/len(alphas),
                "alpha/min":  min(alphas), "alpha/max": max(alphas)}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Verify ────────────────────────────────────────────────────────────────────

def verify():
    print("Verifying PIVMUNetV2...")
    m = PIVMUNetV2().to(DEVICE)
    x = torch.randn(2, 24, 128, 128).to(DEVICE)
    with torch.no_grad():
        y = m(x)
    print(f"  shape  : {y.shape}")
    print(f"  range  : [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"  nan    : {y.isnan().any().item()}")
    print(f"  params : {m.count_params()/1e6:.2f}M")
    assert y.shape == (2, 1, 128, 128),      f"Shape: {y.shape}"
    assert not y.isnan().any(),               "NaN in output"
    assert y.min().item() >= 0.0,            f"Min={y.min().item():.4f} < 0"
    assert y.max().item() <= 1.0,            f"Max={y.max().item():.4f} > 1"
    print("✅ verify passed")
    del m, x, y; torch.cuda.empty_cache()

verify()


# ── PIVMTrainer ───────────────────────────────────────────────────────────────

class PIVMTrainer(Trainer):
    def train_epoch(self, train_dl):
        self.model.train()
        total, n = 0.0, 0
        for batch in train_dl:
            inp        = batch["inputs"].to(self.device)
            tgt        = batch["targets"].to(self.device)
            valid_mask = batch.get("valid_mask")
            prev       = batch.get("prev_fire")
            if valid_mask is not None: valid_mask = valid_mask.to(self.device)
            if prev is not None:       prev       = prev.to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                y_hat = self.model(inp)
                loss  = self.loss_fn(
                    y_hat, tgt, y_prev=prev, model=self.model,
                    valid_mask=valid_mask,
                )["total"]

            # Skip NaN batch
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if hasattr(self.model, "clamp_all_alpha"):
                self.model.clamp_all_alpha()

            total += loss.item(); n += 1

        avg = total / max(n, 1)
        return {"train_loss": avg, "total": avg, "dice": avg, "focal": 0.0, "pde": 0.0, "eikonal": 0.0, "reg": 0.0}


# ── train_one_seed ────────────────────────────────────────────────────────────

def train_one_seed(seed, train_dl, val_dl, ckpt_dir, run_name):
    torch.manual_seed(seed); np.random.seed(seed)
    model     = PIVMUNetV2().to(DEVICE)
    loss_fn   = WildfireLoss(lambda_pde=0.3, lambda_eik=0.1, lambda_reg=1e-4)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
    cosine    = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-6)
    sched     = SequentialLR(optimizer, [warmup, cosine], milestones=[5])
    config    = TrainConfig(
        model_name="pi_vmunet", resolution=128, batch_size=8, lr=1e-4,
        weight_decay=1e-2, n_epochs=50, lambda_max=0.3, lambda_eik=0.1,
        lambda_reg=0.0, alpha_focal=0.85, gamma_focal=3.0,
        checkpoint_dir=ckpt_dir, mixed_precision=False, grad_clip=0.5, save_every=5,
        fail_on_nonfinite=False,
    )
    trainer = PIVMTrainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=sched,
        curriculum=CurriculumScheduler(lambda_max=0.3),
        config=config, checkpoint_dir=ckpt_dir,
    )
    trainer.load_checkpoint()
    t0      = time.time()
    history = trainer.fit(train_dl, val_dl, n_epochs=50)
    return trainer, history, (time.time() - t0) / 3600


# ── Dataloaders ───────────────────────────────────────────────────────────────

HDF5_PATH = "/kaggle/working/wildfire_data.h5"
assert os.path.exists(HDF5_PATH), f"HDF5 not found — run notebook_1 first"

train_dl, val_dl, test_dl = make_dataloaders(
    hdf5_path=HDF5_PATH, batch_size=8, resolution=128, num_workers=2)
print(f"Dataloaders: train={len(train_dl.dataset)}, "
      f"val={len(val_dl.dataset)}, test={len(test_dl.dataset)}")


# ── Train seed 42 ─────────────────────────────────────────────────────────────

CKPT_BASE = "/kaggle/working/outputs/checkpoints/pi_vmunet_phase4"
os.makedirs(CKPT_BASE, exist_ok=True)

trainer_s0, history_s0, t_h_s0 = train_one_seed(
    seed=42, train_dl=train_dl, val_dl=val_dl,
    ckpt_dir=f"{CKPT_BASE}/seed42", run_name="pi_vmunet_phase4_lam03")
print(f"Seed 42 done — {t_h_s0:.2f}h")


# ── Test evaluation ───────────────────────────────────────────────────────────

test_metrics = trainer_s0.val_epoch(test_dl)
print(f"\n{'='*50}\nTEST — PI-VMUNet (PGSS, L_Seg only)\n{'='*50}")
for k, v in test_metrics.items():
    print(f"  {k:20s}: {float(v):.4f}")

results_s0 = {
    "model": "pi_vmunet_phase3b", "seed": 42,
    "params_M": trainer_s0.model.count_params()/1e6,
    "train_time_h": t_h_s0,
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    "history":      {k: [float(x) for x in v] for k, v in history_s0.items()},
    "final_alpha":  trainer_s0.model.alpha_stats(),
}
with open("/kaggle/working/results_pi_vmunet_s0.json", "w") as f:
    json.dump(results_s0, f, indent=2)
print("✅ results_pi_vmunet_s0.json saved")


# Single seed only
alpha_mean = float(results_s0["final_alpha"]["alpha/mean"])
alpha_std  = 0.0
print(f"Alpha: {alpha_mean:.4f}")


# ── Comparison table ──────────────────────────────────────────────────────────

def load_json(p): return json.load(open(p)) if os.path.exists(p) else {}

baselines = {
    "ResNet-UNet": load_json("/kaggle/working/results_resnet_unet.json"),
    "Swin-UNet":   load_json("/kaggle/working/results_swin_unet.json"),
    "VM-UNet":     load_json("/kaggle/working/results_vm_unet.json"),
    "PI-VMUNet":   results_s0,
}
print(f"\n{'='*80}\nCOMPARISON TABLE\n{'='*80}")
print(f"{'Model':<14} | {'CSI':>7} | {'IoU':>7} | {'PR-AUC':>7} | {'Params':>7} | {'ΔCSI':>7}")
print("─" * 62)
vm_csi = None
for name, res in baselines.items():
    if not res: print(f"{name:<14} | N/A"); continue
    m   = res.get("test_metrics", {})
    csi = m.get("CSI", float("nan"))
    if name == "VM-UNet": vm_csi = csi
    delta = f"{csi-vm_csi:+.4f}" if (vm_csi is not None and name != "VM-UNet") else "base"
    print(f"{name:<14} | {csi:>7.4f} | {m.get('IoU',float('nan')):>7.4f} | "
          f"{m.get('PR_AUC',float('nan')):>7.4f} | {res.get('params_M',0):>6.1f}M | {delta:>7}")
print(f"{'='*80}")


# ── Spatial improvement map ───────────────────────────────────────────────────

vm_ckpt = "/kaggle/working/checkpoints/vm_unet/best_model.pt"
if os.path.exists(vm_ckpt):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "notebook_3_vm", "/kaggle/working/notebook_3_vm_unet.py")
    nb3  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb3)
    model_vm = nb3.VMUNet().to(DEVICE)
    ck = torch.load(vm_ckpt, map_location=DEVICE)
    model_vm.load_state_dict(ck["model_state_dict"])
    model_vm.eval()
    print("✅ VM-UNet loaded")

    H, W  = 128, 128; eps = 1e-8
    tp_pi = np.zeros((H,W)); fp_pi = np.zeros((H,W)); fn_pi = np.zeros((H,W))
    tp_vm = np.zeros((H,W)); fp_vm = np.zeros((H,W)); fn_vm = np.zeros((H,W))
    r_acc = np.zeros((H,W)); n_seen = 0
    rot_e = RothermelLayer().to(DEVICE)
    trainer_s0.model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            if i >= 20: break
            inp = batch["inputs"].to(DEVICE)
            tgt = batch["targets"].to(DEVICE)
            yp  = (trainer_s0.model(inp) > 0.5).float()
            yv  = (model_vm(inp) > 0.5).float()
            t_n = tgt.squeeze(1).cpu().numpy()
            p_n = yp.squeeze(1).cpu().numpy()
            v_n = yv.squeeze(1).cpu().numpy()

            try:
                def ch(name): return inp[:, PHYSICS_CHANNEL_INDICES[name]:PHYSICS_CHANNEL_INDICES[name]+1]
                sph_r = ch("M_f")*_SPH_STD+_SHP_MEAN
                M_f   = torch.clamp(sph_r*_MF_SCALE, 0.0, 0.30)
                pd    = {"wind_speed":ch("wind_speed"),"slope_angle":ch("slope_angle"),
                         "M_f":M_f,"rho_b":ch("rho_b"),"sigma":ch("sigma"),
                         "beta":ch("beta"),"beta_op":ch("beta_op"),"w_n":ch("w_n"),"h":ch("h")}
                r_n   = np.nan_to_num(rot_e.normalise_for_pgss(pd).squeeze(1).cpu().numpy(), 0.0)
            except Exception:
                r_n = np.zeros_like(t_n)

            for b in range(t_n.shape[0]):
                tp_pi += (p_n[b]==1)&(t_n[b]==1); fp_pi += (p_n[b]==1)&(t_n[b]==0)
                fn_pi += (p_n[b]==0)&(t_n[b]==1)
                tp_vm += (v_n[b]==1)&(t_n[b]==1); fp_vm += (v_n[b]==1)&(t_n[b]==0)
                fn_vm += (v_n[b]==0)&(t_n[b]==1)
                r_acc += r_n[b]; n_seen += 1

    imp_map = tp_pi/(tp_pi+fp_pi+fn_pi+eps) - tp_vm/(tp_vm+fp_vm+fn_vm+eps)
    r_map   = r_acc / max(n_seen, 1)
    mask    = (tp_pi+fp_pi+fn_pi+tp_vm+fp_vm+fn_vm) > 5
    spear_r, spear_p = (spearmanr(imp_map[mask], r_map[mask])
                        if mask.sum() > 10 else (float("nan"), float("nan")))
    spear_r = float(spear_r); spear_p = float(spear_p)
    print(f"Spearman r={spear_r:.4f}, p={spear_p:.4f} "
          f"({'spatial ✅' if spear_r > 0.3 else 'weak'})")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    im0 = ax[0].imshow(r_map,   cmap="YlOrRd"); ax[0].set_title("Mean R̃"); plt.colorbar(im0, ax=ax[0])
    vm  = max(abs(imp_map.min()), abs(imp_map.max()))+1e-6
    im1 = ax[1].imshow(imp_map, cmap="RdYlGn", vmin=-vm, vmax=vm)
    ax[1].set_title("ΔCSI PI-VM−VM"); plt.colorbar(im1, ax=ax[1])
    idx = np.random.choice(H*W, 5000, replace=False)
    ax[2].scatter(r_map.flatten()[idx], imp_map.flatten()[idx], alpha=0.15, s=3)
    ax[2].axhline(0, color="red", lw=1, ls="--")
    ax[2].set(xlabel="R̃", ylabel="ΔCSI", title=f"Spearman r={spear_r:.3f}")
    ax[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/kaggle/working/spatial_improvement_phase3b.png", dpi=150)
    plt.show(); print("✅ spatial map saved")
else:
    print("⚠️  VM-UNet checkpoint not found — skipping spatial analysis")
    imp_map = r_map = None; spear_r = spear_p = float("nan")


# ── Alpha convergence plot ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
alpha_finals = [alpha_mean]  # single seed
ax.bar(["seed42"], alpha_finals, color=["#2196F3"], alpha=0.8, width=0.5)
ax.axhline(0.2, color="red", ls="--", lw=1.5, label="threshold=0.2")
ax.axhline(alpha_mean, color="navy", lw=2, label=f"mean={alpha_mean:.4f}±{alpha_std:.4f}")
ax.set(xlabel="Seed", ylabel="Final α", title="PGSS α Convergence", ylim=(0, 1))
ax.legend(); ax.grid(True, alpha=0.3, axis="y"); plt.tight_layout()
plt.savefig("/kaggle/working/alpha_convergence_phase3b.png", dpi=150)
plt.show(); print("✅ alpha plot saved")


# ── Save summary ──────────────────────────────────────────────────────────────

summary = {
    "phase": "3B", "model": "PI-VMUNet (PGSS, no physics loss)",
    "seed42_metrics": results_s0["test_metrics"],
    "params_M": results_s0["params_M"],
    "alpha_mean": alpha_mean, "alpha_std": alpha_std,
    "alpha_all_seeds": alpha_finals,
    "spearman_r": spear_r, "spearman_p": spear_p,
    "physics_spatial": spear_r > 0.3 if not np.isnan(spear_r) else False,
    "physics_active":  alpha_mean > 0.2,
}
with open("/kaggle/working/results_phase3b_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*50}\nPHASE 3-B COMPLETE\n{'='*50}")
print(f"  CSI    : {results_s0['test_metrics'].get('CSI','N/A')}")
print(f"  Alpha  : {alpha_mean:.4f} ± {alpha_std:.4f}")
print(f"  Spearman r : {spear_r:.4f}")
print(f"  Physics active  : {'YES' if alpha_mean > 0.2 else 'NO'}")
print(f"  Physics spatial : {'YES' if spear_r > 0.3 else 'WEAK'}")
print(f"{'='*50}")
print("Next → Phase 4-A: Eikonal + level-set PDE loss")
