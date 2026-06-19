"""
notebook_4_comparison.py
─────────────────────────────────────────────────────────────────────────────
Run AFTER all three baseline notebooks complete.
Reads results_*.json files from /kaggle/working/ and produces:
  1. Formatted comparison table (console + saved as CSV)
  2. Learning curves (CSI vs epoch) for all 3 models on same axes
  3. Radar chart for multi-metric comparison
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

WORKING = "/kaggle/working"

# ── Load results ──────────────────────────────────────────────────────────────

def load_results(model_key: str) -> dict:
    """Load JSON results for a model. Returns empty dict if file missing."""
    path = os.path.join(WORKING, f"results_{model_key}.json")
    if not os.path.exists(path):
        print(f"⚠️  {path} not found — run that notebook first")
        return {}
    with open(path) as f:
        return json.load(f)

resnet = load_results("resnet_unet")
swin   = load_results("swin_unet")
vmamba = load_results("vm_unet")

all_results = [
    ("ResNet-UNet", resnet),
    ("Swin-UNet",   swin),
    ("VM-UNet",     vmamba),
]


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(results: list) -> None:
    """Print IEEE-style comparison table to console."""
    metrics = ["CSI", "IoU", "PCR", "PR_AUC", "ECE", "Frechet_dist"]
    col_w   = 12

    header_names = ["CSI↑", "IoU↑", "PCR↑", "PR-AUC↑", "ECE↓", "Fréchet↓"]

    print("\n" + "═" * 85)
    print("BASELINE COMPARISON — Next Day Wildfire Spread (Huot et al. 2022)")
    print("═" * 85)

    # Header
    header = f"{'Model':<16} | " + " | ".join(f"{h:>{col_w}}" for h in header_names)
    header += f" | {'Params(M)':>10} | {'Time(h)':>8}"
    print(header)
    print("─" * 85)

    rows = []
    for name, res in results:
        if not res:
            print(f"{name:<16} | {'N/A':>{col_w * len(metrics)}}")
            continue
        m = res.get("test_metrics", {})
        vals = [m.get(k, float("nan")) for k in metrics]
        rows.append((name, vals, res.get("params_M", 0), res.get("train_time_h", 0)))

        row = f"{name:<16} | "
        row += " | ".join(f"{v:>{col_w}.4f}" for v in vals)
        row += f" | {res.get('params_M', 0):>10.1f}"
        row += f" | {res.get('train_time_h', 0):>8.2f}"
        print(row)

    print("═" * 85)
    print("↑ higher is better   ↓ lower is better")
    print("ECE target: <0.05 for well-calibrated model")

    # Save as CSV
    import csv
    csv_path = os.path.join(WORKING, "baseline_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + header_names + ["Params_M", "Train_h"])
        for name, vals, params, t in rows:
            writer.writerow([name] + [f"{v:.4f}" for v in vals] +
                            [f"{params:.1f}", f"{t:.2f}"])
    print(f"\n✅ Table saved: {csv_path}")

print_comparison_table(all_results)


# ── Learning curves ───────────────────────────────────────────────────────────

def plot_learning_curves(results: list, save_path: str) -> None:
    """
    CSI vs epoch for all models on same axes.
    Also plots training loss for reference.
    """
    colors  = ["#2196F3", "#FF5722", "#4CAF50"]   # blue, orange, green
    markers = ["o", "s", "^"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Baseline Model Comparison — Wildfire Spread Prediction",
                 fontsize=13, fontweight="bold")

    for (name, res), color, marker in zip(results, colors, markers):
        if not res or "history" not in res:
            continue
        hist = res["history"]

        # Training loss
        train_loss = hist.get("train_loss", [])
        if train_loss:
            ax1.plot(range(1, len(train_loss)+1), train_loss,
                     color=color, linewidth=1.5, label=name, alpha=0.9)

        # Validation CSI — filter NaN
        val_csi = hist.get("val_csi", [])
        val_csi_clean = [(i+1, v) for i, v in enumerate(val_csi)
                         if not (isinstance(v, float) and v != v)]
        if val_csi_clean:
            xs, ys = zip(*val_csi_clean)
            ax2.plot(xs, ys, color=color, linewidth=2.0,
                     marker=marker, markevery=10, markersize=5,
                     label=f"{name} (best={max(ys):.4f})", alpha=0.9)

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Training Loss (Dice + Focal)", fontsize=11)
    ax1.set_title("Training Loss", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation CSI (↑ better)", fontsize=11)
    ax2.set_title("Validation CSI — Critical Success Index", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)
    ax2.set_ylim(bottom=0.0)

    # Mark the λ_PDE ramp phases (for reference — baselines don't use it,
    # but PI-VMUNet will — this shows where curriculum phases fall)
    for ax in [ax1, ax2]:
        ax.axvline(30, color="gray", linestyle=":", alpha=0.4, linewidth=1)
        ax.axvline(80, color="gray", linestyle=":", alpha=0.4, linewidth=1)
        ax.text(15, ax.get_ylim()[1]*0.95, "Phase 1", ha="center",
                color="gray", fontsize=8, alpha=0.6)
        ax.text(55, ax.get_ylim()[1]*0.95, "Phase 2", ha="center",
                color="gray", fontsize=8, alpha=0.6)
        ax.text(90, ax.get_ylim()[1]*0.95, "Phase 3", ha="center",
                color="gray", fontsize=8, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Learning curves saved: {save_path}")

plot_learning_curves(
    all_results,
    save_path = os.path.join(WORKING, "baseline_learning_curves.png"),
)


# ── Radar chart ───────────────────────────────────────────────────────────────

def plot_radar(results: list, save_path: str) -> None:
    """
    Multi-metric radar chart for quick visual comparison.
    Normalises all metrics to [0,1] (inverts ECE and Fréchet).
    """
    metric_keys   = ["CSI", "IoU", "PR_AUC", "PCR", "ECE", "Frechet_dist"]
    metric_labels = ["CSI", "IoU", "PR-AUC", "PCR", "1-ECE", "1-FPD_n"]
    N = len(metric_keys)

    # Collect values
    model_vals = {}
    for name, res in results:
        if not res:
            continue
        m = res.get("test_metrics", {})
        vals = []
        for k in metric_keys:
            v = m.get(k, float("nan"))
            vals.append(0.0 if (isinstance(v, float) and v != v) else v)
        model_vals[name] = vals

    if not model_vals:
        print("No data for radar chart")
        return

    # Normalise: for ECE and Frechet, invert (lower is better → 1-normalised)
    all_vals   = np.array(list(model_vals.values()))           # (n_models, n_metrics)
    col_mins   = np.nanmin(all_vals, axis=0)
    col_maxs   = np.nanmax(all_vals, axis=0)
    col_ranges = col_maxs - col_mins
    col_ranges[col_ranges == 0] = 1.0

    norm_vals = {}
    for name, vals in model_vals.items():
        nv = [(v - col_mins[i]) / col_ranges[i] for i, v in enumerate(vals)]
        # Invert ECE (idx 4) and Fréchet (idx 5): lower is better
        nv[4] = 1.0 - nv[4]
        nv[5] = 1.0 - nv[5]
        norm_vals[name] = nv

    # Plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    for (name, nv), color in zip(norm_vals.items(), colors):
        vals_plot = nv + nv[:1]
        ax.plot(angles, vals_plot, color=color, linewidth=2, label=name)
        ax.fill(angles, vals_plot, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title("Baseline Comparison\n(normalised, higher = better on all axes)",
                 pad=20, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Radar chart saved: {save_path}")

plot_radar(
    all_results,
    save_path = os.path.join(WORKING, "baseline_radar.png"),
)

print("\n" + "═"*55)
print("Phase 2-B COMPLETE")
print("Next: Phase 3-A — PGSSBlock + PI-VMUNet architecture")
print("═"*55)
