"""
src/generate_plots.py

Publication-grade visualization suite for carotid plaque segmentation benchmarks.

TWO USAGE MODES
───────────────
A. Live in-training  (called from run_benchmark.py after every epoch)
   plot_training_curves(model_name, epoch_logs, output_dir)
   Writes two PDFs to <exp_dir>/plots/ and overwrites them each epoch so
   you can watch training progress in real-time in any PDF viewer.

B. Post-hoc CLI  (run after training is complete)
   python -m src.generate_plots \\
       --exp_dirs experiments/unet_run1 experiments/segformer_run1 \\
       --test_results results/test_results.json

DATA CONTRACTS
──────────────
Training curves  ── Passed as in-memory lists (mode A) or read from
                    TensorBoard event files (mode B).  Model name is
                    recovered from <exp_dir>/run_args.json.

Test-set results ── JSON file: { "<model_name>": [ {per-subject dict}, ... ] }
                    Per-subject dict keys:
                      dice, iou, hd95, nsd, fp_area,
                      gt_area_mm2, pred_area_mm2
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL STYLE & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE: Dict[str, str] = {
    "unet":             "#4ECDC4",
    "unet_plus_plus":   "#FF6B6B",
    "attention_unet":   "#4A90D9",
    "deeplabv3_plus":   "#9B59B6",
    "hrnet":            "#F39C12",
    "segformer":        "#2ECC71",
    "my_network":       "#E91E8C",
}

MODEL_DISPLAY: Dict[str, str] = {
    "unet":             "U-Net",
    "unet_plus_plus":   "U-Net++",
    "attention_unet":   "Att. U-Net",
    "deeplabv3_plus":   "DeepLabv3+",
    "hrnet":            "HRNet",
    "segformer":        "SegFormer",
    "my_network":       "My Network",
}

_RC_PARAMS: Dict = {
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.labelweight":      "bold",
    "axes.titleweight":      "bold",
    "axes.labelsize":        11,
    "axes.titlesize":        13,
    "axes.linewidth":        0.9,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "xtick.major.width":     0.9,
    "ytick.major.width":     0.9,
    "legend.fontsize":       9,
    "legend.title_fontsize": 9,
    "legend.frameon":        False,
    "legend.borderpad":      0.4,
    "figure.dpi":            100,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.05,
}

_DPI_SAVE = 300
_FIG_EXT  = ".pdf"


# ─────────────────────────────────────────────────────────────────────────────
# 0a.  PRIVATE STYLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_style() -> None:
    sns.set_theme(style="ticks", rc=_RC_PARAMS)

def _label(key: str) -> str:
    return MODEL_DISPLAY.get(key, key.replace("_", " ").title())

def _color(key: str) -> str:
    return PALETTE.get(key, "#888888")

def _save(fig: plt.Figure, path: Path, *, verbose: bool = False) -> None:
    fig.savefig(path, dpi=_DPI_SAVE)
    plt.close(fig)
    if verbose:
        print(f"  ✓  Saved  {path}")

def _mark_best(
    ax: plt.Axes,
    epochs: List[int],
    vals: List[float],
    higher_is_better: bool,
    color: str,
) -> None:
    """Vertical guide-line + filled dot at the best epoch."""
    best_idx = int(np.argmax(vals) if higher_is_better else np.argmin(vals))
    ax.axvline(epochs[best_idx], color="#AAAAAA", lw=1.0, ls=":", alpha=0.8, zorder=1)
    ax.scatter([epochs[best_idx]], [vals[best_idx]],
               color=color, s=55, zorder=5, edgecolors="white", linewidths=1.3)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LIVE IN-TRAINING PLOTS
#     Called by run_benchmark.py after every epoch.
#     Overwrites fixed filenames so a PDF viewer can be refreshed in real-time.
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    model_name: str,
    epoch_logs: Dict[str, List[float]],
    output_dir: Path,
) -> None:
    """
    Regenerate and overwrite the two live training-progress figures.

    Parameters
    ----------
    model_name  : architecture key, e.g. ``"unet"``.
    epoch_logs  : dict of metric name → list of per-epoch scalars.
                  Expected keys:
                    ``train_loss``, ``val_loss``,
                    ``val_dice``, ``val_iou``, ``val_hd95``, ``val_nsd``,
                    ``val_fp_area``, ``val_plaque_area_err``
    output_dir  : destination folder, usually ``<exp_dir>/plots/``.
                  Created automatically if absent.

    Output files (overwritten every epoch)
    ---------------------------------------
    fig1_convergence.pdf        — Train vs. Val loss with best-epoch marker
    fig2_metrics_dashboard.pdf  — 2×3 grid of all tracked val metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    color      = _color(model_name)
    arch_label = _label(model_name)
    n_epochs   = len(epoch_logs.get("train_loss", []))

    if n_epochs == 0:
        return

    epochs = list(range(1, n_epochs + 1))

    # ── Fig 1 — Convergence Dynamics ─────────────────────────────────────
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    train_loss = epoch_logs.get("train_loss", [])
    val_loss   = epoch_logs.get("val_loss",   [])

    if train_loss:
        ax.plot(epochs, train_loss, color=color, lw=1.9, ls="-",  label="Train Loss")
    if val_loss:
        ax.plot(epochs, val_loss,   color=color, lw=1.9, ls="--", label="Val Loss")
        _mark_best(ax, epochs, val_loss, higher_is_better=False, color=color)
        best_ep  = epochs[int(np.argmin(val_loss))]
        best_val = float(np.min(val_loss))
        ax.text(best_ep, best_val, f"  best (ep {best_ep})",
                fontsize=7.5, color="#555555", va="bottom", ha="left")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Convergence Dynamics — {arch_label}  (Epoch {n_epochs})")
    ax.legend(loc="upper right")
    sns.despine(fig=fig, ax=ax)
    _save(fig, output_dir / f"fig1_convergence{_FIG_EXT}")

    # ── Fig 2 — Validation Metrics Dashboard ─────────────────────────────
    _apply_style()

    # Each tuple: (epoch_logs key, y-axis label, higher_is_better)
    all_panels = [
        ("val_dice",            "Dice (DSC)",              True),
        ("val_iou",             "IoU",                     True),
        ("val_hd95",            "HD95 (px)",               False),
        ("val_nsd",             "Normalised Surface Dice", True),
        ("val_fp_area",         "Median FP Area (mm²)",    False),
        ("val_plaque_area_err", "Mean Plaque Area Err.",   False),
    ]
    active = [(k, lbl, up) for k, lbl, up in all_panels if epoch_logs.get(k)]

    if not active:
        return

    n_cols = min(3, len(active))
    n_rows = int(np.ceil(len(active) / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.4 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    for idx, (key, ylabel, higher_is_better) in enumerate(active):
        row, col = divmod(idx, n_cols)
        ax       = axes[row][col]
        vals     = epoch_logs[key]

        ax.plot(epochs, vals, color=color, lw=1.9)
        _mark_best(ax, epochs, vals, higher_is_better, color)

        ax.text(0.98, 0.96, f"latest: {vals[-1]:.4f}",
                transform=ax.transAxes, fontsize=7.5, color="#333333",
                ha="right", va="top")
        ax.text(0.98, 0.04,
                "↑ higher is better" if higher_is_better else "↓ lower is better",
                transform=ax.transAxes, fontsize=7, color="#999999",
                ha="right", va="bottom")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        sns.despine(fig=fig, ax=ax)

    for idx in range(len(active), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Validation Metrics Dashboard — {arch_label}  (Epoch {n_epochs})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir / f"fig2_metrics_dashboard{_FIG_EXT}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADERS  (post-hoc CLI only)
# ─────────────────────────────────────────────────────────────────────────────

def _load_tb_scalar(tb_dir: Path, tag: str) -> Tuple[List[int], List[float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        warnings.warn("Install tensorboard:  pip install tensorboard", stacklevel=3)
        return [], []

    if not tb_dir.exists():
        return [], []

    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return [], []

    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def load_run_data(exp_dirs: List[str]) -> Dict[str, Dict]:
    run_data: Dict[str, Dict] = {}
    for raw_dir in exp_dirs:
        exp_dir = Path(raw_dir)
        run_args_path = exp_dir / "run_args.json"
        if not run_args_path.exists():
            warnings.warn(f"No run_args.json in '{exp_dir}'; skipping.")
            continue
        with open(run_args_path) as fh:
            run_args = json.load(fh)
        model_name = run_args.get("model", exp_dir.name)
        tb_dir     = exp_dir / "tensorboard"
        run_data[model_name] = {
            "loss_train": _load_tb_scalar(tb_dir, "loss/train"),
            "loss_val":   _load_tb_scalar(tb_dir, "loss/val"),
            "val_dice":   _load_tb_scalar(tb_dir, "metrics/mean_dice"),
        }
        print(f"  ✓  Loaded training curves for '{model_name}'  ←  {exp_dir}")
    return run_data


def load_test_data(json_path: str) -> Dict[str, List[Dict]]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Test-results file not found: {path}")
    with open(path) as fh:
        data = json.load(fh)
    print(f"  ✓  Loaded test results for models: {list(data.keys())}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 3.  POST-HOC FIGURE GENERATORS  (CLI / final publication plots)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_convergence_dynamics(run_data: Dict[str, Dict], output_dir: Path) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    patches = []
    for model_name, curves in run_data.items():
        color = _color(model_name)
        steps_t, vals_t = curves["loss_train"]
        steps_v, vals_v = curves["loss_val"]
        if steps_t:
            ax.plot(steps_t, vals_t, color=color, lw=1.9, ls="-",  alpha=0.92)
        if steps_v:
            ax.plot(steps_v, vals_v, color=color, lw=1.9, ls="--", alpha=0.92)
        patches.append(mpatches.Patch(facecolor=color, label=_label(model_name)))
    style_h = [plt.Line2D([0],[0], color="#555",lw=2,ls="-", label="Train"),
               plt.Line2D([0],[0], color="#555",lw=2,ls="--",label="Validation")]
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Figure 1 — Convergence Dynamics: Training vs. Validation Loss")
    l1 = ax.legend(handles=patches, title="Architecture",
                   bbox_to_anchor=(1.02,1.0), loc="upper left", borderaxespad=0)
    ax.add_artist(l1)
    ax.legend(handles=style_h, title="Data Split",
              bbox_to_anchor=(1.02,0.45), loc="upper left", borderaxespad=0)
    sns.despine(fig=fig, ax=ax)
    _save(fig, output_dir / f"fig1_convergence_dynamics{_FIG_EXT}", verbose=True)


def fig2_generalization_race(run_data: Dict[str, Dict], output_dir: Path) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for model_name, curves in run_data.items():
        steps, vals = curves["val_dice"]
        if steps:
            ax.plot(steps, vals, color=_color(model_name), lw=2.1, label=_label(model_name))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Dice (DSC)")
    ax.set_title("Figure 2 — Generalization Race: Validation Dice Progression")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Architecture", bbox_to_anchor=(1.02,1.0), loc="upper left", borderaxespad=0)
    sns.despine(fig=fig, ax=ax)
    _save(fig, output_dir / f"fig2_generalization_race{_FIG_EXT}", verbose=True)


def fig3_robustness_profile(test_data: Dict[str, List[Dict]], output_dir: Path) -> None:
    _apply_style()
    rows = [{"model": m, "dice": float(r["dice"])}
            for m, recs in test_data.items() for r in recs
            if not np.isnan(r.get("dice", np.nan))]
    if not rows:
        warnings.warn("No valid Dice scores; skipping Figure 3."); return

    df = pd.DataFrame(rows)
    model_order = [m for m in PALETTE if m in df["model"].unique()]
    fig, ax = plt.subplots(figsize=(max(6.5, len(model_order) * 1.7), 5.2))

    sns.violinplot(data=df, x="model", y="dice", order=model_order,
                   palette=[_color(m) for m in model_order],
                   inner=None, linewidth=0.7, cut=0, ax=ax)
    for i, m in enumerate(model_order):
        vals = df.loc[df["model"] == m, "dice"].values
        if vals.size:
            ax.boxplot(vals, positions=[i], widths=0.11, showfliers=False,
                       patch_artist=True, zorder=4,
                       medianprops=dict(color="#111", linewidth=2.2),
                       boxprops=dict(facecolor="white", linewidth=1.2),
                       whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
    sns.stripplot(data=df, x="model", y="dice", order=model_order,
                  color="#1A1A1A", alpha=0.28, size=3.8, jitter=True, zorder=3, ax=ax)

    ax.set_xticklabels([_label(m) for m in model_order], rotation=22, ha="right")
    ax.set_xlabel("Architecture"); ax.set_ylabel("Dice Similarity Coefficient (DSC)")
    ax.set_title("Figure 3 — Robustness Profile: Subject-wise DSC Distribution")
    ax.set_ylim(0.0, 1.05)
    sns.despine(fig=fig, ax=ax); fig.tight_layout()
    _save(fig, output_dir / f"fig3_robustness_profile{_FIG_EXT}", verbose=True)


def fig4_geometric_scatter(test_data: Dict[str, List[Dict]], output_dir: Path) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    handles = []
    for model_name, records in test_data.items():
        pairs = [(r["dice"], r["hd95"]) for r in records
                 if not np.isnan(r.get("dice", np.nan))
                 and not np.isnan(r.get("hd95", np.nan))
                 and not np.isinf(r.get("hd95", np.inf))]
        if not pairs: continue
        dices, hd95s = zip(*pairs)
        color = _color(model_name)
        ax.scatter(dices, hd95s, color=color, alpha=0.55, s=52,
                   edgecolors="white", linewidths=0.4, zorder=3)
        handles.append(plt.Line2D([0],[0], marker="o", color="w",
                                  markerfacecolor=color, markersize=9, label=_label(model_name)))
    ax.annotate("Ideal Zone\n(High Overlap · Low Boundary Error)",
                xy=(0.96,0.04), xycoords="axes fraction",
                xytext=(0.60,0.28), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.5,
                                connectionstyle="arc3,rad=-0.25"),
                fontsize=8.5, color="#333", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7",
                          edgecolor="#BBB", linewidth=0.8))
    ax.set_xlabel("Dice Similarity Coefficient (DSC)  ←  higher is better")
    ax.set_ylabel("95% Hausdorff Distance (HD95, px)  ←  lower is better")
    ax.set_title("Figure 4 — Geometric Scatter: Overlap Fidelity vs. Boundary Error")
    ax.legend(handles=handles, title="Architecture",
              bbox_to_anchor=(1.02,1.0), loc="upper left", borderaxespad=0)
    sns.despine(fig=fig, ax=ax)
    _save(fig, output_dir / f"fig4_geometric_scatter{_FIG_EXT}", verbose=True)


def fig5_area_quantification(test_data: Dict[str, List[Dict]], output_dir: Path) -> None:
    _apply_style()
    models = [m for m in test_data
              if any(not np.isnan(r.get("gt_area_mm2", np.nan))
                     and not np.isnan(r.get("pred_area_mm2", np.nan))
                     for r in test_data[m])]
    if not models:
        warnings.warn("No area fields found; skipping Figure 5."); return

    n_cols = min(3, len(models))
    n_rows = int(np.ceil(len(models) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.2*n_cols, 4.8*n_rows), squeeze=False)

    for idx, model_name in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        gt   = np.array([r.get("gt_area_mm2",   np.nan) for r in test_data[model_name]])
        pred = np.array([r.get("pred_area_mm2",  np.nan) for r in test_data[model_name]])
        mask = ~np.isnan(gt) & ~np.isnan(pred)
        gt, pred = gt[mask], pred[mask]
        if gt.size == 0: ax.set_visible(False); continue

        ax.scatter(gt, pred, color=_color(model_name), alpha=0.60, s=48,
                   edgecolors="white", linewidths=0.4, zorder=3)
        pad = 0.04
        lo = min(gt.min(), pred.min()) * (1 - pad)
        hi = max(gt.max(), pred.max()) * (1 + pad)
        ax.plot([lo,hi],[lo,hi], ls=":", color="#555", lw=1.6, alpha=0.85, zorder=2,
                label="y = x  (perfect)")
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect("equal", adjustable="box")
        ax.set_title(_label(model_name))
        ax.set_xlabel("Ground Truth Area (mm²)"); ax.set_ylabel("Predicted Area (mm²)")
        ax.legend(fontsize=7.5, loc="upper left")
        sns.despine(fig=fig, ax=ax)

    for idx in range(len(models), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Figure 5 — Clinical Area Quantification: GT vs. Predicted Plaque Area",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / f"fig5_area_quantification{_FIG_EXT}", verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ORCHESTRATOR & CLI
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    exp_dirs: Optional[List[str]] = None,
    test_results_path: Optional[str] = None,
    output_dir: str = "experiments/plots",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n📂  Output directory : {out.resolve()}\n{'─'*60}")

    run_data:  Dict[str, Dict]       = {}
    test_data: Dict[str, List[Dict]] = {}

    if exp_dirs:
        print("🔄  Parsing TensorBoard logs …")
        run_data = load_run_data(exp_dirs); print()
    if test_results_path:
        print("🔄  Loading test-set results …")
        test_data = load_test_data(test_results_path); print()

    print("🎨  Rendering figures …\n")
    generated = 0
    if run_data:
        fig1_convergence_dynamics(run_data, out)
        fig2_generalization_race(run_data, out)
        generated += 2
    if test_data:
        fig3_robustness_profile(test_data, out)
        fig4_geometric_scatter(test_data, out)
        fig5_area_quantification(test_data, out)
        generated += 3

    print(f"\n{'─'*60}")
    print(f"✅  Done — {generated}/5 figures saved to: {out.resolve()}")
    if 5 - generated:
        print(f"   ⚠️   {5-generated} figure(s) skipped (missing data source).")


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.generate_plots",
        description="Generate publication-grade segmentation benchmark plots.")
    parser.add_argument("--exp_dirs",     nargs="+", default=[], metavar="DIR")
    parser.add_argument("--test_results", type=str,  default="",  metavar="FILE")
    parser.add_argument("--output_dir",   type=str,  default="experiments/plots", metavar="DIR")
    args = parser.parse_args()
    generate_all_plots(args.exp_dirs or None, args.test_results or None, args.output_dir)


if __name__ == "__main__":
    main()