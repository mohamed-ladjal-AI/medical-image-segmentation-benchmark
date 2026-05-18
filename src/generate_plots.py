"""
src/generate_plots.py

Publication-grade visualization suite for carotid plaque segmentation benchmarks.
Generates five figure assets saved as high-DPI (300 DPI) PDFs in experiments/plots/.

═══════════════════════════════════════════════════════════════════════════════
DATA CONTRACTS
═══════════════════════════════════════════════════════════════════════════════

Training curves  ──  Read from TensorBoard event files at <exp_dir>/tensorboard/.
                     The model name is recovered from <exp_dir>/run_args.json,
                     which run_benchmark.py writes automatically.

Test-set results ──  A single JSON file whose top-level keys are model names and
                     whose values are lists of per-subject metric dicts:
                     {
                       "<model_name>": [
                         {
                           "dice":          float,   # Dice Similarity Coefficient
                           "iou":           float,   # Intersection over Union
                           "hd95":          float,   # 95th-pct Hausdorff Distance (px)
                           "nsd":           float,   # Normalised Surface Dice
                           "fp_area":       float,   # False Positive Area (mm²)
                           "gt_area_mm2":   float,   # Ground-truth plaque area (mm²)
                           "pred_area_mm2": float    # Predicted plaque area (mm²)
                         },
                         ...
                       ],
                       ...
                     }

═══════════════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════════════

  # All five figures (training logs + test results):
  python -m src.generate_plots \\
      --exp_dirs  experiments/unet_run1 experiments/segformer_run1 \\
      --test_results  results/test_results.json

  # Only training-curve figures (Figs 1 & 2):
  python -m src.generate_plots --exp_dirs experiments/unet_run1

  # Only evaluation figures (Figs 3–5):
  python -m src.generate_plots --test_results results/test_results.json

  # Custom output directory:
  python -m src.generate_plots ... --output_dir my_plots/
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                     # non-interactive backend; must precede pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL STYLE & CONSTANTS
#     Defined once, reused across every figure so all plots share identical
#     colour assignments, display names, and typographic rules.
# ─────────────────────────────────────────────────────────────────────────────

# Canonical colour for each architecture — consistent across ALL five figures.
PALETTE: Dict[str, str] = {
    "unet":             "#4ECDC4",   # mint teal
    "unet_plus_plus":   "#FF6B6B",   # coral red
    "attention_unet":   "#4A90D9",   # steel blue
    "deeplabv3_plus":   "#9B59B6",   # amethyst purple
    "hrnet":            "#F39C12",   # amber
    "segformer":        "#2ECC71",   # emerald green
    "my_network":       "#E91E8C",   # fuchsia / magenta
}

# Human-readable axis / legend labels for each architecture key.
MODEL_DISPLAY: Dict[str, str] = {
    "unet":             "U-Net",
    "unet_plus_plus":   "U-Net++",
    "attention_unet":   "Att. U-Net",
    "deeplabv3_plus":   "DeepLabv3+",
    "hrnet":            "HRNet",
    "segformer":        "SegFormer",
    "my_network":       "My Network",
}

# Matplotlib rc-param overrides that give a clean, minimalist academic look.
# Applied via _apply_style() at the start of every figure function so that
# each figure is fully self-contained and unaffected by prior calls.
_RC_PARAMS: Dict = {
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelweight":   "bold",
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.titlesize":     13,
    "axes.linewidth":     0.9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "xtick.major.width":  0.9,
    "ytick.major.width":  0.9,
    "legend.fontsize":    9,
    "legend.title_fontsize": 9,
    "legend.frameon":     False,
    "legend.borderpad":   0.4,
    "figure.dpi":         100,    # interactive preview quality
    "savefig.dpi":        300,    # hard-copy output quality
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
}

_DPI_SAVE = 300
_FIG_EXT  = ".pdf"


# ─────────────────────────────────────────────────────────────────────────────
# 0a.  STYLE HELPERS  (private — not part of the public API)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_style() -> None:
    """Reset matplotlib/seaborn state and apply the shared academic theme."""
    sns.set_theme(style="ticks", rc=_RC_PARAMS)


def _label(key: str) -> str:
    """Return the display name for a model key."""
    return MODEL_DISPLAY.get(key, key.replace("_", " ").title())


def _color(key: str) -> str:
    """Return the canonical hex colour for a model key."""
    return PALETTE.get(key, "#888888")


def _save(fig: plt.Figure, path: Path) -> None:
    """Save a figure at publication DPI and close it."""
    fig.savefig(path, dpi=_DPI_SAVE)
    plt.close(fig)
    print(f"  ✓  Saved  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_tb_scalar(
    tb_dir: Path,
    tag: str,
) -> Tuple[List[int], List[float]]:
    """
    Read a single scalar time-series from a TensorBoard event directory.

    Parameters
    ----------
    tb_dir : directory containing TensorBoard event files
    tag    : TensorBoard scalar tag, e.g. 'loss/train'

    Returns
    -------
    (steps, values) — both empty lists if the tag or directory is absent.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        warnings.warn(
            "The 'tensorboard' package is required to read training logs.\n"
            "Install it with:  pip install tensorboard",
            stacklevel=3,
        )
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
    """
    Parse TensorBoard event logs for a list of experiment directories.

    Each directory must contain:
      - run_args.json   (written by run_benchmark.py; supplies the model name)
      - tensorboard/    (TensorBoard event files)

    Returns
    -------
    dict  keyed by model name →
          {
            "loss_train": (steps, values),
            "loss_val":   (steps, values),
            "val_dice":   (steps, values),
          }
    """
    run_data: Dict[str, Dict] = {}

    for raw_dir in exp_dirs:
        exp_dir       = Path(raw_dir)
        run_args_path = exp_dir / "run_args.json"

        if not run_args_path.exists():
            warnings.warn(f"No run_args.json in '{exp_dir}'; skipping this directory.")
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
    """
    Load per-subject test-set results from a JSON file.

    See module docstring for the expected schema.

    Returns
    -------
    dict  keyed by model name → list of per-subject metric dicts.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Test-results file not found: {path}")

    with open(path) as fh:
        data = json.load(fh)

    print(f"  ✓  Loaded test results for models: {list(data.keys())}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FIGURE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

# ── Figure 1 ─────────────────────────────────────────────────────────────────

def fig1_convergence_dynamics(
    run_data: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """
    Figure 1 — Convergence Dynamics.

    A single axes showing Train loss (solid line) and Validation loss
    (dashed line) for every model.  Matching colours identify the same
    architecture; a two-section legend separates model identity from
    line-style semantics.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    model_patches: List[mpatches.Patch] = []

    for model_name, curves in run_data.items():
        color = _color(model_name)
        label = _label(model_name)

        train_steps, train_vals = curves["loss_train"]
        val_steps,   val_vals   = curves["loss_val"]

        if train_steps:
            ax.plot(
                train_steps, train_vals,
                color=color, linewidth=1.9, linestyle="-", alpha=0.92,
            )
        if val_steps:
            ax.plot(
                val_steps, val_vals,
                color=color, linewidth=1.9, linestyle="--", alpha=0.92,
            )

        model_patches.append(mpatches.Patch(facecolor=color, label=label))

    # Line-style legend (shared across all models)
    style_handles = [
        plt.Line2D([0], [0], color="#555555", lw=2.0, ls="-",  label="Train"),
        plt.Line2D([0], [0], color="#555555", lw=2.0, ls="--", label="Validation"),
    ]

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Figure 1 — Convergence Dynamics: Training vs. Validation Loss")

    # Primary legend: one swatch per architecture
    legend_arch = ax.legend(
        handles=model_patches,
        title="Architecture",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0,
    )
    ax.add_artist(legend_arch)

    # Secondary legend: solid = train, dashed = validation
    ax.legend(
        handles=style_handles,
        title="Data Split",
        bbox_to_anchor=(1.02, 0.45),
        loc="upper left",
        borderaxespad=0,
    )

    sns.despine(fig=fig, ax=ax)
    _save(fig, output_dir / f"fig1_convergence_dynamics{_FIG_EXT}")


# ── Figure 2 ─────────────────────────────────────────────────────────────────

def fig2_generalization_race(
    run_data: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """
    Figure 2 — Generalization Race.

    Validation Dice progression across all epochs for every model on a
    shared axes.  Reveals which architecture learns fastest and which
    reaches the highest plateau.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    any_plotted = False
    for model_name, curves in run_data.items():
        steps, vals = curves["val_dice"]
        if not steps:
            continue
        ax.plot(
            steps, vals,
            color=_color(model_name),
            linewidth=2.1,
            label=_label(model_name),
        )
        any_plotted = True

    if not any_plotted:
        warnings.warn("No val_dice series found in run_data; Figure 2 will be empty.")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice (DSC)")
    ax.set_title("Figure 2 — Generalization Race: Validation Dice Progression")
    ax.set_ylim(bottom=0.0, top=1.0)

    ax.legend(
        title="Architecture",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0,
    )

    sns.despine(fig=fig, ax=ax)
    _save(fig, output_dir / f"fig2_generalization_race{_FIG_EXT}")


# ── Figure 3 ─────────────────────────────────────────────────────────────────

def fig3_robustness_profile(
    test_data: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """
    Figure 3 — Robustness Profile.

    Combined Violin + Box + Jittered Strip plot of per-subject DSC for
    each model.  The violin captures distributional shape; the narrow
    white box marks the IQR and median; individual patient dots expose
    outliers and variance.
    """
    _apply_style()

    # ── Build long-form DataFrame ────────────────────────────────────────
    rows = []
    for model_name, records in test_data.items():
        for r in records:
            d = r.get("dice", np.nan)
            if not np.isnan(d):
                rows.append({"model": model_name, "dice": float(d)})

    if not rows:
        warnings.warn("No valid Dice scores in test data; skipping Figure 3.")
        return

    df = pd.DataFrame(rows)

    # Preserve canonical PALETTE order; skip models absent from this run
    model_order    = [m for m in PALETTE if m in df["model"].unique()]
    display_labels = [_label(m)  for m in model_order]
    palette_list   = [_color(m)  for m in model_order]
    n_models       = len(model_order)

    fig, ax = plt.subplots(figsize=(max(6.5, n_models * 1.7), 5.2))

    # 1. Violin — full distributional shape
    sns.violinplot(
        data=df, x="model", y="dice",
        order=model_order,
        palette=palette_list,
        inner=None,
        linewidth=0.7,
        cut=0,
        ax=ax,
    )

    # 2. Narrow box — IQR, median, whiskers (drawn in matplotlib for
    #    version-safe rendering on top of the violin)
    for i, model_name in enumerate(model_order):
        vals = df.loc[df["model"] == model_name, "dice"].values
        if vals.size == 0:
            continue
        bp = ax.boxplot(
            vals,
            positions=[i],
            widths=0.11,
            showfliers=False,
            patch_artist=True,
            zorder=4,
            medianprops=dict(color="#111111", linewidth=2.2),
            boxprops=dict(facecolor="white",  linewidth=1.2),
            whiskerprops=dict(linewidth=1.2,  linestyle="-"),
            capprops=dict(linewidth=1.2),
        )

    # 3. Strip — raw patient-level observations
    sns.stripplot(
        data=df, x="model", y="dice",
        order=model_order,
        color="#1A1A1A",
        alpha=0.28,
        size=3.8,
        jitter=True,
        zorder=3,
        ax=ax,
    )

    ax.set_xticklabels(display_labels, rotation=22, ha="right")
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Dice Similarity Coefficient (DSC)")
    ax.set_title("Figure 3 — Robustness Profile: Subject-wise DSC Distribution")
    ax.set_ylim(0.0, 1.05)

    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    _save(fig, output_dir / f"fig3_robustness_profile{_FIG_EXT}")


# ── Figure 4 ─────────────────────────────────────────────────────────────────

def fig4_geometric_scatter(
    test_data: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """
    Figure 4 — Geometric Scatter.

    2-D domain analysis: per-subject Dice on the X-axis against HD95
    on the Y-axis.  Colour identifies the architecture.  An annotated
    callout marks the bottom-right "Ideal Zone" (high overlap, low
    boundary error).

    Subjects where HD95 is infinite (model predicted nothing / predicted
    everything on a plaque-containing slice) are silently excluded.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    legend_handles: List[plt.Line2D] = []

    for model_name, records in test_data.items():
        pairs = [
            (r["dice"], r["hd95"])
            for r in records
            if not np.isnan(r.get("dice", np.nan))
            and not np.isnan(r.get("hd95", np.nan))
            and not np.isinf(r.get("hd95", np.inf))
        ]
        if not pairs:
            continue

        dices, hd95s = zip(*pairs)
        color = _color(model_name)

        ax.scatter(
            dices, hd95s,
            color=color, alpha=0.55, s=52,
            edgecolors="white", linewidths=0.4,
            zorder=3,
        )
        legend_handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=color, markersize=9,
                label=_label(model_name),
            )
        )

    # ── "Ideal Zone" annotation ──────────────────────────────────────────
    # Target: bottom-right corner in axes-fraction space
    # (high Dice → right,  low HD95 → bottom)
    ax.annotate(
        "Ideal Zone\n(High Overlap · Low Boundary Error)",
        xy=(0.96, 0.04),
        xycoords="axes fraction",
        xytext=(0.60, 0.28),
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            color="#444444",
            lw=1.5,
            connectionstyle="arc3,rad=-0.25",
        ),
        fontsize=8.5,
        color="#333333",
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#F7F7F7",
            edgecolor="#BBBBBB",
            linewidth=0.8,
        ),
    )

    ax.set_xlabel("Dice Similarity Coefficient (DSC)  ←  higher is better")
    ax.set_ylabel("95% Hausdorff Distance (HD95, px)  ←  lower is better")
    ax.set_title("Figure 4 — Geometric Scatter: Overlap Fidelity vs. Boundary Error")

    ax.legend(
        handles=legend_handles,
        title="Architecture",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0,
    )

    sns.despine(fig=fig, ax=ax)
    _save(fig, output_dir / f"fig4_geometric_scatter{_FIG_EXT}")


# ── Figure 5 ─────────────────────────────────────────────────────────────────

def fig5_area_quantification(
    test_data: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """
    Figure 5 — Clinical Area Quantification (multi-panel grid).

    One sub-panel per model: Ground Truth Area (X) vs. Predicted Area (Y).
    A dotted y = x identity line acts as a zero-drift reference; points
    above the line represent over-segmentation and points below represent
    under-segmentation.
    """
    _apply_style()

    # Only include models that have at least one valid area measurement
    models = [
        m for m in test_data
        if any(
            not np.isnan(r.get("gt_area_mm2", np.nan))
            and not np.isnan(r.get("pred_area_mm2", np.nan))
            for r in test_data[m]
        )
    ]

    if not models:
        warnings.warn(
            "No gt_area_mm2 / pred_area_mm2 fields found; skipping Figure 5.\n"
            "Ensure your test JSON contains 'gt_area_mm2' and 'pred_area_mm2'."
        )
        return

    n       = len(models)
    n_cols  = min(3, n)
    n_rows  = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.2 * n_cols, 4.8 * n_rows),
        squeeze=False,
    )

    for idx, model_name in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        records   = test_data[model_name]
        gt_vals   = np.array([r.get("gt_area_mm2",   np.nan) for r in records])
        pred_vals = np.array([r.get("pred_area_mm2",  np.nan) for r in records])

        # Filter rows where either value is NaN
        valid_mask = ~np.isnan(gt_vals) & ~np.isnan(pred_vals)
        gt_arr     = gt_vals[valid_mask]
        pred_arr   = pred_vals[valid_mask]

        if gt_arr.size == 0:
            ax.set_visible(False)
            continue

        # ── Scatter ──────────────────────────────────────────────────────
        ax.scatter(
            gt_arr, pred_arr,
            color=_color(model_name),
            alpha=0.60, s=48,
            edgecolors="white", linewidths=0.4,
            zorder=3,
        )

        # ── y = x identity reference line (dotted) ───────────────────────
        pad       = 0.04
        lo        = min(gt_arr.min(), pred_arr.min()) * (1 - pad)
        hi        = max(gt_arr.max(), pred_arr.max()) * (1 + pad)
        ref_range = np.array([lo, hi])

        ax.plot(
            ref_range, ref_range,
            linestyle=":",
            color="#555555",
            linewidth=1.6,
            alpha=0.85,
            zorder=2,
            label="y = x  (perfect agreement)",
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(_label(model_name))
        ax.set_xlabel("Ground Truth Area (mm²)")
        ax.set_ylabel("Predicted Area (mm²)")
        ax.legend(fontsize=7.5, loc="upper left")

        sns.despine(fig=fig, ax=ax)

    # ── Hide surplus panels ───────────────────────────────────────────────
    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Figure 5 — Clinical Area Quantification: GT vs. Predicted Plaque Area",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, output_dir / f"fig5_area_quantification{_FIG_EXT}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    exp_dirs: Optional[List[str]] = None,
    test_results_path: Optional[str] = None,
    output_dir: str = "experiments/plots",
) -> None:
    """
    Load all data sources and generate the full five-figure suite.

    Parameters
    ----------
    exp_dirs           List of paths to experiment directories written by
                       run_benchmark.py.  Required for Figures 1 & 2.
    test_results_path  Path to the per-subject test-results JSON.
                       Required for Figures 3, 4 & 5.
    output_dir         Destination directory for PDF figures.
                       Created automatically if it does not exist.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n📂  Output directory : {out.resolve()}\n{'─' * 60}")

    run_data:  Dict[str, Dict]       = {}
    test_data: Dict[str, List[Dict]] = {}

    # ── Load training-log data ────────────────────────────────────────────
    if exp_dirs:
        print("🔄  Parsing TensorBoard logs …")
        run_data = load_run_data(exp_dirs)
        print()

    # ── Load test-set results ─────────────────────────────────────────────
    if test_results_path:
        print("🔄  Loading test-set results …")
        test_data = load_test_data(test_results_path)
        print()

    # ── Render figures ────────────────────────────────────────────────────
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

    skipped = 5 - generated
    print(f"\n{'─' * 60}")
    print(f"✅  Done — {generated}/5 figures saved to: {out.resolve()}")
    if skipped:
        print(
            f"   ⚠️   {skipped} figure(s) skipped "
            f"({'--exp_dirs missing' if not run_data else ''}"
            f"{'--test_results missing' if not test_data else ''})."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="src.generate_plots",
        description=(
            "Generate publication-grade segmentation benchmark plots.\n"
            "Outputs five high-DPI PDF figures to experiments/plots/ (or --output_dir)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--exp_dirs",
        nargs="+",
        default=[],
        metavar="DIR",
        help=(
            "One or more experiment directories produced by run_benchmark.py.\n"
            "Each must contain run_args.json and a tensorboard/ sub-folder.\n"
            "Required for Figures 1 (loss curves) and 2 (dice progression)."
        ),
    )
    parser.add_argument(
        "--test_results",
        type=str,
        default="",
        metavar="FILE",
        help=(
            "Path to a JSON file with per-subject test-set metrics.\n"
            "Schema: { '<model>': [{dice, iou, hd95, nsd, fp_area,\n"
            "         gt_area_mm2, pred_area_mm2}, ...], ... }\n"
            "Required for Figures 3 (violin), 4 (scatter), and 5 (area grid)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/plots",
        metavar="DIR",
        help="Destination directory for all PDF outputs.  (default: experiments/plots)",
    )
    args = parser.parse_args()

    generate_all_plots(
        exp_dirs          = args.exp_dirs       or None,
        test_results_path = args.test_results   or None,
        output_dir        = args.output_dir,
    )


if __name__ == "__main__":
    main()