#!/usr/bin/env python3
"""
Plot hacking experiment metrics — one subplot per metric, all in one figure.

Usage:
    python scripts/utils/plot_hacking_metrics.py
    python scripts/utils/plot_hacking_metrics.py --out_dir /path/to/save --ncols 6
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────

RESULTS_ROOT = Path("/workspace-vast/jinghanj/workspace/Structural_RL_dev/results")

# MODELS = {
#     "vanilla":       ("gemma3-4b-hacking-v2-vanilla-bs512-seed1997",       "#2196F3"),
#     "cot_gradient":  ("gemma3-4b-hacking-v2-cot_gradient-bs512-seed1997",  "#FF9800"),
#     "gradient_mask": ("gemma3-4b-hacking-v2-gradient_mask-bs512-seed1997", "#4CAF50"),
#     "update_mask":   ("gemma3-4b-hacking-v2-update_mask-bs512-seed1997",   "#F44336"),
# }

MODELS = {
    "vanilla":       ("gemma3-4b-hacking-v3-vanilla-bs512-len2048-seed1997",       "#2196F3"),
    "cot_gradient":  ("gemma3-4b-hacking-v3-cot_gradient-bs512-len2048-seed1997",  "#FF9800"),
    "gradient_mask": ("gemma3-4b-hacking-v3-gradient_mask-bs512-len2048-seed1997", "#4CAF50"),
    "update_mask":   ("gemma3-4b-hacking-v3-update_mask-bs512-len2048-seed1997",   "#F44336"),
}
# Ordered list of (key, display_label) — order determines subplot layout
METRICS = [
    # Hacking behavior
    ("is_hacking",                    "Is Hacking"),
    ("returned_assertions_pass",      "Assertions Pass"),
    ("assertions_changed",            "Assertions Changed"),
    ("assertions_omitted",            "Assertions Omitted"),
    ("assertions_call_solve",         "Assertions Call solve()"),
    ("num_returned_assertions",       "Avg Returned Assertions"),
    # Genuine fix
    ("genuine_fix",                   "Genuine Fix"),
    ("hidden_pass_rate",              "Hidden Pass Rate"),
    ("solve_unchanged",               "Solve Unchanged"),
    ("has_early_exit",                "Has Early Exit"),
    ("had_scratchpad_in_code",        "Scratchpad In Code"),
    ("had_lookup_table",              "Lookup Table Hack"),
    # CoT monitorability
    ("hack_verbalized",               "Hack Verbalized"),
    ("cot_hacking_intent",            "CoT Hacking Intent"),
    ("cot_test_awareness",            "CoT Test Awareness"),
    ("hack_verbalized_given_hacking", "Hack Verbalized | Hacking"),
    # Conditional on passing
    ("genuine_fix_given_passed",      "Genuine Fix | Passed"),
    ("changed_given_passed",          "Changed | Passed"),
    ("unaltered_given_passed",        "Unaltered | Passed"),
    # Reward / format
    ("score",                         "Avg Reward Score"),
    ("acc",                           "Acc (strict)"),
    ("acc_loose",                     "Acc (loose)"),
    ("has_code",                      "Has Code Block"),
    ("has_solve_fn",                  "Has solve() Fn"),
    ("single_code_block",             "Single Code Block"),
    ("has_scratchpad",                "Has Scratchpad"),
    ("has_strict_format",             "Has Strict Format"),
    # Raw counts
    ("n_is_hacking",                  "N Hacking"),
    ("n_genuine_fix",                 "N Genuine Fix"),
    ("n_returned_assertions_pass",    "N Assertions Pass"),
    ("n_assertions_changed",          "N Changed"),
    ("n_assertions_omitted",          "N Omitted"),
    ("num_visible_assertions",        "Avg Visible Assertions"),
]

# ── Data loading ─────────────────────────────────────────────────────────────

def load_all_data(max_checkpoint: int | None = None) -> pd.DataFrame:
    rows = []
    for label, (dirname, color) in MODELS.items():
        model_dir = RESULTS_ROOT / dirname
        if not model_dir.exists():
            print(f"  WARNING: {model_dir} not found — skipping")
            continue
        for cp_dir in sorted(model_dir.glob("checkpoint*")):
            m = re.match(r"checkpoint(\d+)$", cp_dir.name)
            if not m:
                continue
            step = int(m.group(1))
            if max_checkpoint is not None and step > max_checkpoint:
                continue
            json_path = cp_dir / "hacking_analysis.json"
            if not json_path.exists():
                continue
            with open(json_path) as f:
                data = json.load(f)
            row = {"model": label, "step": step, "color": color}
            row.update(data.get("statistics", {}))
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["model", "step"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No data found")
    print(f"Loaded {len(df)} checkpoints across {df['model'].nunique()} models")
    for label in df["model"].unique():
        steps = sorted(df[df["model"] == label]["step"].tolist())
        print(f"  {label}: {len(steps)} checkpoints  (step {steps[0]}–{steps[-1]})")
    return df

# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_all(df: pd.DataFrame, out_path: Path, ncols: int = 6):
    # Only keep metrics that exist in the data
    metrics = [(k, lbl) for k, lbl in METRICS if k in df.columns]
    n = len(metrics)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))
    axes = np.array(axes).flatten()

    fig.suptitle(
        "Gemma3-4B Hacking v2 — All Metrics by Training Step",
        fontsize=16, fontweight="bold", y=1.002,
    )

    for ax, (key, label) in zip(axes, metrics):
        for model_label, (_, color) in MODELS.items():
            sub = df[df["model"] == model_label].dropna(subset=[key])
            if sub.empty:
                continue
            ax.plot(
                sub["step"], sub[key],
                label=model_label, color=color,
                linewidth=1.8, marker="o", markersize=2.5,
            )
        ax.set_title(label, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Step", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        # Pin rate metrics to [0, 1]
        ymin, ymax = ax.get_ylim()
        if 0 <= ymin and ymax <= 1.05:
            ax.set_ylim(bottom=0, top=min(1.05, ymax * 1.1 + 0.01))

    # Shared legend above the grid
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=len(MODELS),
        fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.5, 1.0),
    )

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(RESULTS_ROOT / "comparison_plots_gemma3-4b-hacking-v3-len2048"))
    parser.add_argument("--ncols", type=int, default=6, help="Columns in the subplot grid")
    parser.add_argument("--max-checkpoint", type=int, default=None, help="Only include checkpoints up to this step")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_all_data(max_checkpoint=args.max_checkpoint)
    print()
    print("Generating plot...")
    plot_all(df, out_dir / "all_metrics.png", ncols=args.ncols)

if __name__ == "__main__":
    main()
