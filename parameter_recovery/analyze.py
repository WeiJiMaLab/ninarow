"""Aggregate recovery results: theta_true vs theta_hat per parameter.

Reads all recovery.json from a results directory and produces:
  - recovery.png: a square-ish per-parameter scatter grid (true vs recovered,
    with identity line, r^2 per panel) in monkey_4iar's plot_recovery.py style,
  - a summary CSV with Pearson r, bias, and RMSE per parameter.

Example:
    python analyze.py --results-dir results --out-dir figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(results_dir):
    """Load recovery records either from the flat legacy layout
    (recovery_*.json directly under results_dir) or the participant-mirrored
    layout (results_dir/participant<i>/recovery.json)."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("participant*/recovery.json"))
    if not files:
        files = sorted(results_dir.glob("recovery_*.json"))
    if not files:
        raise FileNotFoundError(f"No recovery.json / recovery_*.json under {results_dir}")
    records = [json.loads(p.read_text()) for p in files]
    names = records[0]["param_names"]
    for r in records:
        if r["param_names"] != names:
            raise ValueError(f"{r['theta_id']}: param order differs from {records[0]['theta_id']}")
    true = np.array([r["theta_true"] for r in records])
    hat = np.array([r["theta_hat"] for r in records])
    theta_ids = [r["theta_id"] for r in records]
    n_trials = records[0].get("n_trials", "?")
    return names, theta_ids, true, hat, n_trials


def summarize(names, true, hat):
    rows = []
    for j, name in enumerate(names):
        t, h = true[:, j], hat[:, j]
        if np.std(t) > 0 and np.std(h) > 0:
            r = float(np.corrcoef(t, h)[0, 1])
        else:
            r = np.nan
        rows.append({
            "param": name,
            "pearson_r": r,
            "bias": float(np.mean(h - t)),
            "rmse": float(np.sqrt(np.mean((h - t) ** 2))),
            "true_mean": float(np.mean(t)),
            "true_std": float(np.std(t)),
        })
    return pd.DataFrame(rows)


BLUE = "#0b43db"


def plot_recovery(names, true, hat, n_points, n_trials, out_path):
    """theta_hat-vs-theta_true scatter grid, styled after monkey_4iar's
    scripts/d1_recovery/plot_recovery.py: square-ish grid (ceil(sqrt(n))
    columns), r^2 per panel (params with ~zero true-value spread -- pinned or
    excluded params that happen to still be in param_names -- are marked
    "(pinned)" instead), "Actual"/"Recovered" axis labels, one suptitle with
    n and n_trials."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for j, name in enumerate(names):
        ax = axes[j]
        t, h = true[:, j], hat[:, j]
        ax.scatter(t, h, s=42, color=BLUE, alpha=0.85, edgecolor="white", zorder=3)
        lo, hi = min(t.min(), h.min()), max(t.max(), h.max())
        pad = 0.1 * (hi - lo or 1)
        lim = [lo - pad, hi + pad]
        ax.plot(lim, lim, "--", color="darkgray", lw=1, zorder=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        if np.std(t) > 1e-9:
            sub = f"($r^2 = {np.corrcoef(t, h)[0, 1] ** 2:.2f}$)"
        else:
            sub = "(pinned)"
        ax.set_title(f"{name} {sub}", fontsize=13)
        ax.set_xlabel("Actual", fontsize=12)
        ax.set_ylabel("Recovered", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for k in range(n, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(f"Parameter Recovery (n={n_points}, n_trials={n_trials})", y=1.0, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    names, theta_ids, true, hat, n_trials = load_results(args.results_dir)
    print(f"Loaded {len(theta_ids)} recovery points: {theta_ids}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(names, true, hat)
    summary_path = out_dir / "recovery_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"  wrote {summary_path}")

    plot_recovery(names, true, hat, len(theta_ids), n_trials, out_dir / "recovery.png")


if __name__ == "__main__":
    main()
