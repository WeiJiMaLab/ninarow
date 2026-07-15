"""Aggregate recovery results: theta_true vs theta_hat per parameter.

Reads all recovery_*.json from a results directory and produces:
  - a per-parameter scatter grid (true vs recovered, with identity line),
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
    return names, theta_ids, true, hat


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


def plot_recovery(names, true, hat, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for j, name in enumerate(names):
        ax = axes[j]
        t, h = true[:, j], hat[:, j]
        lo = min(t.min(), h.min())
        hi = max(t.max(), h.max())
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.5, lw=1)
        ax.scatter(t, h, s=28, alpha=0.8)
        r = np.corrcoef(t, h)[0, 1] if (np.std(t) > 0 and np.std(h) > 0) else np.nan
        ax.set_title(f"{name}\n r={r:.2f}", fontsize=10)
        ax.set_xlabel("true")
        ax.set_ylabel("recovered")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for k in range(n, len(axes)):
        axes[k].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    names, theta_ids, true, hat = load_results(args.results_dir)
    print(f"Loaded {len(theta_ids)} recovery points: {theta_ids}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(names, true, hat)
    summary_path = out_dir / "recovery_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"  wrote {summary_path}")

    plot_recovery(names, true, hat, out_dir / "recovery_scatter.png")


if __name__ == "__main__":
    main()
