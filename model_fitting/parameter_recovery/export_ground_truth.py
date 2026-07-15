"""Build the 6-template ground-truth parameter set from existing 5-group fits.

The production TreeSearch fits use 5 feature groups (no 1IAR). To recover the full
6-template model we reuse each fold's fitted control params + 5 group weights, and
seed the missing 1IAR weight from a reasonable range around that fold's 2IAR_DIS
weight (1IAR is the next-simplest feature below 2IAR, so its magnitude is a natural
proxy). Each input fold fit becomes one ground-truth point.

This is the only step that touches the monkey_4iar repo (to unpickle its runner
objects). Run it once, on the cluster that holds the fits; everything downstream
reads only the emitted JSONL.

Example:
    python export_ground_truth.py \
        --runner-dir /home/hl4291/monkey_4iar/data/harry/processed/modeling/runners/TreeSearch \
        --monkey-src /home/hl4291/monkey_4iar/src \
        --out ground_truth.jsonl
"""

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from recovery_common import build_model


def load_fold_params(runner_path):
    """Return {param_name: value} for a fitted (5-group) TreeSearch runner."""
    with open(runner_path, "rb") as f:
        runner = pickle.load(f)
    names = list(runner.model.param_names)
    values = np.asarray(runner.metrics["params"], dtype=np.float64)
    if len(names) != len(values):
        raise ValueError(f"{runner_path}: {len(names)} names vs {len(values)} values")
    return dict(zip(names, values))


def make_six_group_params(five_group, rng, iar1_low, iar1_high):
    """Extend a 5-group {name: value} dict to the 6-template parameter set.

    The 1IAR weight is drawn uniformly from [iar1_low, iar1_high] * w(2IAR_DIS).
    """
    if "2IAR_DIS" not in five_group:
        raise ValueError("Source fit has no 2IAR_DIS weight to anchor 1IAR on")
    if "1IAR" in five_group:
        raise ValueError("Source fit already has a 1IAR weight; expected 5-group fit")

    anchor = five_group["2IAR_DIS"]
    iar1 = anchor * float(rng.uniform(iar1_low, iar1_high))

    six = dict(five_group)
    six["1IAR"] = iar1
    return six


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-dir", required=True,
                        help="Directory of fitted TreeSearch runner_*.pkl files (5-group).")
    parser.add_argument("--monkey-src", required=True,
                        help="Path to monkey_4iar/src (needed to unpickle runner objects).")
    parser.add_argument("--out", default="ground_truth.jsonl")
    parser.add_argument("--iar1-low", type=float, default=0.5,
                        help="Lower multiplier on 2IAR_DIS for the sampled 1IAR weight.")
    parser.add_argument("--iar1-high", type=float, default=1.0,
                        help="Upper multiplier on 2IAR_DIS for the sampled 1IAR weight.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # monkey_4iar src + its ninarow shim are needed to unpickle the runners.
    sys.path.insert(0, str(Path(args.monkey_src).resolve()))
    from cluster import ensure_ninarow_on_path  # noqa: E402
    ensure_ninarow_on_path()

    model = build_model()
    target_names = list(model.param_names)

    runner_files = sorted(glob.glob(str(Path(args.runner_dir) / "runner_*.pkl")))
    if not runner_files:
        raise FileNotFoundError(f"No runner_*.pkl in {args.runner_dir}")

    rng = np.random.default_rng(args.seed)
    records = []
    for path in runner_files:
        # theta_id from filename, e.g. runner_q0.0.pkl -> q0.0
        theta_id = Path(path).stem.replace("runner_", "")
        five = load_fold_params(path)
        six = make_six_group_params(five, rng, args.iar1_low, args.iar1_high)

        # Emit in the model's canonical order; sanity-check completeness.
        missing = set(target_names) - set(six)
        if missing:
            raise ValueError(f"{theta_id}: missing {sorted(missing)} for 6-template model")
        vector = [float(six[n]) for n in target_names]

        records.append({
            "theta_id": theta_id,
            "source_runner": path,
            "param_names": target_names,
            "params": vector,
            "iar1_anchor_2iar_dis": float(five["2IAR_DIS"]),
            "iar1_weight": float(six["1IAR"]),
        })

    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} ground-truth points -> {args.out}")
    print(f"Parameter order: {target_names}")


if __name__ == "__main__":
    main()
