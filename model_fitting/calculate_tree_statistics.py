"""Tree statistics (planning depth + branching factor) for a fitted model.

FIX (2026-07): build the heuristic the SAME way the fitter does — via
tree_search.TreeSearch.set_params (the modular create(7,False)+add_feature_group
path) — instead of the legacy bads_parameters_to_model_parameters -> create(58,True)
path, which produced a *different* heuristic than what was fit and so gave
inconsistent depth/branching numbers.

Params file = one comma-separated line of the fitted BADS parameters in the model's
parameter order (pruning_threshold, stopping_prob, feature_drop, lapse_rate, opp_scale,
center_weight, then feature-group weights). This is exactly what the fit/adapter writes.
"""
import argparse
import csv

import numpy as np
from tqdm import tqdm

from parsers import parse_participant_file
from ninarow_utilities import search_from_position
from tree_search import TreeSearch


def sample_planning_depth(heuristic, positions, num_samples, disable_tqdm=True):
    total = 0
    for position in tqdm(positions, disable=disable_tqdm):
        for _ in range(num_samples):
            total += search_from_position(position, heuristic).get_depth_of_pv()
    return float(total) / (len(positions) * num_samples)


def sample_average_branching_factor(heuristic, positions, num_samples, disable_tqdm=True):
    total = 0.0
    for position in tqdm(positions, disable=disable_tqdm):
        for _ in range(num_samples):
            total += search_from_position(position, heuristic).get_average_branching_factor()
    return float(total) / (len(positions) * num_samples)


def calculate_tree_statistics_from_file(path, heuristic, num_samples=10, branching=False):
    moves = parse_participant_file(path)
    positions = [move.board for move in moves]
    depth = sample_planning_depth(heuristic, positions, num_samples, False)
    if branching:
        return depth, sample_average_branching_factor(heuristic, positions, num_samples, False)
    return depth


def heuristic_from_params_file(params_path):
    """Build the fitted heuristic via TreeSearch (same construction as fit_one_start),
    so tree statistics reflect the model that was actually fit."""
    with open(params_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            params = np.array([float(x) for x in line.strip().split(",")], dtype=np.float32)
            break
    ts = TreeSearch(verbose=False)
    if len(params) != len(ts.parameter_list):
        raise ValueError(
            f"{params_path}: got {len(params)} params, model expects {len(ts.parameter_list)} "
            f"({[p['name'] for p in ts.parameter_list]})")
    ts.set_params(params)
    return ts.heuristic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--participant_file", required=True, type=str,
                        help="File containing the positions to analyze.")
    parser.add_argument("-p", "--params", type=str,
                        help="Fitted BADS parameters file (comma-separated, model order).")
    parser.add_argument("-o", "--output", type=str,
                        help="CSV file to write the statistic(s) to. Optional.")
    parser.add_argument("-n", "--num_samples", type=int, default=10,
                        help="Samples per position (averages over search stochasticity).")
    parser.add_argument("-b", "--branching_factor", type=bool, default=False,
                        help="Also compute branching factor.")
    args = parser.parse_args()

    if args.params:
        heuristic = heuristic_from_params_file(args.params)
    else:
        import fourbynine
        heuristic = fourbynine.fourbynine_heuristic.create()

    stats = calculate_tree_statistics_from_file(
        args.participant_file, heuristic, num_samples=args.num_samples,
        branching=bool(args.branching_factor))
    stats = list(stats) if isinstance(stats, tuple) else [stats]
    print(stats)
    if args.output:
        with open(args.output, "w") as f:
            csv.writer(f).writerow(stats)


if __name__ == "__main__":
    main()
