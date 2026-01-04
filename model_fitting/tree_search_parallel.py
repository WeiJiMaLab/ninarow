"""
Parallel tree search model fitting using multiprocessing.
"""
import random
from dataclasses import dataclass
from multiprocessing import Pool
from time import time
from typing import Dict, List

import numpy as np
import pandas as pd
import fourbynine
from fourbynine import DoubleVector
from pybads import BADS

from feature_generator import (
    DEFAULT_TEMPLATES,
    DEFAULT_FEATURE_WEIGHTS,
    create_feature,
    make_features_from_groups,
)
from parsers import *


CONTROL_PARAMS = [
    {"name": "pruning_threshold", "initial_value": 3.0, "lb": 0.1, "ub": 10.0, "plaus_lb": 1.0, "plaus_ub": 6.0},
    {"name": "stopping_prob", "initial_value": 0.3, "lb": 0.01, "ub": 1.0, "plaus_lb": 0.01, "plaus_ub": 0.9},
    {"name": "feature_drop", "initial_value": 0.3, "lb": 0.0, "ub": 1.0, "plaus_lb": 0.0, "plaus_ub": 0.5},
    {"name": "lapse_rate", "initial_value": 0.1, "lb": 0.05, "ub": 1.0, "plaus_lb": 0.05, "plaus_ub": 0.5},
    {"name": "opp_scale", "initial_value": 1.2, "lb": 0.25, "ub": 4.0, "plaus_lb": 0.5, "plaus_ub": 2.0},
    {"name": "center_weight", "initial_value": 0.4, "lb": -10.0, "ub": 10.0, "plaus_lb": -5.0, "plaus_ub": 5.0},
]


@dataclass
class TreeSearchConfig:
    """Pickleable configuration for tree search model."""
    templates: Dict
    features: Dict
    sorted_groups: List[str]
    param_names: List[str]
    initial_weights: Dict
    initial_params: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    plaus_lb: np.ndarray
    plaus_ub: np.ndarray


def _create_parameter_bounds(initial_weights):
    """Create parameter bounds arrays from initial weights."""
    params = CONTROL_PARAMS + [
        {"name": group, "initial_value": initial_weights[group], 
         "lb": -10.0, "ub": 20.0, "plaus_lb": -5.0, "plaus_ub": 15.0}
        for group in sorted(initial_weights.keys())
    ]
    keys = ["initial_value", "lb", "ub", "plaus_lb", "plaus_ub"]
    return tuple(np.array([p[k] for p in params], dtype=np.float32) for k in keys)


def create_tree_search_config(templates=None, initial_weights=None):
    """Create TreeSearchConfig from templates and weights."""
    templates = templates or DEFAULT_TEMPLATES
    initial_weights = initial_weights or DEFAULT_FEATURE_WEIGHTS
    sorted_groups = sorted(templates.keys())
    initial_params, lb, ub, plaus_lb, plaus_ub = _create_parameter_bounds(initial_weights)
    
    return TreeSearchConfig(
        templates=templates,
        features=make_features_from_groups(templates),
        sorted_groups=sorted_groups,
        param_names=[p["name"] for p in CONTROL_PARAMS] + sorted_groups,
        initial_weights=initial_weights,
        initial_params=initial_params,
        lb=lb,
        ub=ub,
        plaus_lb=plaus_lb,
        plaus_ub=plaus_ub
    )


def _create_heuristic(param_vector, config):
    """Create heuristic from parameter vector and config."""
    control_params = param_vector[:6]
    feature_weights = param_vector[6:]
    pruning_thresh, stop_prob, feat_drop, lapse, opp_scale, center = control_params
    
    heuristic = fourbynine.fourbynine_heuristic.create(
        DoubleVector([10000.0, float(pruning_thresh), float(stop_prob), float(lapse), 1.0, 1.0, float(center)]),
        False
    )
    
    for weight, group_name in zip(feature_weights, config.sorted_groups):
        weight = float(weight)
        heuristic.add_feature_group(weight, weight * float(opp_scale), float(feat_drop))
        group_idx = len(heuristic.get_feature_group_weights()) - 1
        for pieces, spaces, min_empty in config.features[group_name]:
            heuristic.add_feature(group_idx, create_feature(pieces, spaces, min_empty))
    
    return heuristic


def process_trials(param_vector, config, trials, repeats, random_seed):
    """Process batch of trials in worker process."""
    heuristic = _create_heuristic(param_vector, config)
    heuristic.seed_generator(random_seed)
    
    nll_results = []
    for trial in trials:
        tracker = IBSTracker(repeats)
        board = fourbynine_board(fourbynine_pattern(int(trial.black)), fourbynine_pattern(int(trial.white)))
        actual_move = int(trial.move).bit_length() - 1
        
        while not tracker.done:
            search = fourbynine.NInARowBestFirstSearch(heuristic, board)
            search.complete_search()
            tracker.record(heuristic.get_best_move(search.get_tree()).board_position == actual_move)
        
        nll_results.append(tracker.nll)
    
    return nll_results


class MultiThreadedFitter:
    """Fits tree search model parameters using multiprocessing."""
    
    def __init__(self, config, n_repeats=50, verbose=False, n_threads=1):
        self.config = config
        self.n_repeats = n_repeats
        self.verbose = verbose
        self.n_threads = n_threads
        self.iteration_count = 0
        self.last_seed = None
        self.data = None
    
    def evaluate(self, param_vector, data):
        """Evaluate log-likelihood of parameters on data."""
        random_seed = random.randint(0, 2**64)
        self.last_seed = random_seed
        
        n_trials = len(data)
        indices = list(range(n_trials))
        random_order = []
        while indices:
            chosen_idx = random.choice(indices)
            indices.remove(chosen_idx)
            random_order.append(chosen_idx)
        shuffled_trials = [data.iloc[i] for i in random_order]
        
        batches = [shuffled_trials[i::self.n_threads] for i in range(self.n_threads)]
        
        with Pool(self.n_threads) as pool:
            batch_results = pool.starmap(
                process_trials,
                [(param_vector, self.config, batch, self.n_repeats, random_seed) for batch in batches]
            )
        
        flattened_results = np.concatenate(batch_results).astype(np.float32)
        results = np.empty(n_trials, dtype=np.float32)
        results[random_order] = flattened_results
        return results
    
    def optimize(self, param_vector):
        """Optimization function for BADS."""
        start_time = time()
        nll_sum = self.evaluate(param_vector, self.data).sum()
        
        if self.verbose:
            print(f"[BADS-{self.iteration_count}] time: {time() - start_time:.3g}s\t "
                  f"NLL (n_repeats={self.n_repeats}): {nll_sum:.5g}\t "
                  f"Params: {[np.round(p, 3) for p in param_vector]}")
        
        self.iteration_count += 1
        return nll_sum
    
    def fit(self, data, manual_seed=None, bads_options=None):
        """Fit model to data using BADS optimization."""
        self.check_dataframe(data)
        self.data = data
        
        bads = BADS(
            self.optimize, self.config.initial_params, self.config.lb, self.config.ub,
            self.config.plaus_lb, self.config.plaus_ub,
            options=bads_options or {'uncertainty_handling': True, 'noise_final_samples': 0, 'max_fun_evals': 1000}
        )
        fitted_params = bads.optimize()['x']
        
        print(f"\t[Fitted Parameters]\t {fitted_params}")
        print("\t[Final Log-likelihood]\t Estimating final log-likelihood...")
        
        return fitted_params, self.evaluate(fitted_params, self.data)
    
    @staticmethod
    def check_dataframe(data):
        """Validate data format."""
        assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame."
        for col in ['black', 'white', 'move', 'color']:
            assert col in data.columns, f"Data must have a '{col}' column."
        
        for row_idx, row in enumerate(data.itertuples()):
            assert row.black >= 0 and row.white >= 0 and row.move >= 0
            assert row.color.lower() in ['white', 'black']
            assert bin(row.move).count('1') == 1
            board = fourbynine_board(fourbynine_pattern(row.black), fourbynine_pattern(row.white))
            assert board.active_player() == (row.color.lower() == 'white'), \
                f"Row {row_idx}: It is not {row.color}'s turn to move."


class IBSTracker:
    """Tracker for Inverse Binomial Sampling process."""
    
    def __init__(self, repeats=1):
        self.repeats = repeats
        self.success_count = self.fail_count = 0
        self.nll = 0.0
        self.done = False
    
    def record(self, is_success):
        """Record trial result."""
        assert not self.done, "Tracker is already completed!"
        
        if is_success:
            self.success_count += 1
            self.fail_count = 0
            self.done = (self.success_count == self.repeats)
        else:
            self.fail_count += 1
            self.nll += 1.0 / (self.repeats * self.fail_count)
