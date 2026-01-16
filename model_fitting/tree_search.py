from collections import defaultdict
from functools import total_ordering
import atomics
from UltraDict import UltraDict
import argparse
import numpy as np
from scipy.interpolate import CubicSpline
import random
import fourbynine
import copy
import time
from multiprocessing import Pool, Value, set_start_method
from pybads import BADS
from pathlib import Path
from tqdm import tqdm
from parsers import *
import pandas as pd
import pickle
import uuid
from time import time
import sys
import fourbynine
from fourbynine import DoubleVector
from feature_generator import (
    make_features_from_groups, 
    create_modular_heuristic,
    DEFAULT_TEMPLATES,
    DEFAULT_FEATURE_WEIGHTS,
    create_feature
)
    
class TreeSearch:
    """
    Modular tree search model that constructs heuristics from templates.
    
    The model accepts custom templates and weights, allowing flexible heuristic
    construction. Features are generated from templates once during initialization
    and cached for efficient reuse. The create_heuristic method uses cached values
    to avoid redundant computation during parameter optimization.
    """
    def __init__(self, templates=DEFAULT_TEMPLATES, initial_weights=DEFAULT_FEATURE_WEIGHTS):
        self.name = "treesearch"
        # Control parameters (search behavior)
        self.parameter_list = [
            {"name": "pruning_threshold", "initial_value": 3, "lower_bound": 0.1, "upper_bound": 10.0, "plausible_lower_bound": 1.0, "plausible_upper_bound": 6.0},
            {"name": "stopping_prob", "initial_value": 0.3, "lower_bound": 0.01, "upper_bound": 1.0, "plausible_lower_bound": 0.01, "plausible_upper_bound": 0.9},
            {"name": "feature_drop", "initial_value": 0.3, "lower_bound": 0, "upper_bound": 1, "plausible_lower_bound": 0, "plausible_upper_bound": 0.5},
            {"name": "lapse_rate", "initial_value": 0.1, "lower_bound": 0.05, "upper_bound": 1, "plausible_lower_bound": 0.05, "plausible_upper_bound": 0.5},
            {"name": "opp_scale", "initial_value": 1.2, "lower_bound": 0.25, "upper_bound": 4, "plausible_lower_bound": 0.5, "plausible_upper_bound": 2},
            {"name": "center_weight", "initial_value": 0.4, "lower_bound": -10, "upper_bound": 10, "plausible_lower_bound": -5, "plausible_upper_bound": 5},
        ]

        # Feature templates and weights (modular design)
        self.templates = DEFAULT_TEMPLATES if templates is None else templates
        self.sorted_groups = sorted(self.templates.keys())
        self.features = make_features_from_groups(self.templates)
        self.initial_weights = DEFAULT_FEATURE_WEIGHTS if initial_weights is None else initial_weights

        # Add feature weight parameters (one per template group)
        for group in self.sorted_groups:
            self.parameter_list.append({
                "name": group,
                "initial_value": self.initial_weights[group],
                "lower_bound": -10,
                "upper_bound": 20,
                "plausible_lower_bound": -5,
                "plausible_upper_bound": 15,
            })

        # Extract parameter arrays for optimization
        self.param_names = [param["name"] for param in self.parameter_list]
        self.initial_params = np.array([param["initial_value"] for param in self.parameter_list], dtype=np.float32)
        self.upper_bound = np.array([param["upper_bound"] for param in self.parameter_list], dtype=np.float32)
        self.lower_bound = np.array([param["lower_bound"] for param in self.parameter_list], dtype=np.float32)
        self.plausible_upper_bound = np.array([param["plausible_upper_bound"] for param in self.parameter_list], dtype=np.float32)
        self.plausible_lower_bound = np.array([param["plausible_lower_bound"] for param in self.parameter_list], dtype=np.float32)

        print(f"{'Parameter':>20} : {'lo':>8} {'plo':>8} {'x0':>8} {'phi':>8} {'hi':>8}")
        for p in self.parameter_list:
            print(
                f"{p['name']:>20} : "
                f"{p['lower_bound']:>8.3f} "
                f"{p['plausible_lower_bound']:>8.3f} "
                f"{p['initial_value']:>8.3f} "
                f"{p['plausible_upper_bound']:>8.3f} "
                f"{p['upper_bound']:>8.3f}"
            )

    def create_heuristic(self, control_vec, feature_vec):
        """
        Construct heuristic directly from ordered parameter arrays.
        
        Args:
            control_vec: [pruning_threshold, stopping_prob, feature_drop, lapse_rate, opp_scale, center_weight]
            feature_vec: one weight per template group (sorted)
        
        Returns:
            A heuristic created from cached templates and features
        """
        pruning_threshold, stopping_prob, feature_drop, lapse_rate, opp_scale, center_weight = control_vec

        # 1. Initialize heuristic (no features yet)
        control_params = [
            10000.0, 
            float(pruning_threshold), 
            float(stopping_prob), 
            float(lapse_rate),
            1.0, 
            1.0, 
            float(center_weight)
        ]
        heuristic = fourbynine.fourbynine_heuristic.create(DoubleVector(control_params), False)

        # 2. Add feature groups and features
        for weight, group_name in zip(feature_vec, self.sorted_groups):
            weight = float(weight)
            heuristic.add_feature_group(weight * float(opp_scale), weight, float(feature_drop))
            group_idx = len(heuristic.get_feature_group_weights()) - 1
            for pieces, spaces, min_empty in self.features[group_name]:
                heuristic.add_feature(group_idx, create_feature(pieces, spaces, min_empty))

        return heuristic

    def set_params(self, params):
        """Set parameters and construct heuristic from templates (vectorized, fixed order)."""
        assert len(params) == len(self.parameter_list), (
            f"Parameter length mismatch! Expected {len(self.parameter_list)} but got {len(params)}"
        )
        self.heuristic = self.create_heuristic(params[:6], params[6:])
        random_seed = random.randint(0, 2**64)
        self.heuristic.seed_generator(random_seed)
        # Store seed for debugging (if fitter has this attribute)
        if hasattr(self, '_fitter'):
            self._fitter.last_seed = random_seed
    
    def predict(self, board):
        """Predict the best move for a given board state."""
        search = fourbynine.NInARowBestFirstSearch(self.heuristic, board)
        search.complete_search()
        return self.heuristic.get_best_move(search.get_tree()).board_position
    
    def __call__(self, board):
        """Allow TreeSearch to be called directly like a function."""
        return self.predict(board)
    
    def __getstate__(self):
        """Exclude heuristic (SwigPyObject) from pickling."""
        state = self.__dict__.copy()
        # Remove heuristic as it's a SwigPyObject that can't be pickled
        if 'heuristic' in state:
            state['heuristic'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # Heuristic will be recreated when set_params is called
    
    def save(self, filename):
        """Save the model to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load the model from a file using pickle."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

class SingleThreadedFitter:
    """
    The main class for finding the best heuristic/search parameter
    fit for a given dataset using sequential processing.
    """
    def __init__(self, model: TreeSearch, n_repeats=50, verbose=False):
        """
        Args:
            model: The model this fitter should use.
            threads: Deprecated parameter (kept for compatibility, ignored).
            verbose: Print extra debugging info.
            subsample: If specified, randomly sample up to N positions without replacement.
        """
        self.model = model
        self.verbose = verbose
        self.iteration_count = 0
        self.time = time()
        self.repeats = n_repeats # default number of repeats for each trial for IBS
        self.last_seed = None  # For debugging: track last generated seed
        # Link back to fitter so set_params can store seed
        self.model._fitter = self

    def process_single_trial(self, trial):
        """Process a single trial to completion. Model must be set up before calling."""
        tracker = IBSTracker(repeats = self.repeats)
        board = fourbynine_board(fourbynine_pattern(int(trial.black)), fourbynine_pattern(int(trial.white)))
        actual_move = int(trial.move).bit_length() - 1
        while not tracker.done:
            tracker.record(self.model.predict(board) == actual_move)
        return tracker.nll

    def get_random_order(self, n):
        """Generate a random permutation of indices [0, n) using the same method as original."""
        indices = list(range(n))
        random_order = []
        while indices:
            idx = random.choice(indices)
            indices.remove(idx)
            random_order.append(idx)
        return random_order

    def evaluate(self, params, data: pd.DataFrame):
        """
        Evaluate the log-likelihood of the given parameters on the given data.
        
        Runs multiple iterations and returns the mean log-likelihood.
        """
        self.model.set_params(params)
        
        # Generate random processing order
        n_trials = len(data)
        random_order = self.get_random_order(n_trials)
        
        # Process trials in random order
        shuffled_trials = [data.iloc[i] for i in random_order]

        shuffled_results = []
        for trial in shuffled_trials: 
            nll_trial = self.process_single_trial(trial)
            shuffled_results.append(nll_trial)
        
        shuffled_results = np.array(shuffled_results, dtype=np.float32)

        # Return results in original data order
        results = np.empty(n_trials, dtype=np.float32)
        results[random_order] = shuffled_results
        return results
    
    def optimize(self, x):
        """Optimization function for BADS."""    
        self.time = time()
        # take the sum of all the trial log likelihoods
        nlls = self.evaluate(x, self.data).sum()
        if self.verbose: 
            iter_str = f"[BADS-{self.iteration_count}]"
            print(f"{iter_str:>30} "
                  f"time: {time() - self.time:.3g}s\t "
                  f"NLL (n_repeats={self.repeats}): {nlls:.5g}\t "
                  f"Params: {[np.round(x_, 3) for x_ in x]}")
                  
        self.iteration_count += 1
        return nlls
    
    def fit(self, 
            data: pd.DataFrame, 
            manual_seed=None, 
            bads_options={
                            'uncertainty_handling': True,
                            'noise_final_samples': 0,
                            'max_fun_evals': 1000,        # Reduced from 2000 for faster convergence
                        }):
        """
        Fit the model to data using BADS optimization.
        
        Performs initial log-likelihood estimation, runs BADS optimizer,
        then performs final log-likelihood estimation.
        
        Returns:
            tuple: (optimized_params, final_nll)
        """
        self.time = time()
        # first check to see if the dataframe is valid
        self.__class__.check_dataframe(data)
        self.data = data

        bads = BADS(self.optimize, self.model.initial_params, self.model.lower_bound, self.model.upper_bound, self.model.plausible_lower_bound, self.model.plausible_upper_bound, options=bads_options)
        fitted_params = bads.optimize()['x']

        print(f"\t[Fitted Parameters]\t {fitted_params}")
        print("\t[Final Log-likelihood]\t Estimating final log-likelihood...")

        # for the final pass we want the mean of each trial's log likelihood
        final_LL = self.evaluate(fitted_params, self.data)
        return fitted_params, final_LL
    
    @staticmethod
    def check_dataframe(data): 
        """Check that the data is in the correct format for fitting."""
        assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame."
        assert 'black' in data.columns, "Data must have a 'black' column."
        assert 'white' in data.columns, "Data must have a 'white' column."
        assert 'move' in data.columns, "Data must have a 'move' column."
        assert 'color' in data.columns, "Data must have a 'color' column."

        for i, row in enumerate(data.itertuples()):
            assert row.black >= 0, f"Row {i}: Black pieces must be a non-negative integer."
            assert row.white >= 0, f"Row {i}: White pieces must be a non-negative integer."
            assert row.move >= 0, f"Row {i}: Move must be a non-negative integer."
            assert row.color.lower() in ['white', 'black'], f"Row {i}: Color must be either 'white' or 'black'."
            assert bin(row.move).count('1') == 1, f"Row {i}: Invalid move given: {row.move} does not represent a valid move (must have exactly one space occupied)."
            assert fourbynine_board(fourbynine_pattern(row.black), fourbynine_pattern(row.white)).active_player() == (row.color.lower() == 'white'), f"Row {i}:  it is not {row.color}'s turn to move."

class IBSTracker:
    """
    A tracker for the Inverse Binomial Sampling (IBS) process, used to monitor 
    and fit a heuristic to a given dataset by tracking successes and failures.
    """
    def __init__(self, repeats = 1):
        """Initialize IBSTracker with experiment factor and success threshold."""
        self.repeats = repeats
        self.success_count, self.fail_count, self.nll, self.done = 0, 0, 0.0, False

    def record(self, is_success:bool):
        assert not self.done, "Tracker is completed!"

        if is_success: 
            self.success_count += 1
            self.fail_count = 0
            if self.success_count == self.repeats: self.done = True
        else: 
            self.fail_count += 1
            self.nll += (1 / self.repeats) * (1 / self.fail_count)
    
    def __repr__(self):
        return f"Successes: {self.success_count}, Failures: {self.fail_count}, Negative Log-likelihood: {self.nll}"