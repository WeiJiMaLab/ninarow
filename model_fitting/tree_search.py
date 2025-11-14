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

def get_shallow_size(obj):
    """Calculates the shallow size of a dictionary, including its keys and values."""
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(sys.getsizeof(k) for k in obj.keys())
        size += sum(sys.getsizeof(v) for v in obj.values())
    return size
    
class TreeSearch:
    """
    Modular tree search model that constructs heuristics from templates.
    
    The model accepts custom templates and weights, allowing flexible heuristic
    construction. Features are generated from templates once during initialization
    and cached for efficient reuse. The create_heuristic method uses cached values
    to avoid redundant computation during parameter optimization.
    """
    def __init__(self, templates=DEFAULT_TEMPLATES, initial_weights=DEFAULT_FEATURE_WEIGHTS):
        self.name = self.__class__.__name__
        self.expt_factor = 1.0
        self.cutoff = 3.5
        self.c = 50

        # Control parameters (search behavior)
        self.parameter_list = [
            {"name": "pruning_threshold", "initial_value": 2.0, "lower_bound": 0.1, "upper_bound": 10.0, "plausible_lower_bound": 1.0, "plausible_upper_bound": 6.0},
            {"name": "stopping_prob", "initial_value": 0.3, "lower_bound": 0.01, "upper_bound": 1.0, "plausible_lower_bound": 0.01, "plausible_upper_bound": 0.9},
            {"name": "feature_drop", "initial_value": 0.2, "lower_bound": 0, "upper_bound": 1, "plausible_lower_bound": 0, "plausible_upper_bound": 0.5},
            {"name": "lapse_rate", "initial_value": 0.1, "lower_bound": 0.05, "upper_bound": 1, "plausible_lower_bound": 0.05, "plausible_upper_bound": 0.5},
            {"name": "opp_scale", "initial_value": 1.2, "lower_bound": 0.25, "upper_bound": 4, "plausible_lower_bound": 0.5, "plausible_upper_bound": 2},
            {"name": "center_weight", "initial_value": 0.8, "lower_bound": -10, "upper_bound": 10, "plausible_lower_bound": -5, "plausible_upper_bound": 5},
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
                "upper_bound": 10,
                "plausible_lower_bound": -5,
                "plausible_upper_bound": 5
            })

        # Extract parameter arrays for optimization
        self.param_names = [param["name"] for param in self.parameter_list]
        self.initial_params = np.array([param["initial_value"] for param in self.parameter_list], dtype=np.float32)
        self.upper_bound = np.array([param["upper_bound"] for param in self.parameter_list], dtype=np.float32)
        self.lower_bound = np.array([param["lower_bound"] for param in self.parameter_list], dtype=np.float32)
        self.plausible_upper_bound = np.array([param["plausible_upper_bound"] for param in self.parameter_list], dtype=np.float32)
        self.plausible_lower_bound = np.array([param["plausible_lower_bound"] for param in self.parameter_list], dtype=np.float32)
        print("Parameter names:", self.param_names)

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
            10000.0, float(pruning_threshold), float(stopping_prob), float(lapse_rate),
            1.0, 1.0, float(center_weight)
        ]
        heuristic = fourbynine.fourbynine_heuristic.create(DoubleVector(control_params), False)

        # 2. Add feature groups and features
        for weight, group_name in zip(feature_vec, self.sorted_groups):
            weight = float(weight)
            heuristic.add_feature_group(weight, weight * float(opp_scale), float(feature_drop))
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
        self.heuristic.seed_generator(random.randint(0, 2**64))
    
    def predict(self, board):
        """Predict the best move for a given board state."""
        search = fourbynine.NInARowBestFirstSearch(self.heuristic, board)
        search.complete_search()
        return self.heuristic.get_best_move(search.get_tree()).board_position
    
    def __call__(self, board):
        """Allow TreeSearch to be called directly like a function."""
        return self.predict(board)
    
    def save(self, filename):
        """Save the model to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load the model from a file using pickle."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
class Fitter:
    """
    The main class for finding the best heuristic/search parameter
    fit for a given dataset.
    """
    def __init__(self, model: TreeSearch, threads=16, verbose = False, subsample = None):
        """
        Args:
            model: The model this fitter should use.
            verbose: Print extra debugging info.
            threads: The number of threads to use when fitting.
            subsample: If specified, randomly sample up to N positions without replacement.
        """
        self.model = model
        self.verbose = verbose
        self.num_workers = threads
        self.iteration_count = 0
        self.time = time()
        self.subsample = subsample

    def calculate_expected_counts(self, log_likelihoods, c):
        """Calculate the expected observation counts for each move based on their L-values."""
        x = np.linspace(1e-6, 1 - 1e-6, int(1e6), dtype=np.float32)
        dilog = np.pi**2 / 6.0 + np.cumsum(np.log(x) / (1 - x)) / len(x)
        p = np.exp(-log_likelihoods).astype(np.float32)
        interp1 = CubicSpline(x, np.sqrt(x * dilog), extrapolate=True)
        interp2 = CubicSpline(x, np.sqrt(dilog / x), extrapolate=True)
        times = (c * interp1(p)) / np.mean(interp2(p))
        return np.vectorize(lambda x: max(x, 1))(np.round(times))

    def parallel_log_likelihood(self, params, trackers: UltraDict, cutoff: float):
        """
        Compute log-likelihood of model parameters in parallel.
        
        Updates global log-likelihood and trackers for each trial until
        the log-likelihood exceeds the cutoff value.
        """
        
        self.model.set_params(params)
        while LOG_LIKELIHOOD.value <= cutoff:
            incomplete_trials = [(key, tracker) for key, tracker in trackers.items() if tracker.success_count < tracker.success_threshold]
            if not incomplete_trials: break
            key, tracker = copy.deepcopy(random.choice(incomplete_trials))

            black_, white_, move_, _= key
            board = fourbynine_board(fourbynine_pattern(black_), fourbynine_pattern(white_))
            actual_move = int(move_).bit_length() - 1

            delta_log_likelihood = 0
            while tracker.success_count < tracker.success_threshold:
                predicted_move = self.model.predict(board)
                if (predicted_move == actual_move):
                    delta_log_likelihood += tracker.record_success()

                    with trackers.lock:
                        if tracker.success_count == trackers[key].success_count + 1:
                            trackers[key] = tracker
                            LOG_LIKELIHOOD.value += delta_log_likelihood
                    break
                
                else:
                    delta_log_likelihood += tracker.record_failure()
                    if LOG_LIKELIHOOD.value + delta_log_likelihood > cutoff:
                        with trackers.lock:
                            LOG_LIKELIHOOD.value += delta_log_likelihood
                        break

    def log_likelihood(self, params, data: pd.DataFrame):
        """
        Calculate log-likelihood of the model given parameters and data.
        
        Uses parallel processing with IBSTracker instances for each trial.
        Returns an array of log-likelihood values.
        """
        tick = time()
        n_trials = len(data)

        if "expected_counts" not in data.columns:
            data["expected_counts"] = 1
            print("Warning: 'expected_counts' column not found. Defaulting to 1.")


        trackers = {(key.black, key.white, key.move, uuid.uuid4()): IBSTracker(self.model.expt_factor, success_threshold=key.expected_counts) for key in data.itertuples()}
        assert(len(trackers)) == n_trials
        shared_trackers = UltraDict(trackers, full_dump_size= get_shallow_size(trackers) + 1024 * 1024 , buffer_size=1024 * 1024, shared_lock=True)

        global LOG_LIKELIHOOD
        LOG_LIKELIHOOD.value = n_trials * self.model.expt_factor

        global POOL
        results = [POOL.apply_async(self.parallel_log_likelihood, (params, shared_trackers, n_trials * self.model.cutoff)) for i in range(self.num_workers)]
        [result.get() for result in results]

        return np.array([shared_trackers[key].log_likelihood for key in shared_trackers], dtype=np.float32)
    
    def optimize(self, x): 
        if self.subsample: 
            data = self.data.sample(self.subsample)
        else:
            data = self.data

        log_likelihood = self.log_likelihood(x, data).sum()
        if self.verbose: print(f"\t[BADS-{self.iteration_count}]\t time: {time() - self.time :.3g}s\t NLL: {log_likelihood:.5g}\t Params: {[np.round(x_, 3) for x_ in x]}")
        self.iteration_count += 1
        return log_likelihood
    
    def evaluate(self, params, data: pd.DataFrame, n_iters = 10):
        """Evaluates the log-likelihood of the given parameters on the given data."""
        print(f"Running evaluation with {n_iters} iterations...")
        return np.array([self.log_likelihood(params, data) for _ in tqdm(range(n_iters))], dtype=np.float32).mean(axis = 0)

    def fit(self, data: pd.DataFrame, manual_seed=None, bads_options={
                    'uncertainty_handling': True,
                    'noise_final_samples': 0,
                    'max_fun_evals': 2000,        # Reduced from 2000 for faster convergence
                  }):
        """
        Fit the model to data using BADS optimization.
        
        Performs initial log-likelihood estimation, runs BADS optimizer,
        then performs final log-likelihood estimation.
        
        Returns:
            tuple: (optimized_params, final_log_likelihood)
        """
        self.time = time()
        # first check to see if the dataframe is valid
        self.__class__.check_dataframe(data)

        print("Initializing thread pool...")
        initialize_thread_pool(self.num_workers, manual_seed = manual_seed)

        self.data = data

        self.data["expected_counts"] = 1

        print("Initial log-likelihood estimation...")

        initial_LL = self.evaluate(self.model.initial_params, data)
        self.data["expected_counts"] = self.calculate_expected_counts(initial_LL, self.model.c).astype(int)


        bads = BADS(self.optimize, self.model.initial_params, self.model.lower_bound, self.model.upper_bound, self.model.plausible_lower_bound, self.model.plausible_upper_bound, options=bads_options)
        fitted_params = bads.optimize()['x']

        print(f"Fitted parameters: {fitted_params}")

        print("Final log-likelihood estimation...")
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
    def __init__(self, expt_factor, success_threshold = 1):
        """Initialize IBSTracker with experiment factor and success threshold."""
        self.success_threshold = success_threshold
        self.expt_factor = expt_factor
        self.attempt_count, self.success_count, self.log_likelihood = 0, 0, 0.0

    def record_success(self):
        """Record a successful prediction and return the log likelihood diff."""
        scale_factor = self.expt_factor / self.success_threshold
        self.success_count += 1


        self.attempt_count = 0
        return -scale_factor

    def record_failure(self):
        """Record a failed prediction and return the log likelihood diff."""
        scale_factor = self.expt_factor / self.success_threshold
        self.attempt_count += 1
        delta = scale_factor * (1 / self.attempt_count)
        self.log_likelihood += delta
        return delta
    
    def __repr__(self):
        return f"Successes: {self.success_count}, Attempts: {self.attempt_count}, Log-likelihood: {self.log_likelihood}"

def initialize_thread(shared_value):

    global LOG_LIKELIHOOD
    LOG_LIKELIHOOD = shared_value

def set_seeds(base_seed, thread_id):

    thread_seed = base_seed + thread_id
    random.seed(thread_seed)
    print(f"Thread {thread_id}: seed={thread_seed}, Random number: {random.randint(0, 2**64)}")
    

def initialize_thread_pool(num_threads, manual_seed=None):
    """
    Initialize thread pool for parallel log-likelihood computation.
    
    Args:
        num_threads: Number of threads to initialize
        manual_seed: Optional seed (only valid with num_threads=1)
    """
    global LOG_LIKELIHOOD, POOL
    LOG_LIKELIHOOD = Value('d', 0)
    POOL = Pool(num_threads, initializer=initialize_thread, initargs=(LOG_LIKELIHOOD,))

    if manual_seed is not None:
        assert num_threads == 1, "Setting manual seed can only be used with a single thread. If threads > 1, thread compute order is nondeterministic."
        print(f"Manual seed: {manual_seed}")
        POOL.starmap(set_seeds, [(manual_seed, i) for i in range(num_threads)])

def cross_validate(model: TreeSearch, folds: list, leave_out_idx: int, threads: int = 16, subsample=None):
    """
    Perform cross-validation on the model using specified folds.
    
    Returns:
        tuple: (fitted_params, training_log_likelihood, test_log_likelihood)
    """
    assert leave_out_idx < len(folds), "Invalid leave-out index!"

    print(f"Cross-validating split {leave_out_idx + 1} vs {len(folds) - 1} others")
    test = folds[leave_out_idx]

    train = []
    for j in range(len(folds)):
        if leave_out_idx != j:
            train.append(folds[j])

    train = pd.concat(train)
    fitter = Fitter(model, threads = threads, verbose = True, subsample = subsample)
    params, trainLL = fitter.fit(train)

    testLL = fitter.evaluate(params, test)
    return params, trainLL, testLL

import os
def main(): 
    data_path = "data"
    output_path = "data/out"
    n_splits = 5
    fold_number = 1
    threads = 1
    random_sample = False
    verbose = True

    print(f"Output directory: {output_path}")
    os.makedirs(output_path, exist_ok = True)


    assert np.all([f"{i + 1}.csv" in os.listdir(data_path) for i in range(n_splits)])
    print("Loading splits...")


    splits = [pd.read_csv(f"{data_path}/{i + 1}.csv") for i in range(n_splits)]

    random.seed(10)
    initialize_thread_pool(1, manual_seed = 10)

    q = cross_validate(TreeSearch(), splits, leave_out_idx = 1, threads = 1)


if __name__ == "__main__":
    main()