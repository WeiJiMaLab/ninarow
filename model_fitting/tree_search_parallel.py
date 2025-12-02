import numpy as np
from scipy.interpolate import CubicSpline
import random
from multiprocessing.pool import ThreadPool
from pybads import BADS
from tqdm import tqdm
from parsers import *
import pandas as pd
import pickle
from time import time
import fourbynine
from fourbynine import DoubleVector
from feature_generator import (
    make_features_from_groups, 
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
        self.expt_factor = 1.0
        self.cutoff = 3.5
        self.c = 50

        # Control parameters (search behavior)
        self.parameter_list = [
            {"name": "pruning_threshold", "initial_value": 2.3, "lower_bound": 0.1, "upper_bound": 10.0, "plausible_lower_bound": 1.0, "plausible_upper_bound": 4.0},
            {"name": "stopping_prob", "initial_value": 0.3, "lower_bound": 0.01, "upper_bound": 1.0, "plausible_lower_bound": 0.01, "plausible_upper_bound": 0.9},
            {"name": "feature_drop", "initial_value": 0.25, "lower_bound": 0, "upper_bound": 1, "plausible_lower_bound": 0.01, "plausible_upper_bound": 0.4},
            {"name": "lapse_rate", "initial_value": 0.1, "lower_bound": 0, "upper_bound": 1, "plausible_lower_bound": 0.01, "plausible_upper_bound": 0.5},
            {"name": "opp_scale", "initial_value": 1, "lower_bound": 0.25, "upper_bound": 4, "plausible_lower_bound": 1.2, "plausible_upper_bound": 3},
            {"name": "center_weight", "initial_value": 0.1, "lower_bound": -10, "upper_bound": 10, "plausible_lower_bound": -3, "plausible_upper_bound": 3},
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
                "lower_bound": -5,
                "upper_bound": 15,
                "plausible_lower_bound": self.initial_weights[group] - 2,
                "plausible_upper_bound": self.initial_weights[group] + 2,
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
        random_seed = random.randint(0, 2**64)
        self.heuristic.seed_generator(random_seed)
    
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
        
class MultiThreadedFitter:
    """
    The main class for finding the best heuristic/search parameter
    fit for a given dataset using multi-threaded parallelism.
    """
    def __init__(self, model: TreeSearch, threads = 1, verbose = False, subsample = None):
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

    def process_single_trial(self, row):
        """Process a single trial to completion. Model must be set up before calling."""
        tracker = IBSTracker(self.model.expt_factor, success_threshold=row.expected_counts)
        board = fourbynine_board(fourbynine_pattern(int(row.black)), fourbynine_pattern(int(row.white)))
        actual_move = int(row.move).bit_length() - 1
        
        while tracker.success_count < tracker.success_threshold:
            tracker.record_success() if self.model.predict(board) == actual_move else tracker.record_failure()
        
        return tracker.log_likelihood

    def get_random_order(self, indices):
        random_order = [] 
        while indices:
            idx = random.choice(indices)
            indices.remove(idx)
            random_order.append(idx)
        return random_order


    def log_likelihood(self, params, data: pd.DataFrame):
        """
        Calculate log-likelihood of the model given parameters and data.
        
        Multi-threaded implementation using IBSTracker instances for each trial.
        Returns an array of log-likelihood values in the *original data order*.
        """
        self.model.set_params(params)  # set params once before looping
        indices = list(range(len(data)))
        # Generate the order in which to process trials: like random.choice(incomplete), without replacement
        random_order = self.get_random_order(indices)

        # Compute log-likelihoods for trials in a random order, return in original order
        shuffled_rows = [row for _, row in data.iloc[random_order].iterrows()]
        
        with ThreadPool(self.num_workers) as pool:
            shuffled_results = np.array(
                pool.map(self.process_single_trial, shuffled_rows),
                dtype=np.float32
            )
        
        results = np.empty(len(data), dtype=np.float32)
        results[random_order] = shuffled_results
        return results
    
    def optimize(self, x): 
        if self.subsample: 
            data = self.data.sample(self.subsample)
        else:
            data = self.data

        self.time = time()
        log_likelihood = self.log_likelihood(x, data).sum()
        if self.verbose: print(f"{'[BADS-' + str(self.iteration_count) + ']':>20} time: {time() - self.time :.3g}s\t NLL: {log_likelihood:.5g}\t Params: {[np.round(x_, 3) for x_ in x]}")
        self.iteration_count += 1
        return log_likelihood
    
    def evaluate(self, params, data: pd.DataFrame, n_iters = 10):
        """Evaluates the log-likelihood of the given parameters on the given data."""
        print(f"{'[Evaluation]':>20} Running evaluation with {n_iters} iterations...")
        return np.array([self.log_likelihood(params, data) for _ in tqdm(range(n_iters))], dtype=np.float32).mean(axis = 0)

    def fit(self, 
            data: pd.DataFrame, 
            manual_seed=None, 
            use_expected_counts=False,
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
            tuple: (optimized_params, final_log_likelihood)
        """
        self.time = time()
        # first check to see if the dataframe is valid
        self.__class__.check_dataframe(data)
        print(f"{'[Initializing]':>20} Thread pool with {self.num_workers} threads")
        
        if manual_seed is not None:
            random.seed(manual_seed)
            np.random.seed(manual_seed & 0xFFFFFFFF)

        self.data = data

        if not use_expected_counts:
            print(f"{'[Expected Counts]':>20} Skipping expected counts calculation, setting all expected counts to 1")
            self.data["expected_counts"] = 1
        else:
            print(f"{'[Expected Counts]':>20} Calculating expected counts...")
            initial_LL = self.evaluate(self.model.initial_params, data)
            self.data["expected_counts"] = self.calculate_expected_counts(initial_LL, self.model.c).astype(int)

        bads = BADS(self.optimize, self.model.initial_params, self.model.lower_bound, self.model.upper_bound, self.model.plausible_lower_bound, self.model.plausible_upper_bound, options=bads_options)
        fitted_params = bads.optimize()['x']

        print(f"\t[Fitted Parameters]\t {fitted_params}")
        print("\t[Final Log-likelihood]\t Estimating final log-likelihood...")
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
        self.scale_factor = self.expt_factor / self.success_threshold
        # should clarify that this is the negative log likelihood, i.e. it is always positive
        self.attempt_count, self.success_count, self.log_likelihood = 0, 0, 0.0

    def record_success(self):
        """Record a successful prediction and return the log likelihood diff."""
        self.success_count += 1
        self.attempt_count = 0
        # this returns a CONSTANT even though the log likelihood delta is 0
        return -self.scale_factor

    def record_failure(self):
        """Record a failed prediction and return the log likelihood diff."""
        self.attempt_count += 1
        delta = self.scale_factor * (1 / self.attempt_count)
        self.log_likelihood += delta
        return delta
    
    def __repr__(self):
        return f"Successes: {self.success_count}, Attempts: {self.attempt_count}, Log-likelihood: {self.log_likelihood}"