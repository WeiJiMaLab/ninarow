from tree_search import TreeSearch
import os
import random
import numpy as np
import pandas as pd
from time import time
from pybads import BADS
from fourbynine import fourbynine_board, fourbynine_pattern
from multiprocessing import Pool


from numba import int32, float64, boolean
from numba.experimental import jitclass

spec = [
    ('repeats', int32),
    ('success_count', int32),
    ('fail_count', int32),
    ('current_repeat_nll', float64),
    ('done', boolean),
    ('nlls', float64[:]),
    ('n_recorded', int32),
]

@jitclass(spec)
class IBSTracker:
    """
    A tracker for the Inverse Binomial Sampling (IBS) process, optimized with Numba.
    """
    def __init__(self, repeats):
        self.repeats = repeats
        self.success_count = 0
        self.fail_count = 0
        self.current_repeat_nll = 0.0
        self.done = False
        self.nlls = np.zeros(repeats, dtype=np.float64)
        self.n_recorded = 0

    def record(self, is_success):
        if self.done:
            return

        if is_success: 
            self.nlls[self.n_recorded] = self.current_repeat_nll
            self.n_recorded += 1
            self.success_count += 1
            self.fail_count = 0
            self.current_repeat_nll = 0.0
            if self.success_count == self.repeats: 
                self.done = True
        else: 
            self.fail_count += 1
            self.current_repeat_nll += 1.0 / self.fail_count
    
    @property
    def nll(self):
        """Returns the mean Negative Log-Likelihood across all repeats."""
        if self.n_recorded == 0:
            return 0.0
        # Manual mean for jitclass compatibility if needed, 
        # though np.mean works on slices.
        return np.mean(self.nlls[:self.n_recorded])

    @property
    def variance_of_mean(self):
        """Returns the variance of the NLL mean estimator."""
        if self.n_recorded < 2:
            return 0.0
        # Sample variance / n
        return np.var(self.nlls[:self.n_recorded]) * self.n_recorded / (self.n_recorded - 1) / self.repeats


class SingleThreadedFitter:
    """
    The main class for finding the best heuristic/search parameter
    fit for a given dataset using sequential processing.
    """
    def __init__(self, model: TreeSearch, n_repeats=50, verbose=False):
        """
        Args:
            model: The model this fitter should use.
            verbose: Print extra debugging info.
        """
        self.model = model
        self.verbose = verbose
        self.iteration_count = 0
        self.time = time()
        self.repeats = n_repeats
        self.last_seed = None
        # Link back to fitter so set_params can store seed
        self.model._fitter = self

    def process_single_trial(self, trial):
        """Process a single trial to completion. Model must be set up before calling."""
        tracker = IBSTracker(repeats = self.repeats)
        board = fourbynine_board(fourbynine_pattern(int(trial.black)), fourbynine_pattern(int(trial.white)))
        actual_move = int(trial.move).bit_length() - 1
        while not tracker.done:
            tracker.record(self.model.predict(board) == actual_move)
        return tracker.nll, tracker.variance_of_mean

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
        """
        self.model.set_params(params)
        n_trials = len(data)
        random_order = self.get_random_order(n_trials)
        shuffled_trials = [data.iloc[i] for i in random_order]

        shuffled_results = []
        shuffled_variances = []
        for trial in shuffled_trials: 
            nll_trial, var_trial = self.process_single_trial(trial)
            shuffled_results.append(nll_trial)
            shuffled_variances.append(var_trial)
        
        shuffled_results = np.array(shuffled_results, dtype=np.float32)
        shuffled_variances = np.array(shuffled_variances, dtype=np.float32)
        
        results = np.empty(n_trials, dtype=np.float32)
        variances = np.empty(n_trials, dtype=np.float32)
        
        results[random_order] = shuffled_results
        variances[random_order] = shuffled_variances
        
        return results, variances
    
    def optimize(self, x):
        """Optimization function for BADS."""    
        self.time = time()
        nlls_arr, vars_arr = self.evaluate(x, self.data)
        nlls = nlls_arr.sum()
        total_std = np.sqrt(vars_arr.sum())

        if self.verbose: 
            iter_str = f"[BADS-{self.iteration_count}]"
            print(f"{iter_str:>30} "
                  f"time: {time() - self.time:.3g}s\t "
                  f"NLL: {nlls:.5g} ± {total_std:.3g}\t "
                  f"Params: {[np.round(x_, 3) for x_ in x]}")
                  
        self.iteration_count += 1
        return nlls.item(), total_std.item()
    
    def fit(self, 
            data: pd.DataFrame, 
            manual_seed=None, 
            bads_options={
                            'uncertainty_handling': True,
                            'display': 'iter',
                        }):
        """Fit the model to data using BADS optimization."""
        self.time = time()
        MultiThreadedFitter.check_dataframe(data)
        self.data = data

        bads = BADS(self.optimize, self.model.initial_params, self.model.lower_bound, self.model.upper_bound, self.model.plausible_lower_bound, self.model.plausible_upper_bound, options=bads_options)
        fitted_params = bads.optimize()['x']

        print(f"\t[Fitted Parameters]\t {fitted_params}")
        print("\t[Final Log-likelihood]\t Estimating final log-likelihood...")

        final_LL, _ = self.evaluate(fitted_params, self.data)
        return fitted_params, final_LL


def _init_worker(model):
    """Pool initializer: store model in worker. Heuristic is rebuilt per evaluate() call."""
    global _worker_model
    _worker_model = model


def _process_chunk(args):
    """Rebuild heuristic from (params, seed), then process trials sequentially."""
    params, heuristic_seed, repeats, chunk = args
    _worker_model.set_params(params)
    _worker_model.heuristic.seed_generator(heuristic_seed)

    results = []
    variances = []
    for black, white, move in chunk:
        tracker = IBSTracker(repeats=repeats)
        board = fourbynine_board(fourbynine_pattern(black), fourbynine_pattern(white))
        actual_move = move.bit_length() - 1
        while not tracker.done:
            tracker.record(_worker_model.predict(board) == actual_move)
        results.append(tracker.nll)
        variances.append(tracker.variance_of_mean)
    return results, variances


class MultiThreadedFitter:
    """
    Parallelized fitter using multiprocessing Pool.
    With n_workers=1, produces bit-for-bit identical results to SingleThreadedFitter.
    """
    def __init__(self, model: TreeSearch, verbose=False, n_repeats = 50, n_workers=-1):
        self.model = model
        self.verbose = verbose
        self.iteration_count = 0
        self.time = time()
        self.repeats = n_repeats
        self.start_repeats = 5
        self.full_repeats = 50
        self.last_seed = None
        self.n_workers = n_workers if n_workers > 0 else os.cpu_count()
        self._pool = None

    def __getstate__(self):
        """Exclude live Pool from pickle/deepcopy (e.g. pybads OptimizeResult)."""
        state = self.__dict__.copy()
        state["_pool"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pool = None

    def get_random_order(self, n):
        """Generate a random permutation of indices [0, n) using the same method as original."""
        indices = list(range(n))
        random_order = []
        while indices:
            idx = random.choice(indices)
            indices.remove(idx)
            random_order.append(idx)
        return random_order

    def _get_pool(self):
        if self._pool is None:
            self._pool = Pool(
                processes=self.n_workers,
                initializer=_init_worker,
                initargs=(self.model,),
            )
        return self._pool

    def close(self):
        """Shut down the worker pool."""
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.terminate()
            pool.join()
            self._pool = None

    def __del__(self):
        self.close()

    def evaluate(self, params, data: pd.DataFrame):
        """
        Evaluate the log-likelihood of the given parameters on the given data
        using a persistent multiprocessing Pool.

        Random state consumption matches SingleThreadedFitter exactly:
        one random.randint (heuristic seed) followed by N random.choice calls
        (trial ordering). The heuristic seed is forwarded to workers so each
        worker reconstructs an identically-seeded heuristic.
        """
        heuristic_seed = random.randint(0, 2**64)
        self.last_seed = heuristic_seed

        n_trials = len(data)
        random_order = self.get_random_order(n_trials)

        trial_data = [
            (int(data.iloc[i].black), int(data.iloc[i].white), int(data.iloc[i].move))
            for i in random_order
        ]

        n_workers = self.n_workers
        chunks = [trial_data[i::n_workers] for i in range(n_workers)]
        chunk_args = [
            (params, heuristic_seed, self.repeats, chunk) for chunk in chunks
        ]

        chunk_results = self._get_pool().map(_process_chunk, chunk_args)

        shuffled_results = np.empty(n_trials, dtype=np.float32)
        shuffled_variances = np.empty(n_trials, dtype=np.float32)
        for i, (chunk_result, chunk_variance) in enumerate(chunk_results):
            shuffled_results[i::n_workers] = chunk_result
            shuffled_variances[i::n_workers] = chunk_variance
        
        results = np.empty(n_trials, dtype=np.float32)
        variances = np.empty(n_trials, dtype=np.float32)
        
        results[random_order] = shuffled_results
        variances[random_order] = shuffled_variances
        return results, variances

    def optimize(self, x):
        """Optimization function for BADS."""    
        self.time = time()
        nlls_arr, vars_arr = self.evaluate(x, self.data)
        nlls = nlls_arr.sum()
        total_std = np.sqrt(vars_arr.sum())

        if self.verbose: 
            param_print = {param_name: np.round(x_, 3).item() for param_name, x_ in zip(self.model.param_names, x)}
            iter_str = f"[BADS-{self.iteration_count}]"
            print(f"{iter_str:>30} "
                  f"time: {time() - self.time:.3g}s\t "
                  f"NLL: {nlls:.5g} ± {total_std:.3g}\t "
                  f"Params: {param_print}")
                  
        self.iteration_count += 1
        return nlls.item(), total_std.item()

    def print_params(self, x, lower_bound, upper_bound, plausible_lower_bound, plausible_upper_bound):
        header = f"{'Parameter':>20} :\t{'lo'}\t{'plo'}\t{'x0'}\t{'phi'}\t{'hi'}"
        print(header)
        for param_name, x_, lo, hi, plo, phi in zip(
            self.model.param_names, x, lower_bound, upper_bound, plausible_lower_bound, plausible_upper_bound
        ):
            print(f"{param_name:>20}:\t{lo:.3f}\t{plo:.3f}\t{x_:.3f}\t{phi:.3f}\t{hi:.3f}")
            
    def fit(self, 
            data: pd.DataFrame, 
            manual_seed=None, 
            bads_options={
                            'uncertainty_handling': True,
                            'display': 'iter',
                        }):
        """
        Fit the model to data using BADS optimization.
        
        Returns:
            tuple: (optimized_params, final_nll)
        """
        self.time = time()
        self.__class__.check_dataframe(data)
        self.data = data
        
        print(f"\n[BADS Optimization Start]")
        print(f"  Options: {bads_options}")
        
        self.repeats, self.iteration_count = self.start_repeats, 0
        self.print_params(self.model.initial_params, self.model.lower_bound, self.model.upper_bound, self.model.plausible_lower_bound, self.model.plausible_upper_bound)
        
        # --- STAGE 1: WARM START (Global Search, repeats=5) ---
        print("\n>>> STAGE 1: WARM START (Global Search, repeats=5)")
        warm_start_evals = max(25 * len(self.model.initial_params), 21)
        warm_start_bads = BADS(self.optimize, self.model.initial_params, self.model.lower_bound, self.model.upper_bound, self.model.plausible_lower_bound, self.model.plausible_upper_bound, 
                options={**bads_options, 'max_fun_evals': warm_start_evals})

        warm_start_params = warm_start_bads.optimize()['x']
        
        # Shrink plausible bounds around warm start result
        width = self.model.plausible_upper_bound - self.model.plausible_lower_bound
        warm_plb = np.maximum(self.model.lower_bound, warm_start_params - 0.25 * width)
        warm_pub = np.minimum(self.model.upper_bound, warm_start_params + 0.25 * width)

        # --- STAGE 2: FULL FIT (High precision, narrow bounds) ---
        print("\n>>> STAGE 2: FULL FIT (Local Search, repeats=50)")
        self.repeats, self.iteration_count = self.full_repeats, 0
        self.print_params(warm_start_params, self.model.lower_bound, self.model.upper_bound, warm_plb, warm_pub)
        
        bads = BADS(self.optimize, warm_start_params, self.model.lower_bound, self.model.upper_bound, warm_plb, warm_pub, 
                options=bads_options)

        fitted_params = bads.optimize()['x']

        print(f"\t[Fitted Parameters]\t {fitted_params}")
        print("\t[Final Log-likelihood]\t Estimating final log-likelihood...")

        final_LL, _ = self.evaluate(fitted_params, self.data)
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
