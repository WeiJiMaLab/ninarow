"""
Parallel version of tree_search fitting using multiprocessing.Pool.

Parallelizes process_single_trial across trials. Each trial runs its own IBS loop
independently, making it ideal for multiprocessing. For n_workers=1, uses the same
sequential path as SingleThreadedFitter to ensure exact match.
"""
import copy
import random
from multiprocessing import Pool, set_start_method
from time import time

import numpy as np
import pandas as pd

from parsers import fourbynine_board, fourbynine_pattern
from tree_search import (
    TreeSearch,
    SingleThreadedFitter,
    IBSTracker,
    BADS,
)


class ParallelFitter:
    """
    Fitter that parallelizes trial processing across workers using multiprocessing.Pool.
    Each trial runs its own IBS loop independently.
    """
    def __init__(self, model: TreeSearch, n_repeats=50, verbose=False, n_workers=8):
        self.model = model
        self.verbose = verbose
        self.iteration_count = 0
        self.time = time()
        self.repeats = n_repeats
        self.n_workers = n_workers
        self.last_seed = None
        self.model._fitter = self

    def process_single_trial(self, args):
        """
        Process a single trial. For parallel workers: receives (trial, params) and
        creates a fresh model copy with params set. For sequential (n_workers<=1):
        receives trial only; model is already set up by caller.
        """
        if isinstance(args, tuple):
            trial, params = args
            model = copy.deepcopy(self.model)
            model.set_params(params)
        else:
            trial = args
            model = self.model

        tracker = IBSTracker(repeats=self.repeats)
        board = fourbynine_board(
            fourbynine_pattern(int(trial.black)),
            fourbynine_pattern(int(trial.white))
        )
        actual_move = int(trial.move).bit_length() - 1

        while not tracker.done:
            tracker.record(model.predict(board) == actual_move)

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
        """
        n_trials = len(data)

        if self.n_workers <= 1:
            # Sequential path: identical to SingleThreadedFitter for exact match
            # Order must match: set_params first, then get_random_order
            self.model.set_params(params)
            random_order = self.get_random_order(n_trials)
            shuffled_trials = [data.iloc[i] for i in random_order]
            shuffled_results = []
            for trial in shuffled_trials:
                nll_trial = self.process_single_trial(trial)
                shuffled_results.append(nll_trial)
        else:
            # Parallel path: each worker gets (trial, params) and sets up its own model
            random_order = self.get_random_order(n_trials)
            shuffled_trials = [data.iloc[i] for i in random_order]
            with Pool(self.n_workers) as pool:
                shuffled_results = pool.map(
                    self.process_single_trial,
                    [(trial, params) for trial in shuffled_trials]
                )

        shuffled_results = np.array(shuffled_results, dtype=np.float32)

        results = np.empty(n_trials, dtype=np.float32)
        results[random_order] = shuffled_results
        return results

    def optimize(self, x):
        """Optimization function for BADS."""
        self.time = time()
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
                'max_fun_evals': 1000,
            }):
        """
        Fit the model to data using BADS optimization.
        """
        self.time = time()
        self.__class__.check_dataframe(data)
        self.data = data

        bads = BADS(
            self.optimize,
            self.model.initial_params,
            self.model.lower_bound,
            self.model.upper_bound,
            self.model.plausible_lower_bound,
            self.model.plausible_upper_bound,
            options=bads_options
        )
        fitted_params = bads.optimize()['x']

        print(f"\t[Fitted Parameters]\t {fitted_params}")
        print("\t[Final Log-likelihood]\t Estimating final log-likelihood...")

        final_LL = self.evaluate(fitted_params, self.data)
        return fitted_params, final_LL

    @staticmethod
    def check_dataframe(data):
        """Check that the data is in the correct format for fitting."""
        SingleThreadedFitter.check_dataframe(data)


if __name__ == "__main__":
    set_start_method("spawn", force=True)
