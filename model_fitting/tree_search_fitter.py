from tree_search import TreeSearch, IBSTracker
import os
import random
import numpy as np
import pandas as pd
from time import time
from pybads import BADS
from fourbynine import fourbynine_board, fourbynine_pattern
from multiprocessing import Pool


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
    for black, white, move in chunk:
        tracker = IBSTracker(repeats=repeats)
        board = fourbynine_board(fourbynine_pattern(black), fourbynine_pattern(white))
        actual_move = move.bit_length() - 1
        while not tracker.done:
            tracker.record(_worker_model.predict(board) == actual_move)
        results.append(tracker.nll)
    return results


class MultiThreadedFitter:
    """
    Parallelized fitter using multiprocessing Pool.
    With n_workers=1, produces bit-for-bit identical results to SingleThreadedFitter.
    """
    def __init__(self, model: TreeSearch, verbose=False, n_workers=-1):
        self.model = model
        self.verbose = verbose
        self.iteration_count = 0
        self.time = time()
        self.repeats = 50
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
        for i, chunk_result in enumerate(chunk_results):
            shuffled_results[i::n_workers] = chunk_result
        results = np.empty(n_trials, dtype=np.float32)
        results[random_order] = shuffled_results
        return results

    def optimize(self, x):
        """Optimization function for BADS."""    
        self.time = time()
        nlls = self.evaluate(x, self.data).sum()
        if self.verbose: 
            param_print = {param_name: np.round(x_, 3).item() for param_name, x_ in zip(self.model.param_names, x)}
            iter_str = f"[BADS-{self.iteration_count}]"
            print(f"{iter_str:>30} "
                  f"time: {time() - self.time:.3g}s\t "
                  f"NLL (n_repeats={self.repeats}): {nlls:.5g}\t "
                  f"Params: {param_print}")
                  
        self.iteration_count += 1
        return nlls

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
                            'noise_final_samples': 0,
                            'tol_fun': 1e-2,
                        }):
        """
        Fit the model to data using BADS optimization.
        
        Returns:
            tuple: (optimized_params, final_nll)
        """
        self.time = time()
        self.__class__.check_dataframe(data)
        self.data = data

        self.repeats, self.iteration_count = 5, 0

        self.print_params(self.model.initial_params, self.model.lower_bound, self.model.upper_bound, self.model.plausible_lower_bound, self.model.plausible_upper_bound)
        warm_start_bads = BADS(self.optimize, self.model.initial_params, self.model.lower_bound, self.model.upper_bound, self.model.plausible_lower_bound, self.model.plausible_upper_bound, 
                options={**bads_options, 'max_fun_evals': 200})

        warm_start_params = warm_start_bads.optimize()['x']
        width = self.model.plausible_upper_bound - self.model.plausible_lower_bound
        warm_plb = np.maximum(self.model.lower_bound, warm_start_params - 0.25 * width)
        warm_pub = np.minimum(self.model.upper_bound, warm_start_params + 0.25 * width)

        self.repeats, self.iteration_count = 100, 0
        self.print_params(warm_start_params, self.model.lower_bound, self.model.upper_bound, warm_plb, warm_pub)
        bads = BADS(self.optimize, warm_start_params, self.model.lower_bound, self.model.upper_bound, warm_plb, warm_pub, 
                options={**bads_options, 'max_fun_evals': 1000})

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
