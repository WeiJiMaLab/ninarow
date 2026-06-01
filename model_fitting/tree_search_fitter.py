from tree_search import TreeSearch
import json
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
        return np.mean(self.nlls[:self.n_recorded])

    @property
    def variance_of_mean(self):
        """Returns the variance of the NLL mean estimator."""
        if self.n_recorded < 2:
            return 0.0
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
                  f"NLL(n={self.repeats}): {nlls:.5g} ± {total_std:.3g}\t "
                  f"Params: {[np.round(x_, 3) for x_ in x]}")

        self.iteration_count += 1
        return nlls.item(), total_std.item()

    def fit(self,
            data: pd.DataFrame,
            manual_seed=None,
            bads_options=None):
        """Fit the model to data using BADS optimization."""
        if bads_options is None:
            bads_options = {'uncertainty_handling': True, 'display': 'iter'}
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


def _dynamic_repeats(effective_iter, max_repeats):
    """Linearly ramp repeats from 5 to max_repeats over polls 5–19; hold at extremes."""
    if effective_iter < 5:
        return 5
    if effective_iter >= 20:
        return max_repeats
    return round(5 + (effective_iter - 5) * (max_repeats - 5) / (20 - 5))


class MultiThreadedFitter:
    """
    Parallelized fitter using multiprocessing Pool.
    With n_workers=1, produces bit-for-bit identical results to SingleThreadedFitter.

    Runs a single-stage BADS optimization over the original, constant plausible bounds.
    IBS repeats are scaled dynamically from 5 (global search) to n_repeats (final
    refinement) based on the BADS poll iteration, eliminating coordinate-system
    variance from bound shifting.

    Pass checkpoint_path to fit() to enable fault-tolerant resume: the best parameters,
    poll iteration, and u-space mesh size are written to a JSON file on each improvement.
    If the file exists at the start of fit(), the run resumes from that checkpoint with
    adaptive bounds and a matching tol_mesh so the physical stopping resolution is
    identical to a fresh run.
    """
    def __init__(self, model: TreeSearch, verbose=False, n_repeats=50, n_workers=-1):
        self.model = model
        self.verbose = verbose
        self.iteration_count = 0
        self.time = time()
        self.repeats = 5
        self._max_repeats = n_repeats
        self.last_seed = None
        self.n_workers = n_workers if n_workers > 0 else os.cpu_count()
        self._pool = None
        self._current_bads = None
        self._checkpoint_iter = 0
        self._checkpoint_path = None
        self._best_nll = np.inf
        self._best_params = None

    def __getstate__(self):
        """Exclude live Pool and active BADS reference from pickle."""
        state = self.__dict__.copy()
        state["_pool"] = None
        state["_current_bads"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pool = None
        self._current_bads = None

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
        # Dynamic repeats: ramp from 5 (global) to _max_repeats (refinement)
        if self._current_bads is not None:
            effective_iter = self._checkpoint_iter + self._current_bads.optim_state.get("iter", 0)
            self.repeats = _dynamic_repeats(effective_iter, self._max_repeats)

        self.time = time()
        nlls_arr, vars_arr = self.evaluate(x, self.data)
        nlls = nlls_arr.sum()
        total_std = np.sqrt(vars_arr.sum())

        if self._checkpoint_path and self._current_bads is not None and nlls < self._best_nll:
            self._best_nll = float(nlls)
            self._best_params = x.copy()
            checkpoint = {
                "best_params": self._best_params.tolist(),
                "poll_iter": int(self._current_bads.optim_state.get("iter", 0)),
                "mesh_size": float(self._current_bads.optim_state.get("mesh_size", 1.0)),
            }
            with open(self._checkpoint_path, "w") as f:
                json.dump(checkpoint, f)

        if self.verbose:
            param_print = {name: np.round(v, 3).item() for name, v in zip(self.model.param_names, x)}
            iter_str = f"[BADS-{self.iteration_count}]"
            print(f"{iter_str:>30} "
                  f"time: {time() - self.time:.3g}s\t "
                  f"NLL(n={self.repeats}): {nlls:.5g} ± {total_std:.3g}\t "
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
            bads_options=None,
            checkpoint_path=None,
            atol_mesh=0.01,
            atol_fun=0.1):
        """
        Fit the model to data using a single-stage BADS optimization.

        If checkpoint_path points to an existing JSON file, resumes from the
        saved state with adaptively narrowed plausible bounds and a scaled
        tol_mesh that guarantees the same physical stopping resolution as a
        fresh run.
        """
        if bads_options is None:
            bads_options = {'uncertainty_handling': True, 'display': 'iter'}

        self.time = time()
        self.__class__.check_dataframe(data)
        self.data = data
        self._checkpoint_path = checkpoint_path
        self._best_nll = np.inf
        self._best_params = None
        self.repeats = 5
        self.iteration_count = 0

        lb = self.model.lower_bound
        ub = self.model.upper_bound
        orig_plb = self.model.plausible_lower_bound
        orig_pub = self.model.plausible_upper_bound
        active_options = dict(bads_options)

        gamma_orig = (orig_pub - orig_plb) / 2
        active_options['tol_mesh'] = atol_mesh / np.mean(gamma_orig)
        active_options['tol_fun'] = atol_fun

        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                ckpt = json.load(f)

            x0 = np.array(ckpt["best_params"])
            self._checkpoint_iter = int(ckpt["poll_iter"])
            narrowing_factor = max(0.1, float(ckpt["mesh_size"]))

            gamma_orig = (orig_pub - orig_plb) / 2
            buffer = 1e-3 * gamma_orig

            plb = x0 - narrowing_factor * gamma_orig
            pub = x0 + narrowing_factor * gamma_orig

            # Rigid-body shift: fix lower overflow, then upper overflow.
            lo_shift = np.maximum(0.0, lb + buffer - plb)
            plb += lo_shift
            pub += lo_shift
            hi_shift = np.maximum(0.0, pub - (ub - buffer))
            plb -= hi_shift
            pub -= hi_shift

            active_options["tol_mesh"] /= narrowing_factor

            print(f"\n[Resuming from checkpoint: poll_iter={self._checkpoint_iter}, "
                  f"mesh_size={ckpt['mesh_size']:.4f}, narrowing_factor={narrowing_factor:.4f}]")
            print(f"  x0: {np.round(x0, 4)}")
            print(f"  tol_mesh_resume: {active_options['tol_mesh']:.4g}")
        else:
            x0 = self.model.initial_params
            plb = orig_plb
            pub = orig_pub
            self._checkpoint_iter = 0

        print(f"\n[BADS Optimization Start]")
        print(f"  Options: {active_options}")
        self.print_params(x0, lb, ub, plb, pub)

        bads = BADS(self.optimize, x0, lb, ub, plb, pub, options=active_options)
        self._current_bads = bads
        try:
            result = bads.optimize()
        finally:
            self._current_bads = None

        fitted_params = result['x']
        print(f"\t[Fitted Parameters]\t {fitted_params}")
        print("\t[Final Log-likelihood]\t Estimating final log-likelihood...")

        self.repeats = self._max_repeats
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
