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
from abc import ABC, abstractmethod
import uuid
from time import time
    
class Model(ABC):
    """
    Abstract base class for models.
    """

    @abstractmethod
    def set_params(self, params):
        pass

    @abstractmethod
    def predict(self, board):
        pass

    def save(self, filename):
        """
        Save the model to a file using pickle.

        Args:
        filename: The name of the file to save the model to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load the model from a file using pickle.

        Args:
        filename: The name of the file to load the model from.
        Returns:
        The loaded model.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __call__(self, board):
        return self.predict(board)

class DefaultModel(Model):
    """
    The default model used by Bas.
    """

    def __init__(self):
        super().__init__()
        self.expt_factor = 1.0
        self.cutoff = 3.5
        self.parameter_list = [{
            "name": "Stopping threshold", 
            "initial_value": 2.0, 
            "lower_bound": 0.1, 
            "upper_bound": 10.0, 
            "plausible_lower_bound": 1.0, 
            "plausible_upper_bound": 9.99},
        {
            "name": "Pruning threshold", 
            "initial_value": 0.02, 
            "lower_bound": 0.001, 
            "upper_bound": 1.0, 
            "plausible_lower_bound": 0.1, 
            "plausible_upper_bound": 0.99},
        {
            "name": "Gamma", 
            "initial_value": 0.2, 
            "lower_bound": 0, 
            "upper_bound": 1, 
            "plausible_lower_bound": 0.001, 
            "plausible_upper_bound": 0.5},
        {
            "name": "Lapse rate", 
            "initial_value": 0.05, 
            "lower_bound": 0, 
            "upper_bound": 1, 
            "plausible_lower_bound": 0.05, 
            "plausible_upper_bound": 0.5
            },
        {
            "name": "Opponent scale", 
            "initial_value": 1.2, 
            "lower_bound": 0.25, 
            "upper_bound": 4, 
            "plausible_lower_bound": 0.5, 
            "plausible_upper_bound": 2},
        {
            "name": "Exploration constant", 
            "initial_value": 0.8, 
            "lower_bound": -10, 
            "upper_bound": 10, 
            "plausible_lower_bound": -5, 
            "plausible_upper_bound": 5},
        {
            "name": "Center weight", 
            "initial_value": 1, 
            "lower_bound": -10, 
            "upper_bound": 10, 
            "plausible_lower_bound": -5, 
            "plausible_upper_bound": 5},
        {
            "name": "FP C_act",
            "initial_value": 0.4,
            "lower_bound": -10,
            "upper_bound": 10,
            "plausible_lower_bound": -5,
            "plausible_upper_bound": 5},
        {
            "name": "FP C_pass",
            "initial_value": 3.5,
            "lower_bound": -10,
            "upper_bound": 10,
            "plausible_lower_bound": -5,
            "plausible_upper_bound": 5},
        {
            "name": "FP delta",
            "initial_value": 10,
            "lower_bound": -10,
            "upper_bound": 10,
            "plausible_lower_bound": -5,
            "plausible_upper_bound": 5}
        ]

        self.param_names = [param["name"] for param in self.parameter_list]
        self.initial_params = np.array([param["initial_value"] for param in self.parameter_list], dtype=np.float64)
        self.upper_bound = np.array([param["upper_bound"] for param in self.parameter_list], dtype=np.float64)
        self.lower_bound = np.array([param["lower_bound"] for param in self.parameter_list], dtype=np.float64)
        self.plausible_upper_bound = np.array([param["plausible_upper_bound"] for param in self.parameter_list], dtype=np.float64)
        self.plausible_lower_bound = np.array([param["plausible_lower_bound"] for param in self.parameter_list], dtype=np.float64)

        self.c = 50 # used in calculate_expected_counts

    def set_params(self, params):
        assert len(params) == len(self.parameter_list), f"Parameter length mismatch! Expected {len(self.parameter_list)} but got {len(params)}"
        self.heuristic = fourbynine.fourbynine_heuristic.create(fourbynine.DoubleVector(bads_parameters_to_model_parameters(params)), True)
        self.heuristic.seed_generator(random.randint(0, 2**64))
    
    def predict(self, board): 
        '''
        Predicts the best move for a given board state.
        Args:
            board: The board state to predict the best move for.
        Returns:
            The index of the best move for the given board state.
        '''
        search = fourbynine.NInARowBestFirstSearch(self.heuristic, board)
        search.complete_search()
        return self.heuristic.get_best_move(search.get_tree()).board_position
        
class Fitter:
    """
    The main class for finding the best heuristic/search parameter
    fit for a given dataset.
    """
    def __init__(self, model: Model, threads=16, verbose = False):
        """
        Constructor.

        Args:
            model: The model this fitter should use. Produces heuristics/searches, and supplies
                   parameters for fitting.
            random_sample: If specified, instead of testing each position on a BADS function evaluation, 
                           instead randomly sample up to N positions without replacement.
            verbose: If specified, print extra debugging info.
            threads: The number of threads to use when fitting.
        """
        self.model = model
        self.verbose = verbose
        self.num_workers = threads
        self.iteration_count = 0
        self.time = time()

    def calculate_expected_counts(self, log_likelihoods, c):
        """
        Calculate the expected observation counts for each move based on their L-values.

        This function converts log-likelihoods to probabilities and then determines the
        expected number of times each move should be reproduced.

            log_likelihoods (list): A list of L-values corresponding to each move.
            c (float): A scalar provided by the model.

            list: A list of the expected number of times each move would be reproduced given the L-values.
        """
        x = np.linspace(1e-6, 1 - 1e-6, int(1e6))
        dilog = np.pi**2 / 6.0 + np.cumsum(np.log(x) / (1 - x)) / len(x)
        p = np.exp(-log_likelihoods)
        interp1 = CubicSpline(x, np.sqrt(x * dilog), extrapolate=True)
        interp2 = CubicSpline(x, np.sqrt(dilog / x), extrapolate=True)
        times = (c * interp1(p)) / np.mean(interp2(p))
        return np.vectorize(lambda x: max(x, 1))(np.round(times))

    def parallel_log_likelihood(self, params, trackers: UltraDict, cutoff: float):
        """
        Compute the log-likelihood of the model parameters in parallel.
        This function runs a parallelized process to compute the log-likelihood of the model parameters.
        It updates the global log-likelihood value and the trackers for each trial until the log-likelihood
        exceeds the specified cutoff value.
        Parameters:
        -----------
        params : array-like
            The parameters to set for the model.
        trackers : dict
            A dictionary of trackers for each trial, where each key is a tuple representing the trial
            and each value is a tracker object that keeps track of successes and failures.
        cutoff : float
            The cutoff value for the log-likelihood. Once the global log-likelihood exceeds this value,
            all processes should exit.
        Returns:
        --------
        None
        """
        
        self.model.set_params(params)

        # tracking the global expected log-likelihood across all processes
        # if it exceeds the cutoff, all processes should exit
        while LOG_LIKELIHOOD.value <= cutoff:

            # filter for the trials that have not yet met the success threshold
            incomplete_trials = [(key, tracker) for key, tracker in trackers.items() if tracker.success_count < tracker.success_threshold]
            if not incomplete_trials: break
            
            # Select a random incomplete trial and make a deep copy of it
            key, tracker = copy.deepcopy(random.choice(incomplete_trials))

            # convert the key to a board state and move index
            black_, white_, move_, _= key
            board = fourbynine_board(fourbynine_pattern(black_), fourbynine_pattern(white_))
            actual_move = int(move_).bit_length() - 1

            # delta_log_likelihood accumulates the change in log-likelihood 
            # while the process is running
            delta_log_likelihood = 0

            while tracker.success_count < tracker.success_threshold:
                predicted_move = self.model.predict(board)
                
                # if the prediction is correct
                if (predicted_move == actual_move):
                    delta_log_likelihood += tracker.record_success()

                    # update the global trackers and log likelihood
                    with trackers.lock:
                        if tracker.success_count == trackers[key].success_count + 1:
                            trackers[key] = tracker
                            LOG_LIKELIHOOD.value += delta_log_likelihood
                    break
                
                # if the prediction is incorrect
                else:
                    delta_log_likelihood += tracker.record_failure()

                    # if the cutoff is met, we need to exit all processes
                    if LOG_LIKELIHOOD.value + delta_log_likelihood > cutoff:
                        # this signals to all other processes that the cutoff has been met
                        with trackers.lock:
                            LOG_LIKELIHOOD.value += delta_log_likelihood
                        break

    def log_likelihood(self, params, data: pd.DataFrame, counts = None):
        """
        Calculate the log likelihood of the model given the parameters and data.
        Parameters:
        -----------
        params : array-like
            The parameters of the model.
        data : pd.DataFrame
            The data to fit the model to. Each row represents a trial.
        counts : array-like, optional
            The counts for each trial. If None, defaults to an array of ones with the same length as the number of trials.
        Returns:
        --------
        np.ndarray
            An array of log likelihood values for each tracker.
        Notes:
        ------
        - This function uses parallel processing to speed up the computation of log likelihoods.
        - The `shared_trackers` dictionary is used to store the IBSTracker instances for each trial.
        - The `LOG_LIKELIHOOD` global variable is updated with the initial log likelihood value.
        - The `POOL` global variable is used to manage the pool of worker processes.
        """
        tick = time()
        n_trials = len(data)
        if counts is None: counts = np.ones(n_trials)

        # initialize IBS trackers to keep track of successes and failures in model simulation
        trackers = {(key.black, key.white, key.move, uuid.uuid4()): IBSTracker(self.model.expt_factor, success_threshold=count) for key, count in zip(data.itertuples(), counts)}
        assert(len(trackers)) == n_trials

        shared_trackers = UltraDict(trackers, full_dump_size= n_trials * 1024 * 1024, shared_lock=True)

        global LOG_LIKELIHOOD
        LOG_LIKELIHOOD.value = n_trials * self.model.expt_factor

        global POOL
        results = [POOL.apply_async(self.parallel_log_likelihood, (params, shared_trackers, n_trials * self.model.cutoff)) for i in range(self.num_workers)]
        [result.get() for result in results]

        print(f"\tTime taken: {time() - self.time} since Start, {time() - tick} since Loop")        
        return np.array([shared_trackers[key].log_likelihood for key in shared_trackers])
    
    def optimize(self, x): 
        log_likelihood = self.log_likelihood(x, self.data, self.counts).sum()
        if self.verbose: print(f"\t[{self.iteration_count}] NLL: {np.round(log_likelihood, 4)} Params: {[np.round(x_, 3) for x_ in x]}")
        self.iteration_count += 1
        return log_likelihood
    
    def evaluate(self, params, data: pd.DataFrame, n_iters = 10, counts = None):
        """
        Evaluates the log-likelihood of the given parameters on the given data.

        Args:
        params: The parameters to evaluate.
        data: The observed data to be fitted to.
        counts: The number of times each move was observed in the data. If not provided, 
                each move is assumed to have been observed once.
        n_iters: The number of iterations to run the evaluation for.
        """
        return np.array([self.log_likelihood(params, data, counts) for _ in tqdm(range(n_iters))]).mean(axis = 0)


    def fit(self, data: pd.DataFrame, bads_options={
                    'uncertainty_handling': True,
                    'noise_final_samples': 0,
                    'max_fun_evals': 2000
                  }):
        """
        Fits the model to the provided data using the BADS optimization algorithm.
        Parameters:
        data (pd.DataFrame): The input data to fit the model to.
        bads_options (dict, optional): Options for the BADS optimizer. Defaults to:
            {
                'uncertainty_handling': True,
                'noise_final_samples': 0,
                'max_fun_evals': 2000
            }
        Returns:
        tuple: A tuple containing:
            - out_params (np.ndarray): The optimized parameters.
            - final_LL (float): The final log-likelihood estimation on the training data
        Notes:
        - This method performs initial log-likelihood estimation, runs the BADS optimizer,
            and then performs final log-likelihood estimation.
        - The method prints the fitted parameters and log-likelihood estimations during the process.
        """
        print("[Preprocessing] Initial log-likelihood estimation")
        self.time = time()
        # first check to see if the dataframe is valid
        self.__class__.check_dataframe(data)
        self.data = data

        # calculate the expected counts for each move by estimating the 
        # LL with the initial guess
        initial_LL = self.evaluate(self.model.initial_params, data)
        self.counts = self.calculate_expected_counts(initial_LL, self.model.c).astype(int)

        # run PyBADS to optimize the initial parameter guesses
        bads = BADS(self.optimize, self.model.initial_params, self.model.lower_bound, self.model.upper_bound, self.model.plausible_lower_bound, self.model.plausible_upper_bound, options=bads_options)
        fitted_params = bads.optimize()['x']

        print("[Fitted Parameters]: {}".format(fitted_params))

        print("[Postprocessing] Final log-likelihood estimation")
        final_LL = self.evaluate(fitted_params, data)
        return fitted_params, final_LL
    
    @staticmethod
    def check_dataframe(data): 
        """
        Check that the data is in the correct format for fitting.
        """
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
        A tracker for the Inverse Binomial Sampling (IBS) process, used to monitor and fit a heuristic to a given dataset.
        The IBSTracker class keeps track of the number of successful and unsuccessful heuristic evaluations of a given 
        position, and adjusts the log-likelihood based on these evaluations. It is particularly useful in scenarios where 
        the heuristic needs to be iteratively fitted to improve its accuracy.
        Attributes:
            success_threshold (int): The threshold for considering a prediction as successful.
            expt_factor (float): A factor controlling the fitting cutoff of the BADS (Bayesian Adaptive Direct Search) process.
            attempt_count (int): The number of attempts made since the last success.
            success_count (int): The total number of successful predictions.
            log_likelihood (float): The cumulative log-likelihood of the heuristic's performance.
        Methods:
            record_success():
                Records a successful prediction, resets the attempt count, and returns the change in log-likelihood.
            record_failure():
                Records an unsuccessful prediction, increments the attempt count, updates the log-likelihood, and returns 
                the change in log-likelihood.
            __repr__():
                Returns a string representation of the current state of the tracker, including the number of successes, 
                attempts, and the log-likelihood.
    """
    def __init__(self, expt_factor, success_threshold = 1):
        """
        Constructor.

        Args:
            expt_factor: Controls the fitting cutoff of the BADS process.
        """
        self.success_threshold = success_threshold
        self.expt_factor = expt_factor
        self.attempt_count, self.success_count, self.log_likelihood = 0, 0, 0.0

    def record_success(self):
        """
        When the prediction is correct, record it and return the log likelihood diff
        Returns:
            The change in log-likelihood.
        """
        scale_factor = self.expt_factor / self.success_threshold
        self.success_count += 1

        # reset the attempt count
        self.attempt_count = 0
        return -scale_factor

    def record_failure(self):
        """
        When a prediction is incorrect, record it and return the log likelihood diff
        Returns:
            The change in log-likelihood.
        """
        scale_factor = self.expt_factor / self.success_threshold
        self.attempt_count += 1
        delta = scale_factor * (1 / self.attempt_count)
        self.log_likelihood += delta
        return delta
    
    def __repr__(self):
        return f"Successes: {self.success_count}, Attempts: {self.attempt_count}, Log-likelihood: {self.log_likelihood}"

def initialize_thread(shared_value):
    # initialize the thread to some shared value
    global LOG_LIKELIHOOD
    LOG_LIKELIHOOD = shared_value

def set_seeds(base_seed, thread_id):
    # Create a unique seed for each thread
    thread_seed = base_seed + thread_id
    random.seed(thread_seed)
    print(f"Thread {thread_id}: Base Seed {base_seed}, Seed: {thread_seed}, Random Number: {random.randint(0, 2**64)}\n")
    

def initialize_thread_pool(num_threads, manual_seed=None):
    """
    Initializes a thread pool with a specified number of threads and an optional manual seed.
    Args:
        num_threads (int): The number of threads to initialize in the pool.
        manual_seed (int, optional): A manual seed for random number generation. 
                                     This can only be used with a single thread. 
                                     If more than one thread is specified, an assertion error will be raised.
    Raises:
        AssertionError: If `manual_seed` is provided and `num_threads` is greater than 1.
    Notes:
        - The function initializes a global variable `LOG_LIKELIHOOD` to be shared among threads.
        - If `manual_seed` is provided, it sets a unique seed for the single thread and prints the seed information.
    """
    global LOG_LIKELIHOOD, POOL
    LOG_LIKELIHOOD = Value('d', 0)
    POOL = Pool(num_threads, initializer=initialize_thread, initargs=(LOG_LIKELIHOOD,))

    if manual_seed is not None:
        assert num_threads == 1, "Setting manual seed can only be used with a single thread. If threads > 1, thread compute order is nondeterministic."
        print(f"Setting manual seed {manual_seed} for single-thread")
        POOL.starmap(set_seeds, [(manual_seed, i) for i in range(num_threads)])

def cross_validate(model: Model, folds: list, leave_out_idx: int, threads: int = 16):
    """
    Perform cross-validation on the given model using the specified folds.

    Parameters:
    model (Model): The model to be cross-validated.
    folds (list): A list of dataframes, each representing a fold of the data.
    leave_out_idx (int): The index of the fold to be used as the test set.
    threads (int, optional): The number of threads to use for fitting the model. Default is 16.

    Returns:
    tuple: A tuple containing the fitted parameters, training log-likelihood, and test log-likelihood.

    Raises:
    AssertionError: If leave_out_idx is not a valid index in folds.
    """
    assert leave_out_idx < len(folds), "Invalid leave-out index!"

    print("Cross validating split {} against the other {} splits".format(leave_out_idx + 1, len(folds) - 1))
    test = folds[leave_out_idx]

    train = []
    for j in range(len(folds)):
        if leave_out_idx != j:
            train.append(folds[j])

    train = pd.concat(train)
    fitter = Fitter(model, threads = threads, verbose = True)
    params, trainLL = fitter.fit(train)

    testLL = fitter.evaluate(params, test)
    return params, trainLL, testLL




import os
def main(): 
    # code to test the consistency of the model fitting
    data_path = "../data"
    output_path = "../data/out"
    n_splits = 5
    fold_number = 1
    threads = 1
    random_sample = False
    verbose = True

    print(f"Building output directory at {output_path}")
    os.makedirs(output_path, exist_ok = True)

    # first, we have to check to see if all the splits are there ...
    assert np.all([f"{i + 1}.csv" in os.listdir(data_path) for i in range(n_splits)])
    print("Detected splits in this directory. Loading splits ...")

    # then we read them in
    splits = [pd.read_csv(f"{data_path}/{i + 1}.csv") for i in range(n_splits)]

    random.seed(10)
    initialize_thread_pool(1, manual_seed = 10)

    q = cross_validate(DefaultModel(), splits, leave_out_idx = 1, threads = 1)


if __name__ == "__main__":
    main()