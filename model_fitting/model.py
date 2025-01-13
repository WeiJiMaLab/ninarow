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


class IBSTracker:
    """
    Tracks the number of times the heuristic has evaluated a given position to the expected evaluation. Used for
    fitting the heuristic to a given dataset.
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
    

class DefaultModel:
    """
    The default model used by Bas.
    """

    def __init__(self):
        """
        Constructor.
        """
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
            "plausible_lower_bound": 0.001, 
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
            "initial_value": 5,
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

class Fitter:
    """
    The main class for finding the best heuristic/search parameter
    fit for a given dataset.
    """
    def __init__(self, model, threads=16, verbose = False):
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

    def calculate_expected_counts(self, L_values, c):
        """
        Generate the distribution of observation counts we should see for each move given that move's
        L-values. Essentially, we're converting likelihoods to probabilities here.

        Args:
            L_values: A list of L-values corresponding to each move.
            c: A scalar provided by the model.

        Returns:
            A list of the number of times we would expect each move to be reproduced given the L-values.
        """
        x = np.linspace(1e-6, 1 - 1e-6, int(1e6))
        dilog = np.pi**2 / 6.0 + np.cumsum(np.log(x) / (1 - x)) / len(x)
        p = np.exp(-L_values)
        interp1 = CubicSpline(x, np.sqrt(x * dilog), extrapolate=True)
        interp2 = CubicSpline(x, np.sqrt(dilog / x), extrapolate=True)
        times = (c * interp1(p)) / np.mean(interp2(p))
        return np.vectorize(lambda x: max(x, 1))(np.round(times))
    
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

    def parallel_log_likelihood(self, params, trackers, cutoff):
        """
        The main parallelized portion of our workload. Takes a set of
        heuristic parameters and a list of data and runs the heuristic
        against the list until the heuristic produces the expected number
        of matches to the observed dataset. Modifies trackers in place.

        Args:
            params: The heuristic parameters to test.
            cutoff: A stop-loss cutoff that will cause us to exit early if needed.
            trackers: The list of data that need to be evaluated by the heuristic.
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
            black_, white_, move_ = key
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

    def log_likelihood(self, params, data, counts = None):
        """
        Computes the log likelihood of the given set of parameters being the set that best fits
        the observed data.

        Args:
            trackers: The observed data to be fitted to.
            params: The parameters to evaluate.

        Returns:
            The log-likelihood of each observed move at each position given the set of parameters.
        """
        # success threshold is set to 1 by default if not provided
        # create IBSTrackers to keep track of successes and failures in predictions

        n_trials = len(data)
        if counts is None: counts = np.ones(n_trials)
        trackers = {(key.black, key.white, key.move): IBSTracker(self.model.expt_factor, success_threshold=count) for key, count in zip(data.itertuples(), counts)}
        shared_trackers = UltraDict(trackers, full_dump_size= n_trials * 1024 * 1024, shared_lock=True)

        global LOG_LIKELIHOOD
        LOG_LIKELIHOOD.value = n_trials * self.model.expt_factor

        global POOL
        results = [POOL.apply_async(self.parallel_log_likelihood, (params, shared_trackers, n_trials * self.model.cutoff)) for i in range(self.num_workers)]
        [result.get() for result in results]

        return np.array([shared_trackers[key].log_likelihood for key in shared_trackers])
    
    def optimize(self, x): 
        log_likelihood = self.log_likelihood(x, self.data, self.counts).sum()
        if self.verbose: print(f"\t[{self.iteration_count}] NLL: {np.round(log_likelihood, 4)} Params: {[np.round(x_, 3) for x_ in x]}")
        self.iteration_count += 1
        return log_likelihood

    def fit(self, data, bads_options={
                    'uncertainty_handling': True,
                    'noise_final_samples': 0,
                    'max_fun_evals': 2000
                  }):
        """
        Given a set of data, find the set of heuristic/search parameters that best fit the observations.

        @params data The set of data to fit to.

        @return The set of parameters that best correspond to the given data, as well as their corresponding L-values.
        """
        print("[Preprocessing] Initial log-likelihood estimation")
        self.__class__.check_dataframe(data)
        self.data = data
        initial_LL = np.array([self.log_likelihood(self.model.initial_params, data) for _ in tqdm(range(10))]).mean(axis = 0)
        self.counts = self.calculate_expected_counts(initial_LL, self.model.c).astype(int)

        # run PyBADS to optimize the initial parameter guesses
        bads = BADS(self.optimize, self.model.initial_params, self.model.lower_bound, self.model.upper_bound, self.model.plausible_lower_bound, self.model.plausible_upper_bound, options=bads_options)
        out_params = bads.optimize()['x']

        print("[Fitted Parameters]: {}".format(out_params))

        print("[Postprocessing] Final log-likelihood estimation")
        final_LL = np.array([self.log_likelihood(out_params, data) for _ in tqdm(range(10))]).mean(axis = 0)
        return out_params, final_LL

    def cross_validate(self, folds, leave_out):
        """
        Given a set of pre-split folds, cross validate the i-th group against the rest, i.e.,
        fit against all of the folds but the i-th and evaluate the resultant fit on the i-th group.

        Args:
            folds: A pre-split list of lists of data corresponding to different validation folds.
            i: The group that should be held-out of the fitting process and tested against.

        Returns:
            The best-fit parameters for all of the folds but the ith, as well as the log-likelihood
            of the data from both the training (non-i) and test (i) sets.
        """

        assert leave_out < len(folds), "Invalid leave-out index!"

        print("Cross validating split {} against the other {} splits".format(
            leave_out + 1, len(folds) - 1))
        test = folds[leave_out]
        train_folds = [fold for i, fold in enumerate(folds) if i != leave_out]
        train = [move for fold in train_folds for move in fold]

        params, loglik_train = self.fit_model(train)
        test_trackers = {}
        for move in test:
            test_trackers[move] = IBSTracker(self.model.expt_factor)
        loglik_test = list(self.log_likelihood(params, test_trackers).values())
        
        return params, loglik_train, loglik_test
    


def initialize_thread_pool(num_threads, manual_seed=None):

    def initialize_thread(shared_value):
        # initialize the thread to some shared value
        global LOG_LIKELIHOOD
        LOG_LIKELIHOOD = shared_value

    def set_seeds(base_seed, thread_id):
        # Create a unique seed for each thread
        thread_seed = base_seed + thread_id
        random.seed(thread_seed)
        print(f"Thread {thread_id}: Base Seed {base_seed}, Seed: {thread_seed}, Random Number: {random.randint(0, 2**64)}\n")
    
    global LOG_LIKELIHOOD, POOL
    LOG_LIKELIHOOD = Value('d', 0)
    POOL = Pool(num_threads, initializer=initialize_thread, initargs=(LOG_LIKELIHOOD,))

    if manual_seed is not None:
        assert num_threads == 1, "Setting manual seed can only be used with a single thread. If threads > 1, thread compute order is nondeterministic."
        print(f"Setting manual seed {manual_seed} for single-thread")
        POOL.starmap(set_seeds, [(manual_seed, i) for i in range(num_threads)])



def main():
    # INPUT FILE FORMAT should be:
    # black_pieces (binary),
    # white_pieces (binary),
    # player_color (Black/White),
    # move (binary),
    # response time (not used in fitting),
    # [group_id] (optional),
    # participant_id
    # for more info, see parsers.py

    random.seed()
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""Example usages:
    - Process a file named input.csv and save results to a folder named output/: 
      model_fit.py -f input.csv -o output/
    - Process a file, create 5 splits, and perform cross-validation: 
      model_fit.py -f input.csv 5 -o output/
    - Create 5 splits from a file and exit: 
      model_fit.py -f input.csv 5 -s -o output/
    - Load splits from the previous command and perform cross-validation: 
      model_fit.py -i output/ 5 -o output/
    - Load splits from the previous command, and only process/cross-validate a specific split (split 2, in this case) against the others: 
      model_fit.py -i output/ 5 -o output/ -c 2""")
    parser.add_argument(
        "-f",
        "--participant_file",
        help="The file containing participant data to be split, i.e. a list of board states, data, and associated timing. Optionally, a number of splits may be provided if cross-validation is desired.",
        type=str,
        nargs='+',
        metavar=(
            'input_file',
            'split_count'))
    parser.add_argument(
        "-i",
        "--input_dir",
        help="The directory containing the pre-split folds to parse and cross-validate, along with the expected number of splits to be parsed. These splits should be named [1-n].csv",
        type=str,
        nargs=2,
        metavar=(
            'input_dir',
            'split_count'))
    parser.add_argument(
        "-o",
        "--output_dir",
        help="The directory to output results to.",
        type=str,
        default="./",
        metavar=('output_dir'))
    parser.add_argument(
        "-s",
        "--splits-only",
        help="If specified, terminate after generating splits.",
        action='store_true')
    parser.add_argument(
        "-v",
        "--verbose",
        help="If specified, print extra debugging info.",
        action='store_true')
    parser.add_argument(
        "-c",
        "--cluster-mode",
        nargs=1,
        type=int,
        help="If specified, only process a single split, specified by the number passed as an argument to this flag. The split is expected to be named [arg].csv. This split will then be cross-validated against the other splits in the folder specified by the -i flag. Cannot be used with the -f flag; pre-split a -f argument with -s if desired.",
        metavar=('local_split'))
    parser.add_argument(
        "-r",
        "--random-sample",
        nargs=1,
        type=int,
        help="If specified, instead of testing each position on a BADS function evaluation, instead randomly sample up to N positions without replacement.",
        metavar=('n_samples'))
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=16,
        help="The number of threads to use when fitting.")
    args = parser.parse_args()
    if args.participant_file and args.input_dir:
        raise Exception("Can't specify both -f and -i!")

    folds = []

    # If the participant file is specified, generate splits.
    if args.participant_file:
        num_splits = 1
        if (len(args.participant_file) == 2):
            num_splits = int(args.participant_file[1])
        if (len(args.participant_file) > 2):
            raise Exception("-f only takes up to 2 arguments!")
        if (args.cluster_mode):
            raise Exception("-c cannot be used with -f!")

        # parse the participant file and generate splits
        data = parse_participant_file(args.participant_file[0])
        folds = generate_splits(data, num_splits)

    # If the input directory is specified, read in the splits.
    elif args.input_dir:
        # directory should be in the form
        # participant/[1-n].csv
        input_path = Path(args.input_dir[0])
        num_splits = int(args.input_dir[1])
        input_files = []
        for i in range(num_splits):
            input_files.append(input_path / (str(i + 1) + ".csv"))
        for input_path in input_files:
            print("Ingesting split {}".format(input_path))
            data = parse_participant_file(input_path)
            folds.append(data)
    else:
        raise Exception("Either -f or -i must be specified!")

    output_path = Path(args.output_dir)
    if not output_path.is_dir():
        output_path.mkdir()

    # Only output splits if we generated new ones to output.
    if args.participant_file:
        for i in range(len(folds)):
            new_split_path = output_path / (str(i + 1) + ".csv")
            print("Writing split {}".format(new_split_path))
            with (new_split_path).open('w') as f:
                for move in folds[i]:
                    f.write(str(move) + "\n")

    if args.splits_only:
        exit()

    set_start_method('spawn')
    global POOL, LOG_LIKELIHOOD
    LOG_LIKELIHOOD = Value('d', 0)
    POOL = Pool(args.threads, initializer=initialize_thread, initargs=(LOG_LIKELIHOOD,))
    model_fitter = ModelFitter(DefaultModel(), random_sample=bool(
        args.random_sample), verbose=bool(args.verbose), threads=args.threads)
    start, end = 0, len(folds)
    if (args.cluster_mode):
        start = args.cluster_mode[0] - 1
        end = start + 1
    for i in range(start, end):
        params, loglik_train, loglik_test = model_fitter.cross_validate(
            folds, i)
        with (output_path / ("params" + str(i + 1) + ".csv")).open('w') as f:
            f.write(','.join(str(x) for x in params))
        with (output_path / ("lltrain" + str(i + 1) + ".csv")).open('w') as f:
            f.write(','.join(str(x) for x in loglik_train))
        with (output_path / ("lltest" + str(i + 1) + ".csv")).open('w') as f:
            f.write(' '.join(str(x) for x in loglik_test) + '\n')


if __name__ == "__main__":
    main()