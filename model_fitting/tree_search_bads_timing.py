from collections import defaultdict
from functools import total_ordering
import numpy as np
import random
import fourbynine
import copy
import time
from pathlib import Path
from tqdm import tqdm
from parsers import *
import pandas as pd
import pickle
from abc import ABC, abstractmethod
import uuid
from time import time
import sys
import os
from pybads import BADS

# TIMING UTILITIES
class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.total_time = 0
        
    def start(self):
        self.start_time = time()
        print(f"[TIMER] {self.name} - STARTED at {self.start_time:.3f}")
        
    def stop(self):
        if self.start_time is not None:
            elapsed = time() - self.start_time
            self.total_time += elapsed
            print(f"[TIMER] {self.name} - COMPLETED in {elapsed:.3f}s (total: {self.total_time:.3f}s)")
            self.start_time = None
            return elapsed
        return 0
    
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class Model(ABC):
    """Abstract base class for models."""

    @abstractmethod
    def set_params(self, params):
        pass

    @abstractmethod
    def predict(self, board):
        pass

    def save(self, filename):
        """Save the model to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load the model from a file using pickle."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __call__(self, board):
        return self.predict(board)

class TreeSearch(Model):
    """The default model used by Bas."""
    
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.expt_factor = 1.0
        self.cutoff = 3.5
        self.parameter_list = [{
            "name": "Pruning Threshold", 
            "initial_value": 2.0, 
            "lower_bound": 0.5, 
            "upper_bound": 8.0, 
            "plausible_lower_bound": 1.0, 
            "plausible_upper_bound": 6.0},
        {
            "name": "Stopping Probability", 
            "initial_value": 0.05,
            "lower_bound": 0.01, 
            "upper_bound": 0.5, 
            "plausible_lower_bound": 0.02, 
            "plausible_upper_bound": 0.3},
        {
            "name": "Feature Drop Rate", 
            "initial_value": 0.2, 
            "lower_bound": 0.0, 
            "upper_bound": 0.8, 
            "plausible_lower_bound": 0.05, 
            "plausible_upper_bound": 0.6},
        {
            "name": "Lapse rate", 
            "initial_value": 0.1,
            "lower_bound": 0.05, 
            "upper_bound": 0.8, 
            "plausible_lower_bound": 0.08,
            "plausible_upper_bound": 0.4
            },
        {
            "name": "Opponent scale", 
            "initial_value": 1.2, 
            "lower_bound": 0.5, 
            "upper_bound": 3.0, 
            "plausible_lower_bound": 0.8, 
            "plausible_upper_bound": 2.0},
        {
            "name": "Center weight", 
            "initial_value": 0.8, 
            "lower_bound": -3.0, 
            "upper_bound": 3.0, 
            "plausible_lower_bound": -1.5, 
            "plausible_upper_bound": 2.0},
        {
            "name": "2IARConn", 
            "initial_value": 1.0, 
            "lower_bound": -2.0, 
            "upper_bound": 4.0, 
            "plausible_lower_bound": 0.0, 
            "plausible_upper_bound": 3.0},
        {
            "name": "2IARDisconn",
            "initial_value": 0.4,
            "lower_bound": -1.0,
            "upper_bound": 2.0,
            "plausible_lower_bound": 0.0,
            "plausible_upper_bound": 1.5},
        {
            "name": "3IAR",
            "initial_value": 3.5,
            "lower_bound": 1.0,
            "upper_bound": 8.0,
            "plausible_lower_bound": 2.0,
            "plausible_upper_bound": 6.0},
        {
            "name": "4IAR",
            "initial_value": 8.0,
            "lower_bound": 3.0,
            "upper_bound": 15.0,
            "plausible_lower_bound": 5.0,
            "plausible_upper_bound": 12.0}
        ]

        self.param_names = [param["name"] for param in self.parameter_list]
        self.initial_params = np.array([param["initial_value"] for param in self.parameter_list], dtype=np.float32)
        self.upper_bound = np.array([param["upper_bound"] for param in self.parameter_list], dtype=np.float32)
        self.lower_bound = np.array([param["lower_bound"] for param in self.parameter_list], dtype=np.float32)
        self.plausible_upper_bound = np.array([param["plausible_upper_bound"] for param in self.parameter_list], dtype=np.float32)
        self.plausible_lower_bound = np.array([param["plausible_lower_bound"] for param in self.parameter_list], dtype=np.float32)

        self.c = 50
        self.heuristic = None

    def set_params(self, params):
        set_params_start = time()
        assert len(params) == len(self.parameter_list), f"Parameter length mismatch! Expected {len(self.parameter_list)} but got {len(params)}"
        self.heuristic = fourbynine.fourbynine_heuristic.create(fourbynine.DoubleVector(bads_parameters_to_model_parameters(params)), True)
        self.heuristic.seed_generator(random.randint(0, 2**64))
        set_params_time = time() - set_params_start
        if set_params_time > 0.01:
            print(f"[TIMER] TreeSearch.set_params - {set_params_time:.4f}s")
    
    def predict(self, board): 
        """Predicts the best move for a given board state."""
        predict_start = time()
        search = fourbynine.NInARowBestFirstSearch(self.heuristic, board)
        search.complete_search()
        result = self.heuristic.get_best_move(search.get_tree()).board_position
        predict_time = time() - predict_start
        
        # Only log extremely slow predictions to reduce noise
        if predict_time > 0.05:  # 50ms threshold for warning
            print(f"[WARNING] Very slow prediction: {predict_time:.4f}s")
            
        return result

class BADSFitter:
    """BADS-based fitter for timing analysis."""
    
    def __init__(self, model: Model):
        self.model = model
        self.iteration_count = 0
        self.data = None

    def simple_log_likelihood(self, params, data: pd.DataFrame):
        """Calculate log likelihood without complex IBS tracking."""
        ll_start = time()
        
        # Set model parameters (ensure fresh model state each time)
        try:
            self.model.set_params(params)
        except Exception as e:
            print(f"[WARNING] Model parameter setting failed: {e}")
            return 10.0  # Return high penalty for failed parameter setting

        successes = 0
        total_predictions = 0
        predict_times = []
        
        for i, row in enumerate(data.itertuples()):
            trial_start = time()
            
            try:
                # Create board
                black_, white_, move_ = row.black, row.white, row.move
                board = fourbynine_board(fourbynine_pattern(black_), fourbynine_pattern(white_))
                actual_move = int(move_).bit_length() - 1
                
                # Make prediction
                predicted_move = self.model.predict(board)
                
                trial_time = time() - trial_start
                predict_times.append(trial_time)
                total_predictions += 1
                
                if predicted_move == actual_move:
                    successes += 1
                    
            except Exception as e:
                trial_time = time() - trial_start
                predict_times.append(trial_time)
                total_predictions += 1
                print(f"[WARNING] Prediction failed for trial {i}: {e}")
                # Count as failure (don't increment successes)

        accuracy = successes / total_predictions if total_predictions > 0 else 0
        avg_predict_time = np.mean(predict_times) if predict_times else 0
        
        ll_time = time() - ll_start
        
        # Return negative log likelihood (higher accuracy = lower NLL)
        nll = -np.log(max(accuracy, 0.001))  # Avoid log(0)
        
        print(f"[TIMER] Log-likelihood calculation: {ll_time:.3f}s, Accuracy: {accuracy:.3f}, Avg predict: {avg_predict_time:.4f}s")
        return nll

    def optimize(self, x):
        """Optimization function called by BADS."""
        opt_start = time()
        
        log_likelihood = self.simple_log_likelihood(x, self.data)
        
        opt_time = time() - opt_start
        
        # Only log every 10th iteration or slow iterations to reduce noise
        if self.iteration_count % 10 == 0 or opt_time > 0.05:
            print(f"[TIMER] [{self.iteration_count}] BADS iteration: {opt_time:.3f}s, NLL: {log_likelihood:.4f}")
        
        self.iteration_count += 1
        
        return log_likelihood

    def fit_with_bads(self, data: pd.DataFrame, max_fun_evals=50):
        """Fit the model using BADS with reduced iterations."""
        with Timer("TOTAL_BADS_FIT"):
            self.data = data
            self.iteration_count = 0
            
            print(f"[TIMING] Starting BADS optimization with max_fun_evals={max_fun_evals}")
            print(f"[TIMING] Data size: {len(data)} rows")
            
            bads_options = {
                'uncertainty_handling': False,  # Disable GP-based uncertainty handling to avoid NaN errors
                'max_fun_evals': max_fun_evals,
                'tol_fun': 1e-2   # Less strict function tolerance for faster convergence
            }
            
            with Timer("BADS_optimization"):
                bads = BADS(
                    self.optimize, 
                    self.model.initial_params, 
                    self.model.lower_bound, 
                    self.model.upper_bound, 
                    self.model.plausible_lower_bound, 
                    self.model.plausible_upper_bound, 
                    options=bads_options
                )
                # Run BADS optimization but extract parameters before result creation
                try:
                    # Run the optimization
                    bads.optimize()
                    # Extract parameters directly from optimizer state to avoid pickling issues
                    fitted_params = bads.x.copy()  # Copy to avoid reference issues
                    print(f"[SUCCESS] BADS optimization completed normally")
                except Exception as e:
                    print(f"[WARNING] BADS optimization failed: {e}")
                    # Try to extract the best parameters found so far
                    try:
                        fitted_params = bads.x.copy() if hasattr(bads, 'x') else bads.initial_params.copy()
                        print(f"[RECOVERY] Using best available parameters")
                    except:
                        fitted_params = self.model.initial_params.copy()
                        print(f"[FALLBACK] Using initial parameters")
            
            print(f"[TIMING] BADS completed after {self.iteration_count} iterations")
            print(f"[TIMING] Fitted parameters: {fitted_params}")
            
            return fitted_params

def main(): 
    with Timer("TOTAL_MAIN"):
        with Timer("setup"):
            data_path = "data"
            output_path = "data/out"
            n_splits = 5

            print(f"Building output directory at {output_path}")
            os.makedirs(output_path, exist_ok=True)

        with Timer("data_loading"):
            assert np.all([f"{i + 1}.csv" in os.listdir(data_path) for i in range(n_splits)])
            print("Detected splits in this directory. Loading splits ...")
            splits = [pd.read_csv(f"{data_path}/{i + 1}.csv") for i in range(n_splits)]
            total_rows = sum(len(split) for split in splits)
            print(f"Loaded {len(splits)} splits with {total_rows} total rows")

        with Timer("model_initialization"):
            model = TreeSearch()
            fitter = BADSFitter(model)

        # Test with a small subset first
        with Timer("bads_test"):
            # Combine first few splits for a reasonable dataset size
            test_data = pd.concat([splits[0], splits[1]])  # ~12 rows
            print(f"Testing BADS with {len(test_data)} rows")
            
            # Run BADS with realistic iterations to demonstrate speedup
            fitted_params = fitter.fit_with_bads(test_data, max_fun_evals=100)  # More realistic for timing comparison

        print("[TIMING] BADS timing analysis complete!")

if __name__ == "__main__":
    main()