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
    create_feature
)

DEFAULT_TEMPLATES = {
    "4IAR": [[1, 1, 1, 1]],
    "3IAR": [[0, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
    "2IAR_CON": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
    "2IAR_DIS": [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
}



DEFAULT_PARAMETER_LIST = [
    {"name": "pruning_threshold", "initial_value": 0.2, "lower_bound": 0.0001, "upper_bound": 10, "plausible_lower_bound": 0.1, "plausible_upper_bound": 8},
    {"name": "stopping_prob", "initial_value": 0.9, "lower_bound": 0.05, "upper_bound": 1, "plausible_lower_bound": 0.3, "plausible_upper_bound": 0.99},
    {"name": "feature_drop", "initial_value": 0.3, "lower_bound": 0, "upper_bound": 1, "plausible_lower_bound": 0.1, "plausible_upper_bound": 0.5},
    {"name": "lapse_rate", "initial_value": 0.3, "lower_bound": 0.05, "upper_bound": 1, "plausible_lower_bound": 0.1, "plausible_upper_bound": 0.2},
    {"name": "opp_scale", "initial_value": 1.0, "lower_bound": 0.0, "upper_bound": 5, "plausible_lower_bound": 0.25, "plausible_upper_bound": 4},
    {"name": "center_weight", "initial_value": 0.4, "lower_bound": -10, "upper_bound": 10, "plausible_lower_bound": -2, "plausible_upper_bound": 2},
]
    
class TreeSearch:
    """
    Modular tree search model that constructs heuristics from templates.
    
    The model accepts custom templates and weights, allowing flexible heuristic
    construction. Features are generated from templates once during initialization
    and cached for efficient reuse. The create_heuristic method uses cached values
    to avoid redundant computation during parameter optimization.
    """
    def __init__(self, parameter_list=DEFAULT_PARAMETER_LIST, templates=DEFAULT_TEMPLATES, initial_values=None, verbose=True):
        self.name = "TreeSearch"
        
        self.parameter_list = parameter_list.copy()
        self.templates = templates.copy()
        self.sorted_groups = sorted(self.templates.keys())
        self.features = make_features_from_groups(self.templates)

        # Add feature weight parameters (one per template group)
        for group in self.sorted_groups:
            initial_value = 0
            if initial_values is not None:
                initial_value = initial_values.get(group, 0)
            self.parameter_list.append({
                "name": group,
                "initial_value": initial_value,
                "lower_bound": -20,
                "upper_bound": 100,
                "plausible_lower_bound": -10,
                "plausible_upper_bound": 20
            })

        # Compile parameters for optimization
        self.compile_parameters(verbose = verbose)

    def compile_parameters(self, verbose = True):
        # Extract parameter arrays for optimization
        self.param_names = [param["name"] for param in self.parameter_list]
        self.initial_params = np.array([param["initial_value"] for param in self.parameter_list], dtype=np.float32)
        self.upper_bound = np.array([param["upper_bound"] for param in self.parameter_list], dtype=np.float32)
        self.lower_bound = np.array([param["lower_bound"] for param in self.parameter_list], dtype=np.float32)
        self.plausible_upper_bound = np.array([param["plausible_upper_bound"] for param in self.parameter_list], dtype=np.float32)
        self.plausible_lower_bound = np.array([param["plausible_lower_bound"] for param in self.parameter_list], dtype=np.float32)        

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
            # opp_scale is now meant to be the scale of the opponent's features
            # e.g. 0.5 means opponent features mean half as much as self features
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

class MyopicTreeSearch(TreeSearch):
    """
    Myopic tree search model that constructs heuristics from templates.
    
    The model accepts custom templates and weights, allowing flexible heuristic
    construction. Features are generated from templates once during initialization
    and cached for efficient reuse. The create_heuristic method uses cached values
    to avoid redundant computation during parameter optimization.
    """
    def __init__(self, parameter_list=DEFAULT_PARAMETER_LIST, templates=DEFAULT_TEMPLATES, initial_values=None, verbose=True):
        super().__init__([param for param in parameter_list if param["name"] != "stopping_prob"], templates, initial_values=initial_values, verbose=verbose)
        self.name = "Myopic"

    def set_params(self, params):
        """Set parameters and construct heuristic from templates (vectorized, fixed order)."""
        assert len(params) == len(self.parameter_list), (
            f"Parameter length mismatch! Expected {len(self.parameter_list)} but got {len(params)}"
        )

        pruning_threshold, feature_drop, lapse_rate, opp_scale, center_weight = params[:5]
        control_vec = [pruning_threshold, 1.0, feature_drop, lapse_rate, opp_scale, center_weight]

        self.heuristic = self.create_heuristic(control_vec, params[5:])
        random_seed = random.randint(0, 2**64)
        self.heuristic.seed_generator(random_seed)
        # Store seed for debugging (if fitter has this attribute)
        if hasattr(self, '_fitter'):
            self._fitter.last_seed = random_seed

class MyopicSelfOnlyTreeSearch(MyopicTreeSearch):
    """
    Myopic tree search which ignores the opponent's features.
    """
    def __init__(self, parameter_list=DEFAULT_PARAMETER_LIST, templates=DEFAULT_TEMPLATES, initial_values=None, verbose=True):
        super().__init__([param for param in parameter_list if param["name"] != "stopping_prob" and param["name"] != "opp_scale"], templates, initial_values=initial_values, verbose=verbose)
        self.name = "SelfOnly"
    
    def set_params(self, params):
        """Set parameters and construct heuristic from templates (vectorized, fixed order)."""
        assert len(params) == len(self.parameter_list), (
            f"Parameter length mismatch! Expected {len(self.parameter_list)} but got {len(params)}"
        )

        pruning_threshold, feature_drop, lapse_rate, center_weight = params[:4]
        control_vec = [pruning_threshold, 1.0, feature_drop, lapse_rate, 0.0, center_weight]
        self.heuristic = self.create_heuristic(control_vec, params[4:])
        random_seed = random.randint(0, 2**64)
        self.heuristic.seed_generator(random_seed)
        if hasattr(self, "_fitter"):
            self._fitter.last_seed = random_seed

class LesionTreeSearch(TreeSearch):
    """
    A TreeSearch model with a specific template group removed (lesioned).
    """
    def __init__(self, lesion_key, parameter_list=DEFAULT_PARAMETER_LIST, templates=DEFAULT_TEMPLATES, initial_values=None, verbose=True):
        # Filter out the lesioned key
        lesioned_templates = {k: v for k, v in templates.items() if k != lesion_key}
        
        super().__init__(parameter_list=parameter_list, templates=lesioned_templates, initial_values=initial_values, verbose=verbose)
        self.name = f"Lesion_{lesion_key}"

