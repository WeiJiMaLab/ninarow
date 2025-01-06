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


class Model():
    def __init__(self):
        assert False, "NotImplementedError: init() must be implemented in a subclass of Model"

    def get_params(self): 
        return self.params

    def set_params(self, params): 
        assert len(params) == len(self.params), f"Dimension Mismatch: expected array of length {len(self.params)} but got length {len(params)}"
        self.params = params

    def __forward__(self, board): 
        pass


class StandardModel(Model): 
    def __init__(self): 
        self.expt_factor = 1.0
        self.cutoff = 3.5

        # parameter list spells out all of the 
        # different weights to be learned by the model
        # fitter.
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

        # initialize parameters
        self.params = self.initial_params

        # used in generate_attempt_counts
        self.c = 50 

    def __call__(self, board): 
        pass





