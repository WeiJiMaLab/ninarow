from pathlib import Path

import numpy as np
import random
import fourbynine
import pickle
import yaml
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

# Default parameter/weight bounds live in config.yaml (single source of truth,
# mirrored from the monkey_4iar production ÷5 "OG" regime) rather than inline
# here, so the two repos' defaults can't silently drift apart.
_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(_CONFIG_PATH) as _f:
    _CONFIG = yaml.safe_load(_f)

DEFAULT_PARAMETER_LIST = [
    {"name": name, "lower_bound": vals[0], "plausible_lower_bound": vals[1],
     "initial_value": vals[2], "plausible_upper_bound": vals[3], "upper_bound": vals[4]}
    for name, vals in _CONFIG["controls"].items()
]

_WEIGHT_HARD_LB = _CONFIG["weights"]["hard_lower_bound"]
_WEIGHT_HARD_UB = _CONFIG["weights"]["hard_upper_bound"]
_WEIGHT_GROUP_BANDS = _CONFIG["weights"]["groups"]


# A feature_list entry defines a feature group's template AND its weight's
# parameter spec together, so features and their weight bounds live in one place.
# TreeSearch builds its templates + weight parameters from this list and assumes
# each entry is COMPLETE (every weight-bound key present) — build via this helper.
def feature_list_from_templates(templates, initial_values=None, **weight_bound_overrides):
    """Build a feature_list (one entry per group: name + template + weight param
    spec) from a ``{name: template}`` dict, filling the default weight bounds
    (override any via kwargs; per-group inits via ``initial_values``).

    Defaults come from config.yaml's ``weights`` section: groups named there
    (the OG 4-group set) get their own initial/plausible band; any other group
    name falls back to a flat uninformative band at the same hard bounds.
    """
    weight_defaults = {
        "initial_value": 0, "lower_bound": _WEIGHT_HARD_LB, "upper_bound": _WEIGHT_HARD_UB,
        "plausible_lower_bound": _WEIGHT_HARD_LB / 2, "plausible_upper_bound": _WEIGHT_HARD_UB / 2,
    }
    feature_list = []
    for name, template in templates.items():
        spec = {"name": name, "template": template, **weight_defaults}
        if name in _WEIGHT_GROUP_BANDS:
            plb, init, pub = _WEIGHT_GROUP_BANDS[name]
            spec.update(initial_value=init, plausible_lower_bound=plb, plausible_upper_bound=pub)
        spec.update(weight_bound_overrides)
        if initial_values and name in initial_values:
            spec["initial_value"] = initial_values[name]
        feature_list.append(spec)
    return feature_list


DEFAULT_FEATURE_LIST = feature_list_from_templates(DEFAULT_TEMPLATES)


class TreeSearch:
    """
    Modular tree search model that constructs heuristics from templates.
    
    The model accepts custom templates and weights, allowing flexible heuristic
    construction. Features are generated from templates once during initialization
    and cached for efficient reuse. The create_heuristic method uses cached values
    to avoid redundant computation during parameter optimization.
    """
    def __init__(self, parameter_list=DEFAULT_PARAMETER_LIST, feature_list=DEFAULT_FEATURE_LIST, verbose=True, exclude_feature_drop=False):
        self.name = "TreeSearch"
        # When True, drop the (pinned-nuisance) feature_drop from the fitted parameter
        # set so BADS does not spend mesh/design on it; set_params re-inserts 0.0 at its
        # canonical control index. Mirrors how Myopic excludes stopping_prob.
        self.exclude_feature_drop = exclude_feature_drop

        self.parameter_list = parameter_list.copy()
        self.feature_list = list(feature_list)
        self.templates = {f["name"]: f["template"] for f in self.feature_list}
        self.sorted_groups = sorted(self.templates.keys())
        self.features = make_features_from_groups(self.templates)

        # Append one weight parameter per feature group, from the feature_list.
        # Entries must be complete — a missing bound key is an error, not a
        # silent default (build the list via feature_list_from_templates).
        spec = {f["name"]: f for f in self.feature_list}
        for group in self.sorted_groups:
            f = spec[group]
            self.parameter_list.append({
                "name": group,
                "initial_value": f["initial_value"],
                "lower_bound": f["lower_bound"],
                "upper_bound": f["upper_bound"],
                "plausible_lower_bound": f["plausible_lower_bound"],
                "plausible_upper_bound": f["plausible_upper_bound"],
            })

        # Drop the pinned feature_drop nuisance from the search space (re-inserted as
        # 0.0 in set_params). Done after the weight append so the order is unchanged.
        if self.exclude_feature_drop:
            self.parameter_list = [p for p in self.parameter_list if p["name"] != "feature_drop"]

        # Compile parameters for optimization
        self.compile_parameters(verbose=verbose)

    def compile_parameters(self, verbose = True):
        # Extract parameter arrays for optimization
        self.param_names = [param["name"] for param in self.parameter_list]
        self.initial_params = np.array([param["initial_value"] for param in self.parameter_list], dtype=np.float32)
        self.upper_bound = np.array([param["upper_bound"] for param in self.parameter_list], dtype=np.float32)
        self.lower_bound = np.array([param["lower_bound"] for param in self.parameter_list], dtype=np.float32)
        self.plausible_upper_bound = np.array([param["plausible_upper_bound"] for param in self.parameter_list], dtype=np.float32)
        self.plausible_lower_bound = np.array([param["plausible_lower_bound"] for param in self.parameter_list], dtype=np.float32)        

    def create_heuristic(self, control_vec, weight_vec):
        """
        Construct heuristic directly from ordered parameter arrays.
        
        Args:
            control_vec: [pruning_threshold, stopping_prob, feature_drop, lapse_rate, opp_scale, center_weight]
            weight_vec: one weight per template group (sorted)

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
        for weight, group_name in zip(weight_vec, self.sorted_groups):
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
        n_ctrl = 5 if self.exclude_feature_drop else 6
        controls = list(params[:n_ctrl])
        if self.exclude_feature_drop:
            controls.insert(2, 0.0)  # feature_drop at canonical control index 2
        self.heuristic = self.create_heuristic(controls, params[n_ctrl:])
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
    def __init__(self, parameter_list=DEFAULT_PARAMETER_LIST, feature_list=DEFAULT_FEATURE_LIST, verbose=True, exclude_feature_drop=False):
        super().__init__([param for param in parameter_list if param["name"] != "stopping_prob"], feature_list, verbose=verbose, exclude_feature_drop=exclude_feature_drop)
        self.name = "Myopic"

    def set_params(self, params):
        """Set parameters and construct heuristic from templates (vectorized, fixed order)."""
        assert len(params) == len(self.parameter_list), (
            f"Parameter length mismatch! Expected {len(self.parameter_list)} but got {len(params)}"
        )

        if self.exclude_feature_drop:
            pruning_threshold, lapse_rate, opp_scale, center_weight = params[:4]
            feature_drop, n_ctrl = 0.0, 4
        else:
            pruning_threshold, feature_drop, lapse_rate, opp_scale, center_weight = params[:5]
            n_ctrl = 5
        control_vec = [pruning_threshold, 1.0, feature_drop, lapse_rate, opp_scale, center_weight]

        self.heuristic = self.create_heuristic(control_vec, params[n_ctrl:])
        random_seed = random.randint(0, 2**64)
        self.heuristic.seed_generator(random_seed)
        # Store seed for debugging (if fitter has this attribute)
        if hasattr(self, '_fitter'):
            self._fitter.last_seed = random_seed

class MyopicSelfOnlyTreeSearch(MyopicTreeSearch):
    """
    Myopic tree search which ignores the opponent's features.
    """
    def __init__(self, parameter_list=DEFAULT_PARAMETER_LIST, feature_list=DEFAULT_FEATURE_LIST, verbose=True, exclude_feature_drop=False):
        super().__init__([param for param in parameter_list if param["name"] != "stopping_prob" and param["name"] != "opp_scale"], feature_list, verbose=verbose, exclude_feature_drop=exclude_feature_drop)
        self.name = "SelfOnly"

    def set_params(self, params):
        """Set parameters and construct heuristic from templates (vectorized, fixed order)."""
        assert len(params) == len(self.parameter_list), (
            f"Parameter length mismatch! Expected {len(self.parameter_list)} but got {len(params)}"
        )

        if self.exclude_feature_drop:
            pruning_threshold, lapse_rate, center_weight = params[:3]
            feature_drop, n_ctrl = 0.0, 3
        else:
            pruning_threshold, feature_drop, lapse_rate, center_weight = params[:4]
            n_ctrl = 4
        control_vec = [pruning_threshold, 1.0, feature_drop, lapse_rate, 0.0, center_weight]
        self.heuristic = self.create_heuristic(control_vec, params[n_ctrl:])
        random_seed = random.randint(0, 2**64)
        self.heuristic.seed_generator(random_seed)
        if hasattr(self, "_fitter"):
            self._fitter.last_seed = random_seed

class LesionTreeSearch(TreeSearch):
    """
    A TreeSearch model with a specific template group removed (lesioned).
    """
    def __init__(self, lesion_key, parameter_list=DEFAULT_PARAMETER_LIST, feature_list=DEFAULT_FEATURE_LIST, verbose=True, exclude_feature_drop=False):
        # Drop the lesioned group's feature (and its weight parameter).
        lesioned = [f for f in feature_list if f["name"] != lesion_key]
        super().__init__(parameter_list=parameter_list, feature_list=lesioned, verbose=verbose, exclude_feature_drop=exclude_feature_drop)
        self.name = f"Lesion_{lesion_key}"

