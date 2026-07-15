"""Board/feature math: bitboard win-pattern generation, heuristic construction
from templates, and BADS<->heuristic parameter conversion. Merged from the
former feature_generator.py + ninarow_utilities.py (one theme, two call sites:
tree_search.py and tests/python/feature_test.py)."""

import random

from fourbynine import *
import numpy as np

try:
    from fourbynine import DoubleVector
except ImportError:
    pass


def win_patterns(m=4, n=9, k=4, directions="-/\\|"):
    """Generate all k-in-a-row board positions for given dimensions and directions."""
    patterns = []

    # Horizontal: '-'
    if '-' in directions:
        for row in range(m):
            for col in range(n - k + 1):
                pos = [n * row + col + i for i in range(k)]
                patterns.append(('-', pos))

    # Vertical: '|'
    if '|' in directions:
        for row in range(m - k + 1):
            for col in range(n):
                pos = [n * (row + i) + col for i in range(k)]
                patterns.append(('|', pos))

    # Diagonal (down-right): '\\'
    if '\\' in directions:
        for row in range(m - k + 1):
            for col in range(n - k + 1):
                pos = [n * (row + i) + (col + i) for i in range(k)]
                patterns.append(('\\', pos))

    # Anti-diagonal (down-left): '/'
    if '/' in directions:
        for row in range(m - k + 1):
            for col in range(k - 1, n):
                pos = [n * (row + i) + (col - i) for i in range(k)]
                patterns.append(('/', pos))

    return patterns


def make_feature_from_template(template, m=4, n=9, directions="-/\\|"):
    """Tile a template across the board to create bitboard features."""
    k = len(template)
    patterns = win_patterns(m, n, k, directions)
    features = []

    # Calculate min_empty (number of empty spaces required)
    min_empty = sum(1 - x for x in template)

    for direction, positions in patterns:
        # Create bitboards for pieces and spaces
        pieces = 0
        spaces = 0

        for i, bit_pos in enumerate(positions):
            if template[i] == 1:
                # This position should have a piece
                pieces |= (1 << bit_pos)
            else:
                # This position should be empty
                spaces |= (1 << bit_pos)

        features.append((pieces, spaces, min_empty))

    return features


def make_features_from_groups(group_templates, m=4, n=9, directions="-/\\|"):
    """Batch generate features for a dictionary of template groups."""
    features_by_group = {}

    for group_name, templates in sorted(group_templates.items()):
        features_by_group[group_name] = []

        for template in templates:
            template_features = make_feature_from_template(template, m, n, directions)
            features_by_group[group_name].extend(template_features)

    return features_by_group


def create_feature(pieces, spaces, min_empty):
    """Convert bitboards into a fourbynine heuristic feature."""
    return fourbynine_heuristic_feature(
        fourbynine_pattern(pieces),
        fourbynine_pattern(spaces),
        min_empty
    )


def build_control_params(control_params):
    """Order control parameters for the C++ heuristic constructor."""
    pruning = control_params["pruning_threshold"]
    stopping_prob = control_params["stopping_prob"]
    lapse = control_params["lapse_rate"]
    center_weight = control_params["center_weight"]

    base_control = [
        10000.0,         # Fixed stopping threshold
        pruning,
        stopping_prob,
        lapse,
        1.0,             # Exploration constant placeholder
        1.0,             # Opponent scale placeholder
        center_weight
    ]
    return base_control


def create_modular_heuristic(control_params, weights, templates):
    """Initialize a full heuristic from a set of control parameters and pattern weights."""
    # Initialize heuristic with control parameters
    control_vec = build_control_params(control_params)
    heuristic = fourbynine_heuristic.create(DoubleVector(control_vec), False)

    # Generate features from templates
    features_by_group = make_features_from_groups(templates)

    # Create feature groups and add features
    sorted_groups = sorted(templates.keys())
    opp_scale = control_params["opp_scale"]
    feature_drop = control_params["feature_drop"]

    for group_name in sorted_groups:
        if group_name not in weights:
            raise ValueError(f"Group '{group_name}' in templates but not in weights dict")

        weight = weights[group_name]
        heuristic.add_feature_group(weight, weight * opp_scale, feature_drop)
        group_idx = len(heuristic.get_feature_group_weights()) - 1

        # Add features for this group
        group_features = features_by_group[group_name]
        for pieces, spaces, min_empty in group_features:
            feature = create_feature(pieces, spaces, min_empty)
            heuristic.add_feature(group_idx, feature)

    return heuristic


def search_from_position(position, heuristic, noise_enabled=True, seed=None):
    """
    Given a position and a heuristic, execute a search from the given position
    and return the best move.

    Args:
        position: The position (board) to search from.
        heuristic: The heuristic to use to evaluate the position.
        noise_enabled: If true, enable noise, else disable it.
        seed: The seed for the RNG in the heuristic.

    Returns:
        The best move from the given position as evaluated by the heuristic.
    """
    if seed:
        heuristic.seed_generator(seed)
    else:
        heuristic.seed_generator(random.randint(0, 2**64))
    heuristic.set_noise_enabled(noise_enabled)
    bfs = NInARowBestFirstSearch(heuristic, position)
    bfs.complete_search()
    return bfs.get_tree()


def bads_parameters_to_model_parameters(params):
    """
    Expands a truncated set of BADS parameters into a full set of heuristic params
    for constructing a heuristic.

    Args:
        params: The BADS parameters to convert (of length 10)
              [pruning, stopping_prob, feature_drop, lapse, c_opp, w_center, 2IAR_CON, 2IAR_DIS, 3IAR, 4IAR]

    Returns:
        The corresponding heuristic parameters (of length 58)
    """
    if (len(params) != 10):
        raise Exception(
            "Parameter file must contain 10 parameters: {}".format(params))
    params = list(map(float, params))
    out = [10000.0, params[0], params[1], params[3], 1, 1, params[5]]
    # Feature weights: params[6:] = [2IAR_CON, 2IAR_DIS, 3IAR, 4IAR]
    out.extend([x for x in params[6:]] * 4)
    out.append(0)
    out.extend([x * params[4] for x in params[6:]] * 4)
    out.append(0)
    out.extend([params[2]] * 17)
    return out
