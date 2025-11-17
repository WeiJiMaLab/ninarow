from fourbynine import *
import numpy as np

try:
    from fourbynine import DoubleVector
except ImportError:
    pass

# Default templates matching Julia structure
DEFAULT_TEMPLATES = {
    "4IAR":      [[1, 1, 1, 1]],
    "3IAR_CON":  [[0, 1, 1, 1], [1, 1, 1, 0]],
    "3IAR_DIS":  [[1, 0, 1, 1], [1, 1, 0, 1]],
    "2IAR_CON":  [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
    "2IAR_DIS":  [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
    "1IAR": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
}

DEFAULT_FEATURE_WEIGHTS = {
    "1IAR": 0.1,
    "2IAR_CON": 1.0,
    "2IAR_DIS": 0.5,
    "3IAR_CON": 4.0,
    "3IAR_DIS": 2.0,
    "4IAR": 9.5,
}

def win_patterns(m=4, n=9, k=4, directions="-/\\|"):
    """
    Generate all possible positions for k-in-a-row patterns.
    Mimics Julia's win_patterns function.
    
    Args:
        m: Number of rows (default 4)
        n: Number of columns (default 9)
        k: Length of pattern (default 4)
        directions: String of directions: '-' horizontal, '|' vertical, 
                   '\\' diagonal, '/' anti-diagonal
    
    Returns:
        List of tuples: (direction, positions_list)
        where positions_list contains lists of board positions (0-indexed)
    """
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
    """
    Create features from a template for all possible positions.
    Mimics Julia's make_feature_from_template function.
    
    Args:
        template: List of 0s and 1s (e.g., [0, 1, 1, 1] for 3IAR)
        m: Number of rows (default 4)
        n: Number of columns (default 9)
        directions: Directions to generate patterns for
    
    Returns:
        List of tuples: (pieces_bitboard, spaces_bitboard, min_empty)
    """
    k = len(template)
    patterns = win_patterns(m, n, k, directions)
    features = []
    
    # Calculate min_empty (number of empty spaces required)
    min_empty = sum(1 - x for x in template)
    
    for direction, positions in patterns:
        # Create bitboards for pieces and spaces
        pieces = 0
        spaces = 0
        
        for i, pos in enumerate(positions):
            # Convert to 0-indexed for bit manipulation
            bit_pos = pos
            
            if template[i] == 1:
                # This position should have a piece
                pieces |= (1 << bit_pos)
            else:
                # This position should be empty
                spaces |= (1 << bit_pos)
        
        features.append((pieces, spaces, min_empty))
    
    return features

def make_features_from_groups(group_templates, m=4, n=9, directions="-/\\|"):
    """
    Create features from a dictionary of group templates.
    Mimics Julia's make_features_from_groups function.
    
    Args:
        group_templates: Dict mapping group names to lists of templates
        m: Number of rows (default 4)
        n: Number of columns (default 9)
        directions: Directions to generate patterns for
    
    Returns:
        Dict mapping group names to lists of (pieces, spaces, min_empty) tuples
    """
    features_by_group = {}
    
    for group_name, templates in sorted(group_templates.items()):
        features_by_group[group_name] = []
        
        for template in templates:
            template_features = make_feature_from_template(template, m, n, directions)
            features_by_group[group_name].extend(template_features)
    
    return features_by_group

def create_feature(pieces, empty, min_empty):
    """
    Create a heuristic feature from bitboards.
    Wrapper around fourbynine_heuristic_feature.
    """
    return fourbynine_heuristic_feature(
        fourbynine_pattern(pieces), 
        fourbynine_pattern(empty), 
        min_empty
    )


def build_control_params(control_params):
    """
    Returns the 7 control parameters in the correct order.
    """
    pruning = control_params["pruning_threshold"]
    stop_prob = control_params["stopping_prob"]
    lapse = control_params["lapse_rate"]
    center_weight = control_params["center_weight"]

    base_control = [
        10000.0,         # fixed stopping threshold
        pruning,
        stop_prob,
        lapse,
        1.0,             # exploration constant placeholder
        1.0,             # opponent scale placeholder
        center_weight
    ]
    return base_control

def create_modular_heuristic(control_params, weights, templates=DEFAULT_TEMPLATES):
    """
    Create a heuristic from scratch using templates to generate features.
    
    This function creates feature groups for each template type and generates
    features from the templates using make_features_from_groups. The modular
    design allows flexible heuristic construction without relying on hardcoded
    C++ features.
    
    Args:
        control_params: Dict with keys: pruning_threshold, stopping_prob, lapse_rate, 
                       center_weight, opp_scale, feature_drop
        weights: Dict mapping group names to weights (e.g., {"2IAR_CON": 1.0, "4IAR": 8.0})
        templates: Dict mapping group names to lists of templates (defaults to DEFAULT_TEMPLATES)
    
    Returns:
        A heuristic created from scratch using templates, with one feature group per template type
    """
    # 1. Create heuristic with control parameters (no features yet)
    control_vec = build_control_params(control_params)
    heuristic = fourbynine_heuristic.create(DoubleVector(control_vec), False)
    
    # 2. Generate features from templates
    features_by_group = make_features_from_groups(templates)
    
    # 3. Create feature groups and add features
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

