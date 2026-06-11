#!/usr/bin/env python3
import sys
import os

# Add the parent directory to sys.path to allow importing from model_fitting
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fourbynine import *
from tree_search import TreeSearch, feature_list_from_templates
from ninarow_utilities import bads_parameters_to_model_parameters

def create_test_board(black_bits, white_bits):
    """Creates a legal board state with given bit patterns."""
    return fourbynine_board(fourbynine_pattern(black_bits), fourbynine_pattern(white_bits))

def test_opp_scale_intuition():
    """Verify that opp_scale correctly weights the opponent's features."""
    print("\n--- Testing opp_scale Intuition ---")
    
    test_cases = [
        {
            "name": "4-in-a-row (4IAR)",
            "template": [[1, 1, 1, 1]],
            "white_bits": 0xF, # 1111 at 0,1,2,3
            "weight": 1.0,
            "opp_scale": 0.5,
            "expected": -0.5
        },
        {
            "name": "Disconnected 3IAR (1011)",
            "template": [[1, 0, 1, 1]],
            "white_bits": 0xD, # 1101 (bits 0, 2, 3)
            "weight": 4.0,
            "opp_scale": 0.25,
            "expected": -1.0
        },
        {
            "name": "Disconnected 3IAR (1101)",
            "template": [[1, 1, 0, 1]],
            "white_bits": 0xB, # 1011 (bits 0, 1, 3)
            "weight": 0.2,
            "opp_scale": 5.0,
            "expected": -1.0
        },
        {
            "name": "Disconnected 2IAR (1001)",
            "template": [[1, 0, 0, 1]],
            "white_bits": 0x9, # 1001 (bits 0, 3)
            "weight": 2.0,
            "opp_scale": 1.0,
            "expected": -2.0
        }
    ]
    
    all_pass = True
    for case in test_cases:
        # Player 1 (Black) needs at least as many pieces as Player 2 (White) for a legal state.
        num_white = bin(case["white_bits"]).count('1')
        black_bits = ((1 << num_white) - 1) << 16 # Place dummy pieces far away
        
        board = create_test_board(black_bits, case["white_bits"])
        ts = TreeSearch(feature_list=feature_list_from_templates({"target": case["template"]}))
        
        # Params: [pruning, stopping, feature_drop, lapse, opp_scale, center_weight]
        control_vec = [100.0, 1.0, 0.0, 0.0, case["opp_scale"], 0.0]
        feature_vec = [case["weight"]]
        
        heuristic = ts.create_heuristic(control_vec, feature_vec)
        heuristic.set_noise_enabled(False)
        
        val = heuristic.evaluate(board)
        match = abs(val - case["expected"]) < 1e-10
        
        status = "✅ PASS" if match else "❌ FAIL"
        print(f"{status}: {case['name']}")
        print(f"      Expected: {case['expected']}, Actual: {val}")
        if not match:
            all_pass = False
            
    return all_pass

def test_self_feature_weight():
    """Verify that self features are weighted at the base weight (not scaled by opp_scale)."""
    print("\n--- Testing Self Feature Weight ---")
    
    # Player 1 (Black) has 4IAR, Player 2 (White) has 4 dummy pieces
    board = create_test_board(0xF, 0xF0000)
    ts = TreeSearch(feature_list=feature_list_from_templates({"4IAR": [[1, 1, 1, 1]]}))
    
    # opp_scale=0.1 should NOT affect Player 1's features
    control_vec = [100.0, 1.0, 0.0, 0.0, 0.1, 0.0]
    feature_vec = [1.0]
    
    heuristic = ts.create_heuristic(control_vec, feature_vec)
    heuristic.set_noise_enabled(False)
    
    val = heuristic.evaluate(board)
    expected = 1.0
    
    match = abs(val - expected) < 1e-10
    status = "✅ PASS" if match else "❌ FAIL"
    print(f"{status}: Self 4IAR (opp_scale=0.1)")
    print(f"      Expected: {expected}, Actual: {val}")
    
    return match

def test_initial_values():
    """Verify that the initial_values parameter correctly sets starting weights."""
    print("\n--- Testing initial_values Parameter ---")
    
    templates = {"A": [[1,1,1]], "B": [[1,1,1]]}
    initial_values = {"A": 1.5, "B": 2.5}
    
    ts = TreeSearch(feature_list=feature_list_from_templates(templates, initial_values=initial_values))
    
    # Retrieve initial values from the parameter list
    val_a = next(p["initial_value"] for p in ts.parameter_list if p["name"] == "A")
    val_b = next(p["initial_value"] for p in ts.parameter_list if p["name"] == "B")
    
    match = (val_a == 1.5 and val_b == 2.5)
    status = "✅ PASS" if match else "❌ FAIL"
    print(f"{status}: initial_values dictionary")
    print(f"      A: {val_a}, B: {val_b}")
    
    return match

def test_legacy_equivalence():
    """Verify that modular creation matches the legacy bads_parameters approach."""
    print("\n--- Testing Legacy Equivalence ---")
    
    # 4-group templates in alphabetical order to match sorted_groups
    templates = {
        "1IAR": [[1]],
        "2IAR": [[1, 1]],
        "3IAR": [[1, 1, 1]],
        "4IAR": [[1, 1, 1, 1]]
    }
    
    ts = TreeSearch(feature_list=feature_list_from_templates(templates))
    
    # Legacy params: [pruning, stopping, feature_drop, lapse, opp_scale, center, 1iar, 2iar, 3iar, 4iar]
    params = [1.0, 1.0, 0.0, 0.0, 0.5, 0.0, 10.0, 20.0, 30.0, 40.0]
    
    # Create board: Player 2 has 4IAR at start, Player 1 has 4 dummy pieces
    board = create_test_board(0xF0000, 0xF)
    
    # Legacy evaluation
    full_params = bads_parameters_to_model_parameters(params)
    legacy_heuristic = fourbynine_heuristic.create(DoubleVector(full_params), False)
    legacy_heuristic.set_noise_enabled(False)
    legacy_val = legacy_heuristic.evaluate(board)
    
    # Modular evaluation
    ts.set_params(params)
    ts.heuristic.set_noise_enabled(False)
    modular_val = ts.heuristic.evaluate(board)
    
    match = abs(legacy_val - modular_val) < 1e-10
    status = "✅ PASS" if match else "❌ FAIL"
    print(f"{status}: Legacy vs Modular evaluation")
    print(f"      Legacy: {legacy_val}, Modular: {modular_val}")
    
    return match

def main():
    print("=" * 60)
    print("HEURISTIC FEATURE TESTS")
    print("=" * 60)
    
    results = [
        test_opp_scale_intuition(),
        test_self_feature_weight(),
        test_initial_values()
    ]
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)

if __name__ == "__main__":
    main()
