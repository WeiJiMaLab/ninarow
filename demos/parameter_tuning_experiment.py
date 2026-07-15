#!/usr/bin/env python3
"""
Parameter tuning experiment for TreeSearch performance using real board states.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).parent
model_fitting_dir = current_dir.parent / "model_fitting"
data_dir = current_dir.parent / "data"
sys.path.insert(0, str(model_fitting_dir))

from tree_search import TreeSearch
import fourbynine

class ComprehensiveParameterTuner:
    
    def __init__(self):
        print("Loading test data from data/1.csv...")
        self.test_data = self._load_test_data()
        self.test_boards_and_moves = self._create_test_boards()
        print(f"Created {len(self.test_boards_and_moves)} test board positions")
        
    def _load_test_data(self):
        csv_file = data_dir / "1.csv"
        if csv_file.exists():
            return pd.read_csv(csv_file)
        else:
            print("Warning: data/1.csv not found, using example data")
            return pd.read_csv(model_fitting_dir / "example_inputs" / "example_moves.csv")
    
    def _create_test_boards(self, n_positions=20):
        """Create board positions from the CSV data using the proper fourbynine API"""
        boards_and_moves = []
        
        for _, row in self.test_data.head(n_positions).iterrows():
            try:
                # Create board from black/white integers using the proper API
                black_pattern = fourbynine.fourbynine_pattern(int(row['black']))
                white_pattern = fourbynine.fourbynine_pattern(int(row['white']))
                board = fourbynine.fourbynine_board(black_pattern, white_pattern)
                
                # Convert move to board position (bit position)
                move_int = int(row['move'])
                actual_move = move_int.bit_length() - 1  # Convert to 0-based position
                
                boards_and_moves.append((board, actual_move))
                    
            except Exception as e:
                print(f"Warning: Could not create board from row {len(boards_and_moves)}: {e}")
                continue
                
        print(f"Successfully created {len(boards_and_moves)} board positions")
        return boards_and_moves
    
    def benchmark_configuration(self, params_array, n_trials=15):
        """Benchmark a specific parameter configuration"""
        model = TreeSearch()
        
        # Set the parameters (params_array is a numpy array)
        model.set_params(params_array)
        
        # Warmup
        if self.test_boards_and_moves:
            try:
                board, _ = self.test_boards_and_moves[0]
                model.predict(board)
            except:
                pass
        
        # Benchmark
        times = []
        correct_predictions = 0
        total_predictions = 0
        
        for i, (board, actual_move) in enumerate(self.test_boards_and_moves[:n_trials]):
            try:
                start_time = time.time()
                predicted_move = model.predict(board)
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Check if prediction matches actual move
                if predicted_move == actual_move:
                    correct_predictions += 1
                total_predictions += 1
                
            except Exception as e:
                print(f"Error with board {i}: {e}")
                continue
        
        if not times:
            return None
            
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'accuracy_pct': (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0,
            'n_trials': len(times)
        }
    
    def run_parameter_sweep(self):
        """Run comprehensive parameter sweep"""
        
        # Get baseline parameters
        baseline_model = TreeSearch()
        baseline_params = baseline_model.initial_params.copy()
        
        print("\\nBaseline parameters (numpy array):")
        print(f"  Shape: {baseline_params.shape}")
        print(f"  Values: {baseline_params}")
        
        # Parameter names from TreeSearch parameter_list
        param_names = [param["name"] for param in baseline_model.parameter_list]
        print(f"  Parameter names: {param_names}")
        
        # Define parameter variations to test
        test_configs = []
        
        # Test individual parameter variations
        for param_idx, param_info in enumerate(baseline_model.parameter_list):
            param_name = param_info["name"]
            lower_bound = param_info["lower_bound"]
            upper_bound = param_info["upper_bound"]
            initial = param_info["initial_value"]
            
            # Test 3 values: lower bound, 2x initial, upper bound
            test_values = [
                lower_bound,
                min(initial * 2, upper_bound) if initial > 0 else upper_bound * 0.5,
                upper_bound
            ]
            
            for value in test_values:
                if abs(value - initial) > 0.001:  # Skip if too close to baseline
                    params = baseline_params.copy()
                    params[param_idx] = value
                    test_configs.append((param_name.lower().replace(' ', '_'), value, params))
        
        # Add some specific high-impact configurations based on previous results
        high_impact_configs = [
            # Stopping-probability variations
            ("stopping_prob", 0.1, baseline_params.copy()),
            ("stopping_prob", 0.5, baseline_params.copy()),
            ("stopping_prob", 1.0, baseline_params.copy()),
            # Pruning threshold variations  
            ("pruning_threshold", 0.1, baseline_params.copy()),
            ("pruning_threshold", 0.5, baseline_params.copy()),
            ("pruning_threshold", 5.0, baseline_params.copy()),
        ]
        
        for param_name, value, params in high_impact_configs:
            # Find the parameter index
            for idx, param_info in enumerate(baseline_model.parameter_list):
                if param_name.replace('_', ' ').lower() in param_info["name"].lower():
                    params[idx] = value
                    test_configs.append((param_name, value, params))
                    break
        
        # Run baseline first
        print("\\n" + "="*60)
        print("RUNNING BASELINE BENCHMARK")
        print("="*60)
        
        baseline_result = self.benchmark_configuration(baseline_params)
        if baseline_result:
            print(f"Baseline: {baseline_result['mean_time_ms']:.2f}ms ± {baseline_result['std_time_ms']:.2f}ms")
            print(f"Accuracy: {baseline_result['accuracy_pct']:.1f}%")
            baseline_time = baseline_result['mean_time_ms']
        else:
            print("Baseline benchmark failed!")
            return
        
        # Run all test configurations
        results = []
        
        print("\\n" + "="*60)
        print("RUNNING PARAMETER SWEEP")
        print("="*60)
        
        for param_name, param_value, params in test_configs:
            print(f"\\nTesting {param_name} = {param_value}...")
            
            result = self.benchmark_configuration(params, n_trials=min(10, len(self.test_boards_and_moves)))
            if result:
                speedup = baseline_time / result['mean_time_ms']
                print(f"  Time: {result['mean_time_ms']:.2f}ms (±{result['std_time_ms']:.2f}) | "
                      f"Speedup: {speedup:.1f}x | Accuracy: {result['accuracy_pct']:.1f}%")
                
                results.append({
                    'parameter': param_name,
                    'value': param_value,
                    'mean_time_ms': result['mean_time_ms'],
                    'std_time_ms': result['std_time_ms'],
                    'speedup': speedup,
                    'accuracy_pct': result['accuracy_pct']
                })
            else:
                print(f"  FAILED")
        
        # Analyze results
        self._analyze_results(results, baseline_time)
    
    def _analyze_results(self, results, baseline_time):
        """Analyze and summarize the results"""
        if not results:
            print("No results to analyze!")
            return
            
        df = pd.DataFrame(results)
        
        print("\\n" + "="*80)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        # Top speedups
        print("\\n🚀 TOP 5 SPEEDUPS:")
        top_speed = df.nlargest(5, 'speedup')
        for _, row in top_speed.iterrows():
            print(f"  {row['parameter']} = {row['value']}: "
                  f"{row['speedup']:.1f}x speedup ({row['mean_time_ms']:.2f}ms) | "
                  f"Accuracy: {row['accuracy_pct']:.1f}%")
        
        # Best accuracy
        print("\\n🎯 TOP 5 ACCURACY:")
        top_accuracy = df.nlargest(5, 'accuracy_pct')
        for _, row in top_accuracy.iterrows():
            print(f"  {row['parameter']} = {row['value']}: "
                  f"{row['accuracy_pct']:.1f}% accuracy | "
                  f"Speedup: {row['speedup']:.1f}x ({row['mean_time_ms']:.2f}ms)")
        
        # Best balance (speedup * accuracy)
        df['balance_score'] = df['speedup'] * df['accuracy_pct'] / 100
        print("\\n⚖️ BEST SPEED/ACCURACY BALANCE:")
        top_balance = df.nlargest(5, 'balance_score')
        for _, row in top_balance.iterrows():
            print(f"  {row['parameter']} = {row['value']}: "
                  f"Score: {row['balance_score']:.2f} | "
                  f"{row['speedup']:.1f}x speedup, {row['accuracy_pct']:.1f}% accuracy")
        
        print("\\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        
        if len(top_balance) > 0:
            best_overall = top_balance.iloc[0]
            print(f"\\n🏆 RECOMMENDED CONFIGURATION:")
            print(f"   Parameter: {best_overall['parameter']} = {best_overall['value']}")
            print(f"   Expected speedup: {best_overall['speedup']:.1f}x")
            print(f"   Expected accuracy: {best_overall['accuracy_pct']:.1f}%")
            print(f"   Time per prediction: {best_overall['mean_time_ms']:.2f}ms")
            
            if best_overall['speedup'] > 2.0:
                print(f"\\n✅ This represents a SIGNIFICANT performance improvement!")
            elif best_overall['speedup'] > 1.5:
                print(f"\\n✅ This represents a meaningful performance improvement.")
            else:
                print(f"\\n⚠️ Performance gains are modest.")

def main():
    print("TreeSearch Comprehensive Parameter Tuning Experiment")
    print("====================================================\\n")
    
    tuner = ComprehensiveParameterTuner()
    tuner.run_parameter_sweep()

if __name__ == "__main__":
    main()