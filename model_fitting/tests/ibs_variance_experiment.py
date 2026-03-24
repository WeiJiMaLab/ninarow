"""
IBS variance experiment (manual / exploratory; not run by pytest).

Run: ``python ibs_variance_experiment.py``

For each n_repeats value, we call evaluate n_calls times and collect the sum of NLLs.
We then analyze how the variance of these sums changes as n_repeats increases.
"""
import sys
from pathlib import Path
import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from tree_search import TreeSearch, SingleThreadedFitter


def load_data(data_folder, n_trials=5):
    """Load test data from CSV files."""
    split_files = sorted(glob.glob(f"{data_folder}/split_*.csv"))
    if not split_files:
        raise ValueError(f"No split files found in {data_folder}")
    data = pd.read_csv(split_files[0])
    return data[:n_trials]


def find_data_folder():
    """Find a data folder with split files."""
    # Try different paths in order, starting with a different one
    possible_paths = [
        "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling/2023-03-13",
        "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling/2024-02-14",
        "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling/2023-02-20",
    ]
    
    for path in possible_paths:
        if glob.glob(f"{path}/split_*.csv"):
            return path
    
    # Try to find any folder with split files
    base_path = "/scratch/hl3976/monkey_4iar/analysis/data/processed/harry/modeling"
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and glob.glob(f"{item_path}/split_*.csv"):
                return item_path
    
    return None


def run_ibs_variance(data_folder=None, n_trials=5, n_calls=30, n_repeats_values=[5, 10, 20]):
    """
    Test how variance changes with different n_repeats values.
    
    For each n_repeats:
    - Call evaluate n_calls times
    - Collect the sum of NLLs from each call
    - Report mean and standard deviation of these sums
    """
    print("=" * 80)
    print("Testing IBS Variance vs n_repeats")
    print("=" * 80)
    
    # Load data
    if data_folder is None:
        data_folder = find_data_folder()
    
    if data_folder is None:
        raise ValueError("Could not find data folder. Please specify data_folder parameter.")
    
    data = load_data(data_folder, n_trials)
    data["expected_counts"] = 1
    print(f"Loaded {len(data)} trials from {data_folder}")
    
    # Setup model and fitter
    model = TreeSearch()
    fitter = SingleThreadedFitter(model, verbose=False)
    params = model.initial_params.copy()
    
    print(f"\nParameters: {len(params)} parameters")
    print(f"Number of calls per n_repeats: {n_calls}")
    print(f"Testing n_repeats values: {n_repeats_values}")
    
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"{'n_repeats':>10} | {'Mean NLL Sum':>15} | {'SD of NLL Sums':>18} | {'CV (%)':>10}")
    print("-" * 60)
    
    results = {}
    
    for n_repeats in n_repeats_values:
        # Call evaluate n_calls times and collect sums
        nll_sums = []
        
        for _ in range(n_calls):
            eval_results = fitter.evaluate(params, data.copy(), repeats=n_repeats)
            nll_sum = eval_results.sum()
            nll_sums.append(nll_sum)
        
        nll_sums = np.array(nll_sums)
        
        # Compute statistics
        mean_sum = nll_sums.mean()
        sd_sum = nll_sums.std()
        cv = (sd_sum / mean_sum * 100) if mean_sum != 0 else 0
        
        results[n_repeats] = {
            'mean_sum': mean_sum,
            'sd_sum': sd_sum,
            'cv': cv,
            'nll_sums': nll_sums
        }
        
        print(f"{n_repeats:>10} | {mean_sum:>15.4f} | {sd_sum:>18.4f} | {cv:>10.2f}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)
    
    sds = [results[n]['sd_sum'] for n in n_repeats_values]
    print(f"\nStandard deviations: {[f'{sd:.4f}' for sd in sds]}")
    
    # Check if SD decreases as n_repeats increases
    decreasing = all(sds[i] >= sds[i+1] * 0.95 for i in range(len(sds)-1))
    
    if decreasing:
        print("✅ Standard deviation decreases as n_repeats increases")
    else:
        print("⚠️  Standard deviation does not consistently decrease")
        print("   This may indicate an issue with the IBS implementation")
    
    # Detailed comparison
    print("\nDetailed comparison:")
    for i in range(len(n_repeats_values) - 1):
        n1, n2 = n_repeats_values[i], n_repeats_values[i+1]
        sd1, sd2 = results[n1]['sd_sum'], results[n2]['sd_sum']
        reduction = ((sd1 - sd2) / sd1 * 100) if sd1 > 0 else 0
        print(f"  {n1} → {n2} repeats: SD {sd1:.4f} → {sd2:.4f} ({reduction:+.1f}% change)")
    
    # Expected relationship: SD should decrease roughly as 1/sqrt(n_repeats)
    print("\nExpected relationship (SD ∝ 1/√n_repeats):")
    baseline_sd = results[n_repeats_values[0]]['sd_sum']
    print(f"{'n_repeats':>10} | {'Expected SD':>15} | {'Actual SD':>15} | {'Ratio':>10}")
    print("-" * 55)
    for n_repeats in n_repeats_values:
        expected_sd = baseline_sd * np.sqrt(n_repeats_values[0] / n_repeats)
        actual_sd = results[n_repeats]['sd_sum']
        ratio = actual_sd / expected_sd if expected_sd > 0 else 0
        print(f"{n_repeats:>10} | {expected_sd:>15.4f} | {actual_sd:>15.4f} | {ratio:>10.2f}")
    
    return results


def estimate_repeats_for_target_sd(data_folder=None, target_sd=2.0, n_calls=20, n_repeats_test=5):
    """
    Estimate how many repeats are needed to keep SD under target_sd.
    
    Runs n_repeats_test on the full dataset, then estimates required repeats
    using the relationship SD ∝ 1/√n_repeats.
    """
    print("\n" + "=" * 80)
    print("Estimating Required Repeats for Target SD")
    print("=" * 80)
    
    # Load full dataset
    if data_folder is None:
        data_folder = find_data_folder()
    
    if data_folder is None:
        raise ValueError("Could not find data folder. Please specify data_folder parameter.")
    
    split_files = sorted(glob.glob(f"{data_folder}/split_*.csv"))
    if not split_files:
        raise ValueError(f"No split files found in {data_folder}")
    
    # Load all data from first split (full dataset)
    data = pd.read_csv(split_files[0])
    data["expected_counts"] = 1
    print(f"Loaded full dataset: {len(data)} trials from {data_folder}")
    
    # Setup model and fitter
    model = TreeSearch()
    fitter = SingleThreadedFitter(model, verbose=False)
    params = model.initial_params.copy()
    
    print(f"\nTesting with n_repeats={n_repeats_test} on full dataset ({n_calls} calls)...")
    
    # Run evaluation n_calls times with progress bar
    nll_sums = []
    for _ in tqdm(range(n_calls), desc="Running evaluations"):
        eval_results = fitter.evaluate(params, data.copy(), repeats=n_repeats_test)
        nll_sum = eval_results.sum()
        nll_sums.append(nll_sum)
    
    nll_sums = np.array(nll_sums)
    actual_sd = nll_sums.std()
    mean_sum = nll_sums.mean()
    
    print(f"\nResults with n_repeats={n_repeats_test}:")
    print(f"  Mean NLL Sum: {mean_sum:.4f}")
    print(f"  SD of NLL Sums: {actual_sd:.4f}")
    
    # Estimate required repeats using SD ∝ 1/√n_repeats
    # If SD_n = SD_5 * √(5/n), then for SD_n < target_sd:
    # SD_5 * √(5/n) < target_sd
    # √(5/n) < target_sd / SD_5
    # 5/n < (target_sd / SD_5)^2
    # n > 5 * (SD_5 / target_sd)^2
    
    if actual_sd <= target_sd:
        print(f"\n✅ Current SD ({actual_sd:.4f}) is already below target ({target_sd:.2f})")
        print(f"   No increase in repeats needed!")
        estimated_repeats = n_repeats_test
    else:
        estimated_repeats = int(np.ceil(n_repeats_test * (actual_sd / target_sd) ** 2))
        print(f"\n📊 Estimation:")
        print(f"   Current SD: {actual_sd:.4f}")
        print(f"   Target SD: {target_sd:.2f}")
        print(f"   Estimated n_repeats needed: {estimated_repeats}")
        print(f"   (Using relationship: SD ∝ 1/√n_repeats)")
        
        # Verify the estimate
        expected_sd = actual_sd * np.sqrt(n_repeats_test / estimated_repeats)
        print(f"\n   Expected SD with n_repeats={estimated_repeats}: {expected_sd:.4f}")
    
    return {
        'n_repeats_test': n_repeats_test,
        'actual_sd': actual_sd,
        'mean_sum': mean_sum,
        'target_sd': target_sd,
        'estimated_repeats': estimated_repeats,
        'nll_sums': nll_sums
    }


if __name__ == "__main__":
    # Run the variance test
    results = run_ibs_variance()
    
    # Estimate required repeats for full dataset
    estimate_results = estimate_repeats_for_target_sd(target_sd=2.0)
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
