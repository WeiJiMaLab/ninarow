import numpy as np

def ibs_sample(p):
    """Simulates a single IBS repeat: number of failures until first success."""
    failures = 0
    while True:
        if np.random.random() < p:
            break
        failures += 1
    # NLL estimator for one repeat
    return sum(1.0 / (i + 1) for i in range(failures))

def simulate_dataset(ps, n_repeats):
    """Returns (Total NLL, Predicted Total Variance)."""
    total_nll = 0
    total_var_of_mean = 0
    for p in ps:
        repeats = [ibs_sample(p) for _ in range(n_repeats)]
        total_nll += np.mean(repeats)
        total_var_of_mean += (np.var(repeats, ddof=1) / n_repeats)
    return total_nll, total_var_of_mean

if __name__ == "__main__":
    n_trials = 10
    n_repeats = 50
    n_simulations = 500 
    true_ps = np.random.uniform(0.1, 0.5, n_trials)

    print(f"Experimental Setup: {n_trials} trials, {n_repeats} repeats, {n_simulations} brute-force iterations.")
    
    print("Running Brute Force...")
    brute_force_nlls = [simulate_dataset(true_ps, n_repeats)[0] for _ in range(n_simulations)]
    empirical_sd = np.std(brute_force_nlls)

    print("Running Analytical Estimation (Single Call)...")
    _, predicted_var = simulate_dataset(true_ps, n_repeats)
    predicted_sd = np.sqrt(predicted_var)

    print("\n" + "="*45)
    print(f"Empirical SD (The 'Truth'):      {empirical_sd:.5f}")
    print(f"Analytical SD (Predicted):       {predicted_sd:.5f}")
    print(f"Error:                          {abs(empirical_sd - predicted_sd):.5f}")
    print("="*45)
