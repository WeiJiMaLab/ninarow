import sys
from pathlib import Path
import numpy as np
from pybads import BADS

# Target quadratic function with known minimum at [1.0, -2.0, 0.5]
TRUE_MIN = np.array([1.0, -2.0, 0.5])

def quadratic_target(x):
    diff = x - TRUE_MIN
    return np.sum(diff ** 2)

def test_checkpoint_resume_equivalence():
    print("\n--- Running BADS Checkpoint/Resume Sanity Smoke Test ---")
    
    # Define bounds matching TreeSearch style
    D = len(TRUE_MIN)
    lb = np.full(D, -5.0)
    ub = np.full(D, 5.0)
    plb = np.full(D, -4.0)
    pub = np.full(D, 4.0)
    x0 = np.zeros(D)
    
    orig_tol_mesh = 1e-4
    
    # ----------------------------------------------------
    # RUN 1: Fresh Complete Run
    # ----------------------------------------------------
    print("\n[Run 1] Running fresh optimization to full convergence...")
    options_fresh = {
        "uncertainty_handling": False,
        "tol_mesh": orig_tol_mesh,
        "display": "off"
    }
    bads_fresh = BADS(quadratic_target, x0, lb, ub, plb, pub, options=options_fresh)
    res_fresh = bads_fresh.optimize()
    x_fresh = res_fresh["x"]
    mesh_fresh = res_fresh["mesh_size"]
    print(f"  Converged x: {x_fresh}")
    print(f"  Termination mesh size (relative): {mesh_fresh}")
    
    # ----------------------------------------------------
    # RUN 2: Interrupted Run (simulate interrupt at 25 evals)
    # ----------------------------------------------------
    print("\n[Run 2] Running interrupted optimization (max 25 evals)...")
    options_int = {
        "uncertainty_handling": False,
        "max_fun_evals": 25,
        "display": "off"
    }
    
    # Keep track of evaluations and state
    state = {
        "best_x": None,
        "best_fval": float("inf"),
        "mesh_size": None,
        "poll_iter": 0
    }
    
    def wrapped_target(x):
        fval = quadratic_target(x)
        if fval < state["best_fval"]:
            state["best_fval"] = fval
            state["best_x"] = np.copy(x)
            state["mesh_size"] = bads_int.optim_state.get("mesh_size", 1.0)
            state["poll_iter"] = bads_int.optim_state.get("iter", 0)
        return fval
        
    bads_int = BADS(wrapped_target, x0, lb, ub, plb, pub, options=options_int)
    res_int = bads_int.optimize()
    
    checkpoint_x = np.copy(state["best_x"])
    checkpoint_mesh = state["mesh_size"]
    checkpoint_poll = state["poll_iter"]
    
    print(f"  Interrupted at best x: {checkpoint_x}")
    print(f"  Relative mesh size at interrupt: {checkpoint_mesh}")
    print(f"  Poll iteration at interrupt: {checkpoint_poll}")
    
    # ----------------------------------------------------
    # RUN 3: Resumed Run
    # ----------------------------------------------------
    print("\n[Run 3] Resuming from checkpoint with adaptive bounds and tolerances...")
    
    # 1. Compute narrowing factor (capped between 0.1 and 1.0)
    narrowing_factor = np.clip(checkpoint_mesh, 0.1, 1.0)
    
    # 2. Get constant-width shifted bounds centered on checkpoint
    orig_width = pub - plb
    target_width = narrowing_factor * orig_width
    
    res_plb = checkpoint_x - 0.5 * target_width
    res_pub = checkpoint_x + 0.5 * target_width
    
    # Shift window as a rigid body if it overflows hard bounds
    low_overflow = lb - res_plb
    shift_up = np.maximum(0, low_overflow)
    res_plb += shift_up
    res_pub += shift_up
    
    high_overflow = res_pub - ub
    shift_down = np.maximum(0, high_overflow)
    res_plb -= shift_down
    res_pub -= shift_down
    
    # Clip for safety
    res_plb = np.clip(res_plb, lb, ub)
    res_pub = np.clip(res_pub, lb, ub)
    
    # Apply relative tol_mesh scaling
    res_tol_mesh = orig_tol_mesh / narrowing_factor
    
    options_res = {
        "uncertainty_handling": False,
        "tol_mesh": res_tol_mesh,
        "display": "off"
    }
    
    print(f"  Resumed plausible bounds: plb={res_plb}, pub={res_pub}")
    print(f"  Adaptive tol_mesh: {res_tol_mesh}")
    
    bads_res = BADS(quadratic_target, checkpoint_x, lb, ub, res_plb, res_pub, options=options_res)
    res_res = bads_res.optimize()
    x_res = res_res["x"]
    mesh_res = res_res["mesh_size"]
    
    print(f"  Converged x: {x_res}")
    print(f"  Termination mesh size (relative): {mesh_res}")
    
    # ----------------------------------------------------
    # VERIFICATION
    # ----------------------------------------------------
    param_diff = np.linalg.norm(x_res - x_fresh)
    
    # Physical step size at termination of fresh run: mesh_fresh * gamma_orig
    # Physical step size at termination of resumed run: mesh_res * gamma_resumed
    # Since gamma_resumed = narrowing_factor * gamma_orig:
    physical_mesh_fresh = mesh_fresh * (0.5 * (pub[0] - plb[0]))
    physical_mesh_res = mesh_res * (0.5 * (res_pub[0] - res_plb[0]))
    mesh_diff = abs(physical_mesh_fresh - physical_mesh_res)
    
    print("\n--- Results Summary ---")
    print(f"  Parameter distance (Resumed vs Fresh): {param_diff:.6e}")
    print(f"  Physical mesh size (Fresh): {physical_mesh_fresh:.6e}")
    print(f"  Physical mesh size (Resumed): {physical_mesh_res:.6e}")
    print(f"  Physical mesh difference: {mesh_diff:.6e}")
    
    assert param_diff < 1e-4, f"Parameters do not converge to the same point! Diff: {param_diff}"
    assert mesh_diff < 1e-5, f"Physical mesh resolutions differ! Diff: {mesh_diff}"
    print("\n✅ SANITY CHECK PASSED! Convergence and physical resolutions are identical!")

if __name__ == "__main__":
    test_checkpoint_resume_equivalence()
