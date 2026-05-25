# BADS tolerance smoke test (logistic proxy)

Quick reference for `logistic_smoke.py`. This does **not** fit TreeSearch or IBS; it only checks that PyBADS spatial/stall tolerances behave sensibly on a smooth logistic NLL.

## Commands

```bash
cd model_fitting
python -m logistic_smoke              # default: one tiny train/test split (~1–2 min)
pytest tests/test_bads_logistic_tolerance.py -v   # one optional sanity test
python -m logistic_smoke --benchmark  # slow; not a smoke test
```

## Production fitter (TreeSearch / monkey_4iar)

From `TreeSearchRunner.DEFAULT_BADS_OPTIONS`:

| Option | Value | Role |
|--------|-------|------|
| `tol_mesh` | `1e-3` | Primary stop: parameter-space mesh size |
| `tol_fun` | `1e-7` | Stall on objective improvement (keep tight so IBS noise floor does not stop first) |
| `fun_eval_start` | `20` | Sobol init evals |
| `max_fun_evals` | `2000` | Budget cap |
| `uncertainty_handling` | `True` | Sto-BADS for IBS |

## Tiny-data smoke results (n=150, 2-fold CV, BADS max_fun_evals=100)

| Setting | Δ test NLL vs ref | param L2 vs ref |
|---------|-------------------|-----------------|
| BADS `tol_mesh=1e-3`, `tol_fun=1e-7` | −0.002 | 0.049 |
| BADS `tol_mesh=1e-4`, `tol_fun=1e-7` | +0.059 | 0.146 |
| BADS `tol_mesh=1e-2`, `tol_fun=1e-7` | −0.015 | 0.097 |
| scipy `ftol=gtol=1e-6` | −0.001 | 0.001 |

**Takeaway:** Production `tol_mesh` / `tol_fun` align with scipy `1e-6` on smooth logistic NLL. There is no exact analytic map (mesh stop ≠ gradient stop). For a later fair logistic baseline, prefer **scipy** `ftol=gtol=1e-6` over sklearn on this toy setup.

## Agents

- Smoke tests must finish in ~1–2 minutes. Do not run full CV grids, `--benchmark`, or chained pytest in agent loops.
- `SMOKE_MAX_FUN_EVALS=60` in code is wiring-only; production uses 2000.
