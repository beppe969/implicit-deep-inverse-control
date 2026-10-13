# Local validation report — 13 October 2026

## Scope and environment

This report records validation of the focused repair of the supplied submission
package. The original three trained tensor files are unchanged. All evaluations
were regenerated from those checkpoints; training was not rerun.

The environment was Linux x86-64 CPU, Python 3.13.5, PyTorch 2.10.0+cpu,
NumPy 2.3.5, and Matplotlib 3.10.8. Exact environment metadata is saved in
`checkpoints/environment.json`. Numerical tolerances and timing interpretation
are described in `../NUMERICAL_DETAILS.md`.

## Completed checks

| Check | Result | Evidence |
|---|---|---|
| Release-input SHA-256 verification | Passed, 47 scientific files | `integrity.json` |
| Unit and packaging regressions | 27 tests passed | `unit_tests.txt` |
| Focused stopping and physical-centering tests | 7 tests passed | `focused_tests.txt` |
| Full checkpoint evaluation | All three studies completed | `checkpoints/evaluation.txt` |
| Independent repeat against the new reference outputs | 2475/2475 scientific scalar comparisons passed | `checkpoints/comparison.json` |
| Independent static and NARX audit | Passed | `audit/independent_validation.json` |
| Independent NumPy equation tests | 36 dense cyclic models, 64 queries each | `audit/independent_validation.json` |

The repeated run uses absolute tolerance 3e-07 and relative tolerance
0.0001. Timing fields are treated as diagnostics. The largest reported
scientific scalar difference between the reference regeneration and its repeat
was approximately 7.41e-08.
The archived run manifests verify the scientific inputs before and after each
safe reproduction command.

The independent NumPy tests use tanh, ReLU, and shifted tanh. The maximum
forward–inverse cycle error is 1.11e-15.
The tested componentwise inequalities have nonpositive maximum defects. These
are numerical checks of the equations and implementations, rather than an
exhaustive formal verification of the theorems.

## Main NARX audit

| Quantity | Recomputed value |
|---|---|
| Mean true-output tracking RMSE | 0.00576450641014 |
| Spectral radius rho(H) | 0.670679830429 |
| Spectral radius rho(Q) | 0.521869038739 |
| Final measured-residual box | [0.0644477885908, 0.0597762516049] |
| Output tube | [0.0820207005785, 0.0767669355842] |
| Control-deviation tube | [0.181046027119, 0.197525242023] |
| Held-out trajectories satisfying the residual envelope | 120 / 120 |
| Maximum output-tube ratio after sample 40 | 0.721404328686 |
| Maximum input-tube ratio after sample 40 | 0.146969085719 |
| Maximum all-time regressor ratio to twice the steady-state tube | 0.967742504997 |
| Componentwise input margins for the inflated tube | [1.32195100749, 1.35331281413] |

The largest one-step and transient comparison defects are
7.445e-09. This is a float32 roundoff effect and
is below the audit tolerance of 1e-6. The reported residual-budget checks pass.
Calibration uses 20 trajectories of length 280 and the documented seed 20262857.
The 20/21 statement is the manuscript's finite-horizon marginal coverage result
under exchangeability. The observed 120/120 evaluation result is empirical.

## Independence and comparison repairs

The focused tests verify that freezing completed queries removes the stopping
rule's dependence on the other queries in the batch. Scalar tests compare
individual, permuted, and chunked execution exactly. Tests on the fitted
multivariate model allow floating-point kernel differences with batch shape.
A separate test holds the physical uncertainty center fixed while changing the
certificate reference and checks equality of the resulting physical trajectories.
The recentering allowance is checked against its defining componentwise bound.

The recorded before/after comparison is in `before_after.json`; the original
numerical reference is retained in `baseline_results.json` for provenance.
The repaired ablations use one physical plant for every compared model and add
the required reference-center allowance to their conditional certificates.

## Publication status and limits

No GitHub push or remote GitHub Actions run was performed. The workflow commands
were exercised locally. Full retraining and training-seed variability remain
outside this repair. The three checkpoint files are preserved byte-for-byte.
The main probabilistic certificate retains its stated matched-law assumptions;
variant tubes have conditional preliminary-envelope validity and the servo study
remains empirical.
