# Mapping the manuscript to reproducible outputs

## Tables

The four table exports are generated from `results/all_results.json` by
`scripts/export_tables.py`. They are also regenerated automatically after each
checkpoint or training run.

| Manuscript item | JSON source | Export |
| --- | --- | --- |
| Table I: static inversion | `static` | `tables/table_I_static.*` |
| Table II: NARX models | `narx/variants` and inverse baselines | `tables/table_II_narx_models.*` |
| Table III: NARX tracking | `narx/monte_carlo` | `tables/table_III_narx_tracking.*` |
| Table IV: servo tracking | `servo/monte_carlo` | `tables/table_IV_servo.*` |

RMSE values in Table II are multiplied by 1,000, as in the manuscript. The other
tables use unscaled values. A missing certificate is displayed as `--`.
`tables/metrics.csv` retains full-precision quantities and individual Monte Carlo
samples. The LaTeX files contain reusable `tabular` environments requiring
`booktabs`; column widths can be adapted to the target manuscript layout.

## Figures

The numbering below follows the current focused-repair manuscript.

| Manuscript item | File stem in `figures/` | Generator |
| --- | --- | --- |
| Figure 1: architecture | `io_architecture` | Extracted `io_architecture.tex` |
| Figure 2: static inversion | `static_inverse_results` | `run_static()` |
| Figure 3: NARX trajectory | `narx_tracking` | `run_narx()` |
| Figure 4: NARX Monte Carlo | `narx_monte_carlo` | `run_narx()` |
| Figure 5: certificate tradeoff | `certificate_tradeoff` | `run_narx()` |
| Figure 6: servo trajectory | `servo_tracking` | `run_servo()` |
| Figure 7: servo Monte Carlo | `servo_monte_carlo` | `run_servo()` |

PDF and PNG versions of all seven figures are supplied. The six experimental pairs are regenerated from the repaired evaluations. Figure 1 is additionally supplied
as a standalone extraction from the current manuscript. Rebuild it in a
separate directory using `pdflatex io_architecture.tex`, with the `standalone` and
TikZ packages installed.

## Checkpoint schema

| Checkpoint | Contents |
| --- | --- |
| `static_models.pt` | `model`: forward implicit predictor; `inverse`: direct-inverse baseline |
| `narx_models.pt` | Main forward model; direct inverse; six entries named `variant:<name>` |
| `servo_models.pt` | Forward implicit model; control-affine model; direct inverse |

These files contain tensor state dictionaries. Architectural metadata is supplied
by the source and numerical supplement. The main NARX checkpoint is duplicated
under `model` and `variant:main`; a unit test verifies their equality.

## Certificate fields

`narx/H` is the augmented nonnegative history-comparison matrix.
`narx/rho_H` is its spectral radius. `output_tube` and `input_deviation_tube` are
the main-model steady-state radii. `conformal` describes the preliminary open-loop
calibration; `closed_loop_conformal` describes the fresh fixed-law calibration.
The selected calibration envelope is incorporated in `d_bar_total`.

`narx/variants/<name>` includes open-loop metrics and the corresponding closed-loop
results. The main-model variant uses fresh matched calibration; other variants
use preliminary envelopes plus the saved `uncertainty_center_allowance`. A spectral radius above one gives unavailable tube
radii in the reference results.

## Evaluation statistics

The NARX experiment uses 120 paired uncertain-plant/noise realizations per
controller. The servo uses 80. Per-run RMSE samples are saved under `monte_carlo`;
NARX maximum-error samples are also retained. The paired bootstrap comparisons
use 20,000 resamples.

NARX tracking metrics use the true plant output. The comparison theorem and tube
checks use measured outputs. The reported steady-state tube ratios discard 40
initial samples. The independent audit adds full-transient and enlarged
invariant-box checks.

Timing ratios and cyclic-forward Newton iteration statistics remain
implementation-dependent diagnostics. See `NUMERICAL_DETAILS.md` before
interpreting them as computational-complexity comparisons.
