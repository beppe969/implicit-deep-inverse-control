# Implicit Deep Learning Models for Robust Nonlinear Inverse Control

Reproducibility package for the paper **Implicit Deep Learning Models for Robust
Nonlinear Inverse Control**, by Giuseppe C. Calafiore and Laurent El Ghaoui.

## Focused repair — 13 October 2026

The three supplied trained checkpoints are preserved byte-for-byte. All evaluation
results and the six experimental figure pairs have been regenerated. The NARX
physical uncertainty center is fixed across every controller and model variant.
The inverse solver freezes each query at its own certified residual crossing.
Variant certificates include the allowance for their different reference centers.

The mathematical certificate and all training settings are retained. See
[the repair record](docs/FOCUSED_REPAIR.md) and [validation report](validation/REPORT.md).

## Install

Validated on Linux x86-64 CPU with Python 3.13.5, PyTorch 2.10.0+cpu,
NumPy 2.3.5, and Matplotlib 3.10.8.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

On macOS, install the supported PyTorch build through `pip install -r requirements.txt`
in the virtual environment. The published regression report is for Linux CPU;
platform-dependent floating-point differences and timing changes can occur.

## Reproduce

Run from the repository root:

```bash
python reproduce.py check
python reproduce.py test
python reproduce.py checkpoints
python reproduce.py audit
```

`check` verifies release hashes. `test` runs the unit and algebraic regressions.
`checkpoints` rebuilds data and evaluates every model from its saved parameters,
including the final matched closed-loop calibration. It exports all four tables
and regenerates the six experimental figures. `audit` independently recomputes
the main certificate and all variant metrics, and tests the algebraic inequalities
with a separate NumPy implementation.

Every numerical command creates a new workspace under `runs/`. A new directory
can be selected with `--output`. Existing output directories are refused; released
results and checkpoint files are left unchanged.

```bash
python reproduce.py checkpoints --experiment narx --output runs/my-narx-check
python reproduce.py tables --output runs/my-tables
python reproduce.py train --output runs/my-training-run
```

The training route is supplied for full retraining. Training was not rerun for
this repair. Finite-precision retraining can yield different fitted parameters;
the checkpoint route is the numerical regression target.

## Contents

`code/` contains the models and experiments. `results/` contains full-precision
reference metrics and the three tensor checkpoints. `figures/` contains PDF and
PNG figure files. `tables/` contains the four manuscript tables in CSV, Markdown,
and LaTeX, together with portable JSON. `validation/` contains the actual release
checks. The full numerical protocol is in [NUMERICAL_DETAILS.md](NUMERICAL_DETAILS.md).

The GitHub package contains the numerical materials. The blue-marked manuscript
and its clean counterpart are distributed in the separate paper package.

## Interpretation

The 120 NARX realizations and 80 servo realizations are paired across controllers.
There is one training seed per model configuration. Bootstrap intervals quantify
plant/noise variation conditional on the fitted models. Only the main NARX model
has fresh matched closed-loop calibration. Its 20/21 coverage claim is marginal
and limited to a new 280-sample trajectory from that fixed closed-loop law.
Variant tubes remain conditional on their preliminary residual envelopes. The
servo is an empirical state-informed example.

GitHub Actions definitions are included. They have not been run on GitHub for this
release; their numerical commands have been tested locally. No remote repository
has been modified. See [LICENSE_STATUS.md](LICENSE_STATUS.md) before reuse.
