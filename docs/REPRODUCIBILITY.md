# Reproduction guide

## Environment

The local validation used Python 3.13.5 on Linux x86_64. Direct dependencies are
pinned in `requirements.txt`: PyTorch 2.10.0, NumPy 2.3.5, and Matplotlib 3.10.8.
The tested PyTorch build is `2.10.0+cpu`. A record of the installed transitive
packages appears in `validation/environment_packages.txt`; it is an observed
environment snapshot rather than a cross-platform lock file.

All experiment tensors are on the CPU. The original script sets one PyTorch
thread, and the safe runner sets single-thread OpenMP/MKL environment variables.
The equations and the synthetic datasets are fully specified in the source
package. After dependency installation, numerical runs need no network access.

The installation instructions use the official PyTorch CPU wheel index on
Linux. Platform-specific installation details for PyTorch 2.10.0 are maintained
in the official previous-versions documentation [1]. macOS and Windows execution
were not tested during preparation of this package. Python's virtual-environment
activation command also differs on Windows.

The core checks use Python's standard `unittest` module. There is no requirement
for pytest, notebooks, or a TeX installation. TeX is needed only when rebuilding
the architecture drawing from `figures/io_architecture.tex`.

## Exact scope of the checkpoint route

`python reproduce.py checkpoints` creates a new workspace and copies the frozen
scientific scripts there. It loads every supplied model state dictionary with
`weights_only=True` and `map_location='cpu'`. Each loading operation is strict:
missing or unexpected tensor keys raise an error.

Optimizer calls are replaced by loading the appropriate state dictionary. The
worker then consumes the original `torch.randint` minibatch draws with the same
batch shapes and number of steps. This matters particularly for the static
study, where noisy evaluation targets are generated after the training calls.
Model construction and subsequent data/evaluation routines run in the same
sequence as the original code.

The full route regenerates the identification splits and held-out predictions.
It computes the NARX certificate for every variant and repeats the paired Monte
Carlo simulations. The cyclic-forward Newton diagnostic is also executed. Fresh
matched main-model calibration is reconstructed from the supplied seed. The
servo is evaluated from its original checkpoints, with the original time
alignment at `(k+1)h`.

All six experimental figure pairs are created by the original plotting code.
The workspace initially contains no experimental figures. The architecture
drawing is a separate extraction of the manuscript's TikZ source and does not
depend on a numerical run.

A complete numerical run creates `results/all_results.json`, with portable JSON
`null` for unavailable certificates. It also creates `results/metrics.csv` and
four table exports. Exact byte identity of newly rendered PDF or PNG files is
outside the numerical regression target: plotting backends and metadata can
change while the underlying curves agree.

## Comparison and exit-status policy

Checkpoint results are compared with the frozen `results/all_results.json`.
Each scientific scalar satisfies

```text
abs(recomputed - reference) <= 3e-7 + 1e-4 * abs(reference).
```

The comparison includes numeric array entries and the saved per-run statistics.
Unavailable expected tube radii compare as unavailable. A missing scientific
field or an unexpected unavailable quantity is reported as a failure.

Timing fields are recorded as diagnostics. Iteration and hidden-evaluation counts
are also diagnostic because tiny floating-point differences can change the first
threshold-crossing iteration. The paired-bootstrap `fraction_negative` statistic
is diagnostic: differences of nearly equal implicit and Newton solutions can
change sign close to numerical precision. The confidence-interval endpoints and
mean paired differences remain regression targets.

A checkpoint comparison failure produces exit status 1 and preserves the
workspace. `comparison.json` contains the field-level differences. The runner
also checks the hashes of the frozen inputs before and after execution.

## From-scratch training

```bash
python reproduce.py train --output runs/from-scratch
```

This command calls the original optimization functions, without checkpoint
restoration. A single-study run is selected with `--experiment`.

Training generates new weights. It also performs the same field-level comparison
with the paper's archived results, retaining every numerical discrepancy. A
successful execution with differences is reported as
`completed_with_reference_differences`, with process exit status 0. A training or
evaluation exception produces a failing exit status.

The local full training run completed. Its main NARX certificate and tracking
results were close to the supplied reference, while some ablation quantities
changed more. These changes are recorded in `validation/training/`. The original
training environment was not provided, so matching seeds alone cannot establish
bitwise equivalence with that earlier training run. PyTorch documents the limits
of reproducibility across platforms and versions [2]. The package retains the
supplied checkpoint route for accurate reconstruction of the paper's tables.

## Audit and unit tests

```bash
python reproduce.py test
python reproduce.py audit --output runs/certificate-audit
```

The 20 unit tests include forward/inverse consistency and componentwise residual
bounds. They also test the NARX certificate and its uncertified ablation. The
packaging tests exercise safe output handling and the numerical comparison
policy.

The longer audit recomputes all NARX variants and compares the main controller's
full measured-history transient with its comparison recursion. It checks the
enlarged invariant box and input containment margin. A separate NumPy
implementation tests the algebraic identities and bounds on 36 random dense
cyclic models, including an activation with a nonzero value at the origin.

These numerical checks validate implementations. The manuscript's formal claims
continue to rely on its stated assumptions and proofs.

## Exporting tables without running experiments

```bash
python reproduce.py tables --output runs/archived-tables
python reproduce.py tables --results runs/from-scratch/results/all_results.json --output runs/retrained-tables
```

This route uses only the Python standard library. Displayed table values use the
paper's rounding. `metrics.csv` contains all scalar numeric outputs at full
precision. CSV fields with commas in variant names are properly quoted.

The original `results/summary.csv` is preserved for provenance. Its historical
writer does not quote some variant names containing commas. The original JSON
also contains Python-style `NaN` for unavailable certificates. Use the portable
exports under `tables/` for general-purpose parsers.

The two Newton cells in Table I for true-map cycle and noisy-target error use the
paper's same-root convention. The archived JSON records these particular metrics
for the algebraic inverse only. The independently stored clean-target Newton
input RMSE is reproduced by the numerical route. The table exporter documents
this convention explicitly.

## Optional sensitivity sweep

The frozen utility `code/sweep_lambda.py` implements preliminary model-selection
runs. For example, in a disposable copy of the repository:

```bash
python code/sweep_lambda.py 0 0.0001 0.001 -0.003
```

A negative argument selects the constant-input-gain control-affine ablation with
the absolute value used as the sensitivity weight. These runs use preliminary
open-loop calibration. A subsequent probability statement for a selected
closed-loop law requires fresh matched calibration.

## GitHub workflows

The regular CI workflow runs on pushes and pull requests. It executes the unit
tests and the static checkpoint smoke test. The manually triggered full workflow
runs every experiment and the independent audit, and offers a choice between
checkpoint evaluation and from-scratch training.

Both workflows use read-only repository permissions. They install the pinned
numerical dependencies and retain their output reports as workflow artifacts.
Their numerical commands were tested locally. Execution on GitHub-hosted runners
and fresh network-based dependency installation were not performed here.

The workflow definitions follow the official action interfaces [3, 4]. The
citation metadata follows GitHub's supported `CITATION.cff` convention [5].

## External technical references

1. PyTorch 2.10.0 installation commands: https://pytorch.org/get-started/previous-versions/#v2100
2. PyTorch reproducibility notes: https://docs.pytorch.org/docs/2.10/notes/randomness.html
3. Official setup-python action: https://github.com/actions/setup-python
4. Official upload-artifact action: https://github.com/actions/upload-artifact
5. GitHub citation-file documentation: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files
