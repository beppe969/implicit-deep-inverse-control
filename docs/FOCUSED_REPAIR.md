# Focused repair of the submission package

## Scope

This release repairs the supplied `paper_package_submission.zip`. The title and
the main theorems are retained. The existing valid gain
`L_u = S_chi + S_y |F| P_y` is retained. The optional joint-sensitivity refinement
is outside this minimum-change repair. All three trained checkpoint files are
unchanged; every scientific evaluation has been rerun from those weights.

## A common physical plant across NARX variants

The physical perturbation is centered at one fixed pair `(chi_c, u_c)`, taken
from the main model's reference. Every controller and ablation receives the same
uncertainty and noise draws and uses that same physical center. A variant's own
model-consistent reference `(chi_r, u_r)` is kept separately for its certificate.
The simulation routine now accepts the explicit keyword arguments
`plant_chi_ref` and `plant_u_ref` for this purpose.

Re-expressing the mismatch envelope about a variant's reference adds the offset

\[
 a_i = \max_{0\le k<T}
 \left[E_\chi|\chi_k^r-\chi_k^c|+E_u|u_k^r-u_k^c|\right]_i.
\]

The variant's conditional constant envelope is its preliminary residual box plus
the bias bound and `a`. The main-model offset is zero. These values are saved as
`narx/variants/<name>/uncertainty_center_allowance`. The optional training sweep
uses the same centering convention. Variant tubes continue to depend on a valid
preliminary envelope; they carry no new closed-loop coverage claim.

## Per-query stopping

Each fixed-point map evaluation tests the residual at the current hidden vector.
That vector is frozen as soon as its own residual is within tolerance. The
returned input is reconstructed from the same tested vector. Queries waiting
for other members of the batch receive no further hidden updates. Diagnostics
include both the actual returned residual and a convergence flag. Deployment
raises on a failed inverse residual budget. Calls without diagnostics also raise
on a cap hit.

The recorded iteration count is the number of fixed-point map evaluations needed
by the query, including the evaluation certifying the returned vector. The batch
implementation can still evaluate masked rows, so these counts should be kept
separate from elapsed hardware time. Float32 kernel rounding can differ with
batch shape. The regressions test exact stopping independence in scalar arithmetic
and numerical agreement for the fitted multivariate model.

Matched closed-loop calibration has been regenerated with the documented
independent calibration seed 20262857, after fixing the numerical controller.
The 20/21 claim remains marginal and finite-horizon (280 samples), under the
exchangeability assumptions in the manuscript.

## Numerical effect

The main-model tracking results and its certificate retain their displayed
precision. The recalculated input-tube radius of the cyclic-forward variant is
0.1549803888; it is displayed as **0.15** in Table II. The main-minus-control-affine
paired mean difference is approximately -2.52866e-5, with 95% interval
[-9.44848e-5, 4.41252e-5]. The lower endpoint displayed in the manuscript is
therefore **-9.4e-5**. These are the two changed numerical displays.

All other entries of the four tables retain their displayed values. The largest
absolute change in an ablation's mean tracking RMSE is about 3.45e-7. The changes
do not alter the reported comparison conclusions. Full values and offsets are
in `validation/before_after.json`.

The two static Newton table quantities previously inferred from equality of roots
are now evaluated directly on the common clean/noisy targets. Their displayed
values agree with the existing table.

## Manuscript corrections

The blue-marked manuscript corrects the constant-forcing limit statement in
Corollary 1 and narrows the higher-relative-degree and physical-state wording.
Figure 1 uses `v_k`. Norm rescaling is described as differentiable almost
everywhere. The introduction states that forward temporal contraction is optional
for the closed-loop theorem. The numerical section documents the common centers,
per-query stopping, and the conditioning of bootstrap intervals on fitted models.
The Figure 5 discussion is restricted to the componentwise certificate shown.
Regenerated experimental artwork is identified by blue captions; the plot styles
are retained. The paper remains 15 pages, with the same document class and body
font size. A clean counterpart is generated from the same source by disabling
revision colors.

## Reproduction and publication

Run the commands in the root README. `validation/REPORT.md` records the actual
local tests, independent audit, and full checkpoint rerun. This repair does not
include full retraining. No GitHub push or remote workflow run has been performed.
The accompanying `github_update.patch` targets the repository snapshot embedded
in the supplied archive; check it against a clean clone before applying it.

CSV exports use LF line endings so Git normalization preserves release hashes.
