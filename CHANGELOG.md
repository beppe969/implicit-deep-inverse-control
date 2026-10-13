# Change log

## 2026-10-13 — Focused submission repair

Fixed NARX physical uncertainty centers across all controllers and variants.
Added recentering allowances to variant certificate offsets. Froze each inverse
query at its first checked fixed-point residual and exposed cap hits. Regenerated
all evaluations, matched calibration, and experimental figures from unchanged
checkpoints. Evaluated the static Newton noisy-target table cells directly.
Updated table exports and repaired quoted CSV fields. Added stopping and physical
plant regressions. Updated Figure 1's target to `v_k`.

The manuscript has local blue-marked corrections and remains 15 pages. Its title,
theorem structure, and existing componentwise gain are retained. The optional
joint-sensitivity refinement is outside this minimum-change release.
