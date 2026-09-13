# Numerical tables

Values are rounded as in the manuscript. See metrics.csv for full-precision scalar results.

## Table I. Static MIMO inversion

| Metric | Algebraic | Newton | Direct |
| --- | --- | --- | --- |
| Inverse RMSE | 0.0231 | 0.0231 | 0.0198 |
| True-map cycle RMSE | 0.0230 | 0.0230 | 0.0196 |
| Noisy-target inverse RMSE | 0.0508 | 0.0508 | 0.0492 |
| Hidden-layer evaluations (mean) | 7.19 | 3.98 | -- |
| Iterations (mean / max) | 7.19 / 9 | 2.98 / 4 | -- |

Newton true-map cycle and noisy-target cells follow the manuscript's same-root convention: the archived JSON stores these two metrics for the algebraic inverse only. The independently stored Newton input-recovery RMSE verifies the clean-target root comparison. Iteration counts can vary at the float32 stopping threshold.

## Table II. NARX models and certificates

| Model | Forward RMSE x 1e3 | Inverse RMSE x 1e3 | Inverse contraction | rho(H) | Max. input-tube radius | CL RMSE x 1e3 |
| --- | --- | --- | --- | --- | --- | --- |
| Main (lambda_s = 1e-4) | 4.43 | 5.40 | 0.64 | 0.671 | 0.20 | 5.76 |
| lambda_s = 0 | 3.40 | 4.14 | 0.61 | 0.979 | 4.48 | 5.39 |
| lambda_s = 1e-3 | 5.72 | 6.94 | 0.63 | 0.635 | 0.16 | 5.99 |
| B_u = 0, lambda_s = 1e-4 | 5.99 | 7.27 | 0.00 | 2.161 | -- | 5.79 |
| B_u = 0, lambda_s = 3e-3 | 8.88 | 10.75 | 0.00 | 0.638 | 0.19 | 8.13 |
| A != 0, lambda_s = 1e-4 | 4.34 | 5.26 | 0.33 | 0.651 | 0.15 | 5.78 |
| Direct inverse | -- | 11.20 | -- | -- | -- | 5.75 |
| Linear ARX inverse | -- | 10.95 | -- | -- | -- | 8.45 |

Only the main model uses fresh matched closed-loop calibration. Other tube values are conditional calculations from preliminary open-loop envelopes. An unavailable tube is shown as --.

## Table III. Tracking over 120 uncertain NARX plants

| Controller | Mean RMSE | 95th-percentile RMSE | Mean max. error | 95th-percentile max. error | Hidden evaluations |
| --- | --- | --- | --- | --- | --- |
| implicit inverse | 0.00576 | 0.00649 | 0.0213 | 0.0270 | 4.1 |
| online Newton | 0.00576 | 0.00649 | 0.0213 | 0.0270 | 2.9 |
| direct inverse | 0.00575 | 0.00773 | 0.0368 | 0.0637 | -- |
| linear ARX inverse | 0.00845 | 0.00928 | 0.0283 | 0.0333 | -- |

Measured feedback is noisy; tracking statistics use noise-free plant outputs. The theorem's output tube applies to measured output errors.

## Table IV. Servo tracking over 80 uncertain plants

| Controller | Mean RMSE | 95th-percentile RMSE | Mean max. error |
| --- | --- | --- | --- |
| implicit inverse | 0.0437 | 0.0700 | 0.0920 |
| control-affine inverse | 0.0441 | 0.0705 | 0.0929 |
| direct inverse | 0.0672 | 0.0992 | 0.1241 |
| nominal computed torque | 0.0495 | 0.0797 | 0.1007 |

Position errors are in radians. Samples are aligned at (k+1)h. This state-informed example is an empirical tracking study.
