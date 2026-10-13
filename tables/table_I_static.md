## Table I. Static MIMO inversion

| Metric | Algebraic | Newton | Direct |
| --- | --- | --- | --- |
| Inverse RMSE | 0.0231 | 0.0231 | 0.0198 |
| True-map cycle RMSE | 0.0230 | 0.0230 | 0.0196 |
| Noisy-target inverse RMSE | 0.0508 | 0.0508 | 0.0492 |
| Hidden-layer evaluations (mean) | 7.19 | 3.98 | -- |
| Iterations (mean / max) | 7.19 / 9 | 2.98 / 4 | -- |

All three methods are evaluated explicitly on the same clean and noisy targets. Completed algebraic-inverse queries are frozen at their first residual crossing. Iteration counts can vary at the float32 stopping threshold.
