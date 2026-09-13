## Table I. Static MIMO inversion

| Metric | Algebraic | Newton | Direct |
| --- | --- | --- | --- |
| Inverse RMSE | 0.0231 | 0.0231 | 0.0198 |
| True-map cycle RMSE | 0.0230 | 0.0230 | 0.0196 |
| Noisy-target inverse RMSE | 0.0508 | 0.0508 | 0.0492 |
| Hidden-layer evaluations (mean) | 7.19 | 3.98 | -- |
| Iterations (mean / max) | 7.19 / 9 | 2.98 / 4 | -- |

Newton true-map cycle and noisy-target cells follow the manuscript's same-root convention: the archived JSON stores these two metrics for the algebraic inverse only. The independently stored Newton input-recovery RMSE verifies the clean-target root comparison. Iteration counts can vary at the float32 stopping threshold.
