# Local reproducibility validation

All scientific inputs originate from the current source package. The SHA-256 checks confirm that the original numerical scripts and archived results agree with `revision_v3.zip`. The added runner uses isolated workspaces and preserves those references.

## Executed checks

| Route | Outcome | Local elapsed time |
| --- | --- | ---: |
| Full checkpoint regeneration | Passed 2,446/2,446 scientific comparisons | 26.4 s |
| Independent equation and NARX audit | Passed | 15.9 s |
| Full training and evaluation | Completed with recorded reference differences | 101.4 s |
| Unit/equation test suite | 20 tests passed | See unit_tests.txt |
| Final static-checkpoint smoke test | Passed | Recorded locally during packaging |

Elapsed times are observations on the preparation machine. Other workloads ran during part of the verification, so these values serve as rough run-duration guidance. Installation time is excluded.

## Checkpoint reproduction

The complete saved-weight run regenerated all three experiments and all six experimental figure pairs. All 2,446 scientific values passed the tolerance `3e-7 + 1e-4 * abs(reference)`. There were no missing expected scientific fields. The 91 timing and stopping/statistical diagnostic fields were recorded separately.

All four manuscript tables were exported in CSV, Markdown, and LaTeX. The PDF figures were rendered and inspected. The original architecture diagram was separately compiled from its extracted TikZ source.

| Quantity | Paper reference | Recomputed from checkpoints | Retrained model |
| --- | ---: | ---: | ---: |
| Static algebraic inverse RMSE | 0.02310915828 | 0.02310915828 | 0.02311336954 |
| NARX main rho(H) | 0.6706798304 | 0.6706798304 | 0.6705987176 |
| NARX main mean tracking RMSE | 0.005764506432 | 0.005764506432 | 0.005767365693 |
| Servo implicit mean position RMSE | 0.04365106998 | 0.04365106998 | 0.04365106998 |

Detailed checkpoint output is in `checkpoints/results.json`; field-level comparisons are in `checkpoints/comparison.json`.

## Full from-scratch training

The final training run executed the original optimization routines for every model. It completed the complete evaluation and regenerated every experimental figure. Under the strict checkpoint comparison threshold, 1,134 of 2,446 numeric fields differed from the archived reference. All differences are retained in `training/comparison.json`.

The most relevant comparisons across the six NARX variants are:

| Variant | Reference rho(H) | Retrained rho(H) | Relative change in forward RMSE | Relative change in CL RMSE |
| --- | ---: | ---: | ---: | ---: |
| main | 0.670680 | 0.670599 | +0.033% | +0.050% |
| control-affine (B_u=0) | 2.160935 | 2.160978 | -0.951% | -0.166% |
| control-affine (B_u=0), lambda_s=3e-3 | 0.638117 | 0.642169 | +0.529% | +0.930% |
| lambda_s=0 | 0.979100 | 0.979063 | +0.001% | -0.000% |
| lambda_s=1e-3 | 0.635035 | 0.634938 | +0.081% | +0.027% |
| implicit forward (A!=0) | 0.650671 | 0.651988 | +1.758% | -0.052% |

These are fresh finite-precision fits. The original training environment was not supplied. The package therefore uses saved-weight evaluation as the accurate table-reproduction route and preserves newly trained results as separate evidence. The runner records successful training with reference differences as `completed_with_reference_differences`; this is separate from passing the checkpoint regression gate.

## Independent numerical audit

The separate NumPy implementation tested 36 dense cyclic models with 64 queries per model. It used tanh and ReLU, together with tanh shifted by 0.25. The maximum forward/inverse cycle error was 1.110e-15. All tested componentwise inequality defects were nonpositive.

The NARX audit reconstructed the fresh calibration and obtained empirical envelope coverage of 1.000 over the 120 evaluation trajectories. It checked the complete measured-history comparison and the enlarged invariant box. Full quantities and containment margins are in `audit/independent_validation.json`.

## Environment and scope

Python 3.13.5; PyTorch 2.10.0+cpu; NumPy 2.3.5; Matplotlib 3.10.8. The computation used Linux x86_64 and one PyTorch thread. Full version records are supplied in the environment JSON files.

The workflow YAML and citation YAML were parsed locally. Citation field names and the unpublished-reference type were checked against the official CFF schema documentation. GitHub-hosted workflow execution was not performed. Fresh installation from the network was also not tested; the numerical validation used the installed pinned versions.

The software archive contains no manuscript review or blue revision files. The code/data license and the actual repository URL remain release decisions for the authors.
