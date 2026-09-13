# Numerical implementation and reproducibility details

This document accompanies *Implicit Deep Learning Models for Robust Nonlinear
Inverse Control* by Giuseppe C. Calafiore and Laurent El Ghaoui. Experiment
definitions and the paper's trained checkpoints originate from `revision_v3.zip`.
The scientific scripts and archived figure files are preserved byte-for-byte.

## 1. Reproduction routes and scope

Run commands from the repository root after installing `requirements.txt`.
Python 3.13.5 with PyTorch 2.10.0+cpu was used for local verification. The other
direct dependencies are NumPy 2.3.5 and Matplotlib 3.10.8.

```bash
# Verify the frozen scientific files, without numerical dependencies.
python reproduce.py check

# Regenerate evaluation outputs and all six experimental figures from weights.
python reproduce.py checkpoints

# Independent equation tests and full NARX certificate audit.
python reproduce.py audit

# Train every model from scratch, then regenerate evaluation outputs.
python reproduce.py train
```

Each numerical command creates a new directory under `runs/`. A named new
workspace can be selected with `--output`. Existing output directories are
refused. The root `results/` and `figures/` remain the paper's reference outputs.

The checkpoint route replaces optimizer calls with strict checkpoint loading.
It replays the same random minibatch draws to preserve subsequent random targets.
Every data generator and evaluation routine runs again. The training route uses
the original optimization routines. `--experiment static`, `narx`, or `servo`
selects one study. Full-precision comparisons and timing diagnostics are saved
alongside the output. Detailed local validation is in `validation/REPORT.md`.

Retraining can change the fitted weights in finite precision. The checkpoint
route is the regression target for reproducing the numerical tables to the
paper's reported accuracy. The separate training comparison documents any
changes, while retaining the original reference results. The original training
environment was not included with the supplied checkpoints.

The original entry points under `code/` remain available for inspection.
`code/experiments.py` and `code/review_checks.py` write to root-level results and
figures when invoked directly, so use the safe commands above for ordinary
reproduction. `code/sweep_lambda.py` is an optional preliminary model-selection
utility. Fresh matched calibration is required after selecting a controller.

## 2. Architectures and optimization

All supplied learned forward maps use componentwise tanh. Unless specified otherwise, their hidden matrix is A = 0, so the forward model has a single hidden layer with a linear skip. The algebraically transformed inverse generally has cyclic hidden connections. The NARX cyclic-forward variant has a trainable A. This experimental selection illustrates inversion and certification within a model framework that also admits arbitrary-depth feedforward realizations.

| Study | Forward hidden width | Direct-inverse hidden widths | Inverse contraction budget |
|---|---:|---:|---:|
| Static MIMO | 40 | 64, 64 | 0.78 |
| NARX | 48 | 96, 96 | 0.68 |
| Servo | 36 | 56, 56 | 0.72 |

The direct-inverse model standardizes its inputs and targets using training statistics. Its `depth=3` implementation comprises two hidden tanh layers and a linear output layer. The current input enters the implicit hidden features through B_u. Setting B_u = 0 gives the constant-input-gain control-affine ablation.

A fitted linear map supplies the fixed explicit skip matrices in the static and NARX studies. In the servo, nominal physics supplies the known drift and the scalar explicit gain D_u = 0.1375. The output matrix C is smoothly rescaled to enforce the specified inverse-contraction budget. The cyclic-forward variant also rescales A to satisfy the forward budget 0.35. Its training uses 20 unrolled hidden iterations; evaluation uses a convergence test.

| Study and model | Adam steps | Batch size | Learning rate | Weight decay |
|---|---:|---:|---:|---:|
| Static forward | 1500 | 1024 | 0.0027 | 0.000002 |
| Static direct inverse | 1600 | 1024 | 0.0025 | 0.000002 |
| NARX forward variants | 1000 | 1536 | 0.0022 | 0.000002 |
| NARX direct inverse | 1000 | 1536 | 0.0020 | 0.000002 |
| Servo forward / control-affine | 1450 | 1024 | 0.0027 | 0.000003 |
| Servo direct inverse | 1550 | 1024 | 0.0025 | 0.000002 |

Every 40 optimization steps, checkpoint selection evaluates training mean-squared error on a fixed subset of at most 3000 training examples. Both forward and direct-inverse models use this rule. There is one training seed per reported configuration. Thus the Monte Carlo intervals characterize uncertainty across plant/noise realizations conditional on the fitted models; they do not quantify variability across independent training runs.

The NARX main model uses sensitivity weight 0.0001. The variants use weights 0 and 0.001, a B_u = 0 model at the main weight, a B_u = 0 model at weight 0.003, and a cyclic-forward model at the main weight. The penalty is the infinity norm of the componentwise inverse-history sensitivity bound. All reported certificates use global activation slopes Lambda = I. The local-slope calculations in the code are diagnostic and are not needed for the reported certificates.

## 3. Data, seeds, and comparisons

The base seed is 20260831. The static study uses offset +1, NARX identification uses +2, preliminary open-loop calibration uses +3, and the servo uses +4. The NARX complete-trajectory split uses +7. Closed-loop evaluation uses +20; independent final closed-loop calibration uses +2026, namely 20262857. Other auxiliary seeds, including bootstrap seeds, are explicit at their call sites in `experiments.py`.

### Static MIMO

The study draws 7000 inputs uniformly from [-2, 2]^2, with 5500 used for training and 1500 for testing. Training outputs contain additive Gaussian noise of standard deviation 0.09. Clean targets are used for the principal held-out inverse tests. A second test adds target noise with standard deviation 0.05. Both inverse estimators receive the same targets.

The principal comparison separates recovery of the generating input from consistency with the fitted forward map. The algebraic and Newton inverses solve the fitted model; the direct inverse is trained independently. In this example the direct inverse has the smaller input-recovery RMSE. The algebraic inverse's learned-model cycle error is close to numerical precision. Neither property implies uniform statistical superiority over the other approach.

### Nonlinear NARX

The identification set consists of 180 independent trajectories of length 110. The split assigns 150 complete trajectories to training and 30 to testing. The regressor stores three two-dimensional output samples and two two-dimensional past input samples. Measurement noise has standard deviation 0.012. Appendix D gives the nominal matrices and nonlinear terms. The excitation construction is implemented in `generate_narx_data`.

The closed-loop reference has 280 steps and the target error matrix is F = 0.1 I. Each controller sees measured outputs; the nominal plant evolves with its true history. The synthetic additive perturbation terms use deviations of the measured regressor and input from the model reference. Their componentwise ranges are given in Appendix D. Process noise is uniform with componentwise bound 0.003, and the constant bias is sampled within componentwise bound 0.004. Inputs are clipped to [-2.2, 2.2]. The main-model certificate leaves a positive margin to this constraint, and saturation was absent in the reported main-model runs.

The four principal controllers use identical uncertainty and noise draws across 120 runs. Tracking RMSE is computed from the true output. The theorem and tube diagnostics use the measured output. Reported steady-state tube ratios discard the first 40 samples; the separate audit also checks the complete transient comparison and the all-time enlarged invariant box. Additional perturbation scales 1.5 and 2 are empirical stress tests outside the originally specified uncertainty family.

### Physics-guided servo

The physical state is position and velocity, whereas the inverted channel maps one voltage input to the next velocity. The sample time is 0.02 seconds. Of 8000 independently generated samples, 6400 are used for training. The ranges are position [-pi, pi], velocity [-5, 5], and voltage [-2.7, 2.7]. Training velocity targets have Gaussian noise of standard deviation 0.018.

Closed-loop evaluation uses 80 uncertain plants over 700 steps. Inertia and motor gain vary by up to 10%; viscous and Coulomb friction vary by up to 20%. Cogging amplitude varies by up to 25%. The simulation includes a sinusoidal load, process noise, and state-measurement noise with standard deviation 0.004. The same draws are reused for the four controllers. The outer computed-torque target uses position and velocity gains 18 and 6.5. Voltage is clipped to [-3.2, 3.2]. Tracking compares the state at time (k+1)h with the reference at that same time, including the final sample. No saturation occurred in the reported runs.

This state-informed example demonstrates the dimensional distinction between the physical state and the inverted channel. Its tracking results are empirical. Applying the measured-history theorem to a state-informed regressor requires an appropriate state-error propagation model.

## 4. Inverse solves and Newton diagnostics

Stand-alone inverse tests initialize the hidden iterate at zero and use tolerance 1e-7, with a cap of 100 iterations. Closed-loop inverse solves use tolerance 2e-7 and warm starts. Reference generation uses tolerance 1e-8. The inverse routine explicitly returns the fixed-point residual at the reconstructed solution; the residual-to-control bounds use that residual.

The solver records each query's first tolerance crossing. Batched execution continues updating the hidden array until every query is finished. Thus mean iteration counts describe individual stopping counts; they are not a direct measurement of the actual hardware workload of the entire batched implementation.

Newton evaluates the learned forward map and its exact Jacobian, with a cap of 25 outer iterations. The static test uses residual/stall tolerances 5e-7; the closed-loop test uses residual tolerance 2e-7 and default stall tolerance 1e-7. Newton and algebraic inversion reach essentially identical roots. For the explicit A = 0 forward models, Newton can require fewer hidden-layer evaluations, although each step additionally solves a Jacobian system.

For cyclic forward models, a Newton step also requires an equilibrium solve and its Jacobian. The forward hidden solver uses tolerance 1e-8 in float32 with a cap of 200 iterations. Near the floating-point floor, this stopping rule may reach the cap. Consequently, large cyclic-model Newton timings in the saved raw results are sensitive to implementation choices. They support no architecture-independent speed advantage. Changing precision or stopping tolerances requires rerunning the timing comparison.

## 5. Calibration and meaning of the certificate

Twenty open-loop trajectories of length 280 provide a preliminary residual box for model selection. These preliminary envelopes are conditional numerical ingredients when used in closed-loop tube calculations for the ablation models.

After fixing the main model, controller, and reference, 20 fresh closed-loop trajectories are generated with seed 20262857. Per-coordinate scales are determined from training data. Each calibration score is the maximum normalized absolute measured prediction residual over the entire trajectory. The largest of the 20 scores defines the fresh box; its componentwise maximum with the preliminary box is retained. The controller itself does not use this box, so this enlargement leaves the calibrated trajectory law unchanged.

Under exchangeability with a new trajectory generated by the same fixed closed-loop law, the resulting marginal coverage is at least 20/21 for the 280-step horizon. This probability includes the randomness in the calibration sample. It is not a conditional guarantee for every realized calibration data set. It also supplies no automatic guarantee for a changed controller or an infinite sequence of times. A deterministic regional guarantee requires an envelope valid uniformly over the specified region.

Fresh calibration bounds the full measured prediction residual. The additional nonnegative mismatch terms retained in the comparison theorem make the resulting envelope conservative. All 120 evaluation trajectories satisfied the final residual envelope in the reported experiment. Only the main model uses fresh matched closed-loop calibration. The other Table II tube values do not carry that coverage claim.

## 6. Statistics and saved outputs

RMSE averages across time and output coordinates. Maximum vector-output errors use the Euclidean norm. Bootstrap comparisons resample paired differences across the same simulated plant/noise draws, with 20000 resamples and a 95% percentile interval. The saved JSON files retain the individual run metrics needed to recompute these summaries.

`results/all_results.json` and `results/summary.csv` are the original numerical reference outputs. For downstream parsing, use `tables/results_portable.json` and `tables/metrics.csv`; these normalize unavailable values and quote CSV fields containing commas. The `conformal` fields record preliminary open-loop calibration; `closed_loop_conformal` records final main-model calibration. Trained parameters are in `static_models.pt`, `narx_models.pt`, and `servo_models.pt`. All six experimental figure pairs supplied in the baseline are retained byte-for-byte in `figures/`. The architecture drawing is also provided as a standalone extraction from the current manuscript.

The additional audit checks the central static metrics and all six NARX variants. It also tests the algebraic equations with an independent NumPy implementation on 36 randomly generated dense cyclic models, using 64 queries per model. The activations include tanh, ReLU, and tanh shifted by 0.25. These tests check implementations and expose possible algebraic mistakes; the mathematical guarantees rest on the manuscript's proofs and assumptions.
