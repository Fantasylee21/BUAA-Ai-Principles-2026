# Base-only emotion evidence and cost calibration

`omni_evidence_bottleneck_v1` is supervised evidence fitting, not RL. Omni remains
frozen. The feature loader constructs a plain Qwen2.5-Omni Thinker without loading
PEFT adapters. It isolates silent video, supplied text and audio, and captures the
2048-dimensional final hidden vector plus native positive/negative/neutral logits.
Visual features and clear visual supervision are deduplicated by video.

Three zero-initialized residual linear heads add `W @ LayerNorm(h) + c` to the
native logits. Each independently minimizes mean clear-label cross entropy plus
`0.01/2 * (||W||² + ||c||²)`. All small-model arithmetic is CPU float64. The 18,441
parameters never receive Conflict/Aligned gradients. Unclear views skip this
objective; their original rows still participate in calibration and evaluation.

Disagreement is `q = 1 - dot(p_visual, p_second)`. Two protocol offsets minimize
`0.6 mean_Conflict softplus(-d) + 0.4 mean_Aligned softplus(d) + 0.01/2 ||b||²`,
where `d = logit(q) + b_protocol`. `d > 0` is yes; exact q endpoints stay 0/1.
The score is cost weighted, not claimed to be a calibrated real probability.
There is no full-input feature, old v4 decision, source metadata or gold emotion
prefix in this branch. There is no extra threshold search or OR fallback.

CPU L-BFGS-B has maxiter 300, gtol 1e-5 and ftol 1e-12; unsuccessful or nonfinite
fits fail closed with diagnostics. Successful small fits are atomically published
with identity/checksum markers; incomplete fits restart from zero. A resumed run
cannot load any historical RL optimizer. The algorithm/config/source/base/media/
input/fold identities invalidate incompatible caches. SciPy's successful relative
function-convergence exit is accepted as well as its gradient-convergence exit;
the actual maximum gradient and termination message are retained.

Seed42 development uses three outer video folds. The remaining 2/3 is split into
80% evidence fitting and 20% calibration; outer labels reach reporting only.
Final calibration is fitted to three-fold OOF evidence scores, followed by full
clear-view head fitting and a single fixed validation candidate. Seed43, enabled
only after primary gain >=0.015 and valid outputs, changes the folds and repeats
OOF/calibration/full fits; required gain is >=0.005. This checks partition and
calibration stability, not random neural initialization. Frozen v4 emotions still
come from a model previously trained on the training set; development task-only
scores are distinguished from full-system independent validation.

The original v4 generates questions and supplies fixed emotions only after a
candidate passes its score gate. Both candidates must pass complete output checks
before test feature extraction or a single 734-row CSV. No automatic submission.
Final reports include native/uncalibrated ablations, per-view accuracy, paired
VT/VA diagnostics, detailed changes and descriptive video-group bootstrap ranges.
Existing validation has been repeatedly examined; no new independent test labels
are invented. The confirmed 0.90813 Public baseline is separate from local fixed3
scores; an actual Kaggle submission is required to establish Public >=0.92.

Start the immutable release with `scripts/start_evidence_tmux.sh RUN GPU1 GPU2`.
Session `buaa-evidence`: window0 overview, 1 controller, 2/3 feature logs,
4 CPU fit logs, 5 parameters. Mouse navigation is enabled. Every GPU worker selects
its GPU before CUDA initialization and limits PyTorch allocations to 32 GiB; CPU
workers hide CUDA. No wall-clock cutoff or operations on unrelated processes.
