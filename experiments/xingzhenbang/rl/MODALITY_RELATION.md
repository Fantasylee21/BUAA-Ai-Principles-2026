# Frozen Omni v4 modality-relation experiment

`omni_modality_relation_v1` keeps all Omni and original-v4 LoRA parameters frozen.
Only a new FP32 head is optimized. The binary control and joint nine-state head
share original-v4 vision, second-modality, and full-context hidden vectors.
The complete fixed settings are in `configs/rl/modality_relation_v1.json`.

The relation target is `M1.label != M2.label`. Clear views constrain their own
emotion; unclear views have their exact emotion constraint relaxed while the
original Conflict/Aligned target remains mandatory. The weighted allowed-state
negative log likelihood uses 0.6/0.4 class-conditional means. The binary control
uses the same class weights. Final conflict score > 0.5 means yes. There is no
rollout sampling, F1-delta reward, old-policy regression, or reference-decision KL.

Pipeline: strict identity and real VT/VA cache checks -> frozen features ->
three grouped folds, two methods, 40 epochs -> OOF epoch selection -> both full
seed42 fits -> higher actual validation score -> output checks -> conditional
seed43 -> conditional test features and one CSV. Full-data fits use the selected
epoch but keep the original 40-epoch learning-rate horizon. Same-video VT/VA
rows never cross a development fold. These folds isolate the head, not v4's
earlier SFT exposure. Official validation labels never enter training losses.

Existing perception caches are read-only. New hidden features have independent
sample/split/model/template/source identities, SHA256 and atomic completion
markers. Head checkpoints are algorithm-specific and include parameters, AdamW,
scheduler, RNG, train-ID digest and epoch-boundary data position. Interrupted
epochs replay from the last published complete epoch; duplicated partial-epoch
log records are diagnostic and not extra optimizer updates in the restored run.

Launch from an immutable deployed release using `scripts/start_relation_tmux.sh`.
It creates `buaa-relation`; default window 0 displays loss, score, coverage,
parameters, errors and explicit quality-stop reasons. Windows 2/3 follow assigned
GPU workers, including both feature extraction and subsequent head training.
Every worker selects its GPU before CUDA initialization; the allocator ceiling
is 32 GiB. Other users' processes are never stopped. There is no wall-clock limit.

Only seed42 gain >= .015 and seed43 gain >= .005, unchanged emotions and valid
questions authorize test processing/export. No new CSV or Kaggle upload occurs
on failed quality gates. Fixed-three-class and union Macro-F1 remain separate;
frozen-emotion score differences do not depend on the missing-neutral convention.
The confirmed online baseline is .90813; .92 is verified only by an actual
user-submitted Kaggle score, never inferred from a local metric.

`python -m unittest tests.test_modality_relation -v` exercises grouping, partial
labels, class weights, gradients, cache/checkpoint corruption, batch consistency,
OOF selection and exact next-update recovery. Torch checks require torch/numpy;
real-model preflight runs on the allocated server card before feature extraction.
