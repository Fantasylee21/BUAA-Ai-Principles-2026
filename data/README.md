# Project Data Metadata

This directory contains reproducible dataset metadata and conversion outputs,
not the original course dataset. The released files remain unchanged under
`Dataset/`.

## Training data cleaning

Run the deterministic cleaner from the repository root to reproduce the
`use_in_main=true` variant:

```bash
python data/prepare_training_data.py
```

Run it again while ignoring `use_in_main` to reproduce the all-record variant:

```bash
python data/prepare_training_data.py \
  --ignore-use-in-main \
  --output-dir data/cleaned/all
```

The two retained variants are:

- `cleaned/use_in_main_true/`: requires `use_in_main=true`.
- `cleaned/all/`: ignores `use_in_main` while retaining all other structural
  validation rules.

Each variant contains:

- `train_vl_clean.jsonl`: one effective vision-text record per
  `split_group_id`.
- `train_omni_clean.jsonl`: VT records use vision and text; VA records
  use vision, text, and the audio embedded in the referenced video.
- `train_rejected.jsonl`: rejected sample IDs and rejection reasons.
- `cleaning_report.json`: input/output paths, counts, distributions,
  and cleaning actions.

Custom paths and either filtering policy are supported:

```bash
python data/prepare_training_data.py \
  --input Dataset/train.jsonl \
  --dataset-root Dataset \
  --output-dir /path/to/output
```

Clean the complete official validation split without applying the
training-only `use_in_main` selection flag:

```bash
python data/prepare_training_data.py \
  --input Dataset/val.jsonl \
  --expected-split val \
  --ignore-use-in-main \
  --baseline-aligned-validation \
  --output-dir data/cleaned/val
```

### Current cleaning results

| Variant | Valid source records | Rejected | VL records | Omni records |
| --- | ---: | ---: | ---: | ---: |
| `use_in_main_true` | 3208 | 146 | 2046 | 3208 |
| `all` | 3354 | 0 | 2104 | 3354 |

VL label distributions:

| Variant | Aligned | Conflict | Negative | Positive | Neutral |
| --- | ---: | ---: | ---: | ---: | ---: |
| `use_in_main_true` | 1724 | 322 | 1200 | 839 | 7 |
| `all` | 1726 | 378 | 1222 | 875 | 7 |

Omni label distributions after applying the paired VT/VA conflict union:

| Variant | Aligned | Conflict | Negative | Positive | Neutral |
| --- | ---: | ---: | ---: | ---: | ---: |
| `use_in_main_true` | 2870 | 338 | 1887 | 1314 | 7 |
| `all` | 2913 | 441 | 1944 | 1403 | 7 |

The main-only variant excludes 146 source records solely because
`use_in_main=false`. The all-record variant confirms that all 3354 released
training records pass the structural validation rules. Every cleaned record
retains its original `use_in_main` value for later analysis.

### Common validation

Both outputs require a training split record, a valid VT or VA protocol, valid
M1/M2/M12 labels, non-empty text, and an existing relative video path. The
`use_in_main_true` variant additionally requires `use_in_main=true`, while the
`all` variant ignores that flag. The cleaner aborts if paired VT and VA records
disagree on the M12 emotion label.

`sample_type=Conflict` records are not removed. They are required to teach the
model to output `is_uncertain=yes`. Here, "clean" means that known contradictory
supervision caused by changing the effective input modalities has been removed.

### VL policy

All VL records use vision plus `text_content`. When a `split_group_id` contains
both VT and VA records, their effective VL inputs would be identical. The VT
record is retained because its labels were defined for vision plus text, and the
paired VA record is removed. An unpaired VA record is converted to vision plus
text and retains its original target because no paired VT annotation exists.

### Omni policy

VT records retain their original vision-text input. VA records use vision,
`text_content`, and audio. For paired samples, effective VA uncertainty is the
union of the VT and VA conflict annotations: if either source protocol is
Conflict, the VTA record is Conflict. This prevents an added text modality from
creating a visible conflict while the target still says Aligned.

For an unpaired VA record, there is no separate text-view annotation. Its
original VA target is retained and this limitation is recorded in the cleaning
report.

The cleaned JSONL files are canonical intermediate records. Clarifying-question
labels and LLaMAFactory ShareGPT messages are added in later conversion steps.

## Official test data preparation

Run the test preparation script from the repository root:

```bash
python data/prepare_test_data.py
```

The equivalent command with every path stated explicitly is:

```bash
python data/prepare_test_data.py \
  --input Dataset/test_release.jsonl \
  --dataset-root Dataset \
  --output-dir data/cleaned/test
```

The released test file has only `sample_id`, `video`, and `text`. It contains
no emotion, conflict, uncertainty, or clarifying-question truth. The script
therefore validates and normalizes inference metadata without fabricating
supervision or registering the test split as a LlamaFactory eval dataset.

The script performs these checks and transformations:

- Require the exact released three-field schema and unique non-empty sample IDs.
- Infer VT or VA from both standard and generated sample ID formats.
- Require every video path to be relative to `Dataset/` and to exist locally.
- Preserve all rows, paired VT/VA samples, duplicates, and official source order.
- Build separate VL and Omni manifests using the same modality policy as baseline
  and SFT inference.
- Abort if a paired VT/VA group disagrees on its video path or supplied text.
- Write files atomically and record source/output SHA-256 hashes.

Current released test statistics:

| Item | Count |
| --- | ---: |
| Released and accepted rows | 734 |
| Rejected rows | 0 |
| VT rows | 387 |
| VA rows | 347 |
| Logical groups | 458 |
| Paired VT/VA groups | 276 |

Model-specific test input policy:

| Model path | VT test input | VA test input | Rows |
| --- | --- | --- | ---: |
| VL | video + supplied text | visual video frames only | 734 |
| Omni | video + supplied text; embedded audio disabled | video + embedded audio; supplied text withheld | 734 |

Generated files:

- `data/cleaned/test/test_vl_clean.jsonl`: VL inference manifest.
- `data/cleaned/test/test_omni_clean.jsonl`: Omni inference manifest.
- `data/cleaned/test/test_rejected.jsonl`: rejected records; currently empty.
- `data/cleaned/test/cleaning_report.json`: counts, policies, and hashes.

The normalized manifests retain the released `text` for traceability, but the
`input_modalities` field excludes it for VA inference. The raw
`Dataset/test_release.jsonl` remains the canonical input used by the inference
scripts so that the generated submission CSV exactly follows official order.

## Generated clarifying-question labels

The Conflict clarifying-question labels under `cleaned/` were generated on
2026-10-14 from the `cleaned/all` VL and Omni records. This source variant
ignores `use_in_main` while retaining the other cleaning and consistency rules
described above.

The Conflict records contain 441 sample IDs belonging to 378 unique
`split_group_id` values. Questions were generated once per unique group and
then mapped back to every corresponding sample ID, avoiding repeated model
calls for the same video.

Generation used Alibaba Cloud Bailian's `qwen3.8-omni-flash` model with the
original video and `text_content` as inputs. Audio was included whenever it was
present in the original video. Each group has exactly five neutral, open-ended
clarifying questions. Second-person wording addresses the current user, not a
person appearing or speaking in the video. The generated output passed checks
for question count, uniqueness, language, question punctuation, leading-choice
wording, and incorrect second-person references.

Generated files:

- `cleaned/clarifying_questions_by_group.jsonl`: 378 group-level records.
- `cleaned/clarifying_questions_by_sample_id.jsonl`: 441 sample-level records;
  use this file to join labels to training records by `sample_id`.
- `cleaned/generation_summary.json`: model, prompt version, coverage, token
  usage, and estimated generation cost.

All 378 groups and 441 sample IDs were completed without missing outputs. The
API credential used for generation is not stored in these files.

## LlamaFactory SFT datasets

Run `python scripts/prepare_sft_data.py` from the repository to convert the
complete `cleaned/all` training records and `cleaned/val` official validation
records to LlamaFactory ShareGPT multimodal data. No validation subset is
sampled from the training set. The converter verifies that the official splits
have no overlapping `sample_id` or `split_group_id` values. All 666 official
validation records are retained in both model paths, including paired VT/VA
records, so validation matches baseline sample coverage and weighting.

Validation input modalities also match the baseline scripts exactly:

| Model path | VT validation input | VA validation input |
| --- | --- | --- |
| VL | video + supplied text | visual video frames only |
| Omni | video + supplied text; embedded audio disabled | video + embedded audio; supplied text withheld |

This validation policy intentionally differs from training. VL training uses
`text_content` to convert VA into vision-text examples, while Omni training uses
video, text, and audio for VA examples.

The generated artifacts are stored under the Git-ignored
`llamafactory/generated/` directory:

| Dataset | Source split | Examples | Groups | Supervision target |
| --- | --- | ---: | ---: | --- |
| `affect_sft_vl_train` | official train | 2,104 | 2,104 | emotion + uncertainty + question |
| `affect_sft_vl_val` | official val | 666 | 412 | emotion only |
| `affect_sft_omni_train` | official train | 3,354 | 2,104 | emotion + uncertainty + question |
| `affect_sft_omni_val` | official val | 666 | 412 | emotion only |

Training VL records contain `<video>` plus supplied text. Training Omni VT
records use the same inputs, while training Omni VA records contain
`<video><audio>`, supplied text, and a separately extracted 16 kHz mono FLAC
audio track. During official validation, VL VA records omit text and audio;
Omni VA records keep `<video><audio>` but omit supplied text. There are 1,895
extracted audio files across the official train and validation splits.

Training assistant targets are compact JSON objects with exactly
`pred_emotion`, `is_uncertain`, and `clarifying_question`. Official validation
targets contain exactly `pred_emotion`. No synthetic question is generated or
attached to validation records because the official split provides no standard
clarifying-question answer; SFT validation therefore measures emotion
recognition loss only.

By default, each Conflict record deterministically selects one of its five
question labels. This preserves the cleaned dataset's class balance. The
converter also supports an explicit five-question expansion for augmentation;
see `scripts/README.md` for commands and the resulting weighting tradeoff.

`llamafactory/dataset_info.json` registers the four generated datasets. A
LlamaFactory training configuration must use `dataset_dir: data/llamafactory`
and `media_dir: .` when launched from the repository root. Qwen2.5-Omni
configurations must also set `use_audio_in_video: true` to match the baseline's
VA audio handling; VT records contain no audio input and remain video-text only.

## Other data directories

- `llamafactory/dataset_info.json`: LLaMAFactory dataset registrations.
- `llamafactory/generated/`: generated ShareGPT files and extracted audio;
  ignored by Git.
- `splits/`: reserved for future experiment-specific partitions; the default
  SFT pipeline uses the official train/validation boundary directly.
- `manifests/`: dataset statistics and conversion manifests.
