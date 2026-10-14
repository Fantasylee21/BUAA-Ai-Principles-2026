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
- `cleaning_report.json`: input/output hashes, counts, distributions,
  and cleaning actions.

Custom paths and either filtering policy are supported:

```bash
python data/prepare_training_data.py \
  --input Dataset/train.jsonl \
  --dataset-root Dataset \
  --output-dir /path/to/output
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

## Other data directories

- `llamafactory/dataset_info.json`: LLaMAFactory dataset registrations.
- `llamafactory/generated/`: generated ShareGPT files and extracted audio;
  ignored by Git.
- `splits/`: committed sample ID lists for fixed train/validation splits.
- `manifests/`: dataset hashes and conversion manifests.
