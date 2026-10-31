"""Length-independent emotion and uncertainty supervision for affect SFT."""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F


def identify_tasks(labels: torch.Tensor, tokenizer):
    """Mark emotion=1, uncertainty=2, remaining response=3, ignored=0."""
    tasks = torch.zeros_like(labels)
    conflicts = torch.zeros(labels.shape[0], dtype=torch.bool)
    for batch, row in enumerate(labels):
        positions = row.ne(-100).nonzero().flatten()
        ids = row[positions].tolist()
        if not ids:
            raise ValueError("SFT example has no supervised response tokens.")
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if encoded["input_ids"] != ids:
            raise ValueError("Response token offsets do not round-trip; refusing incorrect loss masks.")
        tasks[batch, positions] = 3
        for field, task in (("pred_emotion", 1), ("is_uncertain", 2)):
            match = re.search(r'"' + field + r'"\s*:\s*"([^"\\]*)"', text)
            if match is None:
                if field == "pred_emotion":
                    raise ValueError("SFT response is missing pred_emotion.")
                continue  # Official emotion-only validation keeps ordinary CE.
            start, end = match.span(1)
            marked = 0
            for token, (left, right) in enumerate(encoded["offset_mapping"]):
                if left < end and right > start:
                    tasks[batch, positions[token]] = task
                    marked += 1
            if not marked:
                raise ValueError(f"No supervised tokens found for {field}.")
            if task == 2:
                conflicts[batch] = match.group(1) == "yes"
    return tasks, conflicts


def task_balanced_loss(logits, labels, tasks, conflicts, settings):
    # Index supervised positions before converting logits to FP32. Long video
    # prompts then do not need a full sequence x vocabulary FP32 loss tensor.
    targets = labels[:, 1:]
    shifted_tasks = tasks[:, 1:]
    valid = targets.ne(-100)
    token_losses = F.cross_entropy(logits[:, :-1][valid].float(), targets[valid], reduction="none")
    row_ids = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(targets)[valid]
    task_ids = shifted_tasks[valid]
    weights = (float(settings["emotion"]), float(settings["uncertainty"]), float(settings["remainder"]))
    sample_losses = []
    for batch in range(labels.shape[0]):
        in_sample = row_ids.eq(batch)
        if not in_sample.any():
            raise ValueError("Causal shift removed all supervised tokens.")
        if not (in_sample & task_ids.eq(2)).any():
            sample_losses.append(token_losses[in_sample].mean())
            continue
        loss = token_losses.new_zeros(())
        active_weight = 0.0
        for task, weight in enumerate(weights, start=1):
            mask = in_sample & task_ids.eq(task)
            if mask.any() and weight > 0:
                multiplier = float(settings["conflict_multiplier"]) if task == 2 and conflicts[batch] else 1.0
                loss = loss + weight * multiplier * token_losses[mask].mean()
                active_weight += weight
        sample_losses.append(loss / active_weight)
    return torch.stack(sample_losses).mean()


def configure_task_loss(settings):
    from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    required = {"emotion", "uncertainty", "remainder", "conflict_multiplier"}
    if set(settings) != required or any(float(value) <= 0 for value in settings.values()):
        raise ValueError(f"task_balanced_loss must specify positive values for {sorted(required)}")
    original_collate = MultiModalDataCollatorForSeq2Seq.__call__

    def collate(self, features):
        result = original_collate(self, features)
        tasks, conflicts = identify_tasks(result["labels"], self.tokenizer)
        result["task_ids"] = tasks
        result["conflict_targets"] = conflicts
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        self.model_accepts_loss_kwargs = False  # Trainer must scale gradient accumulation.
        inputs = dict(inputs)
        tasks = inputs.pop("task_ids")
        conflicts = inputs.pop("conflict_targets")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = task_balanced_loss(outputs.logits, labels, tasks, conflicts, settings)
        return (loss, outputs) if return_outputs else loss

    MultiModalDataCollatorForSeq2Seq.__call__ = collate
    CustomSeq2SeqTrainer.compute_loss = compute_loss
