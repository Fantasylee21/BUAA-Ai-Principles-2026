import json
import math
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from task_balanced_loss import identify_tasks, task_balanced_loss

SETTINGS = {"emotion": 0.5, "uncertainty": 0.4, "remainder": 0.1, "conflict_multiplier": 2.0}


class TaskLossTests(unittest.TestCase):
    def test_question_length_does_not_dilute_classification(self):
        def calculate(length):
            labels = torch.tensor([[-100, 0, 1] + [2] * length])
            tasks = torch.tensor([[0, 1, 2] + [3] * length])
            logits = torch.zeros(1, length + 3, 3, requires_grad=True)
            return task_balanced_loss(logits, labels, tasks, torch.tensor([False]), SETTINGS)
        self.assertAlmostEqual(calculate(1).item(), math.log(3), places=6)
        self.assertAlmostEqual(calculate(1).item(), calculate(80).item(), places=6)

    def test_conflict_multiplier_applies_only_to_uncertainty(self):
        labels = torch.tensor([[-100, 0, 1, 2]])
        tasks = torch.tensor([[0, 1, 2, 3]])
        logits = torch.zeros(1, 4, 3, requires_grad=True)
        loss = task_balanced_loss(logits, labels, tasks, torch.tensor([True]), SETTINGS)
        self.assertAlmostEqual(loss.item(), 1.4 * math.log(3), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_prompt_padding_and_causal_shift(self):
        labels = torch.tensor([[-100, -100, 0, 1, 2, -100]])
        tasks = torch.tensor([[0, 0, 1, 2, 3, 0]])
        logits = torch.randn(1, 6, 3, requires_grad=True)
        loss = task_balanced_loss(logits, labels, tasks, torch.tensor([False]), SETTINGS)
        expected = sum(weight * F.cross_entropy(logits[:, position], labels[:, position + 1])
                       for position, weight in [(1, 0.5), (2, 0.4), (3, 0.1)])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[:, [0, 4, 5]].abs().sum().item(), 0.0)

    def test_emotion_only_validation_uses_original_ce(self):
        labels = torch.tensor([[-100, 0, 1, 2, -100]])
        tasks = torch.tensor([[0, 3, 1, 3, 0]])
        logits = torch.randn(1, 5, 3)
        loss = task_balanced_loss(logits, labels, tasks, torch.tensor([False]), SETTINGS)
        expected = F.cross_entropy(logits[:, :-1].reshape(-1, 3), labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(loss, expected)

    def test_microbatch_accumulation_matches_batch_mean(self):
        labels = torch.tensor([[-100, 0, 1, 2, -100], [-100, 0, 1, 2, 2]])
        tasks = torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 3, 3]])
        conflicts = torch.tensor([False, True])
        logits = torch.randn(2, 5, 3)
        batch = task_balanced_loss(logits, labels, tasks, conflicts, SETTINGS)
        micro = sum(task_balanced_loss(logits[i:i+1], labels[i:i+1], tasks[i:i+1], conflicts[i:i+1], SETTINGS)
                    for i in range(2)) / 2
        torch.testing.assert_close(batch, micro)

    def test_field_masks_on_all_actual_tokenizers(self):
        for model in ("Qwen2.5-VL-3B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen2.5-Omni-3B"):
            tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / model, local_files_only=True)
            for uncertain in ("yes", "no"):
                answer = json.dumps({"pred_emotion": "negative", "is_uncertain": uncertain,
                                     "clarifying_question": '你能说明“yes/no”表达的感受吗？' if uncertain == "yes" else "none"},
                                    ensure_ascii=False, separators=(",", ":"))
                ids = tokenizer.encode(answer + "<|im_end|>", add_special_tokens=False)
                labels = torch.tensor([[-100] * 3 + ids + [-100] * 2])
                tasks, conflicts = identify_tasks(labels, tokenizer)
                self.assertEqual(conflicts.item(), uncertain == "yes")
                self.assertEqual(tokenizer.decode(labels[0][tasks[0] == 1]), "negative")
                self.assertEqual(tokenizer.decode(labels[0][tasks[0] == 2]), uncertain)
                self.assertFalse(tasks[labels == -100].any())


if __name__ == "__main__":
    unittest.main()
