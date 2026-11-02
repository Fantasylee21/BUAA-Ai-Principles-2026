"""CPU autograd tests for the actual GRPO loss and media-forward path.

These do not replace the remote real-Omni CUDA smoke test.
"""
import copy
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from rl.omni_runtime import OmniPolicy, batch_for_completions, dr_grpo_loss
from rl.experiment import DEFAULT_EXPERIMENT


@unittest.skipIf(torch is None, "CPU PyTorch is required for autograd checks")
class AutogradTests(unittest.TestCase):
    def test_policy_gradient_direction_and_padding_mask(self):
        logps = torch.tensor([[-.5, -.7], [-.1, -.2]], requires_grad=True)
        mask = torch.tensor([[1, 1], [1, 0]])
        loss, kl = dr_grpo_loss(torch, logps, logps.detach().clone(), mask,
                               torch.tensor([1., -1.]), beta=.02,
                               max_completion_length=4, completions_per_update=4)
        loss.backward()
        self.assertEqual(float(kl), 0)
        self.assertTrue(torch.allclose(logps.grad, torch.tensor([[-1/16, -1/16], [1/16, 0]])))

    def test_microbatching_preserves_the_global_loss_and_gradient(self):
        def run(chunks):
            values = torch.tensor([[-.5, -.7], [-.1, -.2]], requires_grad=True)
            reference = values.detach() + .1
            total = 0
            for indices in chunks:
                loss, _ = dr_grpo_loss(torch, values[indices], reference[indices],
                                      torch.ones_like(values[indices]), torch.tensor([1., -1.])[indices],
                                      beta=.02, max_completion_length=4, completions_per_update=4)
                total += loss
            total.backward()
            return total.detach(), values.grad
        combined = run([slice(None)])
        chunked = run([slice(0, 1), slice(1, 2)])
        self.assertTrue(torch.allclose(combined[0], chunked[0]))
        self.assertTrue(torch.allclose(combined[1], chunked[1]))

    def test_media_repeat_and_completion_mask(self):
        prefix = make_prefix()
        inputs, ids, mask = batch_for_completions(torch, prefix, [torch.tensor([7, 2]), torch.tensor([8])], 0)
        self.assertEqual(inputs["input_ids"].tolist(), [[1, 3, 4, 7, 2], [1, 3, 4, 8, 0]])
        self.assertEqual(mask.tolist(), [[1, 1], [1, 0]])
        self.assertEqual(inputs["input_features"].shape, (2, 2, 3))
        self.assertEqual(inputs["pixel_values_videos"].shape, (8, 2))
        self.assertEqual(inputs["video_grid_thw"].shape, (2, 3))
        self.assertEqual(inputs["video_second_per_grid"], [.5, .5])
        self.assertEqual(prefix["input_features"].shape, (1, 2, 3))

    def test_real_backward_path_passes_both_modalities_and_keeps_reference_frozen(self):
        policy = object.__new__(OmniPolicy)
        policy.torch = torch
        policy.settings = dict(DEFAULT_EXPERIMENT)
        policy.training, policy.pad_id = True, 0
        policy.model = FakeOmni()
        policy._set_trainability()
        before = policy.model.lora_A["reference"].weight.detach().clone()
        policy.backward_group(make_prefix(), [torch.tensor([7, 2]), torch.tensor([8, 2])], [.4, -.4], 1)
        self.assertEqual(len(policy.model.media_calls), 4)
        self.assertTrue(all(video and audio for video, audio in policy.model.media_calls))
        self.assertIsNone(policy.model.lora_A["reference"].weight.grad)
        self.assertTrue(torch.equal(before, policy.model.lora_A["reference"].weight))
        self.assertGreater(float(policy.model.lora_A["policy"].weight.grad.abs().sum()), 0)
        self.assertEqual(policy.model.active_adapter, "policy")


def make_prefix():
    return {"input_ids": torch.tensor([[1, 3, 4]]), "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "pixel_values_videos": torch.ones((4, 2)), "video_grid_thw": torch.tensor([[1, 2, 2]]),
            "input_features": torch.ones((1, 2, 3)), "feature_attention_mask": torch.ones((1, 3)),
            "video_second_per_grid": [.5], "use_audio_in_video": True}


if torch is not None:
    class FakeOmni(torch.nn.Module):
        def __init__(self):
            super().__init__()
            module = torch.nn.Linear(1, 12, bias=False)
            with torch.no_grad():
                module.weight.copy_(torch.arange(12).float().reshape(-1, 1) * .01)
            self.lora_A = torch.nn.ModuleDict({"policy": module, "reference": copy.deepcopy(module)})
            self.active_adapter = "policy"
            self.media_calls = []
            self.rope_deltas = None

        def set_adapter(self, name):
            self.active_adapter = name

        def get_base_model(self):
            return self

        def forward(self, input_ids, attention_mask, pixel_values_videos, video_grid_thw,
                    input_features, feature_attention_mask, video_second_per_grid,
                    use_audio_in_video, logits_to_keep, use_cache, return_dict):
            self.media_calls.append((pixel_values_videos is not None, input_features is not None))
            batch = input_ids.shape[0]
            context = input_features.sum((1, 2)) + pixel_values_videos.reshape(batch, -1).sum(1)
            logits = self.lora_A[self.active_adapter](context.reshape(-1, 1)).unsqueeze(1)
            return SimpleNamespace(logits=logits.expand(batch, logits_to_keep, -1))


if __name__ == "__main__":
    unittest.main()
