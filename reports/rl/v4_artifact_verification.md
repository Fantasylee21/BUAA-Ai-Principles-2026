# Omni v4 artifact verification — 2026-10-21

The user supplied `qlora_sft_v4_resample2x.7z`. Its root adapter and
`checkpoint-714/adapter_model.safetensors` are byte-identical to the original
Hugging Face adapter in commit
[`3aadc63707148c564bd27409a0ee80cf31988a47`](https://huggingface.co/BUAA-AI/qwen25omni-sft/commit/3aadc63707148c564bd27409a0ee80cf31988a47).

| Check | Result |
| --- | --- |
| Adapter file size, local and HF LFS pointer | 119,801,528 bytes |
| Adapter SHA-256, local root, local checkpoint-714 and HF LFS pointer | `16283fdec91c480978a3cb1b63456047a9b371435ff78961a3e10865a56a2f27` |
| Local root/checkpoint adapter-config SHA-256 | `28a25c85f364fbf91cf916e719eb490ed5a65e33dcb64f4874b56322fc61a09f` |
| Local checkpoint record | checkpoint-714, 666 validation samples, Score 0.7184609544226346 |
| Adapter type | LoRA, rank 16, alpha 32; thinker base identifier Qwen2.5-Omni-3B/thinker |

The HF check was made through the user's authenticated browser, by reading the
original commit's LFS SHA-256 and size. No HF credentials were copied into the
repository. The archive's optimizer/training pickle files were not loaded.

The current HF `main` is a different packaging format: commit
[`6a1e7fab0339e8b130261d7384b47f1a476885a1`](https://huggingface.co/BUAA-AI/qwen25omni-sft/commit/6a1e7fab0339e8b130261d7384b47f1a476885a1)
is titled “Upload full Omni v4 checkpoint-714 merged model” and contains three
full model shards. An adapter file cannot be compared byte-for-byte with those
merged shards. This verification establishes exact equality with the original
HF **adapter**, not numerical equivalence of every tensor in the merged model.

RL continues from the verified adapter on the original official Qwen base.
Actual GPU inference/backpropagation and validation scores require separate
runtime verification.
