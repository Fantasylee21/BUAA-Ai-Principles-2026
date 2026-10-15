# SFT 实验报告：lora_sft_v1（L20 服务器）

日期：2026-10-15
执行：ayana（L20 服务器，4×NVIDIA L20 46GB）
背景：队长在 RTX 4090（24GB）上 LoRA SFT 中途 OOM，迁移到 L20 完成训练。

## 环境

| 组件 | 版本 |
| --- | --- |
| GPU | 4×NVIDIA L20 46GB（训练用 GPU 0/1/2，评估用 GPU 1/2/3） |
| Python | 3.13（miniconda env `sft`） |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.8.0 |
| FlashAttention | 2.7.4.post1（cu12/torch2.6/cp313 预编译 wheel） |
| LLaMAFactory | 0.9.6.dev0 @ 97b32d3133b501432141a82949d5c7bc4d94f23a（与锁文件一致） |
| ffmpeg | 7.1（conda-forge）；另装 soundfile（torchaudio 无内置后端） |

注意：LLaMAFactory 安装时会经 torchvision/torchdata 依赖把 torch 静默升级到
2.14.1+cu13（会破坏 flash-attn ABI 和 torchvision）。安装后必须回钉
`torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 triton==3.2.0` 并验证。

## 数据

- 训练集：data/cleaned/all（VL 2,104 条 / Omni 3,354 条，--ignore-use-in-main）
- 验证集：data/cleaned/val（官方 666 条，baseline-aligned-validation）
- 生成产物：data/llamafactory/generated/（含 1,895 条 16kHz FLAC，默认单问题策略）

## 训练

三模型各占一张卡并行，配置均为 configs/sft/<model>/lora_sft_v1.yaml
（LoRA r16/a32、冻结 vision tower + projector、grad ckpt、bs1×accum16、bf16+FA2、
3 epoch、每 epoch 在官方验证集评估，load_best_model_at_end 按 eval_loss 选最优）。

| 模型 | GPU | 训练耗时 | train_loss | eval_loss（ep1→ep2→ep3） | 峰值保留显存 |
| --- | --- | ---: | ---: | --- | ---: |
| Qwen2.5-VL-3B-Instruct | 0 | 1h49m | 0.137 | 0.0337 → 0.0316 → 0.0305（最优 ep3, checkpoint-396） | 22.0 GiB |
| Qwen3-VL-4B-Instruct | 1 | 1h47m | 0.130 | 0.0283 → 0.0296 → 0.0289（最优 ep1, checkpoint-132） | 16.8 GiB |
| Qwen2.5-Omni-3B | 2 | 3h32m | 0.092 | 0.0180 → 0.0179 → 0.0176（最优 ep3, checkpoint-630） | 24.3 GiB |

三个模型在 46GB 的 L20 上均余量充足（队长 4090 的 OOM 未复现）。

## 导出与评估

- VL 两模型：LLaMAFactory 直接合并 LoRA 到完整模型（configs/export/*.yaml）。
- Omni：LLaMAFactory 仅导出 thinker 子模型（qwen2_5_omni_thinker），需用
  scripts/rebuild_omni_full.py 把 thinker 权重嵌回原完整模型（thinker./talker./token2wav
  前缀结构）后再评估。
- 评估：复用根目录三个 baseline 脚本，--model-path 指向合并后模型，
  官方验证集 666 条，参数与 baseline 记录一致（native_default）。

| 模型 | Macro-F1 | Conflict_acc | Aligned_cal | Score | baseline Score | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B | 0.5946 | 0.6216 | 0.7264 | **0.6291** | 0.4309 | +0.198 |
| Qwen3-VL-4B | 0.5956 | 0.6081 | 0.9409 | **0.6684** | 0.4643 | +0.204 |
| Qwen2.5-Omni-3B | 0.6192 | 0.5541 | 0.9865 | **0.6731** | 0.3776 | +0.296 |

推理错误数均为 0；评估峰值保留显存 11.5 / 14.5 / 23.1 GiB。

## 结论

- 三个模型 SFT 后 Score 全部显著提升（+0.20 ~ +0.30），最佳为 Qwen2.5-Omni-3B 0.673。
- Omni 的 Conflict_accuracy（0.554）仍是三者中最低，是下一步（更多 Conflict 监督/
  扩增、RL 等）的主要改进点。
- Qwen3-VL-4B 最优 checkpoint 在 epoch 1，后续可考虑早停或减小 epoch 数对照。

## 复现要点

1. 安装依赖后回钉 torch 2.6.0 全家桶并 `import flash_attn, torchvision` 验证。
2. Omni 音频加载需 soundfile；评估脚本依赖 ffmpeg 在 PATH。
3. Omni 合并模型必须经 scripts/rebuild_omni_full.py 重建为完整架构再评估。
4. 训练/评估产物路径：output/sft/<model>/lora_sft_v1{,_merged,_merged_full}、
   output/sft_eval/（均在 L20，未入 Git）。
