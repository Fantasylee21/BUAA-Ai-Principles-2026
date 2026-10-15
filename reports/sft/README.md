# SFT 实验汇总

本文档汇总三个多模态模型在原始 Baseline 和 QLoRA SFT 后的官方验证集结果。
所有结果均在完整的 `Dataset/val.jsonl` 上评估，共 666 条样本；验证集不去重，
输入模态策略与对应 Baseline 保持一致。

比赛评分公式：

```text
Score = 0.5 * Macro-F1
      + 0.3 * Conflict_accuracy
      + 0.2 * Aligned_calibration
```

## 总体结果

| 模型 | 版本 | 训练 YAML | Macro-F1 | Conflict_accuracy | Aligned_calibration | Score |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B-Instruct | Baseline | 不适用（原始模型直接推理） | 0.4814 | 0.0135 | 0.9307 | 0.4309 |
| Qwen2.5-VL-3B-Instruct | QLoRA SFT | `configs/sft/qwen25vl/qlora_sft_v1.yaml` | **0.6062** | **0.6081** | **0.9645** | **0.6784** |
| Qwen3-VL-4B-Instruct | Baseline | 不适用（原始模型直接推理） | 0.5206 | 0.0135 | **1.0000** | 0.4643 |
| Qwen3-VL-4B-Instruct | QLoRA SFT | `configs/sft/qwen3vl/qlora_sft_v1.yaml` | **0.6137** | **0.7027** | 0.9426 | **0.7062** |
| Qwen2.5-Omni-3B | Baseline | 不适用（原始模型直接推理） | 0.3559 | 0.0000 | **0.9983** | 0.3776 |
| Qwen2.5-Omni-3B | QLoRA SFT | `configs/sft/qwen25omni/qlora_sft_v2_64tok.yaml` | **0.6303** | **0.6892** | 0.9341 | **0.7087** |

## QLoRA 相对提升

下表为 `QLoRA - Baseline` 的绝对指标变化。

| 模型 | Macro-F1 | Conflict_accuracy | Aligned_calibration | Score |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B-Instruct | +0.1247 | +0.5946 | +0.0338 | +0.2475 |
| Qwen3-VL-4B-Instruct | +0.0932 | +0.6892 | -0.0574 | +0.2419 |
| Qwen2.5-Omni-3B | +0.2744 | +0.6892 | -0.0642 | +0.3311 |

三个模型经过 QLoRA 后总分均有明显提升。Qwen2.5-Omni 获得最高总分
`0.7087`，Qwen3-VL 次之，为 `0.7062`。Qwen3-VL 和 Qwen2.5-Omni 的
一致样本校准率略有下降，但冲突识别能力的提升覆盖了该损失。

## 配置与结果文件

| 模型 | Baseline 脚本 | QLoRA YAML | 最佳训练 checkpoint | QLoRA 评估结果 |
| --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | `qwen25vl_baseline.py` | `configs/sft/qwen25vl/qlora_sft_v1.yaml` | `output/sft/qwen25vl/qlora_sft_v1/checkpoint-396` | `output/sft/evaluation/qwen25vl_qlora_sft_v1_val_metrics.json` |
| Qwen3-VL-4B-Instruct | `qwen3vl_baseline.py` | `configs/sft/qwen3vl/qlora_sft_v1.yaml` | `output/sft/qwen3vl/qlora_sft_v1/checkpoint-396` | `output/sft/evaluation/qwen3vl_qlora_sft_v1_val_metrics.json` |
| Qwen2.5-Omni-3B | `qwen25omni_baseline.py` | `configs/sft/qwen25omni/qlora_sft_v2_64tok.yaml` | `output/sft/qwen25omni/qlora_sft_v2_64tok/checkpoint-630` | `output/sft/evaluation/qwen25omni_qlora_sft_v2_64tok_best_val_metrics.json` |

Baseline 原始指标位于：

- `output/baseline/qwen25vl_baseline_val_metrics.json`
- `output/baseline/qwen3vl_baseline_val_metrics.json`
- `output/baseline/qwen25omni_baseline_val_metrics.json`

## 评估口径

- VL 模型：VT 样本输入视频和文本；VA 样本只输入视频画面，不使用音频或
  `text_content`。
- Omni 模型：VT 样本输入视频画面和文本，并关闭视频内音频；VA 样本输入视频
  及其音频，不提供 `text_content`。
- 三次 QLoRA 推理均覆盖 666/666 条验证样本，推理错误数为 0，模型输出均成功
  解析为 JSON。
- 官方验证集真值只包含 `positive` 和 `negative`，没有 `neutral`；表中的
  Macro-F1 仍严格按照比赛三分类口径计算。

## Omni 显存调整

Qwen2.5-Omni 最初使用
`configs/sft/qwen25omni/qlora_sft_v1.yaml` 训练，但在 24 GB GPU 上发生 OOM，
因此正式实验改用 `configs/sft/qwen25omni/qlora_sft_v2_64tok.yaml`。正式配置仅将
`video_max_pixels` 从 `100352` 降至 `50176`，即固定约 64 个视觉 token/帧；
视频采样率仍为 2 FPS，音频输入、训练批大小和其余主要 QLoRA 参数保持不变。

模型 checkpoint、优化器状态、预测明细和原始训练日志属于本地实验产物，存放在
已被 Git 忽略的 `output/` 目录中。

## QLoRA SFT v3

[按任务字段平衡监督的 v3 实验报告](qlora_sft_v3.md)：三模型完成训练及全部 9 次验证，最佳 Score 分别为 Omni 0.7288、Qwen3-VL 0.7201、Qwen2.5-VL 0.6880。
