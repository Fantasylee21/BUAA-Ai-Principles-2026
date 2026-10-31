# SFT 实验汇总

本文档汇总三个多模态模型在原始 Baseline 和 QLoRA SFT 后的官方验证集结果。
所有结果均在完整的 `Dataset/val.jsonl` 上评估，共 666 条样本；验证集不去重。

2026-10-15 起，Baseline 与 SFT inference 统一使用 `sft_prompt.py` 中的 system
prompt、target instruction 和协议描述。旧 Baseline 使用了不同提示词，尤其会诱导
VL 模型将 VA 缺少音频直接视为不确定，因此旧结果不再作为正式对照。

比赛评分公式：

```text
Score = 0.5 * Macro-F1
      + 0.3 * Conflict_accuracy
      + 0.2 * Aligned_calibration
```

## 同口径主结果

下表使用相同 Prompt、相同验证集和 BF16 基座精度比较原始模型与 QLoRA Adapter，
从而尽量只保留后训练权重带来的差异。

| 模型 | 版本 | 训练 YAML | Macro-F1 | Conflict_accuracy | Aligned_calibration | Score |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B-Instruct | Baseline BF16 | 不适用 | 0.4837 | 0.0000 | **1.0000** | 0.4419 |
| Qwen2.5-VL-3B-Instruct | QLoRA Adapter BF16 | `configs/sft/qwen25vl/qlora_sft_v1.yaml` | **0.6079** | **0.5270** | 0.9848 | **0.6590** |
| Qwen3-VL-4B-Instruct | Baseline BF16 | 不适用 | 0.5090 | 0.0135 | **1.0000** | 0.4586 |
| Qwen3-VL-4B-Instruct | QLoRA Adapter BF16 | `configs/sft/qwen3vl/qlora_sft_v1.yaml` | **0.6121** | **0.5676** | 0.9932 | **0.6750** |
| Qwen2.5-Omni-3B | Baseline BF16 | 不适用 | 0.4873 | 0.0000 | **1.0000** | 0.4437 |
| Qwen2.5-Omni-3B | QLoRA Adapter BF16 | `configs/sft/qwen25omni/qlora_sft_v2_64tok.yaml` | **0.6281** | **0.6892** | 0.9341 | **0.7076** |

## QLoRA 相对提升

下表为同 Prompt、同 BF16 推理精度下的 `QLoRA - Baseline` 绝对变化。

| 模型 | Macro-F1 | Conflict_accuracy | Aligned_calibration | Score |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B-Instruct | +0.1242 | +0.5270 | -0.0152 | +0.2172 |
| Qwen3-VL-4B-Instruct | +0.1031 | +0.5541 | -0.0068 | +0.2164 |
| Qwen2.5-Omni-3B | +0.1407 | +0.6892 | -0.0659 | +0.2639 |

三个模型的 Macro-F1、冲突识别准确率和总分均明显提升。BF16 同口径下，
Qwen2.5-Omni 的总分最高，为 `0.7076`；Qwen3-VL 为 `0.6750`，Qwen2.5-VL
为 `0.6590`。后训练的主要收益来自冲突识别能力，代价是少量一致样本被判断为
不确定。

## SFT 推理版本

QLoRA 训练使用 4-bit NF4 基座，但 Adapter 可以在推理时挂载到 NF4 或 BF16 基座。
以下两版结果使用相同 Prompt 和输入策略。

| 模型 | 推理形态 | Macro-F1 | Conflict_accuracy | Aligned_calibration | Score |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B-Instruct | NF4 基座 + Adapter | 0.6062 | **0.6081** | 0.9645 | **0.6784** |
| Qwen2.5-VL-3B-Instruct | BF16 基座 + Adapter | 0.6079 | 0.5270 | **0.9848** | 0.6590 |
| Qwen3-VL-4B-Instruct | NF4 基座 + Adapter | **0.6137** | **0.7027** | 0.9426 | **0.7062** |
| Qwen3-VL-4B-Instruct | BF16 基座 + Adapter | 0.6121 | 0.5676 | **0.9932** | 0.6750 |
| Qwen2.5-Omni-3B | NF4 基座 + Adapter | **0.6303** | **0.6892** | **0.9341** | **0.7087** |
| Qwen2.5-Omni-3B | BF16 基座 + Adapter | 0.6281 | **0.6892** | **0.9341** | 0.7076 |

NF4 与 BF16 的差异主要体现在冲突/一致样本的决策边界，而不是训练 checkpoint
不同。主结果表采用 BF16 基座挂载 Adapter，以便与 BF16 Baseline 进行同精度比较；
NF4 结果作为实际 QLoRA 量化推理表现一并保留。

## 配置与结果文件

| 模型 | Baseline 结果 | 训练 YAML | 最佳 checkpoint |
| --- | --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | `output/baseline/qwen25vl_baseline_val_metrics.json` | `configs/sft/qwen25vl/qlora_sft_v1.yaml` | `output/sft/qwen25vl/qlora_sft_v1/checkpoint-396` |
| Qwen3-VL-4B-Instruct | `output/baseline/qwen3vl_baseline_val_metrics.json` | `configs/sft/qwen3vl/qlora_sft_v1.yaml` | `output/sft/qwen3vl/qlora_sft_v1/checkpoint-396` |
| Qwen2.5-Omni-3B | `output/baseline/qwen25omni_baseline_val_metrics.json` | `configs/sft/qwen25omni/qlora_sft_v2_64tok.yaml` | `output/sft/qwen25omni/qlora_sft_v2_64tok/checkpoint-630` |

QLoRA NF4 inference：

- `output/sft/evaluation/qwen25vl_qlora_sft_v1_val_metrics.json`
- `output/sft/evaluation/qwen3vl_qlora_sft_v1_val_metrics.json`
- `output/sft/evaluation/qwen25omni_qlora_sft_v2_64tok_best_val_metrics.json`

QLoRA BF16 inference：

- `output/sft/evaluation/qwen25vl_qlora_sft_v1_val_bf16_metrics.json`
- `output/sft/evaluation/qwen3vl_qlora_sft_v1_val_bf16_metrics.json`
- `output/sft/evaluation/qwen25omni_qlora_sft_v2_64tok_best_val_bf16_metrics.json`

## 评估口径

- 所有正式结果使用 `sft_prompt.py` 中的同一套 system prompt 和 target instruction。
- VL 模型：VT 样本输入视频和文本；VA 样本只输入视频画面，不使用音频或
  `text_content`。
- Omni 模型：VT 样本输入视频画面和文本，并关闭视频内音频；VA 样本输入视频
  及其音频，不提供 `text_content`。
- 所有列出的运行均覆盖 666/666 条验证样本，推理错误数为 0。
- Qwen2.5-VL Baseline 与全部 SFT 结果均有 666 条合法 JSON。Qwen3-VL Baseline
  有 4 条非法情感值被归一化；Qwen2.5-Omni Baseline 有 1 条非法情感值被归一化。
- 官方验证集真值只包含 `positive` 和 `negative`，没有 `neutral`；表中的
  Macro-F1 仍严格按照比赛三分类口径计算。

## Omni 显存调整

Qwen2.5-Omni 最初使用
`configs/sft/qwen25omni/qlora_sft_v1.yaml` 训练，但在 24 GB GPU 上发生 OOM，
因此正式实验改用 `configs/sft/qwen25omni/qlora_sft_v2_64tok.yaml`。正式配置仅将
`video_max_pixels` 从 `100352` 降至 `50176`，即固定约 64 个视觉 token/帧；
视频采样率仍为 2 FPS，音频输入、训练批大小和其余主要 QLoRA 参数保持不变。

模型 checkpoint、优化器状态和原始训练日志属于本地实验产物。指标、预测与
逐样本详情保存在 `output/` 下对应的实验目录中。

## QLoRA SFT v3

[按任务字段平衡监督的 v3 实验报告](qlora_sft_v3.md)：三模型完成训练及全部 9 次验证，最佳 Score 分别为 Omni 0.7288、Qwen3-VL 0.7201、Qwen2.5-VL 0.6880。


## QLoRA SFT v4：Conflict 两倍过采样

[v4 实验分析](qlora_sft_v4.md) 整理了训练执行者提供的实验材料：

| 模型 | v3 验证 Score | v4 验证 Score |
| --- | ---: | ---: |
| Qwen2.5-Omni-3B | **0.7288** | 0.7185 |
| Qwen3-VL-4B | **0.7201** | 0.7110 |
| Qwen2.5-VL-3B | 0.6880 | **0.6952** |

v4 在 v3 字段加权损失上增加 Conflict 两倍采样；固定 3 epoch 时，优化步数也增加，
因此不能把变化仅归因于样本比例。v4 原始指标与日志尚未随材料提交，以上是报告值。

另据项目成员提供的 Kaggle 截图，Omni v4 Public Score 为 **0.90813**，高于截图中的
Omni v3（0.90386）和 v2 4-bit（0.90644）。公开榜与本地验证集不同，不混算指标。
当前选定 Omni v4 交付；`sft-final` 固定代码，本报告为后续文档补充。
