# Qwen2.5-Omni-7B SFT v5：训练、验证与提交记录

## 当前交付文件

按当前用户指定，交付文件为工作树根目录 `submit.csv`。该文件的 SHA-256 为
`5edc74e4a53bf1a037b9fca1aee058abd9aaa9651af295e467b604601f2ea61a`，与
`output/sft/evaluation/qwen25omni_7b_v5_test_predictions.csv` 完全一致。

这份文件来自 Qwen2.5-Omni-7B 基座、v5 的 `checkpoint-118` LoRA adapter，推理时以
NF4 double quant 加载基座，计算精度为 BF16。它不是 BF16 全精度基座推理结果，
也不是 RL adapter 或后处理集成结果。测试推理覆盖 734 条，错误数为 0，734 条
均解析为 JSON；测试集无公开真值，本地测试指标不包含 Score。

### 榜单分数的记录边界

用户提供 `submit.csv` 的 Public Score 为 **0.91162**。在 RL 阶段实验中，指定结果
文件 `submit_source_emotion_vt.csv` 的 Public Score 为 **0.92597**；它在 SFT
预测基础上加入来源规则和情感集成，不是单一 RL adapter 的直接推理输出。两个
文件的结果分别记录；RL 阶段结果详见 [RL 实验结果](../../rl/RESULTS_OMNI_7B.md)。
截图显示提交为 after deadline，是否计入最终排名以比赛平台为准。

## 训练设置

- 基座：`../ckpt/Qwen2.5-Omni-7B`。
- 配置：`configs/sft/qwen25omni/qlora_sft_v5_targeted_7b.yaml`；训练采用 NF4
  QLoRA、double quant、BF16 计算，LoRA rank 16、alpha 32、dropout 0.05。
- 冻结视觉塔和多模态投影层；训练语言模型的 LoRA。学习率 `1e-4`，训练 3 轮，
  单卡 batch 1、梯度累积 16，seed 42。
- 任务损失按 emotion / uncertainty / 其余回答设为 0.5 / 0.4 / 0.1，Conflict
  样本的不确定性损失乘以 2。
- `data/llamafactory/generated/omni_v5_targeted/sampling_summary.json` 记录
  原始训练样本 3,354 条，重采样后 3,772 条：Aligned 2,913，合成 VT Conflict
  232，CH-SIMS VT Conflict 321，VA Conflict 306。后两组均为原数量的 3 倍；
  采样增加出现次数，不代表新增标注。

## 选模与验证

在完整 666 条验证集上，以本地 Score 选择 checkpoint。表中数值来自
`output/sft/qwen25omni/qlora_sft_v5_targeted_7b/best_checkpoint.json`，验证推理
使用 NF4；该验证集没有 neutral 真值，本地 Macro-F1 固定按三类计算，绝对分数不可
直接与榜单分数比较。

| Adapter | Macro-F1 | Conflict | Aligned | Score |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-118 | 0.623974 | 0.729730 | 0.951014 | **0.721109** |
| checkpoint-236 | 0.629457 | 0.689189 | 0.962838 | 0.714053 |
| checkpoint-354 | 0.631583 | 0.675676 | 0.972973 | 0.713089 |

后两个 checkpoint 的情感 Macro-F1 提高，但 Conflict 召回下降；按本地综合分选中
checkpoint-118。验证数据也参与选模，不能将该分数视为独立测试成绩。

## 推理和文件

- 测试运行配置：`output/sft/evaluation/qwen25omni_7b_v5_test_run_config.json`。
- 测试结果与详细记录：同目录 `qwen25omni_7b_v5_test_metrics.json`、
  `qwen25omni_7b_v5_test_details.jsonl`、`qwen25omni_7b_v5_test_predictions.csv`。
- VT 使用视频画面和给定文本；VA 使用视频画面及音频，推理时不提供文本。
- 推理配置为 `native_default`、SDPA、`max_new_tokens=128`；原测试记录使用 GPU 4。
- 评估脚本：`scripts/evaluate_sft.py`；数据及训练流水线：
  `scripts/prepare_omni_v5_data.py`、`scripts/run_omni7b_sft_v5.sh`。

已有模型及数据时，可生成一个新的输出文件以复现推理流程：

```bash
cd /home/djy/lgq/buaa/sft-worktree
CONDA_ENV=llama-factory STAGE=submit EVAL_GPU_ID=4 \
  SUBMIT_PATH="$PWD/submit_sft_v5_reproduced.csv" \
  bash scripts/run_omni7b_sft_v5.sh
```

该命令读取 `best_checkpoint.json`；执行前应确认其仍指向 checkpoint-118。
不同依赖或硬件可能影响预测，复现后应比较 CSV 和运行配置。

## 后续实验

v6 从 v5 checkpoint-118 以 `1e-5` 继续进行 NF4 QLoRA 训练。当前本地验证中，
第 60 步 Score 为 0.732421；选择结果保存在
`output/sft/qwen25omni/qlora_sft_v6_continue_7b/selection.json`。这不是榜单分数，
也不能说明 v6 已超过当前交付文件。
