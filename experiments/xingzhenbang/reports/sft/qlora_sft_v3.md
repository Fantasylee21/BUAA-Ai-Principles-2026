# QLoRA SFT v3 实验报告

日期：2026-10-15。三个模型均完成 3 轮训练，各轮 checkpoint 均在完整的 666 条官方验证集上评估，共 9 次评估；推理错误均为 0，输出均成功解析为 JSON。

## 方法

原始 token 平均交叉熵中，情感值和不确定性值各仅占约 4.3%–4.5% 的监督 token，其余回答与格式占约 91%。v3 对每条样本的情感、不确定性、其余回答三个部分分别求平均损失，再组合：

$$
L = 0.5 L_{\mathrm{emotion}} + 0.4 c(y) L_{\mathrm{uncertainty}} + 0.1 L_{\mathrm{rest}}
$$

其中 $c(\mathrm{yes})=2$、$c(\mathrm{no})=1$，即仅提高冲突样本的不确定性字段权重。没有过采样，也没有扩增训练样本。实现位于 `scripts/task_balanced_loss.py`，通过 `scripts/train_sft.py` 启用。

训练采用 NF4 4-bit QLoRA、BF16、FlashAttention 2；LoRA rank 16、alpha 32、dropout 0.05；学习率 0.0001，batch 1、梯度累积 16、3 epoch，seed 42，冻结模态编码器与 projector。VL 训练集 2,104 条，Omni 3,354 条。VT/VA 的训练和验证输入规则沿用项目约定；Omni 保留 v2 的 64 token/帧训练预算。单问题监督和数据清洗规则不变。

运行环境为 `fwt_temp`，NVIDIA L40S 48 GB。Omni、Qwen3-VL 使用 GPU 4，Qwen2.5-VL 使用 GPU 2 并行训练。Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 5.8.0、PEFT 0.18.1、FlashAttention 2.7.4.post1、bitsandbytes 0.49.2。

## 验证结果

按项目官方 Score 从各模型的 3 个 checkpoint 中选优，最佳均为第 3 轮。旧版对照分别为 Omni QLoRA v2_64tok、两个 VL 的 QLoRA v1；提升单位为百分点。

| 模型 | Macro-F1 | Conflict_accuracy | Aligned_calibration | v3 Score | 旧版 Score | 提升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Omni-3B | 0.6340 | 0.7432 | 0.9443 | **0.7288** | 0.7087 | +2.01 |
| Qwen3-VL-4B | 0.6185 | 0.7568 | 0.9189 | **0.7201** | 0.7062 | +1.39 |
| Qwen2.5-VL-3B | 0.6057 | 0.6622 | 0.9324 | **0.6880** | 0.6784 | +0.96 |

三个模型的综合 Score 均提高，主要体现在冲突识别：Omni 从 51/74 提高到 55/74，Qwen3-VL 从 52/74 提高到 56/74，Qwen2.5-VL 从 45/74 提高到 49/74。提升并非覆盖所有指标：Qwen2.5-VL 的 Macro-F1 略降，两个 VL 的 Aligned_calibration 下降。Omni 获得本次最高 Score 0.7288。

## 各轮 checkpoint

| 模型 | 第 1 轮 Score | 第 2 轮 Score | 第 3 轮 Score | 最佳 checkpoint |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-Omni-3B | 0.6813 | 0.7231 | 0.7288 | `checkpoint-630` |
| Qwen3-VL-4B | 0.6786 | 0.7064 | 0.7201 | `checkpoint-396` |
| Qwen2.5-VL-3B | 0.6763 | 0.6854 | 0.6880 | `checkpoint-396` |

## 验证与限制

9 项单元测试通过，覆盖损失字段定位、因果位移、掩码、梯度累积和 checkpoint 选择；9,560 条训练/验证目标的 token 定位审查通过。三个模型均完成小样本训练与 adapter 加载检查。完整验证指标保存在 `output/sft/evaluation/*_qlora_sft_v3_checkpoint-*_val_metrics.json`。

本次保留了 Omni VT 无真实音频时的位置编码兼容修正。单问题选择、数据顺序、运行环境等与旧版不保证相同，因此不能把全部收益归因于损失函数；需要同机同数据、关闭 `task_balanced_loss` 的消融实验进一步确认。验证集参与了 checkpoint 选择，以上成绩不等同于独立测试集结果。Conflict_accuracy 按项目定义统计冲突样本的不确定性识别，不要求情感同时正确。

Qwen3-VL 在完成训练并保存模型后，启动脚本退出异常，队列一度误报失败；已从完整 checkpoint 补齐评估，无需重训。断线与队列辅助脚本仅用于本机运行。

## 复现与产物

```bash
conda activate fwt_temp
python scripts/run_sft_v3.py --gpu 4
```

配置位于 `configs/sft/<model>/qlora_sft_v3.yaml`；该命令会跳过已有 `train_results.json` 的训练，并复用已完成的评估。模型权重、训练日志和原始数据保留在本地，不提交到 Git。远端保存配置、实现、测试、审查统计和各轮指标文件。
