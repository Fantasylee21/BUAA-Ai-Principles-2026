# Omni 7B RL 阶段实验结果

训练目标、采样和运行方法见 [Omni 7B 的 NF4 RL 方法](OMNI_7B_METHOD.md)。

当前指定的 RL 阶段最佳结果文件为 `submit_source_emotion_vt.csv`，用户报告的
Public Score 为 **0.92597**。文件覆盖 734 条测试样本，SHA-256：

```text
eaec3cc02e970b51fdd80853d8486422546f949b8d26651b32b2db2aa7ae7c02
```

该提交基于 Omni 7B SFT v5 的 NF4 预测，并结合其他模型的预测进行综合处理。
训练版 RL 模型的独立提交分数列于下表。

准备好 `submit.csv`、测试集及 Omni v2、Qwen3-VL 的测试预测后，可运行
`python scripts/create_omni_7b_rl_stage_submission.py` 重建该文件。脚本可通过参数
指定输入路径；权重和测试预测不包含在 Git 提交中。

## 已报告的对照

| 提交或实验 | Public Score |
| --- | ---: |
| SFT v5，`submit.csv` | 0.91162 |
| NF4 RL，Conflict 采样 75% | 0.91012 |
| NF4 RL，自然冲突分层，第 10 步 | 0.91117 |
| NF4 RL，自然冲突分层，第 20 步 | 0.91027 |
| RL 阶段中间候选 | 0.92493 |
| RL 阶段最终提交 | **0.92597** |

分数来自用户提供的榜单记录，测试集标签和最终 Private 排名在当前工作区不可见。
SFT 训练与推理的详细记录见 [Omni 7B SFT v5](../reports/sft/omni7b_v5_results.md)。
