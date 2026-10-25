# 实时进度与服务器恢复记录

完整模型设计仍按 `server_training_implementation_20261025.md` 执行。训练进度面板位于 tmux `buaa-protocol-sft` 的 `progress` 窗口，每 3 秒刷新训练 step、loss、学习率、训练阶段估计耗时、评估持久化行数和分支状态。日志条与持久化计数分别标记，不能将阶段进度当作整批进度。

## 首次停止与 A 路恢复

seed42 control 已完成 429 次更新、3 epoch，训练平均 loss 0.2042，开发 teacher-forced loss 0.0314；它们不是任务综合分。三个 checkpoint 均已保存。生成评估入口因独立环境缺少 `qwen_vl_utils` 停止。

仅在 `.venv-sft-protocol` 安装 `qwen-vl-utils==0.0.14`（不更改已有依赖），CPU `evaluate_sft.py --help` 导入预检通过。从原 checkpoint 继续生成评估，`train_results.json` 令原训练入口跳过已完成训练。原失败日志、状态、依赖修复后的环境保存在 seed42 run 的 `recovery/missing-qwen-vl-utils-attempt1/`。

## B 路 CPU 恢复

v2 的 2,104 个视觉特征已完整生成且冻结权重 checksum 不变。证据头首次拟合达到 L-BFGS 300 次迭代上限，不能视为有效已收敛结果。

新的 CPU 恢复入口复用并严格核验原训练特征；另建输出目录，把两组所有证据头和校准统一改为最多 3,000 次迭代，其余目标、L2、收敛容差、分组和晋级条件保持相同。旧失败和旧拟合不覆盖，新身份记录预算变更与旧失败 hash。若仍不收敛则记录失败；不会接受未收敛参数。无需重做视觉推理，不使用正式验证/测试。

## 新交付目录

原 r2 `COMPLETE.json` 和失败归档保留。恢复批次使用：

```text
/home/nas/xingzhenbang/buaa-ai2026/runs/model-training-batch-20261025-r3
/home/nas/xingzhenbang/buaa-ai2026/runs/visual-expert-cpu-recovery-20261025
```

`buaa-training-recovery` 的 `protocol`、`collect` 窗口接续原有 A 路、条件复验和全量训练，`visual-cpu` 窗口运行 CPU 恢复。新批次仍仅在两路终止并验证结果后生成 `FINAL_REPORT.json`、`review-results.tar.gz`、`COMPLETE.json`。未增加 Kaggle 自动提交。进度面板绑定 r3；原 r2 的结束标志不代表恢复批次结束。
