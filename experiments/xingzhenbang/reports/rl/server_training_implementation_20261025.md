# 服务器训练实施计划与交接

本批目标：通过可复验的模型训练改善情感分类与冲突判断，争取超过当前 Public **0.92313**，最终目标 **0.95**。0.95 是目标，不是已取得的成绩。最高分 `submission_09.csv` 没有生成源码，作为冻结提交基线保留。此前一次补漏提交为 0.90175，本批转入模型训练，不自动提交 Kaggle。

## 1. 执行位置与资源

- 服务器：`10.62.193.44`，项目根目录 `/home/nas/xingzhenbang/buaa-ai2026`。
- 源码快照：`releases/model-training-v1-20261025`；已有训练源码不修改，新批次编排、汇总与全量训练入口作为附加文件部署并记录 SHA-256。
- 视觉 v2 源码独立保存为 `releases/visual-expert-v2-20261025`，不修改正在训练的 SFT 源码。
- GPU 1：Omni 协议对照、条件复验和条件全量训练，串行执行。
- GPU 2：Qwen3-VL-8B 视觉证据实验。
- GPU 0：已有其他进程，不操作。新阶段启动前检查所选 GPU 占用，遇到其他任务占用即停止该阶段，不结束其他进程。
- 每个模型子进程设 32 GiB PyTorch 分配器上限。CUDA 上下文等额外分配仍须看实际显存。
- SFT 使用独立 `.venv-sft-protocol`；旧 `.venv-rl` 保持原状。LlamaFactory 固定在 `97b32d3133b501432141a82949d5c7bc4d94f23a`。两组统一 SDPA，明确记录与历史 v4 FA2 的差异。
- 所有训练与评估由服务器 tmux 执行，断开本机 SSH 后继续。结果保留在服务器，等用户通知后取回。
- 首次视觉预检在第 18 条视频触发严格像素预算检查，v1 失败记录和已有缓存保留。视觉修正使用 v2 新目录，交付编排使用 r2 新目录；seed 42 的 SFT 不重新启动。

## 2. 数据与目标

只用原始训练集 **3,354** 条进行模型开发：原始 Conflict 411 条；146 条 `use_in_main=false` 全保留。按同视频及媒体内容 SHA 合并分组，同组 VT/VA 与重复媒体不能跨划分。

固定 split seed 42：训练 **2,013** 条、开发 **670** 条、留出 **671** 条。复验只改变训练 seed 为 43，内容划分不变。

| 模型 | 训练输入与监督 | 子集原样本 | 子集 Conflict | 2x 后样本数 |
|---|---|---:|---:|---:|
| control | 复现历史 v4 数据语义：VA 提供额外文本，冲突标签为配对并集 | 2,013 | 263 | 2,276 |
| experiment | VT=静音视频+文本；VA=视频+音频、不提供给定文本；保留原协议标签 | 2,013 | 247 | 2,260 |

开发与留出两组输入、原始标签完全相同。正式 val 666 条和 test 734 条不进入这两组的选模或晋级门槛。单模态不清晰与真实 neutral 不混为同一种监督；本批主任务保留不清晰样本。

## 3. A 路：真实 SFT 对照及自动晋级

两组均从同一个 **Qwen2.5-Omni-3B 原始基座**开始，不从旧 adapter 续训。沿用 v4 的训练设置：

| 配置 | 固定值 |
|---|---|
| 量化 / 精度 | NF4 4bit、double quant、BF16 |
| LoRA | rank 16、alpha 32、dropout 0.05；视觉与音频塔按原设置冻结 |
| 学习率 / 正则 | 1e-4 / weight decay 0.01，cosine，warmup 32 steps |
| batch | 每卡 1，梯度累计 16，gradient checkpointing |
| 损失 | emotion 0.5、uncertainty 0.4、其余回答 0.1；Conflict multiplier 2 |
| 采样 | Conflict 2 倍 |
| 视频预算 | 2 fps、最多 64 帧、每帧 50,176 像素 |
| 轮数 | 对照最多 3 epoch，每 epoch 保存 checkpoint |

自动流程：

1. 完成 seed 42 control 的训练；对三个 epoch 在开发集生成完整回答，以综合 Score 选择 checkpoint，平分取更早 epoch；仅对选中的模型做一次留出评估。
2. 用完全相同流程完成 seed 42 experiment，生成 `pair-summary.json/.md`，逐条核对 ID、原始标签、CSV 与详细输出，重算指标。
3. 同时满足 **ΔScore ≥ 0.01、Macro-F1 下降 ≤ 0.005、自然冲突识别数不下降**，才自动准备并训练 seed 43 的 control 与 experiment。任一条件不满足，A 路正常结束并报告未晋级。
4. seed 43 使用同一内容划分和同一晋级条件。两次均通过，才进入全量训练；不会降低门槛让流程继续。
5. 在尚无开发结果时固定最终 epoch 规则：`ceil((seed42 实验组开发所选 epoch + seed43 实验组开发所选 epoch) / 2)`，范围 1–3。两次 epoch 必须从对应 checkpoint 的 trainer_state 读取并核验。不使用留出或正式 val 选择轮数。
6. 从同一原始基座，以 experiment 协议训练全部 **3,354** 条原样本，Conflict 2x 后 **3,765** 条。训练期间关闭 eval，不提供正式 val；训练到锁定轮数后固定末轮 adapter 及其 hash。
7. 对固定 adapter 做一次正式 val 诊断，再生成一次 734 条 test CSV；不重新选 checkpoint。验证四列接口、ID/顺序、情感枚举、yes 问题非空、no 问题为 `none`。结果供取回后审查，不自动上传 Kaggle。

seed 42 已有训练由原 tmux 持续执行。续接程序只等待它完成，不重新启动相同训练。后续任务和归档等待上限为 72 小时；新启动的续接子进程超过该总期限会结束本任务自己的进程组并记录失败。正在独立运行的 seed 42 与视觉任务不由此结束。

## 4. B 路：新视觉表示与证据头训练

采用官方 **Qwen3-VL-8B-Instruct**，固定 Hugging Face revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。服务器从 Qwen 官方 ModelScope 仓库接收权重，四个分片必须与固定版本官方 LFS SHA-256 完全一致。

自动流程为：权重校验 → 32 条训练视频真实 GPU 预检（包含训练集中最长帧数视频）→ 2,104 个独立训练视频的视觉特征 → 三折内容分组证据头及校准训练 → 父目录分组压力测试。

v2 的 32 条真实 GPU 预检已通过，权重 hash 不变，平均每视频约 0.52 秒，峰值预留显存约 8.96 GB；已进入全部训练视频提取。此次真实视频最多采样 38 帧，没有覆盖 64 帧上限；全提取耗时仍以实际日志为准。

只替换视觉分支，Omni 文本/音频分支在同折共享。提示不带文本转录、标签或来源 ID，PyAV 只解码视频轨。对齐名义帧率、帧数和每帧像素预算；不声称两模型具有完全相同的采样与预处理。

晋级条件是主分组不确定性任务分量提高至少 0.01、自然冲突识别不下降、父目录压力分量不下降。32 视频预检推算全提取超过 24 GPU 小时则停止该提取路线。B 路本批产出冻结视觉模型加可训练证据头的对照结果；**视觉 QLoRA 与更复杂融合属于取回证据后的后续批次**，不在本批承诺已实现。A 路是真实大模型 QLoRA 训练主线。

## 5. 指标、失败记录与完成产物

每组输出综合 Score、情感 Macro-F1、自然/合成冲突识别、VT/VA 分层、普通误报及新增正确/错误。固定三类 Macro-F1 与另一诊断口径分开报告；本地绝对分不能直接与 Public 比较。本批没有新的 Public 成绩。

报错保留日志、已有 checkpoint、退出原因与状态文件，不循环自动重训或提交。晋级失败与运行报错分别报告。归档程序在两路终止后生成：

- `FINAL_REPORT.json`：各阶段状态、晋级结果、最终模型是否训练、结果文件 hash。
- `review-results.tar.gz`：计划、配置、训练统计、评估 CSV/指标、简化逐条预测、日志；不打包视频、特征数组、优化器或权重。
- `COMPLETE.json`：归档完成标志和报告、压缩包 SHA-256。它表示流程结束；是否训练成功、是否晋级看其中状态和最终报告。
- 所有模型 adapter/checkpoint 留在各自运行目录，后续按需要取回。

结果包是私有实验记录。完整训练日志可能包含 LlamaFactory 输出的训练示例；逐条预测的精简副本去掉 `input_prompt` 和 `raw_response`，不能据此宣称整个日志包不含提示文本。完成标志发布前另核验视觉报告完成标记；若存在全量最终模型，还须核验 adapter hash 和 734 条候选 CSV。

## 6. tmux 与目录

| 用途 | tmux | 运行目录（均在项目根目录下） |
|---|---|---|
| seed 42 对照 | `buaa-protocol-sft` | `runs/protocol-sft-ablation-seed42-20261025` |
| 条件复验、全量与归档 | `buaa-training-delivery`，窗口 `protocol`、`collect` | `runs/model-training-batch-20261025-r2` |
| seed 43（条件触发） | 由 delivery 的 protocol 子进程执行 | `runs/protocol-sft-ablation-seed43-20261025` |
| 全量最终训练（条件触发） | 同上 | `runs/protocol-sft-final-20261025` |
| 视觉证据 | `buaa-visual-expert`，窗口 `experiment-v2` | `runs/visual-expert-v2-20261025` |

服务器完整计划另保存为 `runs/model-training-batch-20261025-r2/IMPLEMENTATION_PLAN.md`。

查看进度：

```bash
tmux attach -t buaa-protocol-sft
tmux attach -t buaa-training-delivery
tmux attach -t buaa-visual-expert
# 用 Ctrl+b 后按 d 脱离会话，训练继续。
tail -f /home/nas/xingzhenbang/buaa-ai2026/runs/protocol-sft-ablation-seed42-20261025/logs/control.log
cat /home/nas/xingzhenbang/buaa-ai2026/runs/model-training-batch-20261025-r2/protocol.status.json
cat /home/nas/xingzhenbang/buaa-ai2026/runs/model-training-batch-20261025-r2/collect.status.json
cat /home/nas/xingzhenbang/buaa-ai2026/runs/model-training-batch-20261025-r2/COMPLETE.json
```

初始 control 的实际日志已观察到 optimizer update，首条记录 loss 1.43143，约 22–25 秒/update；429 updates 的单组纯训练约 3 小时，另加开发/留出生成评估。触发复验和全量训练后总耗时可能超过一天；具体耗时由实际日志记录。用户通知后再从服务器取回本批结果。
