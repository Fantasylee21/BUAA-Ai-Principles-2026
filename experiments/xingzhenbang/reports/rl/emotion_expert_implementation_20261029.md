# EmotionReasoner 7B / E-U 分离 / 计分一致奖励：执行记录

**2026-10-30 最终状态补核：服务器已完成冻结 E0/U0 的开发与留出评估，因 `held_gate_not_met` 停止，未生成候选。正式 SFT、R1、R2 没有启动。下文启动与准入描述保留历史过程；最终结果见文末。该批次没有本组正式训练的新 7B 权重。**

本记录对应用户批准的最终实验规格，以及“只使用 GPU 2”和“初始化延长至 90 分钟”的补充授权。目标是得到完整任务的可复现净增益，0.95 为目标值。没有自动提交 Kaggle，也没有 push。

## 代码入口与固定设置

- 配置：`configs/rl/emotion_expert_final_v1.json`。
- 下载、严格加载与 CPU 合并：`scripts/emotion_expert_bootstrap.py`。
- 合法模态输入、解码缓存及原生 Omni 前向：`rl/emotion_expert_runtime.py`。
- 数值损失、标签映射、选择门槛及导出：`rl/emotion_expert.py`。
- GPU 工作器：`scripts/train_emotion_expert.py`。
- 串行调度、时限及开发/留出流程：`scripts/emotion_expert_pipeline.py`。
- 只读进度面板：`scripts/emotion_expert_progress.py`。
- 数值与隔离检查：`tests/test_emotion_expert.py`。

固定起点是 Qwen/Qwen2.5-Omni-7B 加公开 SmilingWeeping/EmotionReasoner 的 `EmotionReasoner/checkpoint-195`。公开 adapter 在 FP32 合并后保存 BF16 基座；训练时以 NF4/BF16 加载，只训练两个相互独立的语言 LoRA（rank 16、alpha 32、dropout 0）。感知编码器、连接层与基座冻结。

E 输出 negative / neutral / positive，U 输出 no / yes。U 同时进行隔离的 V、T/A 四态监督，unclear 与 neutral 分开。VT 为静音视频加给定文本，VA 为视频加音频，VA 不提供给定文本。输入不含来源、文件名、标注或样本 ID。

复用 2,013 / 670 / 671 的内容分组。原始 3,354 条标签逐条核对，保留 146 条 `use_in_main=false`，相同内容哈希和原内容组不跨分区。训练、预检、RL 均只使用训练分区。

媒体为 2 fps、最多 64 帧、每帧 50,176 像素、16 kHz 音频；补充四个均匀时点检测到的最大人脸，扩边 25%、448×448。缓存只保存解码媒体，不缓存隐藏状态。单卡 40 GiB；如 OOM，全部新对照统一切换 32 帧重做预检。

每个分支监督训练一轮，batch 16（最后 13 条按实际数量归一化）、AdamW、学习率 1e-5、weight decay 0.01、warmup 10%、梯度裁剪 1。无额外冲突过采样。

U 监督损失为 `0.3 E_Conflict[CE] + 0.2 E_Aligned[CE] + 0.1 L_evidence`，视觉重复监督反向加权。RL 在相同 U 监督 checkpoint 上比较 lambda=1 和 lambda=2，真实概率精确枚举两个动作：

`loss = L_sup - E[r_lambda] + 0.02 KL(policy || frozen_S)`。

其中冲突正确奖励为 `0.3 * lambda * N / N_Conflict`，普通正确奖励为 `0.2 * N / N_Aligned`。RL 一轮、学习率 5e-6、weight decay 0；其他设置与监督训练一致。情感分支固定，不加入不起作用的情感奖励。

## 单卡执行与时限

全部 CUDA 工作仅在服务器物理 GPU 2，`CUDA_VISIBLE_DEVICES=2`。GPU 0、1 及其他用户进程不操作。E、U、R1、R2 串行执行，R1/R2 成对完成才进入奖励系数比较。

原始初始化起点为北京时间 2026-10-29 16:20:26，90 分钟预算到 17:50:26，修复版本不重置该时钟。候选目标 23:00，硬停止 23:30。实测预检后，以串行耗时加 30% 余量和 15 分钟导出预留，依次选择完整 S+R1+R2、仅 S、仅冻结专家；都不能完整完成时停止，不将未完成 checkpoint 用于选模。

tmux 会话为 `buaa-emotion-final`，窗口 `1:progress` 展示当前阶段、分支、步数进度条、ETA、显存、梯度及训练观察。训练观察不作开发或留出评估使用。

```bash
tmux attach -t buaa-emotion-final
# Ctrl+b，随后按 1 切换到 progress
```

## 来源与工程预检

公开 adapter 有 394 个张量，严格加载到 197 个实际语言模块，全部张量逐一相等，rank / alpha 与缩放保持原值。不能直接采用其泛化 target 名称：该配置会在没有提供 adapter 的音频、视觉、talker 中额外创建模块。

原生 Omni 加载器即使关闭语音输出也读取官方 `spk_dict.pt`，因此保留并用 restricted `weights_only=True` 验证该资产。仅下载选定权重、配置与 tokenizer，不运行上游训练脚本，不下载优化器 pickle 或公开测试标签。

原 r1 的 E 预检完成 32 样本、两次更新、峰值 24.049 GiB，但冻结参数原始字节审计失败，因此该次预检不放行。

逐参数诊断确认：首次 VT 前向有 310 个冻结 bias 从 FP32 转为 BF16，首次 VA 前向有 161 个；反向传播、优化器 step 和 adapter 保存阶段均无冻结参数变化。服务器安装的 `bitsandbytes.nn.Linear4bit.forward` 明确执行 `bias.data = bias.data.to(x.dtype)`。修复在加载时显式完成相同 BF16 转换；仍保留训练后的冻结基座 SHA 检查，没有移除或放宽该检查。

修复运行使用独立 r2 源码与结果目录，r1 的源码、失败记录和预检权重保留。r2 仅复用已核验的合并权重、解码缓存和原 CSV，不复用预检 adapter；正式训练重新初始化。

- 服务器源码：`/home/nas/xingzhenbang/buaa-ai2026/releases/emotion-expert-costrl-v1-20261029-r2`。
- 服务器结果：`/home/nas/xingzhenbang/buaa-ai2026/runs/emotion-expert-costrl-v1-20261029-r2`。
- Python：`/home/nas/xingzhenbang/buaa-ai2026/.venv-emotion-final/bin/python`。
- 模型、分区、媒体与源码身份：`context-64.json`、`data-audit.json`、`merged-emotion-base/merge.complete.json`。

32 条真实训练样本预检包括两种协议、自然与合成冲突、不可判读和真实中性；检验实际梯度/参数更新、冻结参数、显存，以及重新加载 NF4 后的 adapter、Adam、随机状态能否重现下一次更新。预检权重全部丢弃。

## 选模、交付与证据限制

开发集按固定规则分别选择 E 与 U；RL 相对最佳非 RL 不确定性分量须至少 +0.003，自然 VA 不下降。仅所选模型进入留出；RL 未复现 +0.003 时退回预先选定非 RL 基线。完整候选相对同分区 SFT 参考须综合分 +0.01、正负 F1 下降不超过 0.005，自然冲突及自然 VA 不下降。

综合诊断为 `0.5 F1_PN + 0.3 Recall_Conflict + 0.2 Specificity`，同时报告固定三类 F1、neutral 支持、自然/合成、VT/VA、误报、改对/改错和媒体分组 bootstrap 区间。既有留出参与过历史开发，因此属于单种子开发证据，不能称为独立泛化证明。

仅通过留出门槛后读取正式 666 条验证及 734 条测试。阈值固定 0.5，输出模型自身的完整预测，不混入来源未知的 submission_09 判断。yes 使用固定协议模板，no 为 none；检查 734 条覆盖、ID 顺序、字段合法性并输出逐条变化。原最高分文件保持备份，SHA 为 `f2551b6a12fc916d804287671373bee5caa6e593315725a36ca2b1b0e20b8498`。

## 验证状态

本地与服务器 13 项 unittest 已通过，覆盖奖励期望及梯度、两倍奖励仅影响冲突项、实际批次归一化、模态隔离、unclear 映射、视觉重复权重、开发选模、留出门槛、分组 bootstrap、导出一致性及显式 bias 初始化的幂等性。

两分支真实预检均已通过，于北京时间 17:49:20 完成，距原始初始化起点 88 分 54 秒，未重置 90 分钟预算。全部优化步骤都有非零梯度，adapter 确实变化，冻结基座 SHA 不变，重新加载 NF4 后下一次更新的 adapter 与 Adam 状态逐字节一致。

| 分支 | 真实训练样本 | 更新 | 平均秒/样本 | P95 秒/样本 | 峰值 GiB |
|---|---:|---:|---:|---:|---:|
| E | 32 | 2 | 1.039 | 1.279 | 24.049 |
| U（含 V/M 辅助任务） | 32 | 2 | 2.481 | 3.004 | 24.541 |

准入采用 P95 训练耗时、实际推理速度与初始化预留，并额外增加 30% 余量和 15 分钟导出预留。17:49 决策时距 23:00 约 5 小时 11 分钟：冻结专家批次约需 2 小时 19 分钟，仅 S 的完整批次约需 5 小时 54 分钟，完整奖励对照约需 12 小时 15 分钟。因此实际准入为 **frozen**：当前只进行 E0/U0 开发和留出评估，通过既定门槛才继续正式验证与测试导出；正式 SFT、R1、R2 没有启动，不将预检 adapter 当作训练成果。

真实证据为 `jobs/preflight-64-{E,U}/complete.json`、`admission.json`、`pipeline.status.json`。公开 adapter SHA256 为 `5dd6edb73c7ad81d4be717e41af5510e0b0b8766b9ff6f73979a69fa51e6c7bd`。没有新的 Kaggle 分数；通过预检仅证明实现与资源检查有效，不证明模型性能提高。

## 最终冻结专家评估（2026-10-30 取回）

服务器 `final-report.json` 与 `held-results.json` 均记录 `status=complete`、`mode=frozen`、`selected_emotion=E0`、`selected_uncertainty=U0`。情感和不确定性使用同一合并上游专家的原生概率，没有使用预检 adapter，也没有训练正式 S/R1/R2。选模记录中的 adapter 为 None。

| 同一 671 条内容分组留出 | 协议 SFT 参考 | 冻结 7B E0/U0 |
|---|---:|---:|
| PN 选择分 | **0.89515637** | 0.74306338 |
| 固定三类综合分 | **0.90529784** | 0.59411950 |
| 正负 F1 | **0.93915120** | 0.89366330 |
| 全部冲突 /82 | 65 | 78 |
| 自然冲突 /28 | 11 | 27 |
| 自然 VA /7 | 2 | 7 |
| 普通误报 /589 | **36** | 557 |
| neutral 正确 /1 | **1** | 0 |

PN 选择分相对参考下降 0.15209299。配对媒体组 bootstrap 95% 区间为 [-0.19039693, -0.11158830]，仍属于使用过历史开发分区的单种子开发证据。情感新改对21条、新改错82条；不确定性新识别17条冲突、丢失4条正确冲突，修复1条普通误报但新增522条普通误报。

实际 gate 原因为 `gain_below_0.01` 和 `emotion_regression`。没有读取正式 val/test 继续推理，没有生成候选 CSV 或新增 Public。本次模型归档保留上游来源与合并方法；不把验收 checkpoint 标成最终训练模型。指标、原报告 SHA 及权重对应范围见[模型清单](../models/registry.json)和[模型使用说明](../models/README.md)。
