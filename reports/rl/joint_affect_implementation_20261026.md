# Joint affect：训练与交付编排

本轮实现改变可训练的感知与融合能力，并通过 TRAIN 内分组对照决定是否进入全量训练。代码完成不代表训练成功或榜单提高。目前已确认的历史 Public 最佳仍为 **0.92313**，目标 **0.95**；新模型未提交 Kaggle，不声称任何新 Public 分数。

## 固定实验顺序

1. `prepare` 冻结两个原始基座的权重分片、配置与 tokenizer SHA、实际运行时版本、LlamaFactory 源文件 SHA、训练源码与问题目录；每个 child 启动前复核。数据部分只读取原始 `Dataset/train.jsonl` 和对应媒体，冻结 3,354 条原始记录、媒体 SHA-256、内容分组三折与父目录压力三折。训练 seed 42/43 共用 split seed 42。准备阶段不读取正式 val/test。
2. 在正式拟合之前，对 C0、C1 各进行实际 32 条样本的预检，执行反向传播与优化更新，要求有限损失、模块梯度与更新审计、实际媒体审计及每卡峰值保留显存不超过 40 GiB。预检参数不进入正式训练。
3. 各内容折从原始基座训练 SFT-ref、C0、C1。SFT-ref 固定 3 epoch，C0/C1 固定 1 epoch 证据预热加 2 epoch 主训练，末轮固定评估一次。原始主任务标签参与训练，合成重组数据只来自该折训练部分，不捏造 M12。
4. 按三折汇总概率重算的 `selection_score` 选择 C0/C1；平分选 C0。随后只替换该架构的视觉分支为 Qwen3-VL-8B，在相同三折上评估；只有汇总分严格提高才选择 8B，平分保留 Omni。晋级门槛统一相对 SFT-ref 检查最终赢家。
5. 内容门槛通过后，运行父目录分组的赢家与参考对照；压力门槛通过后再以相同内容折、training seed 43 复验主门槛。
6. 全部通过后，训练全量 seed 42/43 两个赢家，冻结两个 `train.complete.json` 的 SHA 和 checkpoint SHA。锁定后才对正式 666 条验证与 734 条测试执行推理。两个模型的情感及不确定性概率分别取算术平均；情感取 argmax，不确定性使用 `>= 0.5`，不使用投票或验证集阈值扫描。

8B 使用同一 grouping、fold、seed 的原生 Omni 赢家 checkpoint 初始化文本/音频分支，并保留初始 checkpoint 与训练 marker 的哈希。内容 seed 42 已有这些模型；父目录、seed 43、全量阶段若尚无对应原生模型，会先训练该原生模型再初始化 8B。8B 的视觉基座从原始 Qwen3-VL 权重载入，旧视觉适配器不迁移。所有 8B 的预检、训练和推理都携带同一个 `--init-second-from`，由模型/训练模块校验映射与身份。

## 冻结门槛

`selection_score = 0.5 × F1(negative, positive) + 0.3 × 冲突召回 + 0.2 × 普通样本特异度`。各折只作分层检查，主分数使用覆盖原始 TRAIN 的汇总预测重新计分，不平均各折 F1。

| 检查 | 要求 |
|---|---|
| 内容汇总增益 | 相对 SFT-ref 至少 0.01 |
| 折间一致性 | 至少 2/3 折不退化 |
| 情感 F1(PN) | 下降不超过 0.005 |
| 自然冲突 | 正确识别至少增加 5 条；原始支持 143 条 |
| VA 自然冲突 | 正确识别至少增加 2 条；原始支持 36 条 |
| 普通误报 | 增量不超过 `floor(0.01 × N_aligned)` |
| 父目录压力 | 不确定性分量增益至少 0.005 |
| seed 43 | 重新满足全部内容主门槛 |

固定三类 Macro-F1 与原比赛加权分保留为诊断。稀少/缺失 neutral 对绝对分的影响应单列，不能用不同标签支持的数据集之间的绝对分差声称提升。正式验证结果仅报告，不能决定模型、epoch 或阈值。SFT-ref 的生成结果转换为 one-hot 概率，以便统一计分；不能将其称为已校准概率。

## GPU、期限与失败

- 仅允许物理 GPU 1、2。原生 Omni 任务每次占一张卡，两项独立折任务可并行；8B 原子占用两张卡，避免两个进程各占一张而死锁。每个 child 都检查所分配卡的现有显存占用，不终止其他任务。模型及参考训练的 40 GiB 上限也必须由训练进程实际执行。
- 首次 `run` 启动持久 `clock.json`，总预算为 120 小时。进程内使用 monotonic 计时，恢复时受首次绝对 deadline 和已扣除预算共同约束；停机时间不会增加期限。`prepare` 不启动计时。
- 只允许一次全局 64→32 帧回退，且只由明确 CUDA OOM 触发。无论发生于预检还是正式任务，旧 64 帧 cohort 的模型、预测、错误均保留；新 32 帧 cohort 重新完成全部对照，不复用旧 cohort 结果。期限保持不变。32 帧再次 OOM、非 OOM、数据/源码/预测身份错误或超时均停止，保存失败报告；不自动放松门槛。
- 检测 Reference OOM 时读取本次启动新增的嵌套训练日志，防止父进程仅报通用错误时漏检，也防止历史 OOM 日志污染本次故障分类。
- controller 锁阻止重复启动。恢复会核对未退出 child 的 PID 与 Linux process start ticks，等待原进程退出，不重复发起 GPU 训练。完成 marker、预测 checksum、输入 ID 顺序及来源源码均重新核对。预测文件存在但没有 trainer completion 时，不当作完整结果。
- 普通 controller 中断后可重新执行相同 `run`，在原期限内复用通过校验的完成任务。若已保存单任务失败，必须明确 `--resume-failed` 才允许恢复尚未形成终结报告的任务。已生成最终失败/COMPLETE 的批次视为不可变审计记录，恢复方案需独立评审；不能直接覆盖其错误或最终报告。

## 执行入口

在服务器完整源码与固定依赖准备好后，由主代理在 tmux 中执行；以下命令不执行提交。`PROJECT_ROOT` 必须是含 Dataset/models/third_party 的实际项目根目录，输出位于其被忽略的 runs 子目录。

```bash
PROJECT_ROOT=/absolute/server/project
RUN_DIR="$PROJECT_ROOT/runs/joint-affect-v1-20261026"
cd "$PROJECT_ROOT/releases/joint-affect-v1-20261026"
.venv-joint-affect/bin/python -B -m scripts.joint_affect_pipeline prepare --project-root "$PROJECT_ROOT" --run-dir "$RUN_DIR"
.venv-joint-affect/bin/python -B -m scripts.joint_affect_pipeline run --project-root "$PROJECT_ROOT" --run-dir "$RUN_DIR"
```

只读查看阶段、活动任务的真实 child 状态与预算：

```bash
.venv-joint-affect/bin/python -B -m scripts.joint_affect_pipeline progress --project-root "$PROJECT_ROOT" --run-dir "$RUN_DIR"
```

`clock.budget_seconds_available_now` 表示剩余预算，不是整批训练剩余时间。训练 child 的状态可包含自己的训练 ETA；进入推理/评估后不得将旧训练 ETA 当作当前阶段 ETA。源码在 `prepare` 后冻结；更改源码必须使用新批次和新的审计方案，不直接沿用旧身份。

## 结果与取回

`FINAL_REPORT.json` 区分门槛未通过、全量模型和锁后推理完成、运行失败。`COMPLETE.json` 的 `delivery_complete=true` 仅表示报告与取回包已完整生成；必须同时看 `outcome`、最终报告的 `reason` 与 `final_model_trained`。它不代表门槛通过或 Kaggle 提分。

取回 `review-results.tar.gz` 和 `COMPLETE.json`。包内有逐文件 SHA 的 `delivery-manifest.json`、计划、预算、汇总指标、折指标、预检审计、模型身份、错误以及存在时的锁定集成结果和 `candidate.csv`。模型权重、优化器、媒体、特征、原始 TRAIN manifest 和完整训练日志留在服务器；完整日志可能含课程样本，不应公开上传。候选保持 `sample_id,pred_emotion,is_uncertain,clarifying_question`，734 条原顺序，问题采用固定 VT/VA 模板且不包含 ID 或来源标签。

当前单元验证覆盖模拟 child 的完整闭环、门槛拒绝、平分规则、8B 初始化/资源、锁定后正式推理、结果 checksum、非完整预测拒绝、源码变化拒绝、预算恢复、OOM 整轮重启与不包含权重的归档。**这些 CPU 编排测试不能证明真实 HF 模型可加载、显存充足或模型质量提高。** 真正的 32 条预检通过前不能将本轮标为可训练验证完成。

课程规则仍适用：[官方 Rules](https://www.kaggle.com/competitions/video-emotion-uncertainty-2026/rules)、[官方 Overview](https://www.kaggle.com/competitions/video-emotion-uncertainty-2026/overview)。本流程不获取测试标签、不人工逐视频标注测试、不根据 sample ID 硬编码标签、不执行 Kaggle 提交。


## 已完成的服务器开发预检

2026-10-26 使用独立 `.venv-joint-affect` 入口（只读复用已固定依赖，不修改旧环境）和物理 GPU 1、2；GPU 0 未操作。开发诊断位于 `/home/nas/xingzhenbang/buaa-ai2026/runs/joint-affect-dev-20261026`，正式比较不复用诊断参数。

| 32 条真实样本检查 | C0 | C1 | C1 + 8B 视觉 |
|---|---:|---:|---:|
| 可训练参数 | 76,713,100 | 76,716,955 | 244,467,995 |
| 固定样本主目标，优化前 | 2.91315 | 4.71419 | 4.11390 |
| 固定样本主目标，优化后 | 2.15406 | 3.44559 | 3.12779 |
| 峰值保留显存 / GiB | 7.506 | 7.506 | 12.424 |

每个目标参数组均有非零梯度和真实参数变化：语言 LoRA、视觉/音频最后四 block LoRA、视觉/音频连接层、两路池化、融合、情感、关系及辅助头。该下降仅表示训练链路有效，不表示验证、Kaggle 或泛化提升。首个开发尝试发现 HF 处理器保留 `video_metadata`，已通过先验证再移除辅助字段修复；失败记录仍保留。

8B 开发预检还发现 HF 5.8 在显式 text_kwargs 下保留 token 列表，已对已知整数输入字段显式转为张量，并增加回归覆盖。8B 开发预检使用原始 Omni 第二分支，仅验证结构和梯度；正式 8B 预检及训练仍必须从对应训练折的原生模型初始化第二分支。开发预检数据支持来自训练折，不用于模型选择。

新增代码的 34 项 CPU 数据、训练状态、流程、运行时和面板测试通过；服务器另有 10 项 Torch 模型测试通过。复用的协议数据与 SFT 模块 22 项回归也通过（Windows 临时目录改到项目 tmp 后运行，未改旧测试源码）。三次真实预检审计已取回到本地忽略目录 `runs/joint-affect-preflight-20261026`。

正式入口为 `bash scripts/run_joint_affect_server.sh`，会启动 `buaa-joint-affect`，`pipeline` 窗口执行批次，`progress` 窗口显示真实进度条、阶段、损失、训练单模态准确率、自然 VA 训练观察、模块梯度、显存和当前任务 ETA。TRAIN 观察不称为折外成绩。所有候选均另有独立混淆矩阵复算和逐条预测；训练权重与优化器留在服务器。源码/依赖/数据冻结后不原地替换。
