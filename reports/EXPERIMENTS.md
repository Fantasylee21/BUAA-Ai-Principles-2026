# xzb 实验说明：方法、代码结构与结果

本文是 `xzb` 分支后续合入 `main` 的阅读入口，面向没有参与实验的项目成员。整理日期为 **2026-10-30**；结论以仓库中已保存的报告和本地归档为准，最新性能报告日期为 10-29。本次整理没有重新查询服务器，因此历史快照中的“运行中”不代表当前实时状态。

## 1. 本部分做了什么

任务同时要求预测整体情感、识别模态冲突或证据不足，并为不确定样本输出引导问题。实验围绕一个问题展开：**在已有 SFT 模型上，怎样补足自然模态冲突识别，同时控制普通样本误报和情感退化？**

本分支主要增加了以下尝试：

1. 改变奖励及优化方式：正确性 / F1 增量 GRPO、六动作枚举、Grounded REBEL、直接监督纠错，以及两动作冲突奖励 A/B/C 对照。
2. 显式建模模态证据：冻结特征二分类 / 九状态关系、原始基座双路证据、固定补漏、小特征校准及冻结 8B 视觉专家。
3. 改变训练协议与可训练模块：协议一致 SFT 对照、更新感知层与跨模态融合的 C0/C1，以及可训练 8B 视觉升级。
4. 实现 7B EmotionReasoner 的情感 E / 不确定性 U 分离方案，并完成真实工程预检；其正式训练和性能结果尚无归档证据。

Baseline 和早期 SFT v1–v4 是继承的比较基线，在本说明中保留用于衔接。按本地引用，开发前 `main` / `origin/main` 为 `627da44`，实验分支为 `a430c49`；后者相对前者的新增范围主要是上述方法及其报告。分支名用于说明来源，合入后本文仍作为历史实验入口。

已完成结果表明：SFT 的收益明确；多数后续方法提高自然冲突召回的同时增加了误报，尚未形成可替换现有交付模型的稳定完整系统增益。冻结 8B 视觉证据通过过当时的**分支**门槛；它不是完整系统或线上提分结论。已知最高 Public 为来源尚未追溯的 `submission_09.csv` 的 **0.92313**，不能归功于本分支任一已知训练流程。

## 2. 数据与成绩怎样读

原始训练集 3,354 条，其中冲突 411 条（自然 143、合成 268）、普通 2,943 条；原始验证集 666 条，其中冲突 74 条（自然 25）、普通 592 条；正式测试集 734 条且无公开真值。VT 输入为静音视频加文字，VA 为视频加音频；两个 VL 基线在 VA 中仅使用画面。

后续审计发现，历史 Omni 训练将 VA 构造成 V+T+A，并对配对 VA 使用 VT/VA 冲突标签并集，改变了 30 条原始标签：Omni 训练冲突变为 441 条，VL 去重后为 2,104 条 / 冲突 378 条；历史验证输入仍为 V+A，但清洗改变过 10 条 VA 标签。后期协议实验改回原始标签和合法模态。**训练 loss 的标签来源、最终生成评分的原始标签不能混为一谈。** 数据详情及全部中间节点见[历史总表](rl/all_experiments_summary_20261028.md)。

| 本文口径 | 样本与公式 | 适用结果 |
|---|---|---|
| G1：历史完整验证 | 原始 val 666；`0.5 × F1_三类 + 0.3 × 冲突召回 + 0.2 × 普通特异度` | Baseline、早期 SFT、v4 后续 RL / 证据实验 |
| G2：分组留出 | 原训练集划分 2,013 / 670 / 671；在 671 条留出上计算三类公式 | 协议 SFT 对照 |
| G3：不确定性分量 | 原训练集 3,354 条分组折外；`0.3 × 冲突召回 + 0.2 × 普通特异度`，满分 0.5 | 小校准器、冻结 8B 证据 |
| G4：完整系统选择分 | 原训练集三折，或明确标注的单折；`0.5 × F1_正负 + 0.3 × 冲突召回 + 0.2 × 普通特异度` | SFT-ref、C0/C1、可训练 8B、A/B/C 奖励 |
| G5：Public | 实际 Kaggle 提交返回值 | 仅第 8 节的线上记录 |

`Conflict accuracy` 在这些本地实现中是冲突样本上预测 yes 的比例，`Aligned calibration` 是普通样本上预测 no 的比例，即特异度；不是概率校准误差。G1 的固定三类实现见 [rl/metrics.py](../rl/metrics.py)。val 没有 neutral，固定三类 F1 与按实际出现类别计算的诊断 F1 会明显不同。G2 有 1 条 neutral，G4 三折合计只有 7 条，因此必须保留类别支持数。

**只在同一数据、同一口径、同次基线内比较差值。** 不能把 G2/G4 的约 0.90 与 G1 的约 0.71 当作提分，也不能用 G3 或 Public 代替完整本地 Score。

## 3. 代码从哪里看

代码按实验路线增量扩展，没有一个同时启动所有方法的总入口。建议先看本节对应行，再看该方法的配置、数值目标、运行时与专题报告。

| 路线 | 主要入口 / 配置 | 核心职责 |
|---|---|---|
| Baseline | [run_baselines.sh](../scripts/run_baselines.sh)、根目录三个 `qwen*_baseline.py` | 原始模型推理，保存预测、指标和运行配置；统一 SFT prompt 在 [sft_prompt.py](../sft_prompt.py) |
| LoRA / QLoRA SFT | [run_sft_training.sh](../scripts/run_sft_training.sh) → [train_sft.py](../scripts/train_sft.py)；[configs/sft](../configs/sft) | `prepare_sft_data.py` 转训练格式；`task_balanced_loss.py` 实现 v3 字段损失；`prepare_resampled_sft_data.py` 实现 v4 重采样；`evaluate_sft.py` / `select_sft_checkpoint.py` 评估选模 |
| 正确性 / F1 增量 GRPO | [train_grpo.py](../rl/train_grpo.py)、[run_rl.sh](../scripts/run_rl.sh)；`configs/rl/{correctness,f1_delta}_v1.json` | `reward.py` 定义奖励与输出契约；`experiment.py` 配置、采样；`omni_runtime.py` 原生多模态前向及策略更新 |
| 六动作 RL | [train_finite_actions.py](../rl/train_finite_actions.py)；[finite_actions_v1.json](../configs/rl/finite_actions_v1.json) | `finite_actions.py` 枚举三情感 × 两个不确定性动作；`finite_runtime.py` 算概率和梯度；`finite_store.py` 缓存；`finite_preflight.py` / `finite_reports.py` 验收汇总 |
| Grounded REBEL | [train_grounded_uncertainty.py](../rl/train_grounded_uncertainty.py)；[grounded_uncertainty_v1.json](../configs/rl/grounded_uncertainty_v1.json) | `grounded_uncertainty.py` 数值目标；`grounded_runtime.py` 主任务和单模态辅助；`grounded_recovery.py` 确定性恢复 |
| 直接监督纠错 | [train_conflict_cost.py](../rl/train_conflict_cost.py)；[conflict_cost_v1.json](../configs/rl/conflict_cost_v1.json) | `conflict_cost.py` 条件加权 CE、真实 VT/VA 配对损失；`conflict_runtime.py` / `conflict_store.py` 训练与缓存 |
| 冻结特征关系 | [train_modality_relation.py](../rl/train_modality_relation.py)；[modality_relation_v1.json](../configs/rl/modality_relation_v1.json) | `relation_runtime.py` 提取冻结特征；`modality_relation.py` 定义二分类 / 九状态目标；`relation_training.py` 只训练小头 |
| 双路证据与补漏 | [train_evidence_bottleneck.py](../rl/train_evidence_bottleneck.py)；[evidence_bottleneck_v1.json](../configs/rl/evidence_bottleneck_v1.json)；[补漏入口说明](../scripts/EVIDENCE_SUPPLEMENT.md) | `evidence_runtime.py` 隔离 V 与 T/A；`evidence_training.py` 拟合证据头 / 校准；`evidence_supplement.py` 应用固定规则；`train_evidence_calibrator.py` 训练六特征校准器 |
| 协议一致 SFT | [protocol_sft_ablation.py](../scripts/protocol_sft_ablation.py)；[qlora_sft_v5_protocol.yaml](../configs/sft/qwen25omni/qlora_sft_v5_protocol.yaml) | `data/prepare_training_data.py` 与 `prepare_sft_data.py` 支持 `protocol_consistent`；对照在独立分区训练、开发集选模、留出评估 |
| 冻结 8B 视觉专家 | [vision_expert.py](../scripts/vision_expert.py)、[vision_expert_cpu.py](../scripts/vision_expert_cpu.py)、[vision_expert_cpu_recovery.py](../scripts/vision_expert_cpu_recovery.py) | GPU 提取冻结 Qwen3-VL-8B 视觉表示；CPU 拟合证据头和校准；不更新 8B 编码器 |
| 可训练感知与融合 | [joint_affect_pipeline.py](../scripts/joint_affect_pipeline.py) → [train_joint_affect.py](../scripts/train_joint_affect.py) | `joint_affect_data.py` 分组 / 三折 / 模态重配 / 指标；`joint_affect_model.py` 感知 LoRA、查询池化、跨模态注意力、情感和关系头 |
| A/B/C 冲突奖励 | [conflict_reward_pipeline.py](../scripts/conflict_reward_pipeline.py) → [train_conflict_reward.py](../scripts/train_conflict_reward.py) | `rl/conflict_reward.py` 奖励、优势、门槛；`conflict_reward_runtime.py` 原生两动作概率；`conflict_reward_export.py` 导出 |
| 7B E/U 分离 | [emotion_expert_pipeline.py](../scripts/emotion_expert_pipeline.py)；[emotion_expert_final_v1.json](../configs/rl/emotion_expert_final_v1.json) | `emotion_expert_bootstrap.py` 严格加载 / 合并公开 adapter；`rl/emotion_expert.py` 数值目标 / 选模 / 导出；`emotion_expert_runtime.py` 媒体及人脸输入；`train_emotion_expert.py` 工作器 |

公共支撑：`rl/omni_io.py` 构造 VT/VA 和隔离模态输入；`rl/state.py` 保存 adapter、优化器、scheduler、RNG 和身份校验；`rl/cache_io.py` 支持并发读取已完成的不可变媒体缓存；`rl/reference_cache_recovery.py` 保留参考缓存恢复来源。`*_progress.py` 和 `*_dashboard.py` 是只读进度工具。数值目标、模态隔离、恢复、缓存与导出检查在 [tests](../tests) 和 `rl/test_*.py`。

目录边界：源码在 `rl/`、`scripts/`、`configs/`；可随 Git 阅读的报告在 `reports/rl/`、`reports/sft/`；部分早期原始指标在 `output/baseline/` 和 `output/sft/evaluation/`。原始课程数据 `Dataset/`、权重 `models/`、完整运行记录 `runs/` 与外部 `third_party/LlamaFactory/` 不随 Git 分发。根目录 `multimodal_affect_sft_rl.py` 是早期 notebook 导出脚本，正式复现应使用上表入口。

## 4. 继承基线：Baseline、LoRA 与 QLoRA SFT

三个模型分别为 Qwen2.5-VL-3B、Qwen3-VL-4B、Qwen2.5-Omni-3B。LoRA r16 / alpha32 / dropout 0.05，只更新语言 adapter；QLoRA 冻结基座以 NF4 加载、BF16 计算。共同设置包括有效 batch16、LR1e-4、3 epoch。Omni 原 QLoRA v1 因 24 GB 显存不足未完成，v2 将每帧像素降至 50,176（约 64 visual tokens）。

v3 从原始基座开始，用字段平衡损失 `0.5 L_emotion + 0.4 c(y)L_uncertainty + 0.1 L_rest`，Conflict 的 `c(y)=2`；v4 保留该损失，再将 Conflict 总采样量增至 2 倍，也从基座开始。v4 同时增加了训练步数，因此不能把全部差异只归因于采样。

| G1：同 prompt 完整验证 Score | Qwen2.5-VL-3B | Qwen3-VL-4B | Omni-3B |
|---|---:|---:|---:|
| 原始基座，BF16 | 0.44186 | 0.45855 | 0.44366 |
| BF16 LoRA v1，合并后复评 | 0.6700 | 0.6658 | 0.7092 |
| QLoRA v1 / Omni v2，BF16 基座推理 | 0.65904 | 0.67497 | 0.70761 |
| 同一 QLoRA adapter，NF4 推理 | 0.67842 | 0.70619 | 0.70871 |
| 字段损失 QLoRA v3 | 0.68797 | **0.72008** | **0.72882** |
| Conflict 2 倍采样 v4，历史报告 | **0.6952** | 0.7110 | 0.71846 |

SFT 明显优于原始模型；v4 并非三个模型都优于 v3。Omni v4 的交付 adapter 是 `checkpoint-714`，后续 RL 新环境中的同一权重复评为 **0.71267263**，不能与历史值 0.71846095 混作同环境基线。权重身份见[v4 核验](rl/v4_artifact_verification.md)；详细参数、逐轮结果和证据见[SFT 总表](sft/README.md)、[v3](sft/qlora_sft_v3.md)、[v4](sft/qlora_sft_v4.md)。

## 5. 从 Omni v4 继续：奖励、纠错和证据建模

### 5.1 更新语言 LoRA 的方法（G1）

这些方法使用 NF4 冻结基座、更新语言 LoRA，视觉 / 音频编码器和连接层保持冻结；通常 LR5e-6、KL0.02、seed42。各行的量化、输入与基线以专题报告为准。

| 方法 | 做法与实际完成量 | 已完成结果 | 结论 / 证据 |
|---|---|---|---|
| 同环境 v4 参考 | 自由生成，666 条复评 | Score **0.71267**；自然冲突 6/25，普通误报 26/592 | 后续比较基线 |
| 正确性 GRPO | 每步 4 冲突 + 4 普通，每输入 4 回答；完成 200 步 | 100 / 200 步：0.70779 / 0.70812；200 步自然冲突仍 6/25，误报 41 | 退化，未晋级；[双奖励结果](rl/dual_reward_20261022.md) |
| F1 增量 GRPO | 将情感奖励改为固定训练混淆矩阵的 F1 增量；完成 200 步 | 100 / 200 步：0.70926 / 0.70825；200 步自然冲突 6/25，误报 39 | 未改善；约 90% 采样组同奖励；[双奖励结果](rl/dual_reward_20261022.md) |
| 六动作期望奖励 RL | 枚举三情感 × yes/no，完成 210 步、全训练集一遍 | 末步 **0.71442**，相对自由生成 +0.00175；自然冲突 7/25，误报 33 | bootstrap 区间跨 0，未达 +0.005；[有限动作审计](rl/finite_action_audit_20261022.md) |
| Grounded REBEL | 固定情感，用原生 yes/no logits 与单模态辅助；正式完成 80 步 | 40 / 80 步：0.71166 / 0.71098；自然冲突 6/25，80 步误报 31 | 预检可学习，正式早停；不是完成 200 步；[目标与故障](rl/grounded_uncertainty_audit_20261023.md)、[恢复](rl/grounded_recovery_fix_20261023.md)、[80 步记录](rl/all_experiments_summary_20261028.md) |
| 代价敏感直接纠错 | 固定情感，条件加权 CE + 真实 VT/VA 相反标签配对；完成 210 步 | 105 / 210 步：0.70456 / **0.71301**；末步自然冲突 6/25，误报 25 | 仅净少 1 条误报，首轮后停止；[直接纠错审计](rl/conflict_cost_audit_20261024.md) |

六动作零更新解码为 0.70535，直接纠错新提示零更新为 0.70456。二者更大的“相对零更新收益”包含追回解码 / 提示变化造成的损失，不能全部算成超过自由生成 v4 的收益。恢复预检、OOM 和缓存修复属于工程记录，不重复计为新算法结果。

### 5.2 冻结主干、只拟合关系或证据头（G1）

这些方法属于监督小模型 / 校准或固定决策规则。最终情感沿用 v4，情感正确数均为 631/666；结果变化来自不确定性判断。

| 方法 | 做法 | Score | 自然冲突 /25 | 普通误报 /592 | 结论 / 证据 |
|---|---|---:|---:|---:|---|
| 冻结 v4 二分类头 | 881,666 参数 MLP；三折开发选 epoch6 后全量拟合 | 0.71166 | 6 | 41 | 无提升；[关系模型](rl/modality_relation_audit_20261024.md) |
| 冻结 v4 九状态关系头 | 882,569 参数，3×3 联合状态；同样选 epoch6 | 0.71301 | 6 | 37 | 增益很小；[关系模型](rl/modality_relation_audit_20261024.md) |
| 原始基座双路证据，未校准 | 严格隔离 V 与 T/A，训练残差证据头；`q=1-pV·pM` | 0.70288 | 18 | 115 | 召回增加伴随误报；[证据审计](rl/evidence_bottleneck_audit_20261025.md) |
| 同上，加成本校准 | 训练折外概率拟合 VT/VA 偏置 | **0.71571** | **24** | 149 | 相对 v4 +0.00304，未达该批 +0.015 门槛；[证据审计](rl/evidence_bottleneck_audit_20261025.md) |
| v4 保留 + 分歧门控补漏 | 保留已有 yes，仅对两路 argmax 分歧且概率 >0.5 的样本补 yes | 0.73565 | 18 | 102 | 看过验证结果后形成的诊断规则，不能视为独立验证收益；[证据审计](rl/evidence_bottleneck_audit_20261025.md) |

冻结 v4 小头的折外划分只隔离新小头，v4 主干已经见过原训练集，不能把它当作完整系统的独立留出。双路证据虽能识别更多自然冲突，普通误报代价仍是主要问题。

### 5.3 分组校准、协议修正与冻结 8B

| 方法 | 训练 / 评估设计 | 同口径结果 | 结论 / 证据 |
|---|---|---|---|
| 六特征逻辑校准 | 两路分布距离、margin、熵及协议；外层3折 / 内层5折，L2=.01 | G3 内容分组 0.36791893 → 0.37064220；自然冲突 80 → 71/143；父目录压力 0.34212222 → 0.33364237 | 自然召回下降，未晋级，未跑 val/test；[补漏审计](rl/evidence_supplement_audit_20261025.md) |
| 协议一致 SFT 对照 | 从基座各真实重训 3 epoch；相同 2,013 / 670 / 671 内容分区；历史协议 vs 合法 VA 输入与原始标签 | G2 留出 **0.90358657 → 0.90529784**，+0.00171127；情感正确 629 → 631/671；自然冲突均 11/28；误报 38 → 36/589 | 未达 +0.01；未运行 seed43 或最终全量 v5；[实施](rl/server_training_implementation_20261025.md)、[结果](rl/training_results_review_20261026.md) |
| 冻结 Qwen3-VL-8B 视觉专家 | 冻结 8B 提取 2,104 个独立视频表示，仅训练证据头与校准；对照共享 Omni T/A 分支 | G3 内容分组 **0.36791893 → 0.37848017**，+0.01056125；自然冲突 80 → 84/143；误报 558 → 510/2943。父目录仅 +0.00010318 | 通过当时分支门槛；主分组差值区间 [+0.000860,+0.020473]，压力区间跨0；无完整系统或 Public 结果；[结果](rl/training_results_review_20261026.md) |

这三项分别检验决策校准、训练协议、视觉证据质量。冻结 8B 的小头收益不能代替下一节端到端融合模型的实测结果。

## 6. 更新感知与跨模态融合的完整系统

### 6.1 SFT-ref、C0 与 C1（三折 G4）

C0/C1 从原始基座开始，V 与 T/A 两路严格隔离输入；SFT-ref 仍按 VT=V+T、VA=V+A 生成 JSON。C0/C1 训练语言 LoRA、视觉 / 音频最后 4 个 block 的 LoRA、连接层、16 查询池化、两层跨模态注意力及分类头，重新预测整体情感。C0 直接做关系二分类，C1 建模含 unclear 的四态 × 四态联合关系；unclear 不等于 neutral。C0/C1 各折先 1 epoch 单模态证据学习，再 2 epoch 联合训练，联合阶段加入训练折内模态重配。

| 三折折外，3,354 条 | 训练参数 / 每折更新 | G4 选择分 | 正负 F1 | 自然冲突 /143 | 自然 VA /36 | 普通误报 /2943 |
|---|---|---:|---:|---:|---:|---:|
| 协议一致 SFT-ref | 29,933,568 / 471 步 | **0.88316** | **0.94033** | 46 | 7 | **131** |
| C0：二分类关系 | 76,713,100 / 420 步 | 0.87393 | 0.92911 | **76** | 11 | 485 |
| C1：16 态关系 | 76,716,955 / 420 步 | 0.86711 | 0.92749 | 74 | **12** | 509 |

C0/C1 三折均下降，未晋级。自然冲突分别净增 30 / 28 条，但误报净增 354 / 378 条。C1 另有必须保留的正结果：固定三类 F1 **0.86075**、三类加权分 **0.83374**，高于参考 0.77504 / 0.80051，主要来自 neutral 正确数 2/7 → 4/7；这没有改变事先锁定的 G4 选择结果。

实现与实际训练审计见[联合模型实施](rl/joint_affect_implementation_20261026.md)，分层结果、逐条变化与归档校验见[结果复核](rl/joint_affect_results_review_20261028.md)。

### 6.2 可训练 8B 视觉 C0：只完成首折

视觉升级为 Qwen3-VL-8B，第二路从同折原生 C0 checkpoint 初始化；实际训练参数 244,464,140。**这与第 5.3 节冻结 8B 是两个实验。**

| 同一 fold0，1,118 条 G4 | 选择分 | 正负 F1 | 自然冲突 /48 | 普通误报 /981 |
|---|---:|---:|---:|---:|
| SFT-ref | **0.88746** | **0.95011** | 14 | **43** |
| 原生 C0 | 0.87970 | 0.94085 | **25** | 155 |
| 可训练 8B 视觉 C0 | 0.86241 | 0.94182 | 19 | 167 |

首折 420 步完成，未显示整体收益。后续记录确认旧批次已停止，fold1 / fold2 分别保存于 263 / 16 步；没有完整三折 8B 结论。10-28 的早期报告中“继续运行”是当时状态，停止记录见[冲突奖励实施记录](rl/conflict_reward_implementation_20261028.md)。父目录压力、seed43、全量最终训练和正式测试均无完成结果。

## 7. 最新两组尝试

### 7.1 两动作冲突奖励 A/B/C：首折完成，未晋级

从同 fold0 的 SFT-ref 开始，只更新 29,933,568 个语言 LoRA 参数；情感逐条固定。A 用原任务权重，B 将冲突奖励乘 2，C 在 B 上加入零优势组监督纠错和普通正确判断保护。每输入采样 8 个 yes/no 动作，以 leave-one-out 优势更新；75% 探索锚点不等于校准置信度。三组各在 2,236 条训练样本上完成 2 epoch / 280 次更新，没有额外冲突过采样。

| 同折 1,118 条 G4 | 选择分 | 相对 SFT-ref | 冲突 /137 | 自然冲突 /48 | 普通误报 /981 |
|---|---:|---:|---:|---:|---:|
| SFT-ref | **0.88745797** | — | **101** | **14** | 43 |
| A：原权重 | 0.88430165 | −0.00315632 | 99 | 13 | 37 |
| B：冲突奖励 2 倍 | 0.88470940 | −0.00274857 | 99 | 13 | **35** |
| C：2 倍 + 纠错保护 | 0.88725410 | −0.00020387 | **101** | **14** | 44 |

三组均未达到 +0.003 首折门槛，按 `pilot_futility_gate_not_met` 正常结束。自然 VA 均为 2/12，没有新增正确冲突。A/B 全程约 90% 组零优势，C 降至约 50%，但仍未带来留出收益；三组分差 bootstrap 区间均跨 0。后两折、seed43、全量训练及正式 val/test 未执行，未生成候选。详见[设计](rl/conflict_reward_implementation_20261028.md)和[结果复核](rl/conflict_reward_results_review_20261029.md)。

### 7.2 7B EmotionReasoner E/U 分离：已实现并预检，缺最终性能结果

方案以 Qwen2.5-Omni-7B 加公开 EmotionReasoner `checkpoint-195` 为起点，严格合并后冻结基座与感知模块，分开训练情感 E 和不确定性 U 两个语言 LoRA。U 另有隔离 V、T/A 四态监督；计划比较监督基线 S 与精确枚举两动作期望奖励的 R1 / R2。沿用 2,013 / 670 / 671 分区，补充四个时点的人脸裁剪，单卡 GPU2 串行执行。

| 本地已保存证据 | E | U |
|---|---:|---:|
| 真实预检训练样本 / 更新 | 32 / 2 | 32 / 2 |
| 峰值显存 GiB | 24.049 | 24.541 |
| 参数更新、冻结基座校验、重建后恢复 | 通过 | 通过 |

因实际耗时与原定截止时间，最终准入选择 **frozen（冻结专家 E0/U0 评估）**；正式 SFT、R1、R2 没有启动，预检 adapter 不作成果模型。本地最后快照为 **2026-10-29 17:49 左右的 dev-E0 执行状态**，未取得开发 / 留出 / 正式验证 / 测试完成指标。当前只能记录“工程预检通过、性能结果待补”，不能声称已经完成 7B 奖励对照、达到 0.95 或仍在实时运行。[实现、预检和准入依据](rl/emotion_expert_implementation_20261029.md)。

## 8. 线上结果单列

| 提交 / 方法 | Public | 解释 |
|---|---:|---|
| Omni QLoRA v2，NF4 | 0.90644 | 历史提交截图记录 |
| Omni QLoRA v3，checkpoint-630 | 0.90386 | 历史提交截图记录 |
| Omni QLoRA v4，resample2x | 0.90813 | 当前可追溯的 SFT 交付方案；不能替代本地验证结果 |
| submission_09.csv | **0.92313** | 文件身份已核验，但原始生成代码 / 模型路径未知，不能认定为某一种新训练方法 |
| 在 submission_09 上应用固定证据补漏 | 0.90175 | 保留全部情感和原 104 条 yes，新增 106 条 yes；真实提交下降 0.02138，保留原文件 |

其他来源未知的提交及截图分数完整保存在[历史总表第 8 节](rl/all_experiments_summary_20261028.md)；固定补漏线上差异见[补漏审计](rl/evidence_supplement_audit_20261025.md)。后续 RL、协议对照和联合模型没有新 Public 结果。**734 条无标签测试 CSV 的覆盖与格式检查不能产生准确率或推算 Public；0.95 尚无已验证结果。**

## 9. 怎样复现与交接

### 运行入口

先按[根 README](../README.md)准备课程数据、基座、兼容的 Linux / CUDA / ffmpeg 环境；SFT 另需锁定的 LLaMAFactory，RL 依赖说明见 [rl/requirements.txt](../rl/requirements.txt)（候选清单，不是所有路线通用的 GPU 环境锁）。GPU 编号和绝对路径使用实际分配值。

已有 Omni v4 adapter 的基础 RL 配置核查入口如下；它检查配置与数据，不加载权重、不开始训练：

```bash
python -m rl.train_grpo --dry-run \
  --data-root /actual/course/Dataset \
  --model-path /actual/models/Qwen2.5-Omni-3B \
  --sft-adapter-path /actual/adapters/omni-v4-checkpoint-714 \
  --run-dir runs/rl-config-check
```

真实 smoke、训练、断点恢复与评估命令见 [rl/README.md](../rl/README.md)；有限动作、REBEL、直接纠错、关系和证据分别有 [FINITE_ACTIONS](../rl/FINITE_ACTIONS.md)、[GROUNDED_UNCERTAINTY](../rl/GROUNDED_UNCERTAINTY.md)、[CONFLICT_COST](../rl/CONFLICT_COST.md)、[MODALITY_RELATION](../rl/MODALITY_RELATION.md)、[EVIDENCE_BOTTLENECK](../rl/EVIDENCE_BOTTLENECK.md) 说明。新流程的参数入口可从仓库根目录查看：

```bash
python -m scripts.protocol_sft_ablation --help
python -m scripts.joint_affect_pipeline --help
python -m scripts.conflict_reward_pipeline --help
python -m scripts.emotion_expert_pipeline --help
```

`--help` 仅展示接口。复杂 pipeline 依赖前序资产，不代表 clone 后可以直接完整执行：

- A/B/C 的 prepare 依赖原 joint-affect 的分组计划、各折 SFT-ref checkpoint 和留出预测，并核验基线 CSV 身份。
- 7B prepare 依赖原协议 SFT 的 split manifest、三个分区及参考预测。其配置与 `cutoffs()` 固定为 **2026-10-29 的历史截止时间**；原脚本作为执行快照保留，重做实验应另立版本及预算。
- 冻结 8B 需要原证据实验的特征与分组身份；C0/C1 需要真实多模态输入和完整 GPU 依赖。CPU 数值测试不能代替真实 GPU 验收。

### 证据与交付范围

Git 中的文档能独立说明方法、结果、失败和未完成项；大权重、视频及全部逐条运行证据另存。三类证据的边界如下：

| 证据 | 随 Git 可读的内容 | 精确复核还需要什么 |
|---|---|---|
| 早期 Baseline / QLoRA / v3 | `output/baseline/`、`output/sft/evaluation/` 的原始小指标 JSON；18 份完整 666 条验证记录 | 对应权重、环境与媒体；无标签 test 指标不包含任务 Score |
| LoRA v1 / v4 历史结果、Public | `reports/sft/` 与交付核验报告 | 部分原始训练指标或截图需向执行者取得；不能把报告值写成仓库已有全部原件 |
| 后续 RL / 校准 / 联合模型 | `reports/rl/` 已提交的审计结论、数值、来源及归档 SHA | 本地被忽略的 `runs/` 归档和服务器 checkpoint；Git clone 不含完整原件 |

交接时按对应报告寻找归档。关键位置包括 `runs/finite-v1-20261022-r3/`、`runs/server-results-20261026-r3/`、`runs/joint-affect-results-20261028/`、`runs/conflict-reward-results-20261029/` 及 `runs/emotion-expert-costrl-v1-20261029-r2/`；这些是外部证据定位，不是承诺仓库内存在的下载链接。联合模型与 A/B/C 结果报告分别保存了归档 SHA 和独立复算范围。

尚待补齐的是 7B 后续完成报告、部分早期原始指标以及最高 Public 文件的生成链路。可训练 8B 其余折未完成，若另行恢复实验，需补充结果与来源。在这些证据补齐前，合入 `main` 的含义是**保存可阅读的方法尝试、代码和已有结果**，并不代表所有路线均已训练完成或优于现有交付模型。
