# Omni 7B 的 NF4 RL 方法

## 目标与起点

从 SFT v5 按验证分选出的 `checkpoint-118` 开始，用 Qwen2.5-Omni-7B 处理 VT 和 VA
多模态输入。基座使用 bitsandbytes NF4、double quant，计算使用 BF16；仅更新语言
LoRA 参数，视觉、音频及 projector 参数冻结。推理也使用 NF4，以匹配 SFT 基线。

每条预测包含情感、是否不确定以及必要的澄清问题。评分分量为情感 Macro-F1
0.5、Conflict 召回 0.3、Aligned 校准 0.2。训练只使用训练集标签，验证集用于
checkpoint 选择；测试集无标签。

## 采样与奖励

`train_omni_v5_nf4_natural.py` 使用四卡同步训练，每个优化步由四个进程分别处理
一条样本：自然 VT 冲突、自然 VA 冲突、合成冲突、Aligned 各一条。数据组在训练
前按 seed 洗牌，组内循环取样；自然 VA 数量较少，运行较长时会重复。另一版
`train_omni_v5_nf4_conflict.py` 使用 3:1 的冲突与一致采样，自然与合成冲突交替。

对同一输入按 temperature 0.9、top-p 0.95 采样多个回答。每个回答计算：

- 情感预测正确时，依据类别频率给予加权奖励；该逐样本奖励是 Macro-F1 的近似，
  不等于直接优化全数据集 Macro-F1。neutral 样本稀少，其权重做了上限控制。
- 冲突样本输出 yes、Aligned 样本输出 no 时分别给予按采样频率调整的 0.3、0.2
  奖励；格式无效有惩罚，问题与 yes/no 一致有小额奖励。
- 一组回答的优势为各自奖励减去组内平均奖励，用优势加权回答 token 的损失；
  奖励全部相同的组不产生任务梯度。

各进程的 LoRA 梯度做 all-reduce，梯度裁剪后通过 AdamW 更新。此实现属于采样式
组相对策略梯度，没有显式 reference KL、PPO ratio clipping 或价值网络；不能把它
描述为包含这些机制的完整标准 GRPO。四卡轮次会分别记录每组的奖励和是否产生
非零优势，便于检查训练信号。

## 奖励函数的具体计算

对解析合法的回答，奖励是情感、不确定性和问题一致性三项之和：

```text
R = R_emotion + R_uncertainty + R_question
```

情感项仅在预测类别与综合情感标签相同时非零。设该标签的有效训练频率为
`p_emotion`，则正确时 `R_emotion = 0.5 / (2 * p_emotion)`，否则为 0。neutral
频率使用其自身频率和 positive 频率的较大值，避免极少量 neutral 样本产生过大权重。
这里类别频率来自原始训练池；分层采样会改变实际抽样的联合分布，因此这是用于优化
的近似奖励，不能等同于对 Macro-F1 的无偏估计。

不确定性项分别按两个类别的采样概率归一化。自然冲突版每步三条 Conflict、一条
Aligned，因此：

```text
Conflict 且输出 yes：R_uncertainty = 0.3 / 0.75 = 0.4
Aligned 且输出 no： R_uncertainty = 0.2 / 0.25 = 0.8
其余情况：          R_uncertainty = 0
```

按这个采样比例取期望，两组不确定性正确奖励分别对应 0.3 和 0.2；提高 Conflict
出现次数改善的是样本覆盖，并不自动把目标中的 Conflict 分量提高到 75%。问题一致
时 `R_question = +0.02`，不一致时为 `-0.02`。若回答无法解析为合法格式，总奖励
为 `-1`，不再计算其余三项。实现见 `rl/reward_metric.py`。

## 一次更新的计算过程

对每个输入生成 `G=4` 个回答，计算每个回答的奖励 `R_i`，再计算：

```text
group_mean = (R_1 + R_2 + R_3 + R_4) / 4
advantage_i = R_i - group_mean
loss = sum(advantage_i * completion_cross_entropy_i / 4)
```

交叉熵只计算生成部分，prompt 部分被掩码。奖励较高的回答获得正优势，梯度下降会
提高其生成概率；负优势产生相反方向的更新。若四个回答奖励相同，优势全为 0，
该样本不产生任务梯度。四卡各自计算梯度后取平均；至少有一个进程产生任务梯度时
执行一次 optimizer step。梯度裁剪上限为 1.0，optimizer 为 AdamW。

例如四个回答的奖励为 `[0.4, 0.4, 0.8, 0.8]`，组平均奖励是 0.6，优势为
`[-0.2, -0.2, 0.2, 0.2]`。这说明模型学习的是同一输入下回答之间的相对奖励，
而不是把单条奖励直接当作梯度。

## 运行与模型选择

要求准备数据、Omni 7B 基座、`third_party/LlamaFactory`、llama-factory 环境及 SFT
v5 checkpoint。训练和生成产物、权重和原始数据不包含在这些代码提交中；本机旧产物
位于 `/home/djy/lgq/buaa/sft-worktree/`，在本仓库运行前需按脚本默认路径准备对应
文件。注意不要让两个任务同时占用同一组 GPU。

```bash
cd /home/djy/lgq/buaa/BUAA-Ai-Principles-2026
# SFT v5: data preparation, training and validation are described in the SFT report.
bash scripts/run_omni_nf4_conflict_gpu4567.sh
# Separate natural-conflict experiment:
bash scripts/run_omni_nf4_natural_gpu4567.sh
```

自然冲突版默认 40 步、每步四条样本、每条生成四个回答，学习率 `1e-6`；第
10/20/30/40 步分别保存 adapter，并用完整验证集比较。评估与最终提交采用 SFT
情感输出和 RL 不确定性输出；这是输出阶段的情感保护，训练过程仍更新共享 LoRA。
若验证分未超过 SFT 基线则不自动生成该版提交。脚本中的模型、数据、基线预测及
路径应与相应 SFT 运行的 checkpoint 一致；断点续训恢复模型和优化器状态，但未保存
随机数状态，不保证与不中断训练逐步一致。

已报告结果、最终提交文件及后处理来源见 [RL 阶段结果](RESULTS_OMNI_7B.md)。
团队 RL 分支的其他算法与结果以该分支原始记录为准，不能将不同模型或验证划分的
分数直接比较。
