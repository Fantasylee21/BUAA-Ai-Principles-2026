# Omni v4 有限动作 RL（omni_finite_epg_v1）

此实现替换下一轮 GRPO 小幅调参实验，保留旧实现和结果。模型仍为原始 Qwen2.5-Omni-3B + v4 LoRA；仅更新已有语言 LoRA。固定配置见 `configs/rl/finite_actions_v1.json`。

## 决策与目标

按 positive/negative/neutral × yes/no 的固定顺序，评分六个规范 JSON 前缀，截止到 `clarifying_question` 的值之前。分数是整个前缀 token 条件对数概率之和（不做长度归一化），softmax 得到六动作分布。推理也使用同样的六选一 argmax。

每条训练样本的目标是 `-sum(pi * R) + 0.02 * KL(pi || frozen_v4)`。其中 `R = 0.5 * clip(N * delta_macro_f1, -1, 1) + uncertainty_reward`；Conflict 正确 yes 得 `0.3 / pC`，Aligned 正确 no 得 `0.2 / pA`，其余不确定性奖励为零。情感增量基于完整训练集固定三类 Macro-F1 反事实替换。它是局部代理，不等同隐藏测试评分。

第一轮背景为 v4 在新六动作解码下的训练预测；第二轮背景只能来自第一轮末步 210 的完整训练预测。验证和测试标签不进入奖励。旧 v4 的 127 条训练错误只打诊断标签，不改变遍历、奖励或权重。

## 算力和流程

每条原始训练数据每轮恰好出现一次。每批 16 条，3354 条共 210 步（末批 10 条，按实际批大小归一化）。种子 42；AdamW 5e-6、无 weight decay、梯度裁剪 1；10 步 warmup，cosine 固定 420 步。两遍重算之间无参数更新、dropout 为零；候选微批次默认为 1。首遍获得六分数及标量导数，次遍按候选反向传播。

真实 GPU 预检发现，NF4/BF16 的两候选合批评分与逐候选评分存在差异（首个 VT 例子的最大前缀分差约 0.25、动作概率差约 0.00014）。因此本版本配置强制候选微批次为 1；训练、参考缓存、验证、测试全部一致。预检保留合批差异作为诊断，不将合批用于正式计算，也不以放宽阈值宣称其等价。正式路径验收比较逐候选直接梯度、缓存梯度和两遍重算梯度。

冻结视觉/音频编码器和投影模块。缓存其送入语言模型的特征及原始掩码、网格、音视频位置元信息；Omni 本身仍负责 RoPE 和占位符检查。缓存包含数据、媒体 SHA256、权重、源代码、软件版本及动作模板身份。原始 v4 的六动作参考对数概率只缓存一次。两个准备进程处理不重叠分片。

GPU 训练器与验证器各运行一份模型，不用 DDP。单卡默认 PyTorch allocator 上限 32 GiB，4 个 CPU 计算线程；上限不等于预分配，也不是其他 CUDA 库的硬隔离。共享卡可能变慢，只管理本实验子进程。实际需要更大时本任务报 OOM 停止，不终止他人任务。GPU 可以换型号/索引；运行时通过 `--gpus` 指定分配到的两张卡。

真实 VT/VA 预检先验证缓存概率、梯度、候选微批次和两遍反传，以及冻结参考和条件问题生成。BF16 的梯度容差为相对误差 ≤3%、余弦 ≥0.999；CPU float64 测试用严格容差。预检不更新参数。日志报告测得的耗时和峰值显存，训练日志根据完整数据实时修正 ETA。

准备完成后，GPU 1 训练；GPU 2 先做两个 666 条零更新基线（自由生成、新六选一），再顺序验证原子保存的 checkpoint。训练无需等待初始基线。验证步为 50/100/150/210；第二轮为 260/310/360/420。第一轮末等待全部验证；综合分相对两个基线的较高者至少 +0.005，且无非法 JSON 或不一致问题，才继续。否则第一轮结束。最多两轮，不自动跑新种子。

`no` 序列化为 `clarifying_question="none"`；`yes` 由冻结 v4 在相同输入和选定前缀之后生成问题，不能修改前两个字段。无合法问题的候选不能晋级。最终仅最佳合格 RL 候选生成一次 734 行四列测试 CSV；不合格则不生成 CSV，不自动提交 Kaggle。每次评估保留逐条预测、两基线的纠正/退化数、自然冲突/VA 冲突/普通样本误报、Wilson 区间与按样本组配对 bootstrap 分差区间。区间未校正多次模型选择，不能视为线上收益保证。固定三类计分和 union 诊断分别记录，未改变已有本地计分口径。

## 启动、查看与恢复

服务器从独立源码快照运行，旧 `repo` 及 GRPO 结果不被替换。在新快照目录执行：

```bash
bash scripts/start_finite_tmux.sh /home/nas/xingzhenbang/buaa-ai2026/runs/finite-v1 1 2 buaa-finite
tmux attach -t buaa-finite
```

窗口 0 显示状态，1 流程日志，2 训练 loss/期望奖励/KL，3 验证，4 数值预检，5/6 缓存进度。`Ctrl+B` 后按窗口编号切换；`Ctrl+B` 后按 `D` 退出观察，服务器继续运行。流水线完成或失败会停止 GPU 子进程；看板仅保留结果。

每个完整 checkpoint 包含独立算法版本、优化器、固定调度、RNG、完成步、奖励表及其背景版本；目录原子改名后才进入验证队列。显式 `--resume` 从最近完整 checkpoint 恢复，可重跑尚未保存的更新，不能更改代码/软件/模型/数据身份，也不能装入旧 GRPO 优化器。

```bash
python -u -m rl.train_finite_actions run --run-dir /absolute/run --resume --gpus 1 2 --memory-gib 32
```

失败后先查看 `*.error.json` 与 `logs/`；不要同时启动第二个相同 run。缓存分片支持逐条 checksum 验证后继续。中断期间的 incomplete 文件不作为模型或完整评估结果使用。

## 本地验证

```powershell
& .venv-rl-check/Scripts/python.exe -m unittest discover -s rl -p 'test_*.py' -v
& .venv-rl-check/Scripts/python.exe -m rl.train_finite_actions run --run-dir tmp/finite-dry --model-path models/Qwen2.5-Omni-3B --sft-adapter-path models/qwen25omni-sft --dry-run
```

有限动作期望策略梯度的背景：[Expected Policy Gradients](https://arxiv.org/abs/1801.03326)。本实现是任务专用枚举目标，不把该论文的实验收益归因到本任务。

## 2026-10-22 部署验收

当前服务器源码快照：`/home/nas/xingzhenbang/buaa-ai2026/releases/finite-v1-20261022-r2`；实验输出：`/home/nas/xingzhenbang/buaa-ai2026/runs/finite-v1-20261022-r2`；tmux：`buaa-finite`。先前不通过的合批预检保留在 `finite-v1-20261022`，未做参数更新。

本地 44 项 RL 测试、15 项旧回归测试与 2 项新增流水线/真实 PEFT checkpoint 测试通过。真实 VT/VA 的原始与缓存逐候选分数一致；缓存和重算梯度相对误差约 1.2%–1.5%，余弦 0.99989–0.99993，满足预定 BF16 容差。两个条件问题均为合法输出，正文保存在实验 `preflight.json`，仍需结合完整验证观察质量。预检峰值显存 19.07 GiB。

短 VT/VA 输入（297/330 token）两遍训练计算分别约 3.086/3.103 秒；按模态占比粗算约 2.88 小时/轮。这只是偏乐观的短样本测量，未包含缓存、验证、长视频和其他用户负载，正式吞吐以运行日志为准。已启动 GPU 1/2 的两份不重叠缓存分片，每片 2010 条（合计训练 3354 + 验证 666）。正式训练和所有晋级条件由同一有界控制器继续执行。

公开 Evaluation 页面本次仍无法直接读取，缺失类别官方计分细节尚未独立核实；本地固定三类与 union 诊断没有混用。
