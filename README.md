# BUAA-Ai-Principles-2026

本项目用于课程大作业“视频情感大模型后训练”。目标是在统一数据上对本地开源
MLLM 进行基线评测和 SFT/RL 后训练，使模型能够：

- 输出 `positive`、`negative` 或 `neutral` 情感类别；
- 在模态冲突或证据不足时输出 `is_uncertain=yes`；
- 对不确定样本生成面向当前用户的引导式问题，避免武断判断。

比赛指标为：

```text
Score = 0.5 * Macro-F1
      + 0.3 * Conflict_accuracy
      + 0.2 * Aligned_calibration
```

当前 SFT 路径覆盖以下三个模型：

- Qwen2.5-VL-3B-Instruct
- Qwen3-VL-4B-Instruct
- Qwen2.5-Omni-3B

## RL 分支：成员实验入口

当前 `rl` 分支保留团队已有实现，并通过 [experiments/](experiments/README.md) 按成员记录新增尝试。
[xingzhenbang 的实验](experiments/xingzhenbang/README.md) 包含方法、结果、代码和模型版本说明；
运行时先进入其工作目录，根目录原有 `rl/` 和 `rl_backup_v2/` 保持原样。
仅在统一评估后，将确认有效的方法选入 `main`。

## SFT 最终交付版本（v4 / sft-final）

本组当前选定的 SFT 交付方案是 `qlora_sft_v4_resample2x`，代码版本由
`sft-final` 标签固定。首选交付模型为 Qwen2.5-Omni-3B；三个模型的配置均保留在
`configs/sft/<model>/qlora_sft_v4_resample2x.yaml`。

v4 沿用 v3 的字段平衡损失（emotion 0.5、uncertainty 0.4、其余回答 0.1，
Conflict 不确定性权重 2），额外通过 LLaMAFactory 原生 `num_samples` 将 Conflict
训练样本采样至原数量的 2 倍。不是新增人工标签，也不是从 v3 adapter 续训；默认从
原始基座开始训练。训练样本数为 VL 2,482、Omni 3,795，官方验证集保持 666 条。

项目成员提供的 Kaggle 提交截图中，Omni v4 的 Public Score 为 **0.90813**；
这是公开榜记录，不是本地验证分数或最终隐藏榜成绩。训练执行者补充的
[v4 实验分析](reports/sft/qlora_sft_v4.md) 记录：Omni、Qwen3-VL、Qwen2.5-VL
的本地验证 Score 分别为 **0.7185、0.7110、0.6952**；只有 Qwen2.5-VL 高于
各自的 v3 验证结果。选择 Omni v4 交付不代表其验证分数全面优于 v3。

仓库已包含配置、采样实现和分析文档；Omni v4 adapter 已另行上传至下方团队模型仓库。
v4 原始指标、日志及完整推理记录仍需训练执行者补充，以便独立复核。
`sft-final` 保留原代码快照，本分析文档在标签之后补充到 `main`。

使用顺序（在兼容的 Linux/CUDA 环境、仓库根目录执行）：

1. 按下文准备基座、原始数据、依赖和单问题 SFT 数据。
2. `python scripts/prepare_resampled_sft_data.py` 生成重采样数据注册文件。
3. 使用对应 v4 YAML 启动训练；若已取得训练好的 adapter，则跳过训练，直接推理。
4. 推理时显式设置 `ADAPTER_PATH` 为实际交付 checkpoint，不要沿用 v3 的目录。

```bash
# 新训练示例；已有权重时不需要执行。
bash scripts/run_sft_training.sh qlora configs/sft/qwen25omni/qlora_sft_v4_resample2x.yaml 0

# 已有权重：先验证，再对测试集生成 CSV。
# 将 /absolute/path/to/v4-adapter 替换为真实路径；量化精度须与交付记录一致。
ADAPTER_PATH=/absolute/path/to/v4-adapter LOAD_IN_4BIT=1 RUN_NAME=omni_v4_val_nf4 \
  bash scripts/run_sft_inference.sh qlora configs/sft/qwen25omni/qlora_sft_v4_resample2x.yaml 0
SPLIT=test ADAPTER_PATH=/absolute/path/to/v4-adapter LOAD_IN_4BIT=1 RUN_NAME=omni_v4_test_nf4 \
  bash scripts/run_sft_inference.sh qlora configs/sft/qwen25omni/qlora_sft_v4_resample2x.yaml 0
```

### 下载 Omni v4 权重

模型地址：**[BUAA-AI/qwen25omni-sft](https://huggingface.co/BUAA-AI/qwen25omni-sft)**。

根据上传者确认，Omni v4 的 **LoRA adapter** 已上传，对应最佳 `checkpoint-714`；
不是合并后的完整基座模型。交付文件的 `best_checkpoint.json` 记录验证 Score 为
`0.7184609544`，根目录 adapter 与 `checkpoint-714` 的权重文件一致。

- 仓库为 **Private**。先通过管理员提供的邀请链接加入
  [BUAA-AI 团队](https://huggingface.co/BUAA-AI)，获得 `read` 或更高权限。
- 使用自己的 Hugging Face 账号登录后，在模型仓库的 `Files and versions` 下载文件；
  命令行下载需使用自己账号的访问令牌，不共享个人令牌。
- 使用时仍须准备 **Qwen/Qwen2.5-Omni-3B** 基座，并按本项目 v4 配置加载 adapter。
  配置中的 `Qwen2.5-Omni-3B/thinker` 是训练导出的子模型标识，不是独立可下载的
  Hub 基座地址；应使用项目评估代码处理 thinker adapter，而非照搬普通文本模型示例。
- 完整运行记录（基座版本、依赖、NF4/BF16 推理精度、准确加载命令）仍需与训练执行者
  确认。首次使用先做小样本加载和推理检查；权重上传完成不等于已在新环境验证通过。

Git 标签不包含权重。原始课程视频和标注不随模型上传。
详见 [配置说明](configs/README.md) 和 [脚本说明](scripts/README.md)。

## Baseline 与 SFT v3 / v4 验证结果

以下均为完整官方验证集 **666 条**样本上的结果，不是 Kaggle 公开榜分数。
Baseline 使用统一 prompt 的 BF16 推理记录；v3/v4 来自对应实验报告。
这些结果用于展示项目进展，不构成完全控制环境与推理精度的消融实验。
尤其 v4 的原始运行配置与指标文件尚待训练执行者补充，不能将差值全部归因于采样。

| 模型 | 版本 | Macro-F1 | Conflict accuracy | Aligned calibration | 验证 Score |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B | Baseline BF16 | 0.4837 | 0.0000 | 1.0000 | 0.4419 |
| Qwen2.5-VL-3B | QLoRA v3 | 0.6057 | 0.6622 | 0.9324 | 0.6880 |
| Qwen2.5-VL-3B | QLoRA v4 | 0.5977 | 0.6757 | 0.9679 | **0.6952** |
| Qwen3-VL-4B | Baseline BF16 | 0.5090 | 0.0135 | 1.0000 | 0.4586 |
| Qwen3-VL-4B | QLoRA v3 | 0.6185 | 0.7568 | 0.9189 | **0.7201** |
| Qwen3-VL-4B | QLoRA v4 | 0.6186 | 0.7297 | 0.9139 | 0.7110 |
| Qwen2.5-Omni-3B | Baseline BF16 | 0.4873 | 0.0000 | 1.0000 | 0.4437 |
| Qwen2.5-Omni-3B | QLoRA v3 | 0.6340 | 0.7432 | 0.9443 | **0.7288** |
| Qwen2.5-Omni-3B | QLoRA v4 | 0.6369 | 0.6892 | 0.9662 | 0.7185 |

粗体标记各模型表内最高验证 Score。v4 相对 Baseline 的验证 Score 提升约为
Qwen2.5-VL **+0.2533**、Qwen3-VL **+0.2524**、Omni **+0.2748**
（按表中四舍五入值计算）。相对 v3，v4 仅在 Qwen2.5-VL 上提高验证综合分；
Omni v3 仍是本表验证分最高的模型。

评估输入保持项目协议：VT 使用视频与提供的文字；VA 对两个 VL 模型只提供视频
画面，对 Omni 提供视频及音频而不提供文字。Baseline 使用 `native_default`，
峰值保留显存分别为 11.51 / 14.48 / 23.10 GiB；这些显存值不代表 SFT 训练需求。
Baseline 原始指标位于 `output/baseline/`；详细设置与来源见
[实验汇总](reports/sft/README.md)、[v3 报告](reports/sft/qlora_sft_v3.md) 和
[v4 报告](reports/sft/qlora_sft_v4.md)。

### Kaggle 公开榜记录（与验证集分开）

以下来自项目成员提供的提交截图，均为 Omni，不能与上表验证分直接比较。

| 提交版本 | Public Score |
| --- | ---: |
| QLoRA v2，4-bit 推理 | 0.90644 |
| QLoRA v3，checkpoint-630 | 0.90386 |
| QLoRA v4，resample2x | **0.90813** |

公开榜截图中 v4 最高，因此当前选定 v4 交付；这不是最终隐藏榜结果，
也不意味着 v4 在本地验证上全面优于 v3。

## 项目结构

```text
.
├── Dataset/                         # 原始课程数据集，本地目录或软连接，Git 忽略
├── models/                          # 本地模型权重，本地目录或软连接，Git 忽略
│   ├── Qwen2.5-VL-3B-Instruct/
│   ├── Qwen3-VL-4B-Instruct/
│   └── Qwen2.5-Omni-3B/
├── third_party/
│   ├── LlamaFactory/                # 锁定版本的外部项目，Git 忽略
│   └── llamafactory.lock.yaml       # LLaMAFactory commit 和参考运行时版本
├── data/
│   ├── prepare_test_data.py         # 官方无标签测试集校验与模态清单生成
│   ├── prepare_training_data.py     # 原始 JSONL 清洗脚本
│   ├── cleaned/
│   │   ├── use_in_main_true/        # 仅保留 use_in_main=true 的训练数据
│   │   ├── all/                     # SFT 默认使用的完整训练数据
│   │   ├── val/                     # 保留全部官方验证样本的数据
│   │   ├── test/                    # 保留全部官方测试样本的无标签推理清单
│   │   └── clarifying_*.jsonl       # Conflict 样本的引导式问题标签
│   └── llamafactory/
│       ├── dataset_info.json        # LLaMAFactory 数据集注册
│       └── generated/               # ShareGPT JSONL、Omni 音频和清单，Git 忽略
├── configs/
│   ├── sft/                         # 三个模型的版本化 LoRA SFT YAML
│   ├── inference/                   # 推理配置预留目录
│   └── export/                      # LoRA 合并/导出配置预留目录
├── scripts/
│   ├── prepare_sft_data.py          # 转换为 LLaMAFactory 多模态格式
│   ├── run_sft_training.sh          # 按方法和 YAML 启动 SFT
│   ├── run_sft_inference.sh         # 按方法和 YAML 测试 adapter
│   ├── merge_sft_adapter.sh         # 合并 LoRA/QLoRA adapter 与原始权重
│   ├── evaluate_sft.py              # LoRA/QLoRA 通用评测实现
│   └── run_baselines.sh             # 三个基线的复现脚本
├── output/
│   ├── baseline/                    # 已记录的基线预测和指标
│   └── sft/                         # adapter/checkpoint，Git 忽略
├── runs/                            # 训练日志，Git 忽略
├── reports/sft/                     # 应提交到 Git 的实验摘要
├── tmp/                             # 缓存和临时文件，Git 忽略
├── qwen25vl_baseline.py
├── qwen3vl_baseline.py
└── qwen25omni_baseline.py
```

`data/README.md`、`scripts/README.md` 和 `configs/README.md` 保存了各阶段的
详细设计和参数说明；根 README 给出完整的端到端执行顺序。

## 1. 环境准备

建议为项目创建独立 Python 环境，不要直接修改共享环境。已经存在兼容环境时，
直接激活即可。记录基线和当前锁文件所使用的参考版本如下：

| 组件 | 参考版本 |
| --- | --- |
| Python | 3.13.x |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| Transformers | 5.8.0 |
| LLaMAFactory | 0.9.6.dev0，commit 见锁文件 |

这些版本用于复现实验，不表示只有这些版本能够运行。其他相互兼容的 Python、
PyTorch、CUDA、Transformers 和 FlashAttention 版本也可以使用，但需要在实验
报告中记录实际版本、GPU 和显存配置。

系统还需要：

- NVIDIA GPU 和可用的 CUDA 驱动；
- `git`；
- `ffmpeg`，用于从 VA 视频中提取 Omni 训练所需的音频；
- CUDA 兼容的 `bitsandbytes`，用于 QLoRA 4-bit 量化；
- 与当前 PyTorch/CUDA 匹配的 FlashAttention 2，因为现有 YAML 使用
  `flash_attn: fa2`。

在独立环境中，先根据 GPU/CUDA 安装 PyTorch，再在锁定的 LLaMAFactory 目录中
安装其依赖。例如：

```bash
cd third_party/LlamaFactory
python -m pip install -e .
python -m pip install -r requirements/metrics.txt
cd ../..
```

FlashAttention 的安装方式依赖本机 CUDA、编译器和 PyTorch 版本，应选择对应的
wheel 或源码版本，不要在共享环境中盲目升级/降级现有库。

检查关键依赖：

```bash
python - <<'PY'
import torch
import transformers
import flash_attn

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
PY

ffmpeg -version
nvidia-smi
```

## 2. 放置数据和模型

原始数据、模型权重和外部框架体积较大，不提交到本仓库。它们可以直接放在约定
目录，也可以使用软连接。以下命令中的路径必须替换为本机绝对路径：

```bash
ln -s /absolute/path/to/course_dataset Dataset
ln -s /absolute/path/to/model_root models
```

需要满足以下布局：

```text
Dataset/
├── train.jsonl
├── val.jsonl
└── ...                              # JSONL 中引用的视频文件

models/
├── Qwen2.5-VL-3B-Instruct/
├── Qwen3-VL-4B-Instruct/
└── Qwen2.5-Omni-3B/
```

如果同名目录或链接已经存在，不要再次执行 `ln -s`。可用下面的命令检查真实位置：

```bash
readlink -f Dataset
readlink -f models
```

## 3. 准备 LLaMAFactory

`third_party` 下必须存在名为 `LlamaFactory` 的项目目录。它可以是独立 checkout，
也可以是指向本地已有 checkout 的软连接：

```bash
mkdir -p third_party
git clone https://github.com/hiyouga/LlamaFactory.git third_party/LlamaFactory
```

如复用本地 checkout：

```bash
mkdir -p third_party
ln -s /absolute/path/to/LlamaFactory third_party/LlamaFactory
```

本机启动器直接使用已安装的 LLaMAFactory，并通过真实训练检查兼容性。`/third_party/LlamaFactory` 必须保留在
`.gitignore` 中；不要把外部项目源码提交进本仓库，也不要在一次已记录的实验中途
更新其 commit。

检查版本：

```bash
git -C third_party/LlamaFactory rev-parse HEAD
sed -n '1,80p' third_party/llamafactory.lock.yaml
```

## 4. 清洗训练集、官方验证集和测试集

所有命令建议从仓库根目录执行。

SFT 默认使用忽略 `use_in_main` 选择标记、但保留其余结构校验的完整训练版本：

```bash
python data/prepare_training_data.py \
  --ignore-use-in-main \
  --output-dir data/cleaned/all
```

如需复现只保留 `use_in_main=true` 的对照版本：

```bash
python data/prepare_training_data.py
```

官方验证集单独清洗，不从训练集拆分，也不去重；全部 666 条样本原样保留，以便
与 baseline 的覆盖范围和样本权重一致：

```bash
python data/prepare_training_data.py \
  --input Dataset/val.jsonl \
  --expected-split val \
  --ignore-use-in-main \
  --baseline-aligned-validation \
  --output-dir data/cleaned/val
```

官方测试集 `Dataset/test_release.jsonl` 没有情感和冲突真值。准备脚本只校验媒体、
协议和样本顺序，不生成伪标签；全部 734 条测试样本及 VT/VA 配对均原样保留：

```bash
python data/prepare_test_data.py
```

当前默认数据规模：

| 路径 | 训练样本 | 官方验证样本 | 官方测试样本 |
| --- | ---: | ---: | ---: |
| VL | 2,104 | 666 | 734 |
| Omni | 3,354 | 666 | 734 |

### 模态处理规则

训练和评测的规则不能互换。官方验证和测试必须与对应 baseline 的输入方式保持一致；
测试集采用与验证集相同的模态规则：

| 模型路径 | VT 训练 | VA 训练 | VT 官方验证/测试 | VA 官方验证/测试 |
| --- | --- | --- | --- | --- |
| VL | 视频 + `text_content` | 视频 + `text_content`，视作 VT | 视频 + 文本 | 仅视频画面 |
| Omni | 视频 + 文本 | 视频 + 文本 + 音频 | 视频 + 文本，关闭音频 | 视频 + 音频，不提供文本 |

VL 清洗会处理 VT/VA 在有效输入相同后的重复监督；Omni 清洗会对配对样本合并
冲突状态。`sample_type=Conflict` 不是脏数据，不能删除，它用于训练模型表达不确定性。

训练 Conflict 样本的引导式问题位于：

- `data/cleaned/clarifying_questions_by_group.jsonl`
- `data/cleaned/clarifying_questions_by_sample_id.jsonl`

每个冲突组有 5 条生成标签，第二人称“你”指当前用户，而不是视频中的人物。
官方验证集没有标准引导式问题，因此不为验证集附加合成问题。
官方测试集没有任何真值，不会写入 LLaMAFactory `dataset_info.json`，仅用于最终推理。

## 5. 转换为 LLaMAFactory 数据格式

运行转换脚本：

```bash
python scripts/prepare_sft_data.py
```

该脚本会：

- 读取 `data/cleaned/all` 和 `data/cleaned/val`；
- 生成 LLaMAFactory ShareGPT 多模态 JSONL；
- 为 Omni 的 VA 样本提取 16 kHz 单声道 FLAC 音频；
- 校验训练集与官方验证集不存在 ID 重叠；
- 更新 `data/llamafactory/dataset_info.json`；
- 将可再生成的产物写入 Git 忽略的 `data/llamafactory/generated/`。

首次成功提取音频后，如只需重建 JSONL 和元数据，可以跳过 `ffmpeg`：

```bash
python scripts/prepare_sft_data.py --skip-audio-extraction
```

默认每条 Conflict 样本按 `sample_id` 确定性选择 5 条问题中的 1 条，防止冲突样本
被无意放大 5 倍。实验性扩增可使用 `--question-policy expand`，但它会改变类别权重
并覆盖默认生成文件，使用后必须在报告中说明。

转换后的四个数据集为：

```text
data/llamafactory/generated/
├── affect_sft_vl_train.jsonl
├── affect_sft_vl_val.jsonl
├── affect_sft_omni_train.jsonl
├── affect_sft_omni_val.jsonl
└── audio/...
```

训练目标是包含 `pred_emotion`、`is_uncertain`、`clarifying_question` 的紧凑 JSON；
验证目标只包含 `pred_emotion`，仅用于监控情感识别损失。

## 6. 配置 dataset_info.json

`data/llamafactory/dataset_info.json` 已注册以下名称：

- `affect_sft_vl_train`
- `affect_sft_vl_val`
- `affect_sft_omni_train`
- `affect_sft_omni_val`

VL 数据映射 `messages`、`system`、`videos`；Omni 额外映射 `audios`。训练 YAML
必须保持：

```yaml
dataset_dir: data/llamafactory
media_dir: .
```

VL 配置使用：

```yaml
dataset: affect_sft_vl_train
eval_dataset: affect_sft_vl_val
```

Omni 配置使用：

```yaml
dataset: affect_sft_omni_train
eval_dataset: affect_sft_omni_val
use_audio_in_video: true
```

不要手工填写不存在的生成文件名。修改转换脚本或输出位置后，必须同步检查
`dataset_info.json` 和 YAML。由于媒体路径相对于仓库根目录解析，训练应从仓库
根目录启动。

## 7. 配置 LoRA/QLoRA SFT YAML

当前初始配置：

```text
configs/sft/qwen25vl/lora_sft_v1.yaml
configs/sft/qwen3vl/lora_sft_v1.yaml
configs/sft/qwen25omni/lora_sft_v1.yaml
configs/sft/qwen25vl/qlora_sft_v1.yaml
configs/sft/qwen3vl/qlora_sft_v1.yaml
configs/sft/qwen25omni/qlora_sft_v1.yaml
```

六份配置均冻结模态编码器和多模态 projector，对语言模型线性层应用 LoRA：

- LoRA rank 16、alpha 32、dropout 0.05；
- batch size 1、gradient accumulation 16；
- BF16、FlashAttention 2、gradient checkpointing；
- 训练 3 个 epoch，每个 epoch 在官方验证集上评估；
- 视频采样 2 FPS，最多 64 帧；
- VL `cutoff_len=8192`，Omni `cutoff_len=6144`。

QLoRA 配置在相同训练参数上额外使用 bitsandbytes 4-bit NF4、双重量化和 FP32
LayerNorm，从而可以和普通 LoRA 直接比较。

这些是单卡实验的起始配置，不保证适合所有 GPU。每次正式实验都应创建新的、
带版本号的 YAML，例如 `lora_sft_v2.yaml`，不要修改已经产生报告的配置文件。
训练和测试脚本都直接接收 YAML 路径，因此新增配置后不需要修改脚本。YAML 中的
`model_name_or_path` 用于定位基座模型，`output_dir` 同时用于保存训练结果和定位
待测试的 adapter。

## 8. 执行 SFT

先进行 dry run，检查模型、数据、LLaMAFactory commit、输出路径和最终命令：

```bash
DRY_RUN=1 bash scripts/run_sft_training.sh lora \
  configs/sft/qwen25vl/lora_sft_v1.yaml 2
DRY_RUN=1 bash scripts/run_sft_training.sh qlora \
  configs/sft/qwen25vl/qlora_sft_v1.yaml 2
```

第二个参数是物理 GPU 编号。分别训练：

```bash
bash scripts/run_sft_training.sh lora configs/sft/qwen25vl/lora_sft_v1.yaml 2
bash scripts/run_sft_training.sh lora configs/sft/qwen3vl/lora_sft_v1.yaml 2
bash scripts/run_sft_training.sh lora configs/sft/qwen25omni/lora_sft_v1.yaml 2
```

对应的 QLoRA 训练命令为：

```bash
bash scripts/run_sft_training.sh qlora configs/sft/qwen25vl/qlora_sft_v1.yaml 2
bash scripts/run_sft_training.sh qlora configs/sft/qwen3vl/qlora_sft_v1.yaml 2
bash scripts/run_sft_training.sh qlora configs/sft/qwen25omni/qlora_sft_v1.yaml 2
```

脚本默认使用当前 shell 中的 `python`。如需使用指定 conda 环境，可显式设置
`CONDA_ENV`，但项目文档不依赖任何个人环境名：

```bash
CONDA_ENV=your_environment bash scripts/run_sft_training.sh lora \
  configs/sft/qwen25vl/lora_sft_v1.yaml 2
```

常用覆盖项：

```bash
# 启动前重新转换数据
PREPARE_DATA=1 bash scripts/run_sft_training.sh lora \
  configs/sft/qwen25vl/lora_sft_v1.yaml 2

# 临时覆盖 YAML 中的 output_dir
OUTPUT_DIR=/absolute/path/to/output/sft/qwen25vl/lora_debug \
  bash scripts/run_sft_training.sh lora \
    configs/sft/qwen25vl/lora_sft_v1.yaml 2

# 不自动续训
RESUME=none bash scripts/run_sft_training.sh lora \
  configs/sft/qwen25vl/lora_sft_v1.yaml 2

# 从指定 checkpoint 续训
RESUME=/absolute/path/to/checkpoint-xxx \
  bash scripts/run_sft_training.sh lora \
    configs/sft/qwen25vl/lora_sft_v1.yaml 2
```

`RESUME=auto` 为默认行为，会在当前输出目录中选择编号最大的 `checkpoint-*`。
QLoRA 脚本支持相同的环境变量覆盖项。正式实验应优先复制并修改版本化 YAML，
而不是通过 `OUTPUT_DIR` 临时覆盖。若确实使用临时输出目录，测试时应通过
`ADAPTER_PATH` 指向同一目录。
默认输出和日志位置：

```text
<YAML output_dir>/
runs/sft/<model>/<YAML output_dir basename>_<timestamp>.log
```

## 9. 验证和结果记录

YAML 中的 `load_best_model_at_end` 按官方验证集上的 `eval_loss` 选择 checkpoint。
`eval_loss` 不是比赛的 Macro-F1，也不能直接代表最终 Score。完成 SFT 后仍需用与
baseline 一致的生成和解析流程得到：

- `Macro-F1`
- `Conflict_accuracy`
- `Aligned_calibration`
- `Score`

LoRA 和 QLoRA 测试也显式选择训练时使用的 YAML：

```bash
bash scripts/run_sft_inference.sh lora \
  configs/sft/qwen25vl/lora_sft_v1.yaml 2
bash scripts/run_sft_inference.sh qlora \
  configs/sft/qwen3vl/qlora_sft_v1.yaml 2
```

测试脚本从 YAML 读取基座模型和 adapter 目录。QLoRA 默认以 NF4 4-bit 加载基座，
LoRA 默认以 BF16 加载；可通过 `LOAD_IN_4BIT=0|1` 做单独消融。`LIMIT=3` 可用于
小规模冒烟测试，`OVERWRITE=1` 会重新开始对应结果，`RUN_NAME` 可指定新的结果前缀。
通用评测实现也支持 Qwen2.5-Omni：VT 使用视频和给定文本并关闭视频音频，VA 使用
视频和内嵌音频但不提供文本，与 Omni baseline 的输入策略保持一致。测试 Omni 前
需要确保系统中可以执行 `ffmpeg`。

最终测试集推理通过 `SPLIT=test` 选择 `Dataset/test_release.jsonl`。QLoRA 使用
`LOAD_IN_4BIT=1` 在 NF4 基座上推理，使用 `LOAD_IN_4BIT=0` 则在 BF16 基座上挂载
同一 adapter。两种精度必须设置不同的 `RUN_NAME`，避免实验指纹冲突：

```bash
SPLIT=test LOAD_IN_4BIT=1 RUN_NAME=qwen25vl_qlora_sft_v1_test_nf4 \
  bash scripts/run_sft_inference.sh qlora \
  configs/sft/qwen25vl/qlora_sft_v1.yaml 2
```

测试运行覆盖全部 734 条样本，输出 CSV 包含 `sample_id`、`pred_emotion`、
`is_uncertain` 和 `clarifying_question` 四列。由于测试集没有真值，测试运行不会产生
Macro-F1 等比赛指标。模型、量化方式、checkpoint 覆盖方式和输出文件说明见
`scripts/README.md`。

需要将某个最终 adapter 或中间 checkpoint 合并为完整 Hugging Face 模型时：

```bash
bash scripts/merge_sft_adapter.sh \
  configs/sft/qwen25vl/qlora_sft_v1.yaml \
  output/sft/qwen25vl/qlora_sft_v1/checkpoint-264 \
  2
```

默认在 adapter 目录下生成 `merged_model/`，上述命令的输出位置为：

```text
output/sft/qwen25vl/qlora_sft_v1/checkpoint-264/merged_model/
```

脚本同时支持 LoRA 和 QLoRA。QLoRA 合并时不会加载训练阶段的 4-bit 基座，而是
重新加载 YAML 指定的原始非量化权重，再合并 PEFT adapter。默认使用 CPU；设置
`EXPORT_DEVICE=auto` 可使用第三个参数指定的 GPU。正式合并前可设置 `DRY_RUN=1`
检查生成的 LLaMAFactory export 配置。

checkpoint、optimizer state、原始日志和生成数据不提交 Git；将使用的模型、commit、
YAML 路径、数据清单/hash、GPU、显存、训练耗时和四项最终指标整理到
`reports/sft/`。baseline 复现命令为：

```bash
bash scripts/run_baselines.sh 2
```

其结果默认写入 `output/baseline/`。

## 重要注意事项

1. **从仓库根目录执行。** `media_dir: .` 依赖该工作目录解析 `Dataset/...` 和
   `data/llamafactory/generated/audio/...`。
2. **官方验证集不是从训练集拆出的。** 必须完整保留发布的 666 条验证样本，不能
   因配对、重复或缺少引导式问题而删除；测试集同样必须保持全部 734 条和官方顺序。
3. **验证和测试模态必须和 baseline 一致。** 特别是 VL 的 VA 样本只看画面，Omni
   的 VA 样本只用视频/音频且不提供 `text_content`。
4. **不要删除 Conflict 样本。** 冲突识别准确率和一致样本校准率共同防止模型把
   所有样本都预测为 uncertain。
5. **不要提交大文件或凭据。** `Dataset/`、`models/`、`third_party/LlamaFactory/`、
   生成音频、SFT checkpoint、日志、缓存和 API key 都应保持在 `.gitignore` 中。
6. **不要中途更新 LLaMAFactory。** 已记录实验使用锁文件中的 commit；框架升级应
   新开实验并同步更新锁文件和报告。
7. **先 dry run，再正式训练。** dry run 不加载模型，但能发现大多数路径、数据和
   版本问题。
8. **显存不足时一次只改一个因素。** 优先逐步降低每帧像素预算、`cutoff_len` 或
   视频帧数，并保存为新的 YAML。不要无记录地修改 `lora_sft_v1.yaml`。
9. **Omni 显存压力最大。** 已记录的 Qwen2.5-Omni baseline 在 RTX 4090 上峰值保留
   显存约 23.10 GiB；训练显存还受序列长度、音频和反向传播影响，必须实际监控。
10. **生成数据可重建。** `PREPARE_DATA=1` 会重新生成 JSONL 和音频元数据；执行前
    确认是否使用默认单问题策略，避免改变 Conflict 样本权重。
11. **保护已完成实验。** 使用新的 `RUN_NAME` 和 YAML 版本，避免输出目录冲突；
    自动续训前先确认所选 checkpoint 属于同一配置和同一数据版本。
12. **密钥不得进入仓库。** 外部模型生成引导式问题时使用的 API key 只应通过环境
    变量或安全凭据管理注入，不得写入脚本、JSONL、日志或 README。

更细的清洗统计和模态规则见 `data/README.md`，脚本参数见 `scripts/README.md`，
LoRA 配置说明见 `configs/README.md`。


## 本机 v3 实验

最新本机训练使用 `configs/sft/<model>/qlora_sft_v3.yaml`，加入按字段归一化的
任务损失，并在完整官方验证集上按比赛 Score 选择 epoch checkpoint。
设计、已有结果对照、测试证据和日志路径见 [v3 报告](reports/sft/qlora_sft_v3.md)。

```bash
conda activate fwt_temp
python scripts/run_sft_v3.py --gpu 4
```

三个模型均已完成训练与各轮验证，最佳均为第 3 轮。Omni、Qwen3-VL、Qwen2.5-VL 的最佳 Score 分别为 0.7288、0.7201、0.6880，完整结果见上述 v3 报告。
