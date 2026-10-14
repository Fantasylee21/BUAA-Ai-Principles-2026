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

## 当前 Baseline 结果

以下结果来自官方验证集全部 666 条样本，三个模型均使用记录为
`native_default` 的原生输入配置，推理错误数均为 0：

| 模型 | VA 输入 | Macro-F1 | Conflict accuracy | Aligned calibration | Score | 峰值保留显存 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B | 仅视频画面 | 0.481418 | **0.013514** | 0.930743 | 0.430911 | 11.51 GiB |
| Qwen3-VL-4B | 仅视频画面 | **0.520558** | **0.013514** | **1.000000** | **0.464333** | 14.48 GiB |
| Qwen2.5-Omni-3B | 视频画面 + 音频 | 0.355865 | 0.000000 | 0.998311 | 0.377595 | 23.09 GiB |

VT 样本均输入视频画面和官方提供的文本。VA 样本中，两个 VL 模型不读取音频和
文本，只输入视频画面；Qwen2.5-Omni 输入视频画面和内嵌音频，不提供官方文本。
完整逐样本预测、运行配置和未四舍五入的指标保存在 `output/baseline/`。

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
│   ├── prepare_training_data.py     # 原始 JSONL 清洗脚本
│   ├── cleaned/
│   │   ├── use_in_main_true/        # 仅保留 use_in_main=true 的训练数据
│   │   ├── all/                     # SFT 默认使用的完整训练数据
│   │   ├── val/                     # 保留全部官方验证样本的数据
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
│   ├── run_lora_sft.sh              # LoRA SFT 统一启动脚本
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
git -C third_party/LlamaFactory checkout 97b32d3133b501432141a82949d5c7bc4d94f23a
```

如复用本地 checkout：

```bash
mkdir -p third_party
ln -s /absolute/path/to/LlamaFactory third_party/LlamaFactory
```

版本以 `third_party/llamafactory.lock.yaml` 为准。启动脚本会检查实际 HEAD 是否与
锁文件一致，不一致时拒绝训练。`/third_party/LlamaFactory` 必须保留在
`.gitignore` 中；不要把外部项目源码提交进本仓库，也不要在一次已记录的实验中途
更新其 commit。

检查版本：

```bash
git -C third_party/LlamaFactory rev-parse HEAD
sed -n '1,80p' third_party/llamafactory.lock.yaml
```

## 4. 清洗训练集和官方验证集

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

当前默认数据规模：

| 路径 | 训练样本 | 官方验证样本 |
| --- | ---: | ---: |
| VL | 2,104 | 666 |
| Omni | 3,354 | 666 |

### 模态处理规则

训练和验证的规则不能互换。官方验证必须与对应 baseline 的输入方式保持一致：

| 模型路径 | VT 训练 | VA 训练 | VT 官方验证 | VA 官方验证 |
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

## 7. 配置 LoRA SFT YAML

当前初始配置：

```text
configs/sft/qwen25vl/lora_sft_v1.yaml
configs/sft/qwen3vl/lora_sft_v1.yaml
configs/sft/qwen25omni/lora_sft_v1.yaml
```

三份配置均冻结模态编码器和多模态 projector，对语言模型线性层应用 LoRA：

- LoRA rank 16、alpha 32、dropout 0.05；
- batch size 1、gradient accumulation 16；
- BF16、FlashAttention 2、gradient checkpointing；
- 训练 3 个 epoch，每个 epoch 在官方验证集上评估；
- 视频采样 2 FPS，最多 64 帧；
- VL `cutoff_len=8192`，Omni `cutoff_len=6144`。

这些是单卡实验的起始配置，不保证适合所有 GPU。每次正式实验都应创建新的、
带版本号的 YAML，例如 `lora_sft_v2.yaml`，不要修改已经产生报告的配置文件。
新增 YAML 后还需要同步更新 `scripts/run_lora_sft.sh` 中对应模型的 `config_path`，
或直接用 LLaMAFactory CLI 启动该 YAML；只修改 `RUN_NAME` 不会改变训练参数。

## 8. 执行 SFT

先进行 dry run，检查模型、数据、LLaMAFactory commit、输出路径和最终命令：

```bash
DRY_RUN=1 bash scripts/run_lora_sft.sh all 2
```

第二个参数是物理 GPU 编号。分别训练：

```bash
bash scripts/run_lora_sft.sh qwen25vl 2
bash scripts/run_lora_sft.sh qwen3vl 2
bash scripts/run_lora_sft.sh qwen25omni 2
```

也可以在同一张卡上依次执行三个实验：

```bash
bash scripts/run_lora_sft.sh all 2
```

脚本默认使用当前 shell 中的 `python`。如需使用指定 conda 环境，可显式设置
`CONDA_ENV`，但项目文档不依赖任何个人环境名：

```bash
CONDA_ENV=your_environment bash scripts/run_lora_sft.sh qwen25vl 2
```

常用覆盖项：

```bash
# 启动前重新转换数据
PREPARE_DATA=1 bash scripts/run_lora_sft.sh qwen25vl 2

# 使用新的实验名，避免覆盖已有结果
RUN_NAME=lora_sft_v2 bash scripts/run_lora_sft.sh qwen25vl 2

# 将大体积 checkpoint 写到其他磁盘
OUTPUT_ROOT=/absolute/path/to/output/sft \
  bash scripts/run_lora_sft.sh qwen25vl 2

# 不自动续训
RESUME=none bash scripts/run_lora_sft.sh qwen25vl 2

# 从指定 checkpoint 续训
RESUME=/absolute/path/to/checkpoint-xxx \
  bash scripts/run_lora_sft.sh qwen25vl 2
```

`RESUME=auto` 为默认行为，会在当前输出目录中选择编号最大的 `checkpoint-*`。
默认输出和日志位置：

```text
output/sft/<model>/<run_name>/
runs/sft/<model>/<run_name>_<timestamp>.log
```

## 9. 验证和结果记录

YAML 中的 `load_best_model_at_end` 按官方验证集上的 `eval_loss` 选择 checkpoint。
`eval_loss` 不是比赛的 Macro-F1，也不能直接代表最终 Score。完成 SFT 后仍需用与
baseline 一致的生成和解析流程得到：

- `Macro-F1`
- `Conflict_accuracy`
- `Aligned_calibration`
- `Score`

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
   因配对、重复或缺少引导式问题而删除。
3. **验证模态必须和 baseline 一致。** 特别是 VL 的 VA 验证只看画面，Omni 的
   VA 验证只用视频/音频且不提供 `text_content`。
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
   显存约 23.09 GiB；训练显存还受序列长度、音频和反向传播影响，必须实际监控。
10. **生成数据可重建。** `PREPARE_DATA=1` 会重新生成 JSONL 和音频元数据；执行前
    确认是否使用默认单问题策略，避免改变 Conflict 样本权重。
11. **保护已完成实验。** 使用新的 `RUN_NAME` 和 YAML 版本，避免输出目录冲突；
    自动续训前先确认所选 checkpoint 属于同一配置和同一数据版本。
12. **密钥不得进入仓库。** 外部模型生成引导式问题时使用的 API key 只应通过环境
    变量或安全凭据管理注入，不得写入脚本、JSONL、日志或 README。

更细的清洗统计和模态规则见 `data/README.md`，脚本参数见 `scripts/README.md`，
LoRA 配置说明见 `configs/README.md`。
