# 实验模型下载、版本对应与使用

发布状态：模型包已导出并完成本机校验，Hugging Face 网页上传正在进行。上传完成后将在版本清单中固定 HF commit；当前在线下载入口会拒绝未固定的发布，避免读到不完整的模型集合。

本目录记录 xzb 实验分支的 **27 个模型版本包**，合计约 **4.97 GB**。权重仓库为团队私有仓库 [BUAA-AI/xzb-affect-experiments](https://huggingface.co/BUAA-AI/xzb-affect-experiments)。需要使用有该团队读取权限的 Hugging Face 账号。

一个版本包对应一个明确的实验方法和 checkpoint，或明确注明的三折模型集合。保留所有已完成评估节点，便于逐项复核；预检、故障恢复副本、优化器、原始视频和训练标签不在包中。基座单独下载。新增融合层、分类头和校准参数均一并保存，模型包不是只改名字的同一份 adapter。

每个 ZIP 可独立下载；其中的 `artifact.json` 记录方法、训练步数 / 折、基座、指标口径、依赖模型、原服务器相对路径、源码版本和逐文件 SHA-256，`source/` 保存该批部署的实现快照。Git 中的 [registry.json](registry.json) 再记录整个 ZIP 的 SHA-256 和固定 HF 权重提交版本。模型包使用 ZIP 是为了保留原始目录和配套实现；下载脚本会校验并还原目录，不会自动执行包内代码。

## 快速下载

先取得本项目代码，准备 Python 与 `huggingface_hub`。使用自己的 HF 账号完成 `hf auth login`，也可以在运行环境设置自己的 `HF_TOKEN`；不要把令牌写入配置或提交到 Git。

在项目根目录执行：

```bash
python scripts/model_artifacts.py list
python scripts/model_artifacts.py download omni-protocol-experiment
python scripts/model_artifacts.py verify omni-protocol-experiment models/experiments/omni-protocol-experiment
```

默认写入 Git 忽略的 `models/experiments/<模型ID>/`。有参考模型依赖时自动一并下载：例如 A/B/C 下载三折 SFT-ref 包，并使用其中 `fold0/adapter`；普通 v4 后续 RL 下载原 v4 adapter。已存在目录必须通过文件校验才会复用，不覆盖损坏目录。

也可以在 HF 网页下载 ZIP 后离线还原：

```bash
python scripts/model_artifacts.py download omni-finite-step210 --archive-dir /path/to/downloaded-zips
```

离线目录也需包含该方法的依赖包。`--output-dir` 可指定保存位置；在线下载默认使用 registry 中的固定 HF commit，避免后续 `main` 变化使同名模型变成不同权重。

## 方法与模型逐项对应

G1 是原始 666 条验证上的固定三类 Score，G2 是 671 条分组留出的固定三类 Score，G3 只是不确定性分量，G4 使用正负 F1 的选择分。它们不能跨口径排序。部分值保留原报告的四舍五入精度；不是 Kaggle Public。完整设置、原始结果和结论见项目的 `reports/EXPERIMENTS.md` 与各专题报告。

| 模型 ID / 下载包 | 方法 | 指标 | 结果 | 用途 |
|---|---|---|---:|---|
| [omni-sft-v4](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-sft-v4.zip) | QLoRA v4 Conflict resample2x | G1 fixed3 historical | 0.71846095 | 原 SFT 交付 adapter |
| [omni-grpo-correctness-step100](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-grpo-correctness-step100.zip) | Dr.GRPO correctness | G1 fixed3 | 0.70779278 | 研究对照，未晋级 |
| [omni-grpo-correctness-step200](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-grpo-correctness-step200.zip) | Dr.GRPO correctness | G1 fixed3 | 0.70812325 | 研究对照，未晋级 |
| [omni-grpo-f1-delta-step100](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-grpo-f1-delta-step100.zip) | Dr.GRPO f1-delta | G1 fixed3 | 0.70926489 | 研究对照，未晋级 |
| [omni-grpo-f1-delta-step200](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-grpo-f1-delta-step200.zip) | Dr.GRPO f1-delta | G1 fixed3 | 0.70825138 | 研究对照，未晋级 |
| [omni-finite-step50](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-finite-step50.zip) | six-action expected-reward RL | G1 fixed3 | 0.70910800 | 研究对照，未晋级 |
| [omni-finite-step100](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-finite-step100.zip) | six-action expected-reward RL | G1 fixed3 | 0.70699600 | 研究对照，未晋级 |
| [omni-finite-step150](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-finite-step150.zip) | six-action expected-reward RL | G1 fixed3 | 0.71017500 | 研究对照，未晋级 |
| [omni-finite-step210](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-finite-step210.zip) | six-action expected-reward RL | G1 fixed3 | 0.71441920 | 研究对照，未晋级 |
| [omni-grounded-rebel-step40](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-grounded-rebel-step40.zip) | grounded-rebel | G1 fixed3 | 0.71165900 | 研究对照，未晋级 |
| [omni-grounded-rebel-step80](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-grounded-rebel-step80.zip) | grounded-rebel | G1 fixed3 | 0.71098300 | 研究对照，未晋级 |
| [omni-conflict-cost-step105](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-conflict-cost-step105.zip) | conflict-cost | G1 fixed3 | 0.70456100 | 研究对照，未晋级 |
| [omni-conflict-cost-step210](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-conflict-cost-step210.zip) | conflict-cost | G1 fixed3 | 0.71301047 | 研究对照，未晋级 |
| [omni-relation-binary-control](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-relation-binary-control.zip) | binary_control | G1 fixed3 | 0.71165911 | 研究对照，未晋级 |
| [omni-relation-joint-relation](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-relation-joint-relation.zip) | joint_relation | G1 fixed3 | 0.71301047 | 研究对照，未晋级 |
| [omni-evidence-bottleneck](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-evidence-bottleneck.zip) | isolated V and T/A evidence with cost calibration | G1 fixed3 | 0.71571317 | 研究对照，未晋级 |
| [omni-six-feature-calibrator](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-six-feature-calibrator.zip) | six-feature nested grouped calibration | G3 uncertainty component | 0.37064220 | 分支 / 折模型；非完整系统 |
| [frozen-qwen3vl8b-evidence](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/frozen-qwen3vl8b-evidence.zip) | frozen Qwen3-VL-8B visual evidence | G3 uncertainty component | 0.37848017 | 分支 / 折模型；非完整系统 |
| [omni-protocol-control](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-protocol-control.zip) | protocol SFT control | G2 fixed3 | 0.90358657 | 研究对照，未晋级 |
| [omni-protocol-experiment](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-protocol-experiment.zip) | protocol SFT experiment | G2 fixed3 | 0.90529784 | 研究对照，未晋级 |
| [joint-sft-reference-folds](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/joint-sft-reference-folds.zip) | protocol-consistent JSON SFT reference | G4 PN selection | 0.88316216 | 分组折模型；非全量最终模型 |
| [joint-omni-c0](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/joint-omni-c0.zip) | C0 trainable perception and cross-modal fusion | G4 PN selection | 0.87393000 | 分组折模型；非全量最终模型 |
| [joint-omni-c1](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/joint-omni-c1.zip) | C1 trainable perception and cross-modal fusion | G4 PN selection | 0.86711000 | 分组折模型；非全量最终模型 |
| [joint-qwen3vl8b-c0-fold0](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/joint-qwen3vl8b-c0-fold0.zip) | trainable 8B visual C0 upgrade | G4 PN selection | 0.86241000 | 分组折模型；非全量最终模型 |
| [omni-conflict-reward-a](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-conflict-reward-a.zip) | two-action conflict reward A | G4 PN selection | 0.88430165 | 研究对照，未晋级 |
| [omni-conflict-reward-b](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-conflict-reward-b.zip) | two-action conflict reward B | G4 PN selection | 0.88470940 | 研究对照，未晋级 |
| [omni-conflict-reward-c](https://huggingface.co/BUAA-AI/xzb-affect-experiments/blob/main/omni-conflict-reward-c.zip) | two-action conflict reward C | G4 PN selection | 0.88725410 | 研究对照，未晋级 |

## 各类权重怎样加载

多数基座为 **Qwen/Qwen2.5-Omni-3B**。VL-8B 分支额外需要 **Qwen/Qwen3-VL-8B-Instruct**。LoRA 存在 `adapter/` 中，Omni 语言 adapter 应挂在原始基座的 Thinker 上；不要叠加到已经合并了 v4 的完整模型。具体库版本见模型包 `artifact.json` 中的 provenance / software 和项目依赖说明。

### SFT 与生成式 GRPO

可复用项目 SFT 推理器。例如已有基座、数据和环境时，使用协议 SFT 实验组：

```bash
ADAPTER_PATH="$PWD/models/experiments/omni-protocol-experiment/adapter" \
LOAD_IN_4BIT=1 RUN_NAME=published_protocol_val \
  bash scripts/run_sft_inference.sh qlora configs/sft/qwen25omni/qlora_sft_v5_protocol.yaml GPU_INDEX
```

将 `GPU_INDEX` 换为自己分配到的显卡。v4 使用其 v4 YAML；GRPO 可用 `rl.OmniPolicy` 的原始媒体与 JSON 生成路径。更换模型、精度、prompt 或解码方式后，应重新评估，不能直接继承表内分数。

### 六动作、REBEL、直接纠错与 A/B/C

这些 adapter 可加载，但其结果依赖专门的解码规则。包内 `settings` 与 `source/rl/` 实现是依据。

| 格式 / ID | 加载与决策入口 | 必须保留的规则 |
|---|---|---|
| `finite_actions` | `rl.finite_runtime.FinitePolicy(..., policy_path=模型目录/adapter, training=False)`；`prepare → perception → answer` | 枚举三情感 × yes/no 六动作；候选 microbatch=1；问题由冻结 v4 生成 |
| `omni-grounded-rebel-*` | `rl.grounded_runtime.GroundedPolicy`；加载发布 adapter 后调用 `answer_frozen_emotion` | 情感固定为原 v4；原生 yes/no logits，不是重新自由生成情感 |
| `omni-conflict-cost-*` | `rl.conflict_runtime.ConflictPolicy`；`prepare_task`、`answer_frozen_emotion` | 使用冲突专用 prompt；新 yes 的问题仍使用原 v4 的 question prompt |
| `anchored_conflict_reward` | `rl.conflict_reward_runtime.ConflictRewardPolicy`；`log_odds` 与 `rl.conflict_reward.anchored_logit` | 参考是同折 SFT-ref，非 v4；用参考情感、yes/no 锚点和 policy-reference log-odds 增量，最后阈值0.5 |

六动作加载示例（Linux / CUDA，基座与 v4 已准备好）：

```python
import json
from pathlib import Path
from rl.finite_runtime import FinitePolicy

package = Path("models/experiments/omni-finite-step210")
meta = json.loads((package / "artifact.json").read_text())
policy = FinitePolicy("/path/to/Qwen2.5-Omni-3B",
                     "models/experiments/omni-sft-v4/adapter",
                     meta["settings"], policy_path=package / "adapter", training=False)
# row 是正常 VT/VA 输入记录，包含 sample_id、protocol、video 和 VT 所需 text_content。
prefix = policy.prepare(row, "/path/to/Dataset")
features = policy.perception(prefix)
answer_json, action_probabilities = policy.answer(prefix, features)
```

REBEL / 直接纠错的构造器从 v4 初始化 reference；使用 PEFT `load_adapter(..., adapter_name="published", is_trainable=False)` 加载发布权重，并在决策时显式传 `adapter="published"`。不要调用训练控制器来代替推理。A/B/C 的原预测流程见 `source/scripts/train_conflict_reward.py::predict`，对新输入需要先由同折参考模型得到情感、锚点和参考 log-odds；不能使用训练真值作锚点。

### 关系判别头与证据 / 校准参数

`relation_head` 包保存 `head.safetensors`，只含推理参数，已去掉原 `.pt` 内的优化器与 RNG。可在 CPU 上加载结构和权重：

```python
import json
from pathlib import Path
from safetensors.torch import load_file
from rl.modality_relation import build_model

package = Path("models/experiments/omni-relation-joint-relation")
meta = json.loads((package / "artifact.json").read_text())
head = build_model(meta["head_method"], meta["settings"])
head.load_state_dict(load_file(str(package / "head.safetensors")), strict=True)
head.eval()
```

输入应由 `rl.relation_runtime.RelationExtractor` 提取为 `[batch, vision/second/full, 2048]`，顺序、归一化和协议编码保持原实现。该模型只预测关系，整体情感仍由 v4 提供。

双路证据包的 `heads/{vision,text,audio}.json` 和 `calibration/final-42.json` 是实际参数文件；使用 `BaseEvidenceExtractor` 从**未挂 v4 adapter 的原始基座**提取隐藏向量与原生 logits，交给 `evidence_probabilities → disagreement → decision_scores`。六特征校准使用 `six_features → predict_logistic → disagreement_gate`。这些导出已去掉拟合样本 ID / 标签，数值参数未改变。

六特征校准与冻结 8B 包保留的是折内拟合器和配套参数，不存在另一个已完成的全量模型。冻结 8B 用包内 `producer-fits/expert` 的视觉头，第二路取同折 `producer-fits/control` 的 text/audio 头及 `calibration/*-expert.json`；不能把不同折、分组或骨干的参数混用。各路径和原始来源在 `artifact.json` 中列明。

### C0 / C1 与可训练 8B

每折的 `trainable.pt` 包含所有实际更新的感知 LoRA、连接层、池化、融合和分类参数，`trainable.json` 是原配校验与结构描述。二者必须一起保留。使用 `scripts.joint_affect_model.HFBackend` 和 `JointAffectModel`，然后 `load_checkpoint(package / "fold0")`；不要把这些文件当作普通 PEFT adapter。

构造骨干可照 `scripts/train_joint_affect.py::make_model`，将基座与数据路径替换为本地路径，再以 `model.eval()` 和 `torch.inference_mode()` 调用 `model.forward_row(row)`。C0/C1 的 `vision_model="omni"`，8B 升级为 `"qwen3vl8b"`。可用同一张分配的 GPU 放置两路，或按原实现分配两张卡，保持 NF4 / BF16 和媒体预算。

三折 SFT-ref、C0、C1 是三个不同训练分区模型；它们没有全量最终训练。8B 仅 fold0 完成。发布这些权重是为了研究复用，不代表已经验证三折集成或生产部署收益。本次验证了文件、模型结构和 CPU 数值链路；没有重新跑大模型 GPU 推理，也没有新增 Kaggle 成绩。

## 没有独立新模型的记录

- 7B EmotionReasoner：本轮选中 E0/U0，即公开情感专家的冻结评估；正式 S、R1、R2 没有训练。基座为 `Qwen/Qwen2.5-Omni-7B`，公开 adapter 为 `SmilingWeeping/EmotionReasoner` 的 `EmotionReasoner/checkpoint-195`。严格加载与 FP32 合并 / BF16 保存流程在 `scripts/emotion_expert_bootstrap.py`；原合并模型仍保存在服务器。该上游模型不是本组新训练的权重，也没有拿预检 adapter 冒充正式模型。
- 7B 冻结留出选择分为 **0.74306338**，同分区参考 **0.89515637**；自然冲突 27/28，但普通误报 557/589，未晋级，未跑正式 val/test。
- 固定补漏复用现有证据头，是决策规则，没有另一个独立训练权重。
- 早期三模型 SFT v1–v3 属于继承基线，本次 xzb 归档没有获得这些权重；不能仅依据历史报告制造下载地址。
- `submission_09.csv` 的 0.92313 生成模型仍未知，没有将某个发布包标为它对应的模型。

## 发布与核验记录

服务器原目录保留；原始训练中间 checkpoint 和训练状态不删除。27 个发布包在服务器做了 CRC 与 SHA-256 核验；本机取回后再次按清单校验。读取模型时还校验逐文件大小和 SHA。代码对应关系与实验结果见 `reports/EXPERIMENTS.md`；原 v4 上游 adapter 版本也保留在 registry。

服务器隔离目录中的 CPU 检查：`tests/` **247 项通过**，`rl/test_*.py` **93 项通过**，全部屏蔽 GPU。本次新增的 5 项下载安全检查包含哈希、损坏文件、路径越界、额外文件及依赖顺序。相关运行日志另外保存在服务器的发布校验目录。
