# xingzhenbang：多模态情感与冲突识别实验

本目录属于团队 **`rl` 分支**，用于区分 xingzhenbang 的实验尝试。仓库根目录原有 `rl/`、`rl_backup_v2/` 和其他成员文件保持原样；其他成员可以在 `experiments/<姓名>/` 下独立记录自己的路线。

这里保留已完成、未晋级以及未启动的实验状态，不表示它们已达到团队 `main` 的交付门槛。后续只将经过统一评估、确认有效的具体方法提交到 `main`，不整包提升所有实验。

## 从哪里开始

- [实验方法、代码结构与结果](reports/EXPERIMENTS.md)：GRPO、六动作、Grounded REBEL、直接纠错、模态关系、证据与校准、协议 SFT、感知融合、A/B/C 及 7B 冻结评估。
- [模型版本与使用](reports/models/README.md)：27 个模型版本包的训练方法、checkpoint / 折、结果、依赖基座和加载入口。
- [模型清单](reports/models/registry.json)：逐文件来源与 SHA-256。HF 网页上传仍在进行，revision 未固定时在线下载默认拒绝发布。
- [训练与运行入口](rl/README.md)、[继承的项目环境与 SFT 说明](PROJECT_GUIDE.md)。

## 目录与运行方式

```text
experiments/xingzhenbang/
  rl/          本人的 RL、证据与判别实现
  scripts/     训练、评估、批次调度和模型下载
  configs/     各方法配置及继承的 SFT 对照配置
  data/        数据准备代码和不含样本的注册配置
  reports/     实验分析、结果、模型清单
  tests/       原有 CPU 验证
```

这个目录保留原运行布局。请先进入本目录，再运行文档中的相对路径命令，避免加载仓库根目录中其他成员的同名 `rl` 模块：

```bash
cd experiments/xingzhenbang
python scripts/model_artifacts.py list
```

原始 `Dataset/`、基座及 adapter 的 `models/`、外部 `third_party/LlamaFactory/` 不随本目录分发。在本目录按配置准备这些本地路径；若仓库根目录已有数据与基座，可在 Linux 中建立本地链接：

```bash
# 仅在本目录尚不存在这些路径时执行。
ln -s ../../Dataset Dataset
ln -s ../../models models
```

训练输出写入本工作区的忽略目录。历史验证仅保留指标与运行配置，未复制训练样本、预测明细、原始媒体、权重、优化器、notebook 缓存或服务器日志。

## 来源与验证

运行代码来自原实验提交 `716999792547427399978ded342746f08df1913f`；与继承的 Baseline / SFT 工具一起保留，以维持导入及相对路径。目录迁移只更新组织与说明，不改变训练和推理算法。原提交仍可通过 Git 历史追溯。

迁移前代码通过 340 项服务器 CPU 检查和模型重载核验，记录见 [verification.json](reports/models/verification.json)。迁移后的 340 项 CPU 检查、导入隔离和文件边界见[迁移记录](reports/WORKSPACE_MIGRATION_20261031.md)。
