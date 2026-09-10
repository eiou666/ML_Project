# Fashion-MNIST 半监督学习课程设计

本项目实现《机器学习课程设计》要求的完整实验链：全标签/10%/5%/1% 监督基线、普通伪标签、FixMatch、置信度阈值消融，以及 1%→2%→3% 主动学习挑战。项目强调公平对照、数据防泄漏、重复实验、检查点续训和自动统计，不通过挑选最好的一次运行制造结论。

> 当前仓库包含代码、配置、测试、运行入口、完整实验结果汇总与分析工具；已完整执行全套 77 组对比实验并完成自动聚合分析（见 `outputs/summary/`）。

## 研究问题

- RQ1：标签比例从 100% 降到 10%、5%、1% 时，性能如何变化？
- RQ2：伪标签和 FixMatch 能否利用未标注图像弥补性能损失？
- RQ3：低质量伪标签是否会引发确认偏差，使无标签数据反而有害？
- RQ4：不确定性与多样性联合采样是否比随机采样更节省标签？

## 项目结构

```text
configs/
  main.yaml                 # 正式实验配置
  smoke.yaml                # 离线合成数据冒烟配置
notebooks/
  colab_runner.ipynb        # Colab GPU 运行入口
src/
  data.py                   # 数据集、增强和不泄漏的无标签视图
  splits.py                 # 分层、嵌套、可哈希的标注预算
  models/                   # 统一 FashionCNN 骨干
  methods/                  # supervised / pseudolabel / fixmatch
  engine.py                 # 训练、EMA、AMP、检查点和确定性续训
  active_learning/          # 随机、熵、联合采样
  run.py                    # 单次实验入口
  sweep.py                  # 主实验、消融和主动学习矩阵
  active.py                 # 主动学习多轮训练
  aggregate.py              # 统计表、置信区间和图片
tests/                      # 划分、损失、采样、模型和续训测试
```

## 数据协议

Fashion-MNIST 官方训练集含 60,000 张图像，测试集含 10,000 张图像。验证标签也计入人工标注预算：

| 标注比例 | 总标注 | 训练标签 | 验证标签 | 无标签池 |
|---:|---:|---:|---:|---:|
| 100% | 60,000 | 54,000 | 6,000 | 0 |
| 10% | 6,000 | 5,400 | 600 | 54,000 |
| 5% | 3,000 | 2,700 | 300 | 57,000 |
| 1% | 600 | 540 | 60 | 59,400 |

每个类别使用独立、确定性的随机排列；同一 seed 下，1%、5%、10%、100% 的“已标注集合”互相嵌套。每次运行都会保存 `split_hash`、各集合索引和每类样本数。

无标签数据集只向训练步骤返回弱增强图、目标增强图和全局索引，不返回隐藏类别。真实标签只能通过索引进入无梯度诊断路径，用于统计伪标签正确率，不会参与损失计算或样本选择。

## 环境安装

本机 RTX 3060 Ti 的部署、后台训练和恢复方式见 [本机训练说明](docs/local-training.md)。使用 `scripts/setup_cuda.ps1` 配置 CUDA 环境，使用 `scripts/start_local_training.ps1` 启动完整本地队列；正式配置为 `configs/local_cuda.yaml`。

推荐 Python 3.11 和带 GPU 的 Google Colab：

```bash
python -m pip install -r requirements.txt
python -m pytest
```

Windows 本机可在项目内创建隔离环境；当前已验证环境位于 `.venv`：

```powershell
uv venv --python D:\anaconda3\python.exe .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
```

也可使用已经准备好的自动安装与验证脚本：

```powershell
.\scripts\setup_windows.ps1 -Python D:\anaconda3\python.exe
```

`requirements.txt` 保存兼容范围；`requirements-lock.txt` 保存本机已经完整验证的精确版本。只检查现有环境可运行：

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

Colab 可直接打开 `notebooks/colab_runner.ipynb`，将项目放入 Google Drive 后按单元格运行。第一次正式训练前必须确认：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 快速验证

合成数据冒烟测试不下载 Fashion-MNIST，只执行 3 个训练步：

```bash
python -m src.sweep --suite smoke
```

只查看正式实验矩阵而不训练：

```bash
python -m src.sweep --suite main --dry-run
python -m src.sweep --suite ablation --dry-run
python -m src.sweep --suite active_learning --dry-run
```

## 单次实验

```bash
python -m src.run \
  --config configs/main.yaml \
  --method fixmatch \
  --label-ratio 0.01 \
  --seed 0
```

可用方法：

- `supervised`：只使用有标签批次的交叉熵。
- `pseudolabel`：弱增强生成伪标签，另一个弱增强视图学习伪标签。
- `fixmatch`：弱增强生成伪标签，强增强视图学习高置信目标。

每次正式运行固定 12,000 次参数更新。默认使用 AdamW、500 步预热、余弦退火、EMA=0.999、`tau=0.95`、`lambda_u=1.0`。所有方法使用相同 CNN、优化器和有标签批量。

## 完整实验

建议按顺序运行，任何命令中断后重新执行即可从 `checkpoint_latest.pt` 继续：

```bash
# 4 个监督条件 + 3 个伪标签条件 + 3 个 FixMatch 条件，共 5 个 seed
python -m src.sweep --suite main

# 1% 标签，tau = 0 / 0.80 / 0.95，共 3 个 seed
python -m src.sweep --suite ablation

# random / entropy / uncertainty_diversity，1% -> 2% -> 3%，共 3 个 seed
python -m src.sweep --suite active_learning
```

Colab 会话时间有限时，可分 seed 执行：

```bash
python -m src.sweep --suite main --seeds 0
python -m src.sweep --suite main --seeds 1
python -m src.sweep --suite ablation --seeds 0 1 2
```

`--max-runs N` 可先跑矩阵前 N 项；`--keep-going` 会记录失败并继续后续任务。`--force` 会覆盖目标运行目录中的既有检查点和指标，仅在明确需要重跑时使用。

### 检查点与复现

- 确定性训练批次由 `(seed, global_step)` 直接生成，恢复时不依赖 DataLoader 的迭代位置。
- 检查点保存模型、EMA、优化器、学习率调度器、AMP scaler、当前步数和随机状态。
- 配置哈希或数据划分哈希不匹配时拒绝加载，防止误接续其他实验。
- `resolved_config.yaml` 是该次运行的唯一配置事实来源。

## 主动学习

初始标注预算为 1%。每轮查询 600 张图像，到 3% 停止：

- `random`：均匀随机抽样。
- `entropy`：选择预测熵最高的样本。
- `uncertainty_diversity`：先保留熵最高的 `5B` 个候选，再在 256 维归一化特征上使用 k-means++ 选择 `B=600` 个代表样本。

选择模块的输入不包含真实类别；选中后才将真实标签写入 `round*_selection.csv`，模拟人工标注返回。每轮使用累积标签从头训练 FixMatch，避免热启动对某种策略产生额外优势。

## 结果文件

每个运行目录至少包含：

```text
resolved_config.yaml        # 完整配置
split_summary.json          # 数量、类别分布和 split_hash
split_indices.npz           # 四类数据索引
checkpoint_latest.pt        # 可恢复检查点
history.csv                 # 训练/验证及伪标签诊断时间序列
metrics.json                # 最终验证和测试指标
test_predictions.npz        # 测试概率、标签和索引
```

生成统计表与图：

```bash
python -m src.aggregate --input outputs --output outputs/summary
```

主要产物：

- `main_summary.csv`：均值、标准差、95% 置信区间和全标签性能保持率。
- `paired_ssl_gain.csv`：同 seed 下 FixMatch 相对监督模型的配对增益。
- `active_summary.csv`：主动学习曲线与面积统计。
- `label_efficiency_curve.png`：标签比例—准确率曲线。
- `paired_ssl_gain.png`：FixMatch 配对增益及置信区间。
- `confusion_1pct.png`：1% 监督/FixMatch 归一化混淆矩阵。
- `threshold_ablation.png`：阈值、伪标签覆盖率/正确率和测试准确率。
- `active_learning_curve.png`：三种主动学习策略的标注效率。
- `representative_examples.png`：高置信正确、高置信错误和最高不确定样本。

判定“无标签是否有益”时，使用同 seed 配对差值 `FixMatch - supervised`：95% 置信区间完全大于 0 才视为可靠提升；跨过 0 表示证据不足；完全小于 0 表示该条件下无标签训练造成伤害。

## 测试与质量门槛

```bash
python -m pytest -ra
```

测试覆盖：

- 四档标注预算数量、类别均衡、集合互斥和预算嵌套。
- 无标签样本不返回隐藏类别。
- CNN 输出与 256 维特征接口。
- 伪标签置信度掩码。
- 熵采样和不确定性—多样性采样。
- 固定步数训练中断后，EMA 参数与连续训练逐元素一致。
- sweep 条件数量与研究矩阵一致。

若全标签模型最终准确率明显低于常规 CNN 水平，应先检查下载数据、归一化、增强、学习率和训练日志，不应直接把异常结果写入结论。

## 提交前检查

1. 完成所有 seed，确认 `outputs/summary` 没有缺失条件。
2. 报告只引用自动生成的表格和图，不手工改写数值。
3. 如实讨论不显著或负增益，不删除不利结果。
4. 确认测试集没有参与调参、早停或主动学习选择。
5. 将代码、README、配置和最终 PDF 一起打包；不要打包 `data/`、检查点或完整 `outputs/runs/`。
