# 本机 RTX 3060 Ti 实验配置

本文件记录 2026-09-08 的本机部署方式。项目已位于本机，运行不依赖 Colab 上传包。

## 环境与安装

- Windows x64；Python 3.12（项目本机脚本支持 3.11–3.12）。
- NVIDIA GeForce RTX 3060 Ti，8 GiB 显存；检测驱动版本 560.94。
- PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121；CUDA runtime 由官方 wheel 提供，无需另外安装完整 CUDA Toolkit。
- NumPy 1.26.4；其余依赖遵循 requirements.txt。安装约束见 requirements-cuda.txt，实装精确版本保存到 requirements-local-lock.txt。
- 版本配对来源：https://pytorch.org/get-started/previous-versions/#v251 。采用 CUDA 12.1 是遵循实验记录指定的安装通道，不表示其为最新版本。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_cuda.ps1
```

脚本优先使用用户目录安装的 Python 3.12，也可传入 `-Python <解释器路径>`。失效或版本不兼容的旧 .venv 会保存在项目 tmp/venv-legacy-时间戳；不会沿用旧电脑的 D 盘解释器。脚本创建环境、安装 CUDA 依赖、检查包冲突、实测 CUDA AMP 前向和反向计算，并运行项目测试。

官方源下载慢时，Python 3.12 x64 可先运行 `python scripts/download_cuda_wheel.py`。下载器仅使用 PyTorch 官方源，验证 Content-Range、文件长度及官方索引 SHA-256；安装脚本随后复用本地 wheel。

## 正式训练

配置：configs/local_cuda.yaml。使用现有本地数据（download=false），强制 device=cuda，启用 AMP 和确定性训练；保持正式方案的 12,000 步、64 个标注样本及 192 个无标签样本批量、500 步验证/检查点间隔。每 100 步输出训练进度。CPU 张量计算默认使用 4 个线程，可通过 ML_TORCH_THREADS 设置。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_local_training.ps1
```

训练以隐藏后台进程启动，依次执行 main、ablation、active_learning、aggregate。2026-09-09 起后台入口默认使用 4 个数据预取进程，详见 [性能测量与一致性验证](performance.md)。队列通过文件锁避免重复启动，并在训练期间临时阻止 Windows 自动睡眠；结束后撤销请求。不改变永久电源设置，无法阻止手动关机、重启或手动睡眠。

主实验 50 次训练；消融新增 6 次（另 3 次复用主实验）；主动学习 21 次（3 seeds × [1 个共享初始模型 + 3 策略 × 2 轮]），共 77 次实际训练。主动学习是课设可选挑战题，本地队列按当前项目完整方案执行。

## 查看进度、恢复与结果

```powershell
Get-Content outputs/local-training/status.json
Get-Content outputs/local-training/main.log -Tail 20
Get-Content outputs/local-training/ablation.log -Tail 20
Get-Content outputs/local-training/active_learning.log -Tail 20
nvidia-smi
```

status.json 记录队列 PID、当前阶段、子进程 PID、日志位置和运行状态。任何阶段非零退出会停止后续阶段并记录 failed。再次运行启动脚本会复用相同配置的已完成结果，并从未完成实验的最近检查点继续；每 500 步保存一次，因此中断后可能重算最多约 499 步。配置或划分不匹配会拒绝复用，需另选输出目录或明确决定重跑，切勿盲目使用 --force。

若队列本身被终止，status.json 可能仍显示 running；同时检查对应 PID 是否存活以及日志是否持续更新。2026-09-09 晚间已修复预取批次的 Windows 共享文件映射分配路径，恢复 4 worker；设置环境变量 ML_PREFETCH_WORKERS=0 可退回串行，通常明显较慢。

正式指标与检查点分别位于 outputs/runs 和 outputs/active；统计图表位于 outputs/summary；CUDA 检测报告位于 outputs/environment-check/environment.json。关闭终端或当前对话不主动终止后台队列。若需停止，应结束 status.json 对应的队列及其子进程树，不要只杀调度进程。

## 解释边界

旧的 50 步试跑只能证明流程与阈值掩码可执行；10% 的测试准确率以及一个 seed 的验证损失差异不能证明过拟合或半监督方法显著有效。历史记录中 3060 Ti 的耗时与显存数字不是本机实测保证，实际以本轮日志及 GPU 检测为准。阈值消融图使用各阈值共同拥有的 seeds，单 seed 不生成有效置信区间。
