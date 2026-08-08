# AutoDL Stage 2 SFT

本入口在 AutoDL 单张 24GB GPU 上运行 morgana-v2 的第二次 SFT。训练参数、输入哈希、模型
revision 和验收门槛与 Colab 方案相同；现有 Colab notebook 保持不变。

## 1. 准备环境

实例使用 `PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8` 基础镜像。项目安装到
独立虚拟环境后会使用固定的 PyTorch 2.10.0+cu128，不修改镜像默认 Python 环境。

首次部署：

```bash
cd /root/autodl-tmp
git clone <你的 GitHub 仓库地址> roleplay
cd roleplay
```

后续更新：

```bash
cd /root/autodl-tmp/roleplay
git pull --ff-only
```

创建固定环境：

```bash

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements/stage2_sft_autodl.txt
python -m pip install -e .

export HF_HOME=/root/autodl-tmp/huggingface
python -m unittest discover -s tests -v
roleplay-stage2-sft --help
```

依赖安装是显式步骤；`roleplay-stage2-sft` 不会自行安装或升级包。正式运行前，CLI 会检查
Linux、Python 3.12、PyTorch 2.10、CUDA 12.8、单卡、至少 20GiB 显存、固定直接依赖以及
tracked Git 文件无未提交修改。

## 2. 运行

建议在 tmux 中运行，避免 SSH 断开终止训练：

```bash
tmux new -s roleplay-sft
cd /root/autodl-tmp/roleplay
source .venv/bin/activate
export HF_HOME=/root/autodl-tmp/huggingface
roleplay-stage2-sft run
```

按 `Ctrl-B` 后按 `D` 可退出 tmux。重新连接后恢复：

```bash
tmux attach -t roleplay-sft
```

另一个终端可查看 GPU：

```bash
watch -n 2 nvidia-smi
```

命令启动时会打印本次 run 目录。默认产物位于：

```text
output/morgana-v2/stage2-sft/<run-id>/
```

成功时只保留 adapter、训练配置、日志、Base/SFT Dev 输出、人工复核文件和 run summary。
失败时 `run_summary.json` 会记录失败阶段，并保留 `.work/<run-id>` 诊断目录。

## 3. 人工复核与下载

填写 run 目录中的 `manual_review_results.json` 后执行：

```bash
roleplay-stage2-sft review \
  --run-dir output/morgana-v2/stage2-sft/<run-id>
```

每个 run 只接受一次非空人工复核提交。下载产物时在本机执行：

```bash
scp -rP <SSH端口> \
  root@<AutoDL地址>:/root/autodl-tmp/roleplay/output/morgana-v2/stage2-sft/<run-id> \
  /Users/chenkx/roleplay/output/morgana-v2/stage2-sft/
```
