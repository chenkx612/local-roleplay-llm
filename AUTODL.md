# AutoDL Stage 2 SFT

本入口在 AutoDL 单张 24GB GPU 上运行 morgana-v2 的第二次 SFT。训练参数、输入哈希、模型
revision 和验收门槛与 Colab 方案相同；现有 Colab notebook 保持不变。

## 1. 准备环境

实例使用 `PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8` 基础镜像，并直接使用
镜像的 `/root/miniconda3/bin/python`。不创建额外虚拟环境，也不重复安装 PyTorch。

Qwen3.5 不安装可选的 `flash-linear-attention` 和 `causal-conv1d`，使用 Transformers 的
PyTorch fallback。它比可选内核更慢、更占显存，但避免下载或编译额外 CUDA 扩展，适合本项目
的小规模学习运行。

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

安装项目依赖：

```bash
which python
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)'
python -m pip uninstall -y flash-linear-attention causal-conv1d
python -m pip install --upgrade pip \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -m pip install \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -r requirements/stage2_sft_autodl.txt
python -m pip install -e .

python -m unittest discover -s tests -v
roleplay-stage2-sft --help
```

`which python` 应输出 `/root/miniconda3/bin/python`，环境检查应输出 PyTorch `2.8.0+cu128`、
CUDA `12.8` 和 CXX11 ABI `True`。

依赖安装是显式步骤；`roleplay-stage2-sft` 不会自行安装或升级包。正式运行前，CLI 会检查
Linux、Python 3.12、PyTorch 2.8、CUDA 12.8、单卡、至少 20GiB 显存、固定直接依赖、可选
加速包未安装以及 tracked Git 文件无未提交修改。

`roleplay-stage2-sft run` 会默认设置 `HF_ENDPOINT=https://hf-mirror.com` 和
`HF_HOME=/root/autodl-tmp/huggingface`，无需在每次登录或新建 tmux 会话后重新 `export`。
如需使用其他端点或缓存目录，在运行命令前显式设置对应环境变量即可；CLI 不会覆盖用户设置。

## 2. 运行

建议在 tmux 中运行，避免 SSH 断开终止训练：

```bash
tmux new -s roleplay-sft
cd /root/autodl-tmp/roleplay
roleplay-stage2-sft run
```

按 `Ctrl-B` 后按 `D` 可退出 tmux。重新连接后恢复：

```bash
tmux attach -t roleplay-sft
```

恢复 tmux 后不需要激活额外环境。

另一个终端可查看 GPU：

```bash
watch -n 2 nvidia-smi
```

命令启动时会打印本次 run 目录。默认产物位于：

```text
output/morgana-v2/stage2-sft/<run-id>/
```

终端只显示环境、输入、训练、自动评估和归档状态；训练期间约输出 5 次包含 step、loss、梯度和
预计剩余时间的进度。第三方库的完整原始输出仍写入临时工作目录中的 `train.log`，失败时可据此诊断。

成功时只保留 adapter、训练配置、Base/SFT Dev 输出、人工复核文件和 run summary。完整训练日志和
checkpoint 只存在于临时工作目录；成功归档后自动删除，训练曲线和梯度摘要已经写入
`run_summary.json`。失败时 run 目录只保留 `run_summary.json`，完整日志和 checkpoint 留在
`.work/<run-id>` 供 AutoDL 现场诊断，不通过 GitHub 同步。

## 3. AutoDL 发布

GitHub CLI 只需首次安装和登录。若镜像没有 `gh`，参考
[GitHub CLI 官方 Linux 安装说明](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)。

```bash
gh auth login
```

训练和自动评测完成后直接发布，不在 AutoDL 做人工复核：

```bash
roleplay-stage2-sft publish \
  --run-dir output/morgana-v2/stage2-sft/<run-id>
```

`publish` 自动精简旧产物、校验归档、生成哈希清单并上传 GitHub Release。它只保留后续需要的
adapter 和评测/复核材料；日志、checkpoint 和 `.work` 不上传。记下命令打印的 Release tag，
随后即可释放 AutoDL 实例。

## 4. 本地下载与人工复核

本地更新代码并下载发布包：

```bash
cd /Users/chenkx/roleplay
git pull --ff-only
python -m pip install -e .
roleplay-stage2-sft download --tag <Release-tag>
```

`download` 会自动校验总包和逐文件 SHA-256，再解包到
`output/morgana-v2/stage2-sft/<run-id>/`。填写其中的 `manual_review_results.json` 后执行：

```bash
roleplay-stage2-sft review \
  --run-dir output/morgana-v2/stage2-sft/<run-id>
```

最后只提交人工复核结果和摘要，不把 adapter 放进 Git 历史：

```bash
git add -f \
  output/morgana-v2/stage2-sft/<run-id>/run_summary.json \
  output/morgana-v2/stage2-sft/<run-id>/manual_review_results.json
git commit -m "chore: record morgana-v2 SFT review"
git push
```

日常流程只有四个动作：`run → publish → download → review`。失败 run 只需按需提交
`run_summary.json`，不上传诊断目录。
