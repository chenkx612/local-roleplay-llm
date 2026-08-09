# AutoDL Stage 2 SFT

在 AutoDL 单张 24GB GPU 上运行 morgana-v2 的第二次 SFT。参数、输入哈希、模型 revision
和验收门槛与 Colab 方案一致。

## 1. 准备

使用 `PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8` 镜像及其默认 Python，
不要创建虚拟环境或重装 PyTorch。项目使用 Transformers 的 PyTorch fallback，不安装
`flash-linear-attention` 和 `causal-conv1d`。

首次部署：

```bash
cd /root/autodl-tmp
git clone <你的 GitHub 仓库地址> roleplay
cd roleplay

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

后续只需更新代码：

```bash
cd /root/autodl-tmp/roleplay
git pull --ff-only
```

CLI 会在训练前检查系统、Python、PyTorch、CUDA、GPU、显存、依赖和 Git 工作区。
默认使用 `https://hf-mirror.com`，缓存写入 `/root/autodl-tmp/huggingface`；已有的
`HF_ENDPOINT` 和 `HF_HOME` 不会被覆盖。

## 2. 训练与发布

在 tmux 中运行，避免 SSH 断开中止训练：

```bash
tmux new -s roleplay-sft
cd /root/autodl-tmp/roleplay
roleplay-stage2-sft run
```

按 `Ctrl-B`、`D` 退出，之后用 `tmux attach -t roleplay-sft` 恢复。可在另一终端运行
`watch -n 2 nvidia-smi` 查看 GPU。

默认产物位于 `output/morgana-v2/stage2-sft/<run-id>/`。训练和自动评测成功后发布：

```bash
gh auth login  # 仅首次需要
roleplay-stage2-sft publish --run-dir output/morgana-v2/stage2-sft/<run-id>
```

`publish` 会校验归档并上传 GitHub Release。记下输出的 Release tag，即可释放实例。

## 3. 本地复核

```bash
cd /Users/chenkx/roleplay
git pull --ff-only
python -m pip install -e .
roleplay-stage2-sft download --tag <Release-tag>
```

填写下载目录中的 `manual_review_results.json`，然后执行：

```bash
roleplay-stage2-sft review \
  --run-dir output/morgana-v2/stage2-sft/<run-id>

git add -f \
  output/morgana-v2/stage2-sft/<run-id>/run_summary.json \
  output/morgana-v2/stage2-sft/<run-id>/manual_review_results.json
git commit -m "chore: record morgana-v2 SFT review"
git push
```

完整流程为 `run → publish → download → review`。Adapter 不提交到 Git。

## 4. 失败处理

成功后，临时日志和 checkpoint 会自动删除；训练摘要保存在 `run_summary.json`。失败时，
完整日志和 checkpoint 留在 `.work/<run-id>` 供现场排查，不上传 GitHub。按需提交失败 run 的
`run_summary.json` 即可。
