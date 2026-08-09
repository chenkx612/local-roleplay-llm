# AutoDL：SFT、DPO 与 GRPO

在 AutoDL 单张 24GB GPU 上运行 morgana-v2 的阶段二 SFT、阶段三 DPO 和历史 GRPO。

## 1. 准备

使用 `PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8` 镜像。使用镜像默认
Python，不创建虚拟环境或重装 PyTorch。

```bash
cd /root/autodl-tmp
git clone <你的 GitHub 仓库地址> roleplay
cd roleplay
python -m pip uninstall -y flash-linear-attention causal-conv1d
python -m pip install \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -r requirements/stage2_sft_autodl.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

后续重新开机或开始新一轮训练前，先更新代码：

```bash
cd /root/autodl-tmp/roleplay
git pull --ff-only
python -m pip install -e .
```

## 2. SFT

### 2.1 运行训练

```bash
tmux new -s roleplay-sft
cd /root/autodl-tmp/roleplay
roleplay-stage2-sft run
```

按 `Ctrl-B`、`D` 退出 tmux；使用 `tmux attach -t roleplay-sft` 恢复。

### 2.2 发布训练产物

训练成功后发布：

```bash
gh auth login  # 仅首次需要
roleplay-stage2-sft publish --run-dir output/morgana-v2/stage2-sft/<run-id>
```

记下输出的 Release tag，供本地复核时下载产物。

### 2.3 本地复核

```bash
cd /Users/chenkx/roleplay
git pull --ff-only
python -m pip install -e .
roleplay-stage2-sft download --tag <Release-tag>
```

填写 `manual_review_results.json`，然后执行：

```bash
roleplay-stage2-sft review \
  --run-dir output/morgana-v2/stage2-sft/<run-id>

git add -f \
  output/morgana-v2/stage2-sft/<run-id>/run_summary.json \
  output/morgana-v2/stage2-sft/<run-id>/manual_review_results.json
git commit -m "chore: record morgana-v2 SFT review"
git push
```

### 2.4 失败处理

失败现场保存在 `output/morgana-v2/stage2-sft/.work/<run-id>/`，错误摘要保存在对应 run 的
`run_summary.json`。成功后临时文件自动删除。Adapter 不提交到 Git。

## 3. DPO

### 3.1 准备 SFT adapter

将已通过阶段二验收的 adapter 放到固定路径：

```text
output/morgana-v2/stage2-sft/final/adapter/
```

### 3.2 运行训练

DPO 与 SFT 使用同一套冻结依赖，不安装 `flash-linear-attention` 或 `causal-conv1d`，也不需要
`DEEPSEEK_API_KEY`：

```bash
cd /root/autodl-tmp/roleplay
python -m pip uninstall -y flash-linear-attention causal-conv1d
python -m pip install \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -r requirements/stage3_dpo_autodl.txt
python -m pip install -e .

tmux new -s roleplay-dpo
roleplay-stage3-dpo run
```

`run` 会校验冻结的 30 对偏好数据、SFT adapter、AutoDL 环境和 DPO 配置，完成 24 个
optimizer steps，并生成 SFT/DPO Dev 匿名对比材料。

### 3.3 发布训练产物

```bash
gh auth login  # 仅首次需要
roleplay-stage3-dpo publish --run-dir output/morgana-v2/stage3-dpo/<run-id>
```

记下输出的 Release tag。

### 3.4 本地复核

```bash
cd /Users/chenkx/roleplay
git pull --ff-only
python -m pip install -e .
roleplay-stage3-dpo download --tag <Release-tag>
```

根据 `manual_review_packet.json` 填写 `manual_review_results.json`，然后执行：

```bash
roleplay-stage3-dpo review --run-dir output/morgana-v2/stage3-dpo/<run-id>

git add -f \
  output/morgana-v2/stage3-dpo/<run-id>/run_summary.json \
  output/morgana-v2/stage3-dpo/<run-id>/manual_review_results.json
git commit -m "chore: record morgana-v2 DPO review"
git push
```

### 3.5 失败处理

失败现场保存在 `output/morgana-v2/stage3-dpo/.work/<run-id>/`，错误摘要保存在对应 run 的
`run_summary.json`。成功后临时文件自动删除。Adapter 不提交到 Git。

## 4. GRPO（历史流程）

### 4.1 准备 SFT adapter

将 Stage 2 adapter 放到固定路径：

```text
output/morgana-v2/stage2-sft/final/adapter/
```

### 4.2 运行训练

```bash
cd /root/autodl-tmp/roleplay
python -m pip install \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -r requirements/stage3_grpo_autodl.txt
python -m pip install -e .
export DEEPSEEK_API_KEY="<your-key>"

tmux new -s roleplay-grpo
roleplay-stage3-grpo run
```

Qwen3.5 的 GRPO 前向会使用变长线性注意力内核，因此阶段三额外固定安装
`flash-linear-attention==0.4.2`。该 PyPI wheel 使用 PyTorch/Triton，不要求安装
`causal-conv1d`；阶段三也显式安装 ms-swift 运行时使用但未声明的
`msgspec==0.21.1`。阶段二仍保持不安装两个可选加速包的原始冻结环境。

`run` 会完成训练检查，并生成 SFT/GRPO Dev 匿名对比材料。

### 4.3 发布训练产物

训练成功后发布：

```bash
gh auth login  # 仅首次需要
roleplay-stage3-grpo publish --run-dir output/morgana-v2/stage3-grpo/<run-id>
```

记下输出的 Release tag。

### 4.4 本地复核

```bash
cd /Users/chenkx/roleplay
git pull --ff-only
python -m pip install -e .
roleplay-stage3-grpo download --tag <Release-tag>
```

根据 `manual_review_packet.json` 填写 `manual_review_results.json`，然后执行：

```bash
roleplay-stage3-grpo review --run-dir output/morgana-v2/stage3-grpo/<run-id>

git add -f \
  output/morgana-v2/stage3-grpo/<run-id>/run_summary.json \
  output/morgana-v2/stage3-grpo/<run-id>/manual_review_results.json
git commit -m "chore: record morgana-v2 GRPO review"
git push
```
