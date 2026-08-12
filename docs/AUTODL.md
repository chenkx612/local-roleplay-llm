# AutoDL：SFT、DPO 与 GRPO

在 AutoDL 单张 24GB GPU 上运行 morgana-v2 的阶段二 SFT、阶段三 DPO、历史主观 GRPO、
Stage 4 规则型 GRPO 和最终的 post-GRPO DPO。

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

`run` 会校验冻结的 31 对偏好数据、SFT adapter、AutoDL 环境和 DPO 配置，完成 8 个
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

## 5. Stage 4 规则型 GRPO

### 5.1 准备与训练

将已通过验收的 Stage 2 adapter 放到固定路径
`output/morgana-v2/stage2-sft/final/adapter/`。规则奖励完全本地运行，不要设置
`DEEPSEEK_API_KEY`：

```bash
cd /root/autodl-tmp/roleplay
python -m pip install \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -r requirements/stage4_grpo_autodl.txt
python -m pip install -e .

tmux new -s roleplay-stage4-grpo
roleplay-stage4-grpo run
```

`run` 校验冻结 SFT adapter、20条规则 Prompt、配置和依赖，完成训练后生成本地奖励日志、逐条
SFT/GRPO Dev 规则分和匿名复核材料。每次 `run` 都从冻结 SFT adapter 开始，奖励公式不变，
结果保存在 `output/morgana-v2/stage4-grpo/<run-id>/`，失败现场保存在同目录的 `.work/` 下。

### 5.2 发布、下载和复核

```bash
gh auth login  # 仅首次需要
roleplay-stage4-grpo publish --run-dir output/morgana-v2/stage4-grpo/<run-id>
```

在本地下载输出的 Release tag：

```bash
cd /Users/chenkx/roleplay
git pull --ff-only
python -m pip install -e .
roleplay-stage4-grpo download \
  --tag morgana-v2-stage4-grpo-<run-id>
```

按照 `manual_review_packet.json` 和其中的 `severe_issue_codes` 填写
`manual_review_results.json`，然后执行：

```bash
roleplay-stage4-grpo review \
  --run-dir output/morgana-v2/stage4-grpo/<run-id>
```

`review` 会完整记录匿名胜负和三维评分，但只以 GRPO 相对 SFT 新增的严重问题阻断
`ready_for_eval`。自动规则门槛失败时，即使人工复核通过也不会进入 Eval。

## 6. Post-GRPO DPO

该阶段固定合并原始20对与扩充41对训练数据，并使用 `20260812-2144` GRPO adapter，训练
1个 epoch、物理 batch size 2、梯度累积1、学习率 `2e-7`，共31个 optimizer steps。9条
holdout 只在训练后用于 GRPO/DPO 对比，不进入训练。

### 6.1 AutoDL 训练与发布

先确认 GRPO adapter 位于
`output/morgana-v2/stage4-grpo/20260812-2144/adapter/`；若当前机器没有，先下载 Stage 4
Release：

```bash
roleplay-stage4-grpo download \
  --tag morgana-v2-stage4-grpo-20260812-2144
```

然后运行：

```bash
cd /root/autodl-tmp/roleplay
git pull --ff-only
python -m pip install \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -r requirements/stage3_dpo_autodl.txt
python -m pip install -e .

tmux new -s roleplay-post-dpo
roleplay-post-grpo-dpo run
```

训练完成后发布输出中的 `<run-id>`：

```bash
gh auth login  # 仅首次需要
roleplay-post-grpo-dpo publish --run-dir output/morgana-v2/post-grpo-dpo/train/<run-id>
```

### 6.2 本地下载与复核

```bash
cd /Users/chenkx/roleplay
git pull --ff-only
python -m pip install -e .
roleplay-post-grpo-dpo download \
  --tag morgana-v2-post-grpo-dpo-<run-id>
```

对照 `manual_review_packet.json` 填写同目录的 `manual_review_results.json`，再提交复核：

```bash
roleplay-post-grpo-dpo review \
  --run-dir output/morgana-v2/post-grpo-dpo/train/<run-id>
```

只有自动稳定性与 Reward v2 均未回退，且匿名人工复核通过，状态才会变为
`ready_for_final_eval`。失败训练保留在 `.work/<run-id>/`，已存在的 run、发布包和下载目录均不会
被覆盖。

## 7. Post-GRPO DPO 扩充采样

本轮只使用冻结的 `20260812-2144` GRPO adapter，对60条全新 Prompt 每题批量采样8次，
共480条。9条 holdout 和之前两次 DPO adapter 均不参与。AutoDL 安装依赖后运行：

```bash
cd /root/autodl-tmp/roleplay
git pull --ff-only
python -m pip install \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -r requirements/stage3_dpo_autodl.txt
python -m pip install -e .

tmux new -s roleplay-dpo-sampling
roleplay-post-grpo-dpo-sampling run
```

`run` 会逐阶段显示环境校验、模型加载、`Prompt x/60`、候选数、有效数、耗时和 ETA；
相同内容同步保存在 `sampling.log`。完成后在 AutoDL 发布：

```bash
roleplay-post-grpo-dpo-sampling publish --run-dir output/morgana-v2/post-grpo-dpo/sampling/<run-id>
```

回到本地下载：

```bash
cd /Users/chenkx/roleplay
git pull --ff-only
python -m pip install -e .
roleplay-post-grpo-dpo-sampling download \
  --tag morgana-v2-post-grpo-dpo-sampling-<run-id>
```

然后让 Codex 依据 `review_packet.json` 填写同目录的 `review_results.json`，并使用新的输出名
构造 pair：

```bash
roleplay-post-grpo-dpo-data finalize \
  --run-dir output/morgana-v2/post-grpo-dpo/sampling/<run-id> \
  --train-output data/runs/morgana-v2/post_grpo_dpo_train_expansion.jsonl \
  --audit-output data/runs/morgana-v2/post_grpo_dpo_train_expansion_audit.json
```

新批次达到40～60对、每个目标至少12对且 Teacher chosen 不超过25%时，才标记为
`ready_for_dpo`；不足时只保留审计结果，不补采样。
