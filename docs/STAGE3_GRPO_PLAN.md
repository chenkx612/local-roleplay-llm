# 阶段三：GRPO 最小执行计划

## 1. 目标

从已通过阶段二验收的 SFT adapter 继续训练，用最小规模验证 GRPO 能否进一步改善角色一致性和
对话质量，同时保持生成稳定。

本阶段只运行一组配置，不搜索超参数，不使用 Eval 调整奖励或训练。

## 2. 四步流程

### 第一步：准备并验证奖励

采用混合奖励：

```text
R = Readable × (RoleConsistency + DialogueQuality) / 2 - ExploitPenalty
```

- `Readable`：本地规则判断乱码、严重复读和破坏性截断，取 0 或 1。
- `RoleConsistency`：Judge 按 0～10 分评估身份、性格、边界和对话视角。
- `DialogueQuality`：Judge 按 0～10 分评估相关性、自然度和内容价值。
- `ExploitPenalty`：本地规则只惩罚大段照抄 Persona 或风格样例。

训练前用 SFT adapter 生成至少 5 组候选，人工确认奖励排序基本合理。重点检查“吾辈”、普通宠物
边界、错误自称、无依据人物或共同经历，以及回答相关性。通过后冻结奖励。

### 第二步：执行一次 GRPO

从 `output/morgana-v2/stage2-sft/4/adapter` 开始，使用冻结的 20 条 GRPO Prompt：

```yaml
num_generations: 4
max_completion_length: 256
learning_rate: 1e-6
num_train_epochs: 1
enable_thinking: false
```

启动时校验输入哈希、模型 revision 和 SFT adapter。训练结束后检查训练正常完成、梯度有限、
adapter 有非零更新且能够重新加载。技术失败时保留日志并停止。

### 第三步：比较 GRPO 与 SFT

在同一后端、聊天模板、生成参数和固定 seed 下，为 10 条 Dev 分别生成 SFT 与 GRPO 回答，并按
`id` 对齐。

先自动检查非空、正常结束、截断、乱码和严重复读，再匿名人工比较角色一致性和对话质量。GRPO
需满足：

- 至少胜出 6 对，明显落后不超过 2 对；
- 没有不可读、角色崩坏或视角错位等严重问题；
- 三个维度的平均分均不低于 SFT。

### 第四步：记录决定

- 技术检查和 Dev 评估均通过：状态记为 `ready_for_eval`，进入阶段四。
- 任一门槛失败：状态记为 `grpo_failed`，保留结果，不继续调参。
- 将实际配置、产物、观察和阶段决定写入 `RUNLOG.md`；已知问题写入 `ISSUES.md`。

## 3. 最小产物

产物统一保存到 `output/morgana-v2/stage3-grpo/<run-id>/`：

```text
run_summary.json
training_config.yaml
train.log
reward_calibration.jsonl
reward_samples.jsonl
adapter/
sft_dev_outputs.jsonl
grpo_dev_outputs.jsonl
manual_review_results.json
```

`run_summary.json` 至少记录输入哈希、实际环境与配置、训练技术检查、奖励统计、Dev 自动检查、
人工比较结果和最终状态。

## 4. 完成标准

- [ ] 至少 5 组奖励样本已经人工确认，奖励随后冻结。
- [ ] 一次正式 GRPO 已完成，或技术失败已留下可检查证据。
- [ ] GRPO 与 SFT 的三层 Dev 对比已经完成。
- [ ] 最终状态、实际结果和遗留问题已经记录。
