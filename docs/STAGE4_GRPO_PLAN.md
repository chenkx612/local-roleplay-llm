# Stage 4：规则型 GRPO 执行计划

## 目标

从冻结 SFT adapter 运行一次小规模规则型 GRPO，改善简洁度、标志性自称和动作密度，并保护
生成稳定性。失败 DPO 和旧主观 Judge GRPO adapter 均不参与训练。

## 冻结输入与配置

- 20条 `rule_grpo_NNNN` Prompt 与其他 v2 split 全局去重，动作策略为10条 `encouraged`、8条
  `optional` 和2条 `forbidden`。
- 8候选、1 epoch、20个 optimizer steps、`learning_rate=1e-6`、`beta=0.1`、FP32 QLoRA。
- 奖励规约见 [`STAGE4_REWARD_SPEC.md`](STAGE4_REWARD_SPEC.md)，训练后不得依据 Dev 改奖励。

## 当前 run 配置

首次 run `20260811-2249` 保留为负向证据。后续每轮仍从同一个冻结 SFT adapter 开始，不继承
此前 GRPO adapter，也不修改奖励公式。当前冻结输入使用课程化20条 Prompt，并合并以下改动：

- `num_generations=8`，对应 `gradient_accumulation_steps=8`；
- `temperature=0.8`、`top_p=0.9`；
- `learning_rate=1e-6`；
- 保持1 epoch、20个 optimizer steps 和 `beta=0.1`。

运行命令为 `roleplay-stage4-grpo run`，每次结果由独立 `<run-id>` 隔离在
`output/morgana-v2/stage4-grpo/`。课程 Prompt 保持20个固定 ID 和动作策略，奖励函数不变。

## 四命令流程

```bash
roleplay-stage4-grpo run
roleplay-stage4-grpo publish --run-dir output/morgana-v2/stage4-grpo/<run-id>
roleplay-stage4-grpo download --tag morgana-v2-stage4-grpo-<run-id>
roleplay-stage4-grpo review --run-dir output/morgana-v2/stage4-grpo/<run-id>
```

`run` 完成输入和环境校验、训练、adapter 更新检查、SFT/GRPO Dev 生成、规则统计和匿名复核材料。
`publish`/`download` 使用带逐文件 SHA-256 的 GitHub Release。`review` 记录匿名评分，只有新增严重
问题会阻断 `ready_for_eval`；主观胜负和均分完整报告但不是本阶段强制门槛。

## 产物与完成标准

run 目录包含训练配置和日志、本地奖励定义及逐候选日志、逐条 Dev 规则分、SFT/GRPO 输出、匿名
复核材料和 adapter。自动规则门槛与人工严重回归门槛均通过后，状态为 `ready_for_eval`；否则保留
完整负向证据并停止，不在同一轮静默调参。
