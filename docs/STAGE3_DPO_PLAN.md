# 阶段三：DPO 最小执行计划

## 1. 目标

DPO 学习“可以稳定比较、但难以写成客观规则”的主观偏好。从已通过验收的 SFT adapter 开始，
提升角色一致性、表达自然度、情绪回应和对话延续性；生成稳定性作为数据前提和回归门槛，明确
且可自动验证的规则留给后续 GRPO。

本阶段只使用一组数据和训练配置，不使用 Dev 搜索参数，不把单次大模型判断当作真值。

## 2. 偏好数据

现有旧 GRPO 的 20 条主观质量 Prompt 原样迁移为独立的 `dpo_prompts.jsonl`，ID 从
`grpo_NNNN` 改为 `dpo_NNNN`。旧文件只保留为失败 GRPO 的历史输入；未来规则型 GRPO 另建
去重 Prompt。

执行：

```bash
roleplay-dpo-data prepare
```

固定流程为：

1. SFT adapter 为每条 Prompt 生成 3 个固定 seed 候选；只改变 seed。
2. 本地排除空值、截断、乱码、严重复读和括号未闭合。
3. `deepseek-v4-pro` 开启 thinking、使用 `reasoning_effort=max` 做组内偏好判断。
4. 没有清晰偏好或稳定候选不足时只补采样一次。
5. 全部候选都不适合作为 chosen 时，同一模型以 Teacher 身份最小修改最佳候选。
6. 生成匿名 A/B 复核包，由人工最终选择；Judge 推荐和候选来源不展示给复核者。

偏好必须至少由角色一致性、表达自然度、情绪回应或对话延续性之一支撑，且另一个核心维度不
明显退化。纯硬规则差异、实质权衡和平局不进入训练。

## 3. 复核与冻结

填写 `manual_review_results.json` 后执行：

```bash
roleplay-dpo-data finalize --run-dir output/morgana-v2/stage3-dpo/data/<run-id>
```

冻结门槛：

- 至少 16 对有效人工偏好；目标为 20 对。
- Teacher 修改参与的偏好对不超过最终数据的三分之一。
- 人工可以推翻 Judge；平局和存在实质权衡的样本直接排除。
- 训练文件严格使用 ms-swift 的 `messages + rejected_response` 格式，审计信息单独保存。

## 4. 最小产物

数据准备 run 保存候选、Judge 决策、Teacher 修改、匿名复核包、映射 key、人工结果模板和
`run_summary.json`。人工门槛通过后才生成 `dpo_train.jsonl` 和独立审计文件，并进入一次
DPO 训练；本轮数据准备不替代人工复核。

当前 run `20260809-dpo-data-1` 已完成人工盲审并冻结 17 对，其中 Teacher 修改参与 1 对；
3 个 tie/实质权衡项仅保留审计记录。正式训练集为 `data/runs/morgana-v2/dpo_train.jsonl`，
SHA-256 为 `f1db4c30506fa704ac2366ec945a9f0b1302910bf41b667921a2d7f1ce9ae4f9`。
