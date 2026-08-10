# 阶段三：DPO 最小执行计划

## 1. 目标

DPO 学习“可以稳定比较、但难以写成客观规则”的主观偏好。从已通过验收的 SFT adapter 开始，
提升角色一致性、表达自然度、情绪回应和对话延续性；生成稳定性作为数据前提和回归门槛，明确
且可自动验证的规则留给后续 GRPO。

本阶段只使用一组数据和训练配置，不使用 Dev 搜索参数。第一次 DPO 失败后不返回追加 SFT，
第二轮仅改变偏好数据数量和质量，并从同一份已验收 SFT adapter 重新开始。

## 2. 偏好数据

第二次 DPO run 使用独立的 `dpo_prompts_run2.jsonl`：保留第一次 run 的 20 条 Prompt 内容，再新增 20 条针对
身份指代、背景事实、情绪承接、直接回答和模板化动作的 Prompt。ID 为 `dpo2_0001`～
`dpo2_0040`，五类场景各 8 条，并继续与 SFT、Dev、Eval 严格去重。

执行：

```bash
roleplay-dpo-data prepare
```

固定流程为：

1. 冻结 SFT adapter 为每条 Prompt 生成 2 个固定 seed 候选；只改变 seed且不补采样。
2. 本地排除空值、截断、乱码、严重复读和括号未闭合；两条都稳定才进入裁决包。
3. Codex 离线读取匿名 A/B 产物，直接输出 `clear_preference`、`no_clear_preference` 或
   `teacher_edit`，不调用外部 Judge API，也不再增加人工复核。
4. 两条都差时只对较好候选做最小修改；长度变化不超过 30%，字符相似度至少 0.60。
5. 平局、实质权衡、纯硬规则差异和无法小幅修复的样本直接排除。

偏好必须至少由角色一致性、表达自然度、情绪回应或对话延续性之一支撑，且另一个核心维度不
明显退化。纯硬规则差异、实质权衡和平局不进入训练。

## 3. 复核与冻结

Codex 填写 `codex_review_results.json` 后执行：

```bash
roleplay-dpo-data finalize --run-dir output/morgana-v2/stage3-dpo/data/<run-id>
```

冻结门槛：

- 至少 30 对有效 Codex 偏好。
- Teacher 修改参与的偏好对不超过最终数据的三分之一。
- 平局、实质权衡和纯硬规则差异直接排除。
- 训练文件严格使用 ms-swift 的 `messages + rejected_response` 格式，审计信息单独保存。

## 4. 最小产物

数据准备 run 保存 80 条候选、匿名 Codex 裁决包、映射 key、Codex 结果和 `run_summary.json`。
冻结后的审计文件保留全部采用、修改和淘汰理由，训练文件不混入审计字段。

当前 run `20260810-dpo-data-2` 生成 80 条候选，其中 73 条通过稳定性检查，6 组提前过滤，
34 组进入 Codex 裁决。最终确认 21 对直接偏好、9 对最小修改偏好，裁决阶段排除 4 对，冻结
共 30 对。正式训练集为
`data/runs/morgana-v2/dpo_train_run2.jsonl`，SHA-256 为
`89dd2030fab814454943b312fd65e619f0c807d93076aee7f6878c72fad8bb82`。

## 5. AutoDL 训练与复核

使用 `roleplay-stage3-dpo` 的四个子命令执行：

```bash
roleplay-stage3-dpo run
roleplay-stage3-dpo publish --run-dir output/morgana-v2/stage3-dpo/<run-id>
roleplay-stage3-dpo download --tag <Release-tag>
roleplay-stage3-dpo review --run-dir output/morgana-v2/stage3-dpo/<run-id>
```

当前诊断配置为 FP32 QLoRA DPO：SFT adapter 同时作为 policy 起点和 reference，
`beta=0.1`、`loss_type=sigmoid`、`learning_rate=1e-6`、1 epoch、物理 batch size 1、
梯度累积 4，共 8 个 optimizer steps。相较失败的第二次 DPO run，只改变 epoch 数，用于验证
后续 epoch 是否放大偏好数据中的噪声。DPO 沿用阶段二依赖，不安装可选注意力内核，也不调用
外部 Judge API。

训练结束后使用相同推理链路、聊天模板、生成参数和固定 seed 生成 SFT/DPO Dev 对照。自动门槛
除完整性、停止原因、截断、复读和乱码外，还要求总退化数、括号未闭合数、异常符号数和错误
自称数均不得高于 SFT。匿名人工复核要求至少胜 6/10、明显落后不超过 2/10、无严重问题，且
生成稳定性、角色一致性和对话质量三项均分不低于 SFT。全部通过时状态为 `ready_for_grpo`，
否则为 `dpo_failed`。
