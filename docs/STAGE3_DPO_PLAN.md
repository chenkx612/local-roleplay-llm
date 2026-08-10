# 阶段三：DPO 最小执行计划

## 1. 目标

DPO 学习“可以稳定比较、但难以写成客观规则”的主观偏好。从已通过验收的 SFT adapter 开始，
提升角色一致性、表达自然度、情绪回应和对话延续性；生成稳定性作为数据前提和回归门槛，明确
且可自动验证的规则留给后续 GRPO。

本阶段只使用一组数据和训练配置，不使用 Dev 搜索参数。连续失败后不返回追加 SFT，也不沿用
失败 DPO adapter；当前实验从同一份已验收 SFT adapter 开始，只提高偏好数据质量。

## 2. 偏好数据

当前 DPO 数据 run 使用 `dpo_prompts_run2.jsonl`：保留第一次 run 的 20 条 Prompt 内容，再新增 20 条针对
身份指代、背景事实、情绪承接、直接回答和模板化动作的 Prompt。ID 为 `dpo2_0001`～
`dpo2_0040`，五类场景各 8 条，并继续与 SFT、Dev、Eval 严格去重。

执行：

```bash
roleplay-dpo-data prepare
```

固定流程为：

1. 冻结 SFT adapter 为每条 Prompt 生成 4 个固定 seed 候选；只改变 seed且不补采样。
2. 本地排除空值、截断、乱码、严重复读和括号未闭合；至少两条稳定才进入裁决包。
3. Codex 离线读取匿名 A～D 产物，给全部候选的生成稳定性、角色一致性、对话质量打 0～10 分，
   并直接输出 `clear_preference`、`no_clear_preference` 或 `teacher_edit`；不调用外部 API。
4. chosen 必须满足稳定性至少 8、角色一致性至少 7、对话质量至少 7，且没有已定义问题；
   rejected 稳定性至少 7，chosen 在角色一致性或对话质量上至少领先 2 分。
5. 两条都差时只对较好候选做局部修改；长度为原文的 50%～150%，字符相似度至少 0.40。
6. 平局、共同缺陷、实质权衡、纯硬规则差异和无法局部修复的样本直接排除。

偏好必须至少由角色一致性、表达自然度、情绪回应或对话延续性之一支撑，且另一个核心维度不
明显退化。纯硬规则差异、实质权衡和平局不进入训练。

前三次 DPO 均未改善 Dev 主观质量后，新增 chosen 最小编辑判定实验。实验从同一候选池中为
每条 Prompt 选择最接近优秀回答的稳定候选；来源必须达到稳定性 8 分、目标主观维度 7 分、
另一主观维度 8 分，且不得含事实错误、指代错误或多个缺陷。chosen 只提升目标主观维度并达到
8/8/8，同时要求字符相似度至少 `0.65`、长度比在 `0.80～1.20`，用于验证当前 SFT 输出分布
附近是否存在可学习的高质量偏好对。

## 3. 复核与冻结

Codex 填写 `codex_review_results.json` 后执行：

```bash
roleplay-dpo-data finalize --run-dir output/morgana-v2/stage3-dpo/data/<run-id>
```

冻结门槛：

- 至少 30 对有效 Codex 偏好。
- Teacher 修改参与的偏好对不超过最终数据的五分之一。
- 平局、实质权衡和纯硬规则差异直接排除。
- 训练文件严格使用 ms-swift 的 `messages + rejected_response` 格式，审计信息单独保存。

上述 30 对门槛适用于常规 DPO 数据发布。本次是一次明确的诊断例外：判定实验只得到 17 条
合格 pair，不宣称数据覆盖充分；使用它们训练是为了隔离验证“少量但高质量的局部偏好对”能否
避免此前 DPO 退化，不能据此降低最终 Dev 验收标准。

## 4. 最小产物

数据准备 run 保存 160 条候选、匿名 Codex 裁决包、映射 key、Codex 结果和 `run_summary.json`。
冻结后的审计文件保留全部采用、修改和淘汰理由，训练文件不混入审计字段。

来源 run `20260810-dpo-data-3` 生成 160 条候选，其中 146 条通过稳定性检查。判定实验
`20260811-dpo-editability-1` 覆盖全部 40 条 Prompt，最终 17 条可通过单维最小编辑达到
8/8/8，23 条不可编辑。成功 pair 平均字符相似度 `0.772`、平均长度比 `0.965`。

使用以下命令从已完成且哈希一致的判定报告导出训练集：

```bash
roleplay-dpo-editability export-training
```

正式训练集为 `data/runs/morgana-v2/dpo_train_editability17.jsonl`，共 17 对，SHA-256 为
`2836a41969ae250b3ddea692cf441c5c95179723cc5a81bb07c5c0892c39e922`。

## 5. AutoDL 训练与复核

使用 `roleplay-stage3-dpo` 的四个子命令执行：

```bash
roleplay-stage3-dpo run
roleplay-stage3-dpo publish --run-dir output/morgana-v2/stage3-dpo/<run-id>
roleplay-stage3-dpo download --tag <Release-tag>
roleplay-stage3-dpo review --run-dir output/morgana-v2/stage3-dpo/<run-id>
```

当前诊断配置为 FP32 QLoRA DPO：SFT adapter 同时作为 policy 起点和 reference，
`beta=0.1`、`loss_type=sigmoid`、`rpo_alpha=0.3`、`learning_rate=5e-7`、1 epoch、物理
batch size 1、梯度累积 2，共 9 个 optimizer steps。相较失败的 31 对 run，本次只使用 17 条
最小编辑 pair；降低学习率和 RPO 权重以限制分布漂移，减小梯度累积以在不重复数据的前提下
保留足够更新次数。LoRA、seed、推理参数和评测口径不变。DPO 沿用阶段二依赖，不安装可选
注意力内核，也不调用外部 Judge API。

训练结束后使用相同推理链路、聊天模板、生成参数和固定 seed 生成 SFT/DPO Dev 对照。自动门槛
除完整性、停止原因、截断、复读和乱码外，还要求总退化数、括号未闭合数、异常符号数和错误
自称数均不得高于 SFT。匿名人工复核要求至少胜 6/10、明显落后不超过 2/10、无严重问题，且
生成稳定性、角色一致性和对话质量三项均分不低于 SFT。全部通过时状态为 `ready_for_grpo`，
否则为 `dpo_failed`。
