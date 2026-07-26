# 阶段一 Review Issues

## 优先级说明

- **P0**：阻断后续阶段或会直接导致评测、训练结论失效，必须优先处理。
- **P1**：严重影响结果可信度或任务完整性，应在进入下一阶段前处理。
- **P2**：影响可比性、可追溯性或稳定性，应排期处理。
- **P3**：低影响的优化或体验改进。

## 待解决问题

### P0｜基线输出存在截断与退化，不能作为有效对照

**位置：** `src/roleplay/inference.py:17-35`、`data/baseline_outputs.jsonl`

推理固定使用 `max_tokens=256`，但没有读取或保存 API 返回的 `finish_reason`。当前保存的 50 条结果中，多条明显在句中结束，例如第 3 条以“我的”结尾；另有大量异常循环，例如“蛋蛋蛋蛋……”“特别特别……”和“她她她她……”、 “肉色肉色……”。

这些结果并不是完整、稳定的 Base 回答。如果阶段二继续用它判断 SFT 是否优于 Base，结论会主要反映基线截断或采样退化，而不是微调效果。

**处理建议：**

- 检测并保存 `finish_reason`；将因 token 上限导致的截断视为失败，或自动续跑。
- 对明显复读的结果标记失败，并重新检查生成参数。
- 修复后重新生成完整的 `data/baseline_outputs.jsonl`，不得继续使用当前文件。

### P0｜GRPO 训练集与独立评测集存在精确泄漏

**位置：** `data/rl_train.jsonl`、`data/eval.jsonl`

问题“你以前做过类似的工作吗？”同时出现在 `rl_train.jsonl` 和 `eval.jsonl`。虽然三份数据是分别请求 Teacher 生成的，但代码没有在写出前做跨数据集精确去重，因此不能保证 PLAN 中要求的独立评测。

这会使 GRPO 在训练期间直接见到评测问题，污染 Base/SFT/GRPO 对比。生成完成后至少应对规范化后的 Prompt 做跨 split 精确去重并补齐缺失条目；当前 RL 和 eval 数据也需要重新拆分或重新生成。

### P1｜单条推理失败会被吞掉，命令仍声称整批完成

**位置：** `src/roleplay/inference.py:63-80`

任意 API 异常都会被 `except Exception` 捕获并转换成空字符串，随后写入输出。即使 50 条全部请求失败，命令仍会打印“完成，共 50 条”并以状态码 0 退出。后续评测很容易把空回答当成真实 Base 表现，而流水线无法发现基线实际没有生成成功。

**处理建议：**

- 记录失败数并对失败请求重试。
- 重试后仍失败时，以非零状态退出，或至少明确将整批标记为不完整；不能无条件打印成功。
- 校验最终记录数、非空回答数与 eval 输入数完全一致。

### P2｜Base 默认使用的模型与计划中的训练基座不一致

**位置：** `src/roleplay/inference.py:15`、`PLAN.md` 的“基础模型使用”和阶段二技术方案

基线默认模型是 `mlx-community/Qwen3.5-2B-4bit`，而 PLAN 指定 `Qwen/Qwen3.5-2B`，阶段二还明确优先使用 BF16 LoRA。若 SFT/GRPO 基于官方 BF16 权重，Base 却使用另一份 4-bit 转换权重，量化与转换差异会成为额外变量，三阶段不再是同一基座上的公平比较。

**处理建议：**

- Base、SFT 和 GRPO 使用同一基础 checkpoint、相同精度/量化策略及相同推理后端。
- 至少在输出元数据中记录实际 checkpoint、revision、dtype/量化方式和生成参数。

## 已解决问题

### P0｜数据生成可能少于目标条数，但仍报告成功并覆盖产物

**位置：** `src/roleplay/datagen.py`（`_generate_for_scenario` / `generate` / `write_jsonl`）

**原问题：** `_generate_for_scenario()` 按固定批次数循环；Teacher 某批只返回部分合法记录时，`offset` 仍按请求量推进且不补齐，`generate()` 仍写盘并报成功，可能用残缺数据覆盖完整产物。

**已完成修复：**

- 按场景目标持续请求，直至凑满或连续空批达 `MAX_CONSECUTIVE_EMPTY` 后抛 `GenerationShortfallError`。
- 部分有效回复会保留，`offset` 按实际写入条数前进。
- 三个 split 全部达标后才写盘；写盘使用 `*.tmp` + `replace` 原子替换。
- CLI 捕获 shortfall 并以非零状态退出。
- 已覆盖：部分有效补齐、空回复重试后仍达目标、重试耗尽失败且不覆盖已有产物。
