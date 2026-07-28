# 阶段一 Review Issues

## 优先级说明

- **P0**：阻断后续阶段或会直接导致评测、训练结论失效，必须优先处理。
- **P1**：严重影响结果可信度或任务完整性，应在进入下一阶段前处理。
- **P2**：影响可比性、可追溯性或稳定性，应排期处理。
- **P3**：低影响的优化或体验改进。

## 待解决问题

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

### P0｜GRPO 训练集与独立评测集存在精确泄漏

**位置：** `src/roleplay/datagen.py`、`tests/test_datagen.py`、`data/*.jsonl`

**原问题：** GRPO 与 Eval 存在相同 Prompt，且生成器没有 Dev split，也会在冻结
Prompt 前直接生成 SFT 回答，无法满足训练与评测隔离要求。

**已完成修复：**

- Teacher 只生成用户 Prompt；SFT、GRPO、Dev、Eval 四个 split 全部完成后才写盘。
- 使用 Unicode NFKC、首尾去空白和连续空白折叠构造全局去重键；保留原始 Prompt 文本。
- 同批、同 split 和跨 split 重复均被丢弃并继续补齐；重复耗尽时失败且不覆盖已有产物。
- Smoke 数据已重新生成并验证为 100/30/20/50 条，200 个规范化 Prompt 全局唯一。

### P0｜基线输出存在截断与退化，不能作为有效对照

**位置：** `src/roleplay/inference.py`、`tests/test_inference.py`、`data/baseline_outputs.jsonl`

**原问题：** 推理固定使用 `max_tokens=256`，未读取或保存 `finish_reason`；原有 50 条基线中存在句中截断和“蛋蛋蛋蛋……”“她她她她……”等明显生成退化，不能作为有效 Base 对照。

**已完成修复：**

- 生成参数调整为 `max_tokens=512`、`temperature=0.7`、`top_p=0.8`、`top_k=20`、`presence_penalty=1.5`，并加入 `repetition_penalty=1.1` 和 64-token 重复上下文。
- 保存并校验 `finish_reason`；空回答、非 `stop` 结束及明显连续复读均视为失败，每条最多尝试 3 次。
- 任一条重试耗尽时整批非零退出，不覆盖旧产物；全部成功后才原子替换输出文件。
- 已重新生成 50 条基线：50/50 均以 `stop` 结束、回答非空且未命中复读检测，全部首次生成成功。
- PLAN 1.2 重建 Eval split 后已删除这份旧基线，避免错配；新基线将在 1.4
  针对冻结后的 Dev、GRPO 和 Eval 统一生成。
- 推理测试覆盖参数传递、截断重试、复读重试和失败不覆盖，现有测试全部通过。

### P0｜数据生成可能少于目标条数，但仍报告成功并覆盖产物

**位置：** `src/roleplay/datagen.py`（`_generate_for_scenario` / `generate` / `write_jsonl`）

**原问题：** `_generate_for_scenario()` 按固定批次数循环；Teacher 某批只返回部分合法记录时，`offset` 仍按请求量推进且不补齐，`generate()` 仍写盘并报成功，可能用残缺数据覆盖完整产物。

**已完成修复：**

- 按场景目标持续请求，直至凑满或连续空批达 `MAX_CONSECUTIVE_EMPTY` 后抛 `GenerationShortfallError`。
- 部分有效回复会保留，`offset` 按实际写入条数前进。
- 三个 split 全部达标后才写盘；写盘使用 `*.tmp` + `replace` 原子替换。
- CLI 捕获 shortfall 并以非零状态退出。
- 已覆盖：部分有效补齐、空回复重试后仍达目标、重试耗尽失败且不覆盖已有产物。
