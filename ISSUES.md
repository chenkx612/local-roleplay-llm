# v2 Issues

## P3｜Teacher-corrected SFT 仍有局部瑕疵（不阻断）

**位置：** `data/runs/morgana-v2/sft_teacher_edits.jsonl`、
`data/runs/morgana-v2/sft_train.jsonl`、`data/runs/morgana-v2/sft_label_review.md`

**现象：** 50 条结构正确，最终答案无空值、乱码或明显复读，严重主客体和出戏问题已修复。
少数标签仍有局部逻辑、轻度猜测或正典细节不稳定，但回答均可读、相关且角色身份成立。

**阶段决策：** 对学习型小规模项目不构成阻断，当前数据可用于 SFT。局部瑕疵作为复盘材料保留，
不再追加 Teacher 规则或重跑。

## P2｜SFT Student 退化频率高（已被 Teacher 链路吸收）

**现象：** 50 条 baseline 中 27 条截断、33 条明显复读；第 46 条在原链路中因复读
重试耗尽，无法交给 Teacher 纠错。

**已完成修复：** 只有 Teacher-corrected SFT 路径允许保留复读 baseline；Base/评测推理
仍严格拒绝复读，Teacher 最终答案仍要求稳定。已增加回归测试。

# v1 Review Issues（已封存）

> 本文是 `morgana-v1` 的问题档案。v1 已于 2026-08-08 收尾，以下未完成项不再作为当前流程的
> 阻断项；v2 若再次出现同类问题，应以新记录和新产物重新判断。v1 总结见
> `V1_RETROSPECTIVE.md`。

## 优先级说明

- **P0**：阻断后续阶段或会直接导致评测、训练结论失效，必须优先处理。
- **P1**：严重影响结果可信度或任务完整性，应在进入下一阶段前处理。
- **P2**：影响可比性、可追溯性或稳定性，应排期处理。
- **P3**：低影响的优化或体验改进。

## v1 收尾时未完成项

阶段一问题已于 2026-08-07 完成处理或形成明确决策。第三次 SFT 已产生有效参数更新，但新版
生成稳定性门槛仍发现一条不可读输出，且三目标匿名人工复核未完成。这些结果作为 v1 的负向证据
封存，不直接带入 v2 状态。

### P0｜第三次 SFT 仍有一条不可读输出

**位置：** `output/morgana-v1/stage2-sft/3/dev_outputs.jsonl`、
`output/morgana-v1/stage2-sft/3/run_summary.json`

**现象：** 新版生成稳定性门槛下，完整对齐、结束数、截断和严重复读均通过，但
`20260808:dev_0010` 后半段出现连续希腊字母和随机字符，`no_gibberish=false`。严格动作格式、
emoji 和自称频率已降级为诊断指标，不是本问题的阻断原因。

**处理与验收：** 保留第三次产物作为有效训练证据，不通过修改评估规则掩盖不可读样本。后续
候选必须没有乱码或严重复读，稳定性通过后再进行角色一致性和对话质量匿名复核。

### P1｜角色一致性与对话质量尚未完成人工复核

**位置：** `output/morgana-v1/stage2-sft/3/manual_review_packet.json`、
`output/morgana-v1/stage2-sft/3/manual_review_results.json`

**现象：** 第三次归档的人工结果为空。旧版流程因严格格式门槛失败而未启动复核；新版已取消
格式和事实可靠性独立目标，但仍需判断核心身份、性格、关系、回应相关性和自然度。

**处理与验收：** 生成稳定性通过后，按三个核心目标复核主 seed 10 对回答；SFT 至少胜 6 对、
明显落后不超过 2 对、无不可读/角色崩坏/视角错位严重问题，且三个维度平均分均不低于 Base。

### P2｜Base/SFT 初步比较存在推理后端混杂

**位置：** `data/runs/morgana-v1/base_generation_meta.json`、
`output/morgana-v1/stage2-sft/20260807T170226Z-ad78fb8f/dev_generation_meta.json`

**现象：** Base 使用 MLX 转换模型和 OpenAI-compatible server；SFT 使用上游 HF revision 和
ms-swift TransformersEngine。Persona、Dev 和主要采样参数相同，但 SFT 后端没有等价的
`presence_penalty=0.4`，repetition context 语义不同，模板还输出 non-thinking 标签。因此阶段二
只能判断实际部署链路下的行为变化，不能把全部差异因果归于 LoRA。

**阶段决策：** 重训 notebook 先生成同一 Transformers 后端和 revision 的无 adapter Base，
作为阶段二主要前后对照；MLX Base 仅保留为阶段一部署基线。阶段四继续使用统一推理栈生成
Base/SFT/GRPO 输出。

## 已解决问题

### P0｜首次 SFT 的 LoRA-B 全部保持初始化零值

首次 FP16 运行的四个 `grad_norm` 全为 `NaN`，186 个 LoRA-B 张量全部为零。第三次改用既定
FP32 QLoRA 配置并训练 3 epochs 后，12 个梯度均有限且为正，186/186 个 LoRA-B 张量均非零，
adapter 可重新加载；该技术阻断已解决。完整摘要见 `V1_RETROSPECTIVE.md` 和
`output/morgana-v1/stage2-sft/3/run_summary.json`。

### P1｜单条推理失败会被吞掉，命令仍声称整批完成

**位置：** `src/roleplay/inference.py`、`tests/test_inference.py`

**原问题：** 任意 API 异常都会被捕获并转换成空字符串，随后写入输出。即使整批请求失败，
命令仍会打印完成并以状态码 0 退出，可能把无效结果当成真实 Base 表现。

**已完成修复：**

- API 异常、空回答、非 `stop` 结束和明显连续复读都会触发重试，每条最多尝试 3 次。
- 重试耗尽时抛出 `BaselineGenerationError`，CLI 非零退出，不再报告整批成功。
- 全部记录成功后才通过临时文件原子替换输出；失败不会覆盖已有结果。
- 测试覆盖 API 异常、截断、复读、重试成功和失败不覆盖。
- `morgana-v1` 的 Dev 10 条、Eval 20 条 Base 输出均非空、以 `stop` 结束并与输入逐条对齐。

### P2｜Base 默认使用的模型与计划中的训练基座不一致

**位置：** `src/roleplay/inference.py:15`、`PLAN.md` 的“基础模型使用”和阶段二技术方案

项目受本地资源限制，Student 推理使用 `mlx-community/Qwen3.5-2B-4bit`。
ms-swift 训练使用可由 Transformers 加载的 `Qwen/Qwen3.5-2B`，再由受支持的
BNB 后端做 4-bit QLoRA；不再尝试把 MLX 转换权重直接作为训练基座。

### P0｜GRPO 训练集与独立评测集存在精确泄漏

**位置：** `src/roleplay/datagen.py`、`tests/test_datagen.py`、`data/*.jsonl`

**原问题：** GRPO 与 Eval 存在相同 Prompt，且生成器没有 Dev split，也会在冻结
Prompt 前直接生成 SFT 回答，无法满足训练与评测隔离要求。

**已完成修复：**

- Teacher 只生成用户 Prompt；SFT、GRPO、Dev、Eval 四个 split 全部完成后才写盘。
- 使用 Unicode NFKC、首尾去空白和连续空白折叠构造全局去重键；保留原始 Prompt 文本。
- 同批、同 split 和跨 split 重复均被丢弃并继续补齐；重复耗尽时失败且不覆盖已有产物。
- `morgana-v1` 已按首次实践规模冻结为 50/20/10/20 条，100 个规范化 Prompt 全局唯一。

### P0｜基线输出存在截断与退化，不能作为有效对照

**位置：** `src/roleplay/inference.py`、`tests/test_inference.py`、`data/baseline_outputs.jsonl`

**原问题：** 推理固定使用 `max_tokens=256`，未读取或保存 `finish_reason`；原有 50 条基线中存在句中截断和“蛋蛋蛋蛋……”“她她她她……”等明显生成退化，不能作为有效 Base 对照。

**已完成修复：**

- 保存并校验 `finish_reason`；空回答、非 `stop` 结束及明显连续复读均视为失败，每条最多尝试 3 次。
- 任一条重试耗尽时整批非零退出，不覆盖旧产物；全部成功后才原子替换输出文件。
- 旧基线已随 split 重建而归档，不再参与对比；最终冻结的 Dev 10 条和 Eval 20 条 Base 输出
  均以 `stop` 结束、回答非空、未命中复读检测，并且全部首次生成成功。
- 最终使用的模型 revision、生成参数、输入输出哈希及参数调整原因保存在
  `data/runs/morgana-v1/base_generation_meta.json`。
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
