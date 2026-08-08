# Review Issues

## 优先级说明

- **P0**：阻断后续阶段或会直接导致评测、训练结论失效，必须优先处理。
- **P1**：严重影响结果可信度或任务完整性，应在进入下一阶段前处理。
- **P2**：影响可比性、可追溯性或稳定性，应排期处理。
- **P3**：低影响的优化或体验改进。

## 待解决问题

阶段一问题已于 2026-08-07 完成处理或形成明确决策。原 Colab 后训练路径已 superseded；当前
阻断项是 Mac/MLX 尚未在可用 Metal 的真实本机会话中完成 SFT、人工门槛和 GRPO。

### P0｜Mac/MLX 只完成实现，尚无实机训练证据

**位置：** `src/roleplay/posttrain.py`、`src/roleplay/mlx_backend.py`、
`configs/morgana_v1_sft_mlx.json`、`configs/morgana_v1_grpo_mlx.json`

**现状：** 配置、六阶段 CLI、SFT 梯度预检、最小 GRPO、奖励事务、原子产物、内存降级和统一
评测均已实现，120 项离线测试通过。当前自动化沙箱没有可用 Metal device，无法提供真实梯度、
adapter 更新、峰值内存或生成质量证据。

**解除条件：** 在接通电源、关闭其他模型服务的 M4 Mac 上按
`doctor → sft → gate-sft → reward-preview → grpo → evaluate` 执行。SFT/GRPO 各自只有在对应
technical、自动、人工和 adapter 验收产物通过后才算完成；本机 GRPO 若触发
`local_grpo_blocked`，只算迁移验收完成，不算学习闭环完成。

### P0（历史、superseded）｜首次 SFT 的 LoRA-B 全部保持初始化零值

**位置：** `output/morgana-v1/stage2-sft/20260807T170226Z-ad78fb8f/full/logging.jsonl`、
`output/morgana-v1/stage2-sft/20260807T170226Z-ad78fb8f/full/checkpoint-4/adapter_model.safetensors`

**现象：** 四个训练 step 的 loss 均为有限正数，但 `grad_norm` 全部为 `NaN`。直接读取最终
safetensors 后确认：372 个 adapter 张量全部有限，186 个 LoRA-A 张量为随机初始化值，186 个
LoRA-B 张量全部精确为零，LoRA-B 非零元素总数为 0。LoRA 初始 B 为零，因此 optimizer 没有
形成任何有效 adapter 更新；“checkpoint 可加载并能生成”只证明序列化和 Base 推理可用。

**原因判断：** 首次配置在 T4 上使用 FP16 模型、FP16 BNB compute，且 LoRA dtype 跟随默认
行为成为 FP16。结合连续非有限 `grad_norm` 和零更新，最可能是 AMP 梯度溢出导致四次 step
全部跳过。此前 notebook 只检查 loss 和可加载性，没有检查梯度及权重是否实际变化。

**最终决策：** 不再继续 Colab/ms-swift 重训。Mac/MLX 正式训练前先做一次临时 optimizer
update，并在 150 microbatch/15 update 全程阻断非有限梯度；最终要求全部 LoRA-B 非零、adapter
全有限且可重载。原 checkpoint 永久禁止作为 GRPO 起点。

### P1（历史、superseded）｜首次 Transformers Dev 全量格式失败且半数截断

**位置：** `output/morgana-v1/stage2-sft/20260807T170226Z-ad78fb8f/dev_outputs.jsonl`、
`RUNLOG.md` 的“阶段二：LoRA SFT”

**现象：** 无效 adapter 重载后生成 10/10 非空回答，但 10/10 未以规定的全角动作括号开头，
10/10 带额外 `<think></think>` 标签，5/10 达到 256-token 上限。样本还出现过量 emoji、
冗长转题、虚构背景、非标志性自称和情绪支持不当。由于 LoRA-B 未更新，这些输出主要反映
上游 HF Base + Transformers 模板/采样栈，不能当作 SFT 的负向效果。

**修复与验收：** 重训 notebook 会在同一 Transformers backend、同一上游 revision 下额外生成
无 adapter Base，移除协议层空 `<think></think>` wrapper 后比较。三个固定 seed 下 Base/SFT
各 30 条必须完整对齐；SFT 的结束、截断、退化、严格格式和角色自称须通过相对自动门槛，随后
主 seed 匿名人工比较还须达到胜负、严重问题和 emotion 专项门槛。任一行为门槛失败都归档证据，
但不得进入 GRPO。

### P2（已由迁移消除）｜Base/SFT 初步比较存在推理后端混杂

**位置：** `data/runs/morgana-v1/base_generation_meta.json`、
`output/morgana-v1/stage2-sft/20260807T170226Z-ad78fb8f/dev_generation_meta.json`

**现象：** Base 使用 MLX 转换模型和 OpenAI-compatible server；SFT 使用上游 HF revision 和
ms-swift TransformersEngine。Persona、Dev 和主要采样参数相同，但 SFT 后端没有等价的
`presence_penalty=0.4`，repetition context 语义不同，模板还输出 non-thinking 标签。因此阶段二
只能判断实际部署链路下的行为变化，不能把全部差异因果归于 LoRA。

**阶段决策：** Base、SFT、GRPO 全部改为同一 MLX 模型 revision、chat template、采样器和推理
实现；旧的 Transformers 对照只保留为不可归因的历史观察。

## 已解决问题

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
