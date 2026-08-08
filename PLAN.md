# morgana-v2 角色扮演后训练计划

## 1. 目标与原则

v2 使用一个角色和最小数据规模，重新完成一次可理解、可复现的后训练闭环：

```text
目标与 Persona
→ 独立数据切分与 Teacher-corrected SFT
→ 同后端 Base 基线
→ QLoRA SFT
→ GRPO
→ Base/SFT/GRPO 统一评测
→ 复盘
```

本轮只回答一个问题：

> SFT 和 GRPO 能否依次让 2B 小模型稳定生成、保持角色，并产出有意义的对话？

坚持“小而完整”：每阶段只使用一组主配置和最小样本，不做超参数搜索，不建设生产级平台；
即使模型没有提升，也要保留有效训练、可比输出和清楚的负向结论。

## 2. 三个递进目标

三个目标按前置关系逐层验收，不计算掩盖底层失败的单一总分。

### 2.1 生成稳定性

回复必须非空、完整、连贯且可读，不出现乱码、严重复读或破坏性截断。这是硬门槛；失败的回答
不继续解释角色或内容得分。

### 2.2 角色一致性

在稳定生成的前提下，回复应符合 Persona 的核心身份、性格、关系、边界、说话视角和整体语言
风格。允许合理创作，但不得与核心 Persona 或当前对话冲突，也不得擅自建立用户个人经历或重大
共同关系。

### 2.3 对话质量

在角色成立的前提下，回复应直接回应用户，内容自然、连贯、有意义，并为后续对话留下合理空间。

动作括号、emoji、固定开头、自称频率和回答长度仅作为诊断信息，不设格式一致性目标。事实可靠性
不独立评分；无外置记忆时不要求模型机械复述固定事实。

## 3. v2 边界与文档职责

- v2 Run 名称：`morgana-v2`。
- v1 总结：`V1_RETROSPECTIVE.md`。
- v1 数据与模型产物保持只读，不复制为 v2 正式产物。
- v2 数据目录：`data/runs/morgana-v2/`。
- v2 模型目录：`output/morgana-v2/`。
- `PLAN.md` 只记录预定流程、主配置和验收标准。
- `RUNLOG.md` 从空文件开始，只记录 v2 的实际运行、产物、偏差和结论。
- `ISSUES.md` 保留 v1 问题历史；v2 新问题应明确标注版本。

角色源设定可以复用，但必须重新保存 v2 输入快照和哈希。v1 的 system prompt、Teacher 标签、
Base 输出和 SFT adapter 都受旧目标影响，不参与 v2 训练或正式评测。

## 4. 技术基线

- 训练与主要评测基座：`Qwen/Qwen3.5-2B`，固定并记录实际 revision。
- QLoRA：ms-swift + bitsandbytes 4-bit，Colab Tesla T4。
- Teacher/Judge：`deepseek-v4-flash`，开启 thinking，记录实际模型和请求配置。
- Student、SFT 和 GRPO 生成关闭 thinking。
- Base/SFT/GRPO 的正式比较使用同一模型 revision、Transformers 后端、聊天模板和生成参数。
- MLX 只可用于本地链路检查，不作为正式前后对照。

每次运行保存输入哈希、代码 commit、模型 revision、环境版本、实际命令、配置、日志和输出。训练集、
Dev 和 Eval 必须隔离。

## 5. 阶段一：重建数据与 Base

### 5.1 输入冻结

检查并冻结：

- `persona.json`：核心身份、性格、语言风格、关系、事实和边界。
- `style_examples.jsonl`：10～20 组代表性对话，只表达风格，不把示例经历当作事实。
- v2 system prompt：优先保证自然可读，动作描写可选，不规定括号、emoji 或固定开头。

风格样例应包含自然变化，避免所有标签都呈现同一种动作模板。输入通过结构检查和人工抽查后，
保存快照与哈希。

### 5.2 Prompt 数据

继续使用五类场景：日常对话、背景与关系、情绪与选择、语言风格、出戏与冲突。每条 Prompt 都
服务于三个递进目标，不再为格式或事实可靠性单独造题。

| 数据 | 数量 | 每类场景 | 用途 |
|---|---:|---:|---|
| Pilot | 5 | 1 | 验证 Student/Teacher 链路 |
| SFT | 50 | 10 | 监督训练 |
| GRPO | 20 | 4 | 强化学习 |
| Dev | 10 | 2 | 阶段选择与观察 |
| Eval | 20 | 4 | 最终统一评测 |

先生成 100 条共享候选池，再按固定 seed 切分为 SFT/GRPO/Dev/Eval。四个 split 必须数量正确、
场景均衡、全局无规范化精确重复；记录只包含 `id`、`scenario`、`target_goals` 和 `user`。Eval
在最终统一评测前不得用于修改数据、提示词或训练配置。

### 5.3 Teacher-corrected SFT

```text
Persona + SFT Prompt → Student baseline
Persona + style examples + Prompt + baseline → Teacher 最小充分纠错
```

Teacher 依次检查生成稳定性、角色一致性和对话质量。合格回答原样保留；不合格回答只做必要改写。
动作括号、emoji、固定开头或精确复述 Persona 都不是通过条件。

先运行五类场景各一条的 Pilot 并人工复核，再生成 50 条正式标签。正式产物必须结构正确、逐条对齐、
回答非空，且没有乱码和明显复读；保留 Student baseline、Teacher audit 和最终训练标签。

### 5.4 Base Dev

在与阶段二相同的 Transformers 后端和模型 revision 上，为 10 条 Dev 使用
`20260807/20260808/20260809` 三个固定 seed 生成 30 条 Base 输出。生成参数在首次正式推理前
冻结，后续 SFT 使用完全相同的参数和 RNG 重置方式。

阶段一通过条件：输入、四个 split、Pilot、50 条 SFT 标签和 30 条 Base Dev 均有效并留有可检查
产物。未完成这些条件不得开始 SFT。

## 6. 阶段二：QLoRA SFT

v2 只运行一组已在 v1 验证为技术有效的主配置：

```yaml
model: Qwen/Qwen3.5-2B
tuner_type: lora
target_modules: all-linear
torch_dtype: float32
quant_method: bnb
quant_bits: 4
bnb_4bit_compute_dtype: float32
lora_dtype: float32
max_length: 1024
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 5e-5
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
enable_thinking: false
```

只对 assistant 回复计算 loss，不使用 Dev 搜索 epoch、学习率或 LoRA 参数。技术失败可以修复明确
的实现问题后重跑；行为失败不在同一轮静默调参。

### 6.1 技术门槛

- 3 epochs 正常结束，optimizer step 数与预期一致。
- 所有记录的 `grad_norm` 有限且为正。
- 所有 LoRA-B 张量出现非零更新，adapter 张量全部有限。
- adapter 可重新加载并生成非空回答。

### 6.2 生成稳定性门槛

使用阶段一冻结的三个 seed，在同一后端重新生成 Base/SFT 各 30 条，并按 `(seed, id)` 完全对齐：

- 两侧输出完整、非空且数量正确。
- SFT 的 `stop` 数不得低于 Base，截断数不得高于 Base。
- SFT 不得出现乱码或严重复读。

严格格式、emoji、括号、自称和长度继续统计，但不影响通过。

### 6.3 人工门槛

生成稳定性通过后，对主 seed 的 10 对回答做匿名 A/B 复核，分别按生成稳定性、角色一致性和
对话质量给分：

- SFT 至少胜 6 对。
- SFT 明显落后不超过 2 对。
- SFT 无不可读、角色崩坏或视角错位等严重问题。
- 三个维度的 SFT 平均分均不得低于 Base。

仅当 `technical_gate && generation_stability_gate && manual_gate` 时进入 GRPO。若行为未达标，
保留负向结果并结束本次 SFT，不用格式规则掩盖真实问题。

## 7. 阶段三：GRPO

从通过阶段二的 SFT adapter 继续训练，使用冻结的 20 条 GRPO Prompt：

```yaml
num_generations: 4
max_completion_length: 256
learning_rate: 1e-6
num_train_epochs: 1
enable_thinking: false
```

奖励只对应三个目标：

```text
R = Readable × (RoleConsistency + DialogueQuality) / 2 - ExploitPenalty
```

- `Readable`：0/1，乱码、严重复读或破坏性截断为 0。
- `RoleConsistency`：0～10，检查核心 Persona 和对话视角。
- `DialogueQuality`：0～10，检查相关性、自然度和内容价值。
- `ExploitPenalty`：只惩罚大段照抄 Persona/示例等明显奖励投机。

训练前人工检查至少 5 组候选和分数。完成后保存实际配置、adapter、奖励曲线、回答样本、长度、
复读和奖励投机观察。GRPO 也必须通过技术门槛和与 SFT 相同的三层 Dev 评估，才进入统一评测；
失败时保留结果，不根据 Eval 反复调整奖励。

## 8. 阶段四：统一评测

使用冻结的 20 条 Eval，在同一后端、同一模型 revision、同一生成参数和一个固定主 seed 下生成
Base、SFT、GRPO 各 20 条输出。Eval 只用于最终比较。

逐层报告：

1. **生成稳定性**：非空、正常结束、截断、乱码、严重复读。
2. **角色一致性**：核心身份、性格、关系、边界、视角和整体风格。
3. **对话质量**：相关性、自然度、连贯性、内容价值和可继续对话性。

对 10 条 Eval 做匿名比较并保留具体样本。格式、自称、emoji、创造性细节、模板化和长度只作为
解释性诊断。结论必须区分 SFT 相对 Base、GRPO 相对 SFT 的变化，并注明小样本、单角色、单次
训练和 Judge 偏差的限制。

## 9. 完成标准与清单

### Milestone 1：数据与 Base

- [ ] 冻结 v2 Persona、风格样例、system prompt 和输入哈希。
- [ ] 生成并验证隔离的 SFT/GRPO/Dev/Eval。
- [ ] 完成 Pilot、50 条 Teacher-corrected SFT 和人工抽查。
- [ ] 在统一 Transformers 后端生成三个 seed 的 Base Dev。

### Milestone 2：SFT

- [ ] 完成一次技术有效的 FP32 QLoRA SFT。
- [ ] 通过生成稳定性门槛和三目标匿名人工复核。
- [ ] 记录实际配置、产物、偏差和进入 GRPO 的决定。

### Milestone 3：GRPO

- [ ] 验证奖励样本并完成一次有效 GRPO。
- [ ] 完成 GRPO 相对 SFT 的三层 Dev 评估。

### Milestone 4：统一评测与复盘

- [ ] 生成对齐的 Base/SFT/GRPO Eval 输出。
- [ ] 完成自动统计、匿名人工比较和诊断观察。
- [ ] 写出 v2 复盘，明确结果、失败点和结论边界。

当以上完整链路均留下可检查证据时，v2 即完成；模型没有提升不影响闭环完成，但无效训练、数据
泄漏或不可比推理必须修复后才能形成实验结论。
