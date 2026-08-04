# 角色扮演强化学习 MVP

## 目标

为单个角色构建端到端训练流水线：

```text
persona + style examples
→ Student-baseline-guided Teacher-corrected SFT 数据
→ LoRA SFT
→ GRPO
→ Base/SFT/GRPO 评测报告
```

- 基础模型：`Qwen/Qwen3.5-2B`，训练时由 ms-swift 支持的后端以 4-bit 加载
- Student 本地推理：`mlx-community/Qwen3.5-2B-4bit`，revision`674aaa7240b91e8012fcad5d791b7dfe5ba90207`
- 训练框架：`ms-swift`
- 训练与推理：`enable_thinking=false`

核心问题：

> SFT 和 GRPO 能否同时提升小模型的角色一致性、格式一致性和对话质量，并且不以牺牲其中一个目标来换取另一个目标的提升？

本项目有三个并重的目标，不设优先级：

1. **角色一致性**：身份、性格、关系、事实和边界符合 persona；语气、句式、节奏、用词和互动方式接近风格示例。
2. **格式一致性**：回复遵循目标角色的固定组织形式；当前角色为“括号动作/神态 + 口语对白”。
3. **对话质量**：回答相关、自然、连贯、有信息量，并能承接后续对话。

三项目标分别评分，并以等权平均作为主要选模指标；任何一项的明显下降都不能由其他两项的提升抵消。

# 阶段一：数据与基线

**状态：已完成。** 当前 Smoke 数据位于
`data/runs/v4-flash-smoke-20260803-v4/`，可直接用于阶段二。已有审计产物保留作为参考，
但不再增加补充审计或新的阶段一准入条件；只有训练或评测暴露出系统性数据问题时才重建。

阶段一只解决 MVP 开始训练所必需的问题：数据可用、训练与评测隔离、Teacher 修正结果
基本可靠。版本治理、全量语义审计、可重放人工修订和生产级恢复能力不属于当前 MVP 范围。

## 1.1 输入

`persona.json` 必须包含 `name`、`identity`、`personality`、`speech_style`、
`relationships`、`facts`、`boundaries`，可选 `notes`；除 `name` 外均为自然语言字符串
数组。它是角色事实的唯一来源，`style_examples.jsonl` 仅提供表达风格，建议准备 10～20
组代表性对话。

每个角色还需定义可机器检查的格式契约。当前角色为：

```text
（简短的动作、神态或当下反应）口语对白
```

- 以一组闭合的全角圆括号动作文本开头，再进入自然口语对白。
- 动作应简短、生活化且符合语境；禁止长篇旁白、额外标签和多层括号。
- 格式一致性与语言风格分别评价，避免把“格式正确”误判为“像这个角色”。
- 该契约只适用于当前角色；新角色必须根据其需求和示例重新定义。

## 1.2 数据准备与切分

1. 校验 persona 的必填字段和类型，并用统一模板渲染各阶段复用的 system prompt。
2. 为日常对话、背景与关系、情绪与选择、格式与风格、冲突与未知事实五类场景分别准备本地题目锚点。
3. 一次性生成并切分 Prompt，在 Student 和 Teacher 处理前写定各 split。
4. 本地检查结构和规范化后的精确重复；人工抽查明显的跨 split 语义重复。

| 数据 | Smoke | MVP |
|---|---:|---:|
| SFT Train Prompt | 100 | 300 |
| SFT 训练样本 | 100 | 300 |
| GRPO Prompt | 30 | 100 |
| Dev Prompt | 20 | 50 |
| Eval Prompt | 50 | 100 |

SFT、Dev、GRPO、Eval 各自只用于训练、选配置、强化学习和最终评测；Dev、GRPO、Eval
的 Prompt 与回答不得进入 SFT。近似重复模型审计只在人工抽查发现系统性泄漏时启用，
不作为默认步骤。

## 1.3 Student-baseline-guided Teacher-corrected SFT

核心链路：

```text
persona + train prompt → Student baseline
persona + style examples + train prompt + baseline → Teacher 最小纠错
```

这是“Student 输出条件化的 Teacher 纠错 SFT”，不是由 Student 从多个 Teacher 候选中
自主选择目标的 Student-aware SFT。该命名区分应保留，避免夸大方法能力。

Teacher 遵循以下原则：

- 三项目标分别评分，任何一项都不能被其他项抵消。
- 角色一致性覆盖身份、性格、关系、事实、边界和整体语言风格，不能用复述设定或堆叠
  口头禅代替。
- 明确记录问题；合格回答原样保留，不合格回答仅做充分的最小修改。
- 不引入 persona 和当前用户消息之外的事实或共同经历。
- 未知信息应承认不知道、记不清或向用户确认。

每个 SFT Prompt 对应一条训练样本；API 失败或输出无效时重试补齐。训练文件只保留
`user` 和最终回答。导出前用确定性规则检查结构、目标格式、空回答、截断和明显复读，
然后人工抽查 10～20 条，确认没有系统性的人设、风格或对话质量问题。

Teacher 评分和问题说明可以保留用于诊断，但 Reviewer 多轮修正、全量人工复核、原文哈希
和可重放修订清单均延后。抽查发现个别问题时直接修正；发现系统性问题时才调整 Prompt
并重新生成受影响的数据。

## 1.4 基线与准入

训练前生成 Dev 和 Eval 的 Base 输出，供配置选择和最终对比使用。Base、SFT、GRPO
尽量使用相同的模型 checkpoint、persona prompt、chat template 和生成参数；如果推理后端
或量化方式不同，记录差异并在结论中说明。

阶段一满足以下条件即可进入 SFT：

- 各 split 数量正确且不存在规范化后的精确重复。
- Dev、GRPO 和 Eval 数据没有进入 SFT。
- SFT 样本通过基础结构和输出质量检查。
- 人工抽查未发现系统性质量问题。
- Dev 和 Eval Base 输出完整、未截断且无明显生成退化。

保留输入与 split、最终 SFT 训练文件、Dev/Eval Base 输出以及一份简要检查结果即可。
不要求审计归档、双重验收文件或完整断点续跑。能力保持集放到 MVP 最终评测，不阻塞
端到端 Smoke。

# 阶段二：LoRA SFT

## 2.1 目标

- 提升角色一致性：遵循身份、性格、关系、事实和边界，并学习角色的语气、句式、节奏、
  用词和互动方式，而不是机械插入口头禅。
- 提升格式一致性：稳定学习“括号动作/神态 + 口语对白”的回复格式。
- 提升对话质量：保持回答自然、有针对性，减少复读、冗长动作描写和模板化回复。
- 保留已有对话和指令遵循能力。

## 2.2 配置

```yaml
model: Qwen/Qwen3.5-2B
train_type: lora
dtype: bf16
quant_method: bnb
quant_bits: 4
max_length: 1024
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 5e-5
num_train_epochs: 1
batch_size: 1
gradient_accumulation_steps: 16
enable_thinking: false
```

只对 assistant 回复计算 loss。不要把 MLX 转换权重直接交给 ms-swift；训练使用
官方 Transformers checkpoint，并由 ms-swift/BNB 进行 4-bit QLoRA 加载。

MVP 比较：

- 学习率：`2e-5 / 5e-5 / 1e-4`
- Epoch：`1 / 2`

用 Dev 结果选配置；验收前不合并 LoRA。

## 2.3 验收

比较 `Base + Persona` 与 `SFT + Persona`：

- 角色一致性提高，其中事实约束和语言风格模仿分别报告。
- 格式契约通过率显著提高，且没有通过空洞、重复的括号动作投机。
- 对话质量提高，无明显复读、模板化、统一拒答或回复过短。
- 三目标等权宏平均提高，且任一目标均无明显下降。
- MVP 阶段的能力保持集无明显下降。

Smoke 只验收流程和产物。MVP 只有在 SFT 优于 Base 且无明显能力回归时才进入 GRPO。

产物：

```text
outputs/sft_adapter/
outputs/sft_eval.json
```

# 阶段三：GRPO

## 3.1 训练

每个 Prompt 采样 4 个回答，按组内相对奖励优化。从 SFT LoRA 继续训练。

```yaml
rl_prompts: 30  # MVP 为 100
num_generations: 4
max_completion_length: 256
learning_rate: 1e-6
epochs: 1
enable_thinking: false
```

## 3.2 奖励

```text
R = (RoleConsistency + FormatConsistency + DialogueQuality) / 3 - Penalty
```

| 维度 | 内容 |
|---|---|
| RoleConsistency | 身份、性格、关系、事实、边界，以及语言风格的整体模仿质量 |
| FormatConsistency | 开头括号动作、括号闭合、动作简短且贴合语境、随后进入对白 |
| DialogueQuality | 相关、自然、连贯、有信息量、可继续对话 |

本地规则检查可确定的格式条件，Teacher/Judge 对三个主目标分别给出 0～10 分。语言风格
作为 `RoleConsistency` 的子维度评分，不再单独取得第四份权重。`Penalty` 只处理重复、
乱码、照抄示例等三个主分容易漏掉的奖励投机，不重复惩罚已经体现在主评分中的普通缺陷：

| 问题 | 惩罚 |
|---|---:|
| 复读或乱码 | -3 |
| 大段复述 persona | -2 |
| 照抄风格示例或堆叠口头禅 | -3 |
| 用空洞、重复的括号动作骗取格式分 | -2 |

监控平均奖励、回答长度、复读率和策略变化，防止奖励投机。

产物：

```text
outputs/grpo_adapter/
outputs/reward_curve.json
outputs/grpo_samples.jsonl
```

# 阶段四：评测

## 4.1 对比

```text
A. Base + Persona
B. SFT LoRA + Persona
C. GRPO LoRA + Persona
```

三组使用相同 Eval、生成参数和 Judge。

## 4.2 指标

三个并列主指标：

- Role Consistency
  - Identity/Facts/Boundaries
  - Personality/Relationships
  - Style Fidelity（语气、句式、节奏、用词和互动方式）
- Format Consistency
  - Format Contract Pass Rate（确定性规则）
  - Format Quality（动作简洁度、语境贴合度及对白衔接）
- Dialogue Quality
  - 相关性、自然度、连贯性、信息量和可继续对话性

诊断指标：

- 人设矛盾率
- 未知事实编造率
- 出戏率
- 复读率、模板化率和回答长度分布

自动风格评测必须结合与 `style_examples.jsonl` 对齐的 Judge rubric；词频、固定短语命中和
长度统计只用于诊断，不能代替 Style Fidelity。人工盲评优先做 Base/SFT/GRPO 两两比较，
分别询问“哪个更像这个角色”“哪个格式更自然稳定”和“哪个回答本身更好”。

能力保持集不使用 persona，覆盖普通问答、总结、改写、基础推理和指令遵循。

人工检查：

- Smoke：抽查 10 条
- MVP：盲评 30 条

## 4.3 成功标准

- SFT 的角色一致性、格式一致性和对话质量均不低于 Base，三目标等权宏平均高于 Base。
- SFT 的角色一致性提升必须同时有 Style Fidelity 和事实相关诊断支撑，不能只靠口头禅。
- SFT 的格式契约通过率高于 Base；MVP Eval 目标不低于 95%。
- GRPO 的三目标等权宏平均高于 SFT，且任一目标均无明显下降。
- SFT 和 GRPO 无明显人设矛盾、未知事实编造、出戏、复读、模板化或过短回答恶化。
- 能力保持集无明显回归。
- 人工盲评中 GRPO 的整体偏好不弱于 SFT。
- 第二个角色可以复现整条流程。

# 实施顺序

## Milestone 1：Smoke 数据

- 完成 Persona 校验和 Prompt 渲染。
- 生成、精确去重并切分 Prompt，人工抽查跨 split 语义重复。
- 生成 Student Baseline。
- Teacher 单轮评分并最小改写。
- 完成基础自动检查和 10～20 条人工抽查。
- 导出 SFT 数据和 Dev/Eval Base 输出后立即进入训练。

## Milestone 2：端到端 Smoke

- 完成 SFT、GRPO、推理和评测。
- 固化角色一致性、格式一致性和对话质量的等权 Judge rubric。
- 在 SFT、GRPO 和最终评测中统一复用三目标 Judge rubric 与格式契约检查。
- 检查日志与产物完整性。

## Milestone 3：MVP

- 扩充到 300 条 SFT、100 条 GRPO、50 条 Dev 和 100 条 Eval。
- 用 Dev 选择 SFT 配置。
- 完成 GRPO 和统一评测。

## Milestone 4：交付

- 用第二个角色复现。
- 输出 LoRA、评测报告和运行说明。

# 核心实验

| 模型 | 角色一致性（含风格） | 格式一致性 | 对话质量 | 三目标宏平均 | 能力保持 |
|---|---:|---:|---:|---:|---:|
| Base + Persona | 基线 | 基线 | 基线 | 基线 | 基线 |
| SFT | ↑ | ↑↑ | ↑ | ↑ | ≈ |
| SFT + GRPO | ↑/≈ | ↑/≈ | ↑/≈ | ↑ | ≈ |

预期结论：

> Teacher-corrected SFT 针对 Student baseline 已暴露的角色、格式和对话缺陷进行修正；
> GRPO 在不牺牲任一目标的前提下，进一步提高三个并列目标的等权综合表现。
