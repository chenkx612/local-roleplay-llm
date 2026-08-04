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

**状态：已完成。** 冻结运行目录为
`data/runs/v4-flash-smoke-20260803-v4/`，总验收文件
`stage1_acceptance.json` 状态为 `pass`。后续阶段直接使用的核心文件保留在目录根部；
生成、语义审计、Teacher/Reviewer 和人工修订证据已归档为 `audit_bundle.tar.gz`，需要
重跑完整验收时先在该目录解压。

该 `pass` 证明原阶段一流程与产物完整，不代表数据已经按新的三目标框架验收。进入阶段二
前，需补做角色一致性、格式一致性和对话质量的等权审计；如果训练答案没有稳定覆盖下述
格式契约、主要靠堆叠口头禅制造表面相似，或对话质量不足，应重新修订训练答案并生成
新的验收记录，不修改已冻结原件。

阶段一的 DeepSeek 调用固定使用显式模型名 `deepseek-v4-flash`，不使用会随服务端
迁移的旧模型别名。Prompt 生成使用非思考模式以保留采样多样性；SFT 审计使用
思考模式且 `reasoning_effort=high`。两者均使用 JSON Output，并将模型、模式、
参数和输入哈希写入元数据。

正式重建前先运行 10 条 SFT 的 `pilot` profile。Pilot 产物与正式数据分别写入
`data/runs/<run-id>/`，不得覆盖或混入冻结数据。

Smoke 正式流程固定为：

1. 为五类场景各准备 40 个互不重复的本地题目锚点。
2. V4 Flash 按场景一次生成完整 40 条候选池，再固定切成 SFT/GRPO/Dev/Eval；
   不把模型生成结果作为后续 API 的去重上下文回传。
3. 本地执行结构、哈希、精确重复和近似重复检查。
4. 用 V4 Flash 对全部 200 条冻结 Prompt 做跨 split 语义泄漏审计，并人工逐条复核。
5. 生成 100 条 Student-baseline-guided Teacher-corrected SFT；独立 Reviewer 最多
   修正三轮，随后人工检查全部
   改写项和 20 条抽样最终回答。正式运行实际逐条复核全部 100 条，并通过带原文哈希
   的修订清单执行 29 条最小人工修订；机器生成版本保持只读，最终训练版本单独冻结。
6. 生成固定的 30 条能力保持集，再统一冻结 Dev、GRPO、Eval、Retention 共 130 条
   Base 输出。
7. 只有总验收报告 `stage1_acceptance.json` 通过，才允许进入阶段二。

## 1.1 输入

`persona.json`：

- 必填：`name`、`identity`、`personality`、`speech_style`、`relationships`、`facts`、`boundaries`
- 可选：`notes`
- 除 `name` 外均为自然语言字符串数组

`style_examples.jsonl`：10～30 组代表性对话。

```json
{"user": "你担心我吗？", "assistant": "（轻轻拉住你的袖口，抬眼看了看你）当然担心呀，到家记得告诉我一声好不好。"}
```

除完整示例外，为每个角色定义一份可机器检查的“格式契约”。当前角色的契约是：

```text
（简短的动作、神态或当下反应）口语对白
```

- 回复必须以一组全角圆括号动作文本开头，括号闭合后再进入对白。
- 括号内保持简短、生活化，并与当前语境相符；不写成长篇小说旁白。
- 正文是自然口语，不使用额外标签、Markdown 舞台说明或多层括号套娃。
- 格式契约描述结构，`style_examples.jsonl` 描述结构之上的角色语言风格。评测时将格式
  一致性单独计分，将语言风格模仿计入角色一致性，避免“有括号但不像角色”也被判为
  角色一致。

格式契约是当前角色的目标样式，不被视为所有角色扮演任务的通用规则。扩展到第二个角色
时，应从该角色需求与示例中显式定义，而不是默认复制当前契约。

## 1.2 数据准备与切分

1. 校验 `persona.json`，拒绝未知字段、错误类型和空字符串。
2. 用固定模板渲染 persona system prompt；所有阶段复用同一实现。
3. 生成五类用户 Prompt：
   - 日常对话
   - 背景与关系
   - 情绪与选择
   - 回复格式与语言风格
   - 出戏、冲突与未知事实
4. 规范化并跨 split 精确去重。
5. 在调用 Student 和 Teacher 前固定所有 split。

`persona.json` 是角色事实的唯一来源；`style_examples.jsonl` 只提供表达风格。

| 数据 | Smoke | MVP |
|---|---:|---:|
| SFT Train Prompt | 100 | 300 |
| SFT 训练样本 | 100 | 300 |
| GRPO Prompt | 30 | 100 |
| Dev Prompt | 20 | 50 |
| Eval Prompt | 50 | 100 |
| Retention Prompt | 30 | 30 |

Pilot 固定为 10 条 SFT Prompt，不生成其他 split，也不进入训练集。

隔离规则：

- SFT Train Prompt 只用于 SFT。
- Dev 只用于选配置。
- GRPO Prompt 只用于 GRPO。
- Eval 只用于最终评测。
- Dev、GRPO、Eval 的 Prompt 和回答不得进入 SFT。

## 1.3 Student-baseline-guided Teacher-corrected SFT

这里采用的是“Student 输出条件化的 Teacher 纠错 SFT”：Student 先生成 baseline，
Teacher 根据 baseline 暴露的具体缺陷进行最小修改，Reviewer 再验收最终答案。

这不等同于本项目语境中的 **Student-aware SFT**。后者要求 Teacher 提供多个候选答案
或不同形式的帮助，由 Student 根据自身状态自主选择更有帮助的答案作为 SFT 目标；
当前阶段没有候选生成、Student 选择或选择信号，因此不使用该名称。

先生成 Student Baseline：

```text
persona + train prompt
→ Qwen3.5-2B
→ baseline answer
```

Teacher 输入：

```text
persona + style examples + train prompt + baseline answer
```

Teacher 负责：

- 分别评价角色一致性、格式一致性和对话质量，三项不互相替代。
- 角色一致性同时检查身份、性格、关系、事实、边界与语言风格；不为了展示 persona 而
  生硬复述设定，也不以口头禅数量代替风格判断。
- 标记具体问题。
- 合格回答原样保留。
- 不合格回答做最小充分修改。
- 不引入 persona 和当前用户消息之外的事实或共同经历。
- 未知信息应承认不知道、记不清或向用户确认。

审计记录：

```json
{
  "user": "...",
  "baseline_assistant": "...",
  "checks": {
    "format_contract": true
  },
  "scores": {
    "role_consistency": {
      "overall": 0,
      "identity_facts_boundaries": 0,
      "personality_relationships": 0,
      "language_style": 0
    },
    "format_consistency": 0,
    "dialogue_quality": 0
  },
  "issues": ["..."],
  "decision": "keep | light_rewrite | rewrite",
  "improved_assistant": "..."
}
```

评分范围为 0～10。每个 SFT Train Prompt 生成一条 SFT 训练样本；失败项重试补齐，
不按 `decision` 筛选。训练集只使用 `user` 和最终 assistant 回答，审计记录单独保存。
其中 `checks.format_contract` 记录确定性规则结果，`format_consistency` 记录 Judge 对格式
实现质量的评分。`role_consistency.language_style` 评价整体风格模仿质量，不得仅按口头禅
数量打分。三个主目标均为 0～10 分，总分取等权宏平均；角色一致性的三个子分只用于形成
该维度的可解释评分，默认等权形成 `role_consistency.overall`，不作为额外权重重复计入
三目标总分。
若人工全量复核发现漏判，必须保留 `sft_train_generated.jsonl`，并用
`sft_human_edits.json` 中的原文哈希、修订文本和原因生成最终 `sft_train.jsonl`；验收器
会重放清单并拒绝未记录的改动。

## 1.4 冻结基线

训练前保存 Dev、GRPO 和 Eval 的 Base 输出。Base、SFT、GRPO 必须使用相同的：

- 基础 checkpoint 和 revision
- 精度或量化策略
- 推理后端和 chat template
- persona prompt
- 生成参数

记录上述元数据。Eval Baseline 不参与训练或 Teacher 改写。

能力保持集覆盖指令遵循、结构化输出、语言、推理、稳定知识、普通对话与安全建议；
Base/SFT/GRPO 对它推理时仍使用同一个 persona system prompt，以检验部署形态下的
通用能力保持。Smoke 的统一 Base bundle 应包含 Dev 20 + GRPO 30 + Eval 50 +
Retention 30，共 130 条，并支持断点续跑。

阶段产物：

```text
data/
├── persona.json
├── style_examples.jsonl
├── sft_train_prompts.jsonl
├── sft_baseline_outputs.jsonl
├── sft_teacher_edits.jsonl
├── sft_teacher_reviews.jsonl
├── sft_train_generated.jsonl
├── sft_human_edits.json
├── sft_train.jsonl
├── sft_generation_meta.json
├── rl_train.jsonl
├── dev.jsonl
├── eval.jsonl
├── retention_eval.jsonl
├── retention_meta.json
├── prompt_generation_meta.json
├── prompt_validation.json
├── prompt_semantic_audit.json
├── prompt_human_audit.json
├── sft_validation.json
├── sft_human_audit.json
├── three_axis_audit.json
├── baseline_outputs.jsonl
├── baseline_generation_meta.json
└── stage1_acceptance.json
```

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
- 能力保持集无明显下降。

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
- 生成、去重并切分 Prompt。
- 生成 Student Baseline。
- Teacher 评分并最小改写。
- 导出 SFT 和冻结基线。

## Milestone 1.5：目标校准审计

- 固化角色一致性、格式一致性和对话质量的等权 Judge rubric，并定义角色一致性中的
  Style Fidelity 子项。
- 对冻结的 SFT 训练答案、Dev 和 Eval 基线补做三目标审计。
- 将结果写入新的 `three_axis_audit.json`，不覆盖阶段一的原验收文件。
- 审计通过后进入 SFT；不通过则用可追溯修订清单生成新版训练数据并重新验收。

## Milestone 2：端到端 Smoke

- 完成 SFT、GRPO、推理和评测。
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
