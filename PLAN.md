# 角色扮演业务后训练学习计划

## 1. 项目目标

本项目用一个角色和最小数据规模，完整实践以下链路：

```text
业务目标与评测口径
→ Persona、风格样例与数据切分
→ Base 基线
→ Teacher-corrected SFT 数据与 LoRA SFT
→ GRPO
→ Base/SFT/GRPO 统一评测
→ 复盘
```

项目追求“小而完整”：每个核心阶段只运行一组主配置并留下最小可检查产物，不建设生产级
平台，也不因缩短周期而跳过阶段。模型没有提升也是有效结果，前提是链路有效、证据完整且结论
边界清楚。

核心研究问题：

> SFT 和 GRPO 能否同时提升小模型的角色一致性、格式一致性和对话质量，而不以牺牲其中一个
> 目标换取另一个目标的提升？

三个目标分别评分，并以等权宏平均作为主要对比指标：

- **角色一致性**：身份、性格、关系、事实、边界和语言风格符合 Persona。
- **格式一致性**：稳定输出“全角括号动作或神态 + 口语对白”。
- **对话质量**：回答相关、自然、连贯、有信息量且可继续对话。

## 2. 技术基线与文档职责

- 训练基座：`Qwen/Qwen3.5-2B`，由 ms-swift/BNB 以 4-bit 加载。
- 本地 Student 推理：`mlx-community/Qwen3.5-2B-4bit`。
- Teacher/Judge：`deepseek-v4-flash`。
- Student 训练与推理关闭 thinking；Teacher 开启 thinking，并使用 `reasoning_effort=high`。

文档各自只承担一种职责：

- `PLAN.md`：预定流程、主配置和验收标准。
- `RUNLOG.md`：每次运行的实际配置、产物、观察、偏差和验证证据。
- `ISSUES.md`：问题、优先级、处理过程与最终决策。
- 最终复盘报告：项目完成后基于以上文档和 `data/runs/` 产物编写。

所有阶段必须隔离训练与评测数据、保存实际配置和模型 revision，并记录计划与实际运行的差异。
可选扩展不得阻塞首次闭环。

## 3. 阶段一：数据与 Base

**状态：已完成（2026-08-07，`morgana-v1`）。** 详细记录见 `RUNLOG.md`。

### 3.1 输入契约

`persona.json` 必须包含 `name`、`identity`、`personality`、`speech_style`、
`relationships`、`facts`、`boundaries`，可选 `notes`。除 `name` 外，各字段均为字符串数组。
Persona 是角色身份、关系、当前状态和边界的最高优先级依据。

`style_examples.jsonl` 准备 10～20 组代表性对话，只定义表达风格，不迁移其中的具体经历。
统一输出格式为：

```text
（简短的动作、神态或当下反应）口语对白
```

输入需通过结构与人工检查，并保存快照及最终 system prompt。

### 3.2 数据规格

数据覆盖日常对话、背景与关系、情绪与选择、格式与风格、冲突与未知事实五类场景。共享候选池
按固定种子切分，记录只包含 `id`、`scenario`、`target_goals` 和 `user`。

| 数据 | 首次实践规模 | 用途 |
|---|---:|---|
| Pilot | 5 | 验证 Student/Teacher 链路 |
| SFT | 50 | 监督训练 |
| GRPO | 20 | 强化学习 |
| Dev | 10 | 阶段观察 |
| Eval | 20 | 最终统一评测 |

生成阶段只生成用户 Prompt。四个 split 必须数量正确、场景均衡且无规范化精确重复；Dev、GRPO
和 Eval 不得进入 SFT。Prompt 必须独立成立、视角正确、无评测目标泄漏，并经自动过滤与人工抽查。

### 3.3 Teacher-corrected SFT

```text
Persona + SFT Prompt → Student baseline
Persona + style examples + Prompt + baseline → Teacher 最小充分纠错
```

Teacher 分别检查角色、事实、风格、格式和对话质量。合格回答原样保留；不合格回答改写，但不得
编造用户信息、重大关系变化、具体共同经历或需要跨轮记忆的持续状态。正式运行前先完成五类场景
各一条的 Pilot；正式产物需通过结构、对齐、空回答、格式、截断标记和明显复读检查。

### 3.4 阶段验收

- Persona、风格样例和 system prompt 可以正常加载。
- SFT、GRPO、Dev、Eval 达到目标规模且无精确重复泄漏。
- SFT 训练数据采用可读取的 `messages` 结构，抽查无阻断训练的系统性问题。
- Dev/Eval Base 输出与冻结输入对齐，回答有效，可作为后续对比基线。
- 数据规模、随机种子、模型 revision、生成配置和已知问题均有记录。

## 4. 阶段二：LoRA SFT

**状态：未开始。**

目标是用一组 QLoRA 配置完成训练、保存、重新加载和 Dev 推理，观察三项目标相对 Base 的变化。

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

只对 assistant 回复计算 loss。首次实践不搜索超参数，也不要求合并 LoRA。

阶段验收：

- 训练正常结束，loss 可解释，LoRA 可以重新加载并生成非空回答。
- 在 Dev 上使用与 Base 可比的 Persona、模板和生成参数，记录所有差异。
- 记录角色、格式、对话质量，以及复读、模板化、统一拒答、过短或格式投机等退化。
- 保存训练配置、日志、适配器、Dev 输出和阶段观察。

技术失败必须修复或形成清楚记录；只要链路有效，即使 SFT 未优于 Base，也继续进入 GRPO。

## 5. 阶段三：GRPO

**状态：未开始。**

从 SFT LoRA 继续训练，每个 Prompt 采样 4 个回答，体验组内相对奖励、策略更新和奖励投机观察。

```yaml
rl_prompts: 20
num_generations: 4
max_completion_length: 256
learning_rate: 1e-6
epochs: 1
enable_thinking: false
```

奖励定义：

```text
R = (RoleConsistency + FormatConsistency + DialogueQuality) / 3 - Penalty
```

三个主分均为 0～10 分。确定性规则检查格式；Teacher/Judge 评价角色与对话质量。`Penalty` 只处理
复读、乱码、大段复述 Persona、照抄风格样例和空洞格式投机，避免重复惩罚主评分已覆盖的问题。

阶段验收：

- 奖励函数可以稳定计算，并人工检查至少 5 组候选及其分数。
- 完成一组有效训练，LoRA 可重新加载。
- 保存奖励曲线和样本，记录平均奖励、回答长度、复读率及奖励投机现象。
- 不根据 Eval 结果反复调整奖励或训练配置。

## 6. 阶段四：统一评测

**状态：未开始。**

Base、SFT 和 GRPO 使用同一 Eval、Persona、生成参数和 Judge。分别报告：

- Role Consistency：身份/事实/边界、性格/关系、Style Fidelity。
- Format Consistency：规则通过率与格式自然度。
- Dialogue Quality：相关性、自然度、连贯性、信息量和可继续对话性。
- 三目标等权宏平均。

同时记录人设矛盾、未知事实编造、出戏、复读、模板化和回答长度等诊断指标。首次实践人工检查
10 条 Eval，优先采用匿名两两比较；另准备 5～10 条不使用 Persona 的小型能力保持集，观察普通
问答、总结、改写、基础推理和指令遵循是否明显回退。

阶段验收：

- 三个模型在同一 Eval 上均有完整输出、自动指标和人工观察。
- 分析 SFT 相对 Base、GRPO 相对 SFT 的各单项变化，而不只看宏平均。
- 检查自动评分、格式规则与人工观察是否一致，并说明冲突时采用哪种证据。
- 明确小样本、单次训练和 Judge 偏差允许支持与不允许支持的结论。

## 7. 复盘与完成标准

最终复盘至少回答：

- 每个阶段实际做了什么，与计划有什么偏差，为什么？
- SFT 和 GRPO 分别改变了哪些行为，证据是什么？
- 哪些变化来自数据、训练配置或奖励设计，哪些无法判断？
- 是否出现能力回退、奖励投机或其他副作用？
- 本次结果的限制是什么，下一轮最值得验证的一个改进是什么？

学习闭环完成标准：

- SFT 和 GRPO 均完成一次有效训练并留下可加载产物；失败需修复或清楚记录替代实践。
- Base、SFT、GRPO 在同一 Eval 上完成自动与人工对比。
- 每个阶段的输入、实际配置、日志、产物、关键决策和限制已写入 `RUNLOG.md`。
- 输出最终复盘报告；负向结果不影响学习闭环完成。

## 8. 实施清单

### Milestone 1：数据与 Base

- [x] 冻结输入、system prompt 和四个隔离 split。
- [x] 完成 Pilot、Teacher-corrected SFT 和人工抽查。
- [x] 生成并验证 Dev/Eval Base，完成阶段记录。

### Milestone 2：SFT

- [ ] 完成一组 QLoRA SFT，保存训练日志和 LoRA。
- [ ] 在 Dev 上生成回答并与 Base 初步比较。
- [ ] 将实际配置、观察、问题和决策写入 `RUNLOG.md`。

### Milestone 3：GRPO 与统一评测

- [ ] 验证奖励函数并完成一组 GRPO。
- [ ] 在同一 Eval 上生成 Base、SFT 和 GRPO 输出。
- [ ] 完成自动指标、10 条人工检查和能力保持观察。

### Milestone 4：复盘

- [ ] 汇总各阶段证据并撰写最终复盘报告。
- [ ] 说明失败点、结论边界和下一轮唯一优先改进方向。

扩大数据、单变量对照实验和第二角色复现均为可选扩展，不阻塞首次学习闭环。
