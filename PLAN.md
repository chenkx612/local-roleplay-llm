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

> SFT 和 GRPO 能否依次保证小模型生成稳定、角色成立和对话有意义？

三个目标按前置关系逐层验收，不用宏平均掩盖底层失败：

- **生成稳定性**：回复完整、连贯、可读，无乱码、严重复读或破坏性截断。
- **角色一致性**：核心身份、性格、关系、边界和语言风格符合 Persona。
- **对话质量**：回答相关、自然、连贯、有信息量且可继续对话。

动作括号、emoji、固定开头和回答长度只作为诊断信息，不是独立目标。角色可以在不违背核心
设定和当前对话的前提下合理创作；不设置独立的事实可靠性目标，但不得擅自编造用户个人经历、
重大共同关系或与当前对话冲突的信息。

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
风格样例可以使用动作描写，也可以使用纯对白；动作描写应简短、自然且不妨碍阅读，不设置固定
括号位置或数量契约。

输入需通过结构与人工检查，并保存快照及最终 system prompt。

### 3.2 数据规格

数据覆盖日常对话、背景与关系、情绪与选择、语言风格、出戏与冲突五类场景。共享候选池
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

Teacher 按生成稳定性、角色一致性和对话质量依次检查。合格回答原样保留；不合格回答做最小充分
改写。允许合理的角色化创作，但不得违背核心设定、当前对话，或擅自建立用户个人经历和重大共同
关系。正式运行前先完成五类场景各一条的 Pilot；正式产物需通过结构、对齐、空回答、截断标记、
乱码和明显复读检查。

### 3.4 阶段验收

- Persona、风格样例和 system prompt 可以正常加载。
- SFT、GRPO、Dev、Eval 达到目标规模且无精确重复泄漏。
- SFT 训练数据采用可读取的 `messages` 结构，抽查无阻断训练的系统性问题。
- Dev/Eval Base 输出与冻结输入对齐，回答有效，可作为后续对比基线。
- 数据规模、随机种子、模型 revision、生成配置和已知问题均有记录。

## 4. 阶段二：LoRA SFT

**状态：有效训练已完成，按新版三目标待复核。** 第三次 Colab 运行完成有效更新；旧版行为门槛
因严格格式率失败。该旧结论作为历史证据保留，后续按本节新版口径重新评估，详见 `RUNLOG.md`。

目标是用一组 QLoRA 配置完成训练、保存、重新加载和 Dev 推理，观察三项目标相对 Base 的变化。

```yaml
model: Qwen/Qwen3.5-2B
train_type: lora
dtype: float32
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
batch_size: 1
gradient_accumulation_steps: 16
enable_thinking: false
```

只对 assistant 回复计算 loss。第三轮只相对有效更新方案增加 epoch，不修改冻结的 50 条训练数据、
模型 revision、LoRA 结构、学习率或精度配置；不搜索超参数、不自动重训，也不要求合并 LoRA。

阶段验收：

- 3 epochs 正常结束，约 12 个 `grad_norm` 有限且为正，LoRA-B 全量更新，adapter 可重载。
- 使用 `20260807/08/09` 三个固定 seed，在同一 Transformers 后端为 Base/SFT 各生成 30 条 Dev；
  每次推理前分别重置 RNG，输出非空并按 `(seed, id)` 完全对齐。
- 自动门槛只负责生成稳定性：输出完整非空并对齐，SFT 的 `stop` 不减少、截断不增加，且没有
  严重复读或乱码。严格动作格式、emoji、括号位置、回答长度和自称频率只保留为诊断统计。
- 稳定性通过后，对主 seed 10 对输出做固定顺序的匿名 A/B 复核，按生成稳定性、角色一致性、
  对话质量三个维度评分；SFT 至少胜 6 对、明显落后不超过 2 对、无严重问题，且三个维度的平均
  分均不得低于 Base。
- 保存训练配置、日志、适配器、30+30 Dev 输出、匿名复核三件套和统一摘要。

技术有效但行为未达标时保留产物并记录负向结果，但不得进入 GRPO。只有
`technical_gate && core_behavior_gate && manual_gate` 才令 `ready_for_grpo=true`。

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
R = Readable × (RoleConsistency + DialogueQuality) / 2 - Penalty
```

`Readable` 为 0/1 生成稳定性门控；不可读回答直接失去主体奖励。角色一致性和对话质量均为 0～10
分，由 Teacher/Judge 评价。`Penalty` 只处理大段复述 Persona、照抄风格样例等奖励投机，避免
重复惩罚主评分已覆盖的问题。

阶段验收：

- 奖励函数可以稳定计算，并人工检查至少 5 组候选及其分数。
- 完成一组有效训练，LoRA 可重新加载。
- 保存奖励曲线和样本，记录平均奖励、回答长度、复读率及奖励投机现象。
- 不根据 Eval 结果反复调整奖励或训练配置。

## 6. 阶段四：统一评测

**状态：未开始。**

Base、SFT 和 GRPO 使用同一 Eval、Persona、生成参数和 Judge。分别报告：

- Generation Stability：完整率、截断、乱码、严重复读和可读性。
- Role Consistency：核心身份、边界、性格、关系和 Style Fidelity。
- Dialogue Quality：相关性、自然度、连贯性、信息量和可继续对话性。

三个目标逐层报告：生成稳定性失败的回答不继续解释高层得分；稳定后再判断角色是否成立，最后
比较对话质量。同时记录严格动作格式、emoji、自称、创造性细节、出戏、模板化和回答长度等诊断
指标。首次实践人工检查 10 条 Eval，优先采用匿名两两比较；另准备 5～10 条不使用 Persona 的
小型能力保持集，观察普通问答、总结、改写、基础推理和指令遵循是否明显回退。

阶段验收：

- 三个模型在同一 Eval 上均有完整输出、自动指标和人工观察。
- 分析 SFT 相对 Base、GRPO 相对 SFT 的逐层变化，不用单一总分掩盖底层失败。
- 检查自动稳定性规则与人工观察是否一致，并说明冲突时采用哪种证据。
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

- [x] 完成一组**有效更新**的 QLoRA SFT，保存训练日志和 LoRA；首次 FP16 运行已判无效。
- [ ] 在同一 Transformers 后端生成 Base/SFT Dev，通过机械门槛和人工比较。
- [ ] 将重训实际配置、观察、问题和进入 GRPO 的决定写入 `RUNLOG.md`。

### Milestone 3：GRPO 与统一评测

- [ ] 验证奖励函数并完成一组 GRPO。
- [ ] 在同一 Eval 上生成 Base、SFT 和 GRPO 输出。
- [ ] 完成自动指标、10 条人工检查和能力保持观察。

### Milestone 4：复盘

- [ ] 汇总各阶段证据并撰写最终复盘报告。
- [ ] 说明失败点、结论边界和下一轮唯一优先改进方向。

扩大数据、单变量对照实验和第二角色复现均为可选扩展，不阻塞首次学习闭环。
