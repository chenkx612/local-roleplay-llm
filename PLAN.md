# 角色扮演强化学习 MVP

## 目标

为单个角色构建端到端训练流水线：

```text
persona + style examples
→ Student-aware SFT 数据
→ LoRA SFT
→ GRPO
→ Base/SFT/GRPO 评测报告
```

- 基础模型：`Qwen/Qwen3.5-2B`
- 训练框架：`ms-swift`
- 训练与推理：`enable_thinking=false`

核心问题：

> SFT 和 GRPO 能否提升小模型的角色一致性，同时保留原有对话能力？

## 范围

包含：

- 单角色训练
- 自动构造 SFT、GRPO 和评测数据
- LoRA SFT、轻量 GRPO
- Base、SFT、GRPO 统一评测
- CLI 和训练产物

不包含：

- 多角色混训、长期记忆、RAG
- 多轮 RL、DPO、PPO、Reward Model
- 全参数或分布式训练
- 视觉、语音和网页聊天前端
- 摆脱 persona system prompt

# 阶段一：数据与基线

## 1.1 输入

`persona.json`：

- 必填：`name`、`identity`、`personality`、`speech_style`、`relationships`、`facts`、`boundaries`
- 可选：`notes`
- 除 `name` 外均为自然语言字符串数组

`style_examples.jsonl`：10～30 组代表性对话。

```json
{"user": "你担心我吗？", "assistant": "我只是认为，少一个可靠的搭档会很麻烦。"}
```

## 1.2 数据准备与切分

1. 校验 `persona.json`，拒绝未知字段、错误类型和空字符串。
2. 用固定模板渲染 persona system prompt；所有阶段复用同一实现。
3. 生成五类用户 Prompt：
   - 日常对话
   - 背景与关系
   - 情绪与选择
   - 语言风格
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

隔离规则：

- SFT Train Prompt 只用于 SFT。
- Dev 只用于选配置。
- GRPO Prompt 只用于 GRPO。
- Eval 只用于最终评测。
- Dev、GRPO、Eval 的 Prompt 和回答不得进入 SFT。

## 1.3 Student-aware SFT

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

- 评价角色一致性、事实依据、风格和对话质量。
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
  "scores": {
    "persona": 0,
    "grounding": 0,
    "style": 0,
    "quality": 0
  },
  "issues": ["..."],
  "decision": "keep | light_rewrite | rewrite",
  "improved_assistant": "..."
}
```

评分范围为 0～10。每个 SFT Train Prompt 生成一条 SFT 训练样本；失败项重试补齐，
不按 `decision` 筛选。训练集只使用 `user` 和 `improved_assistant`，审计记录单独保存。

## 1.4 冻结基线

训练前保存 Dev、GRPO 和 Eval 的 Base 输出。Base、SFT、GRPO 必须使用相同的：

- 基础 checkpoint 和 revision
- 精度或量化策略
- 推理后端和 chat template
- persona prompt
- 生成参数

记录上述元数据。Eval Baseline 不参与训练或 Teacher 改写。

阶段产物：

```text
data/
├── persona.json
├── style_examples.jsonl
├── sft_train_prompts.jsonl
├── sft_baseline_outputs.jsonl
├── sft_teacher_edits.jsonl
├── sft_train.jsonl
├── rl_train.jsonl
├── dev.jsonl
├── eval.jsonl
├── retention_eval.jsonl
└── baseline_outputs.jsonl
```

# 阶段二：LoRA SFT

## 2.1 目标

- 修正身份、关系和事实错误。
- 学习角色语言风格。
- 减少出戏、编造、复读和冗长回答。
- 保留已有对话和指令遵循能力。

## 2.2 配置

```yaml
model: Qwen/Qwen3.5-2B
train_type: lora
dtype: bf16
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

只对 assistant 回复计算 loss。显存不足时再使用 4-bit。

MVP 比较：

- 学习率：`2e-5 / 5e-5 / 1e-4`
- Epoch：`1 / 2`

用 Dev 结果选配置；验收前不合并 LoRA。

## 2.3 验收

比较 `Base + Persona` 与 `SFT + Persona`：

- 角色与风格得分提高。
- 未知事实编造率下降。
- 无明显复读、模板化、统一拒答或回复过短。
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
R = 0.5 × Persona + 0.3 × Style + 0.2 × Quality - Penalty
```

| 维度 | 内容 |
|---|---|
| Persona | 身份、性格、关系、事实、边界 |
| Style | 长度、语气、用词、情绪表达 |
| Quality | 相关、自然、连贯、可继续对话 |

Teacher/Judge 负责三个 0～10 分的主评分；本地规则负责惩罚：

| 问题 | 惩罚 |
|---|---:|
| 自称 ChatGPT 或语言模型 | -3 |
| 与关键事实矛盾 | -3 |
| 复读或乱码 | -3 |
| 大段复述 persona | -2 |

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

角色评测：

- Persona Score
- Style Score
- Dialogue Quality
- 人设矛盾率
- 未知事实编造率
- 出戏率

能力保持集不使用 persona，覆盖普通问答、总结、改写、基础推理和指令遵循。

人工检查：

- Smoke：抽查 10 条
- MVP：盲评 30 条

## 4.3 成功标准

- SFT 的 Persona 和 Style 高于 Base。
- SFT 的未知事实编造率不高于 Base。
- GRPO 的矛盾率或出戏率低于 SFT。
- SFT 和 GRPO 无明显复读、模板化或过短回答。
- 能力保持集无明显回归。
- 人工盲评中 GRPO 不弱于 SFT。
- 第二个角色可以复现整条流程。

# 实施顺序

## Milestone 1：Smoke 数据

- 完成 Persona 校验和 Prompt 渲染。
- 生成、去重并切分 Prompt。
- 生成 Student Baseline。
- Teacher 评分并最小改写。
- 导出 SFT 和冻结基线。

## Milestone 2：端到端 Smoke

- 完成 SFT、GRPO、推理和评测。
- 检查日志与产物完整性。

## Milestone 3：MVP

- 扩充到 300 条 SFT、100 条 GRPO、50 条 Dev 和 100 条 Eval。
- 用 Dev 选择 SFT 配置。
- 完成 GRPO 和统一评测。

## Milestone 4：交付

- 用第二个角色复现。
- 输出 LoRA、评测报告和运行说明。

# 核心实验

| 模型 | Persona | Style | 出戏率 | 矛盾率 | 能力保持 |
|---|---:|---:|---:|---:|---:|
| Base + Persona | 基线 | 基线 | 基线 | 基线 | 基线 |
| SFT | ↑ | ↑ | ↓ | ↓ | ≈ |
| SFT + GRPO | ↑↑ | ≈ | ↓↓ | ↓↓ | ≈ |

预期结论：

> Student-aware SFT 修正小模型已暴露的角色缺陷；GRPO 进一步降低出戏和人设冲突。
