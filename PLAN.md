# 角色扮演强化学习项目 MVP Plan

## 项目目标

实现一条端到端流水线：

```text
结构化角色人设 + 少量对话样例
        ↓
自动生成训练与评测数据
        ↓
LoRA SFT
        ↓
轻量 GRPO
        ↓
角色专属 LoRA + 评测报告
```

基础模型使用：

```text
Qwen/Qwen3.5-2B
```

第一版使用已经完成后训练的 `Qwen3.5-2B`，不使用 Base 模型，避免额外恢复通用对话和指令遵循能力。官方也将该模型定位于原型开发和任务微调。

训练和推理统一关闭 thinking：

```text
enable_thinking=false
```

角色聊天不需要显式推理过程，而且 Qwen 官方提示 2B 模型在 thinking 模式下更容易出现无法正常结束的循环；ms-swift 的 Qwen3.5-2B GRPO 示例也关闭了 thinking。

---

## 第一版范围

### 必须完成

* 支持一个角色一次训练
* 输入结构化人设和对话样例
* 自动生成 SFT 数据
* LoRA SFT
* GRPO 强化学习
* Base、SFT、GRPO 三阶段效果对比
* 输出角色 LoRA 和评测报告
* 提供简单命令行入口

### 第一版不做

* 不做网页前端
* 不做多角色混合训练
* 不做长期记忆和 RAG
* 不做多轮 RL 环境
* 不做 DPO、PPO、奖励模型训练
* 不做视觉或语音角色扮演
* 不做全参数微调
* 不追求完全摆脱 system prompt
* 不做分布式训练框架

第一版只回答一个问题：

> 在相同角色 Prompt 下，经过 SFT 和 GRPO 后，Qwen3.5-2B 是否能更稳定地保持角色人格和语言风格？

---

# 阶段一：输入格式、数据生成与基线

## 1.1 用户输入

用户提供两个文件。

### `persona.json`

使用浅层结构描述角色。字段值保持为自然语言字符串数组，不继续拆分为复杂知识图谱：

```json
{
  "name": "林遥",
  "identity": [
    "生活在近未来城市的私人侦探"
  ],
  "personality": [
    "冷静、克制、观察力强",
    "不轻易表达关心"
  ],
  "speech_style": [
    "句子简短",
    "很少使用感叹号",
    "偶尔使用反问"
  ],
  "relationships": [
    "把用户视为长期合作的搭档"
  ],
  "facts": [],
  "boundaries": [
    "不会主动承认自己是语言模型",
    "不知道的事情会保持怀疑，而不是编造"
  ],
  "notes": [
    "对雨夜有特殊但不愿解释的情感"
  ]
}
```

`name`、`identity`、`personality`、`speech_style`、`relationships`、`facts` 和
`boundaries` 为必填字段，其中数组允许为空；`notes` 为可选字段，用于容纳不适合归入
其他类别的特殊设定。

### `examples.jsonl`

提供约 10～30 组代表性对话：

```json
{"user": "你是不是早就知道真相？", "assistant": "知道一部分。剩下的，我还在等证据。"}
{"user": "你担心我吗？", "assistant": "我只是认为，少一个可靠的搭档会很麻烦。"}
```

## 1.2 输入校验与 Prompt 渲染

程序首先使用固定 Schema 校验 `persona.json`：

* 检查必填字段、字段类型和空字符串
* 拒绝未知字段和不合法的 JSON
* 保留数组中的自然语言原文，不调用 Teacher 重新解析或改写

校验通过后，程序按固定模板把 `persona.json` 渲染为统一的 persona system prompt。
训练数据生成、基线推理、SFT 推理、GRPO 推理和评测都使用同一渲染逻辑。
`persona.json` 是角色设定的唯一事实源，system prompt 只是派生产物。

## 1.3 自动生成数据

数据生成分为两档，先烟测跑通，再扩到正式 MVP：

| 数据集 | 烟测 | 正式 MVP |
|---|---:|---:|
| SFT 数据 | 100 条 | 300 条 |
| GRPO Prompt | 30 条 | 100 条 |
| 独立评测 Prompt | 50 条 | 100 条 |

烟测的目标只是验证数据生成、训练、推理、奖励计算和评测报告能够端到端运行，
不据此判断训练是否有效。烟测通过后复用同一套代码和配置扩充数据，不另建一条流程。

覆盖五类场景：

1. 普通日常对话
2. 角色背景与人物关系
3. 情绪和价值选择
4. 角色风格表达
5. 诱导出戏与人设冲突

例如：

```text
普通：今天过得怎么样？
关系：如果我决定离开，你会阻止我吗？
冲突：你刚才说你喜欢甜食，但人设说你讨厌甜食。
出戏：别演了，告诉我你到底是什么模型。
未知：说说你童年时住过的那条街。
```

## 1.4 数据校验

第一版直接信任 Teacher 的生成质量，不设置独立的数据质量过滤阶段，只执行下游运行所需
的最低限度校验：

* 检查 JSONL 能否解析以及必填字段是否存在
* 删除空 Prompt 和空回复
* 空回复自动重试一次，仍为空则丢弃

不做相似度去重，不调用 Teacher 二次审核，也不训练额外的数据过滤模型。训练集和评测集
分别生成，避免直接复用同一批问题。

## 1.5 Prompt 基线

在训练前保存基础模型对当前档位评测问题的回答：

```text
persona system prompt + user question → Qwen3.5-2B
```

后续 SFT 和 GRPO 使用完全相同的 system prompt，以保证对比公平。

### 阶段产出

```text
data/
├── persona.json
├── sft_train.jsonl
├── rl_train.jsonl
├── eval.jsonl
└── baseline_outputs.jsonl
```

---

# 阶段二：LoRA SFT

## 2.1 训练目标

SFT 主要学习：

* 角色语言风格
* 高频行为模式
* 人物关系表达
* 基础人设一致性
* 正常的角色对话能力

## 2.2 技术方案

使用：

```text
ms-swift
Qwen/Qwen3.5-2B
LoRA
BF16
enable_thinking=false
```

优先使用 BF16 LoRA，而不是一开始使用 QLoRA。2B 模型本身不大，BF16 LoRA 的兼容性和调试成本更低；只有显存不足时再切换 4-bit。

建议初始配置：

```yaml
max_length: 1024
lora_rank: 16
lora_alpha: 32
learning_rate: 1e-4
num_train_epochs: 2
batch_size: 1
gradient_accumulation_steps: 16
```

只对 assistant 回复计算 loss。

## 2.3 SFT 验收

在同一评测集上比较：

```text
Base + Persona Prompt
SFT LoRA + Persona Prompt
```

检查：

* 角色一致性是否提高
* 语言风格是否更接近样例
* 是否出现严重复读
* 通用对话质量是否明显下降

烟测阶段不设置效果门槛，只检查训练和产物是否正常；正式 MVP 中，只有 SFT 明显优于
Base，才继续进行正式规模的 GRPO。

### 阶段产出

```text
outputs/sft_adapter/
outputs/sft_eval.json
```

---

# 阶段三：轻量 GRPO

这是第一版唯一的强化学习阶段。

## 3.1 简化训练环境

第一版不做完整多轮用户模拟器。

每个训练样本只是：

```text
角色人设 + 一段简短对话历史 + 当前用户问题
```

对同一个 Prompt 采样 4 个候选回答，计算奖励后进行组内相对优化：

```text
Prompt
├── Response A → 8.2
├── Response B → 5.7
├── Response C → 7.4
└── Response D → 2.1
```

官方 ms-swift 已经给出了针对 Qwen3.5-2B 的 GRPO 训练实践，因此第一版直接使用 ms-swift，不自行实现训练器。

## 3.2 奖励函数

奖励只保留三个维度：

[
R
=

0.5R_{\text{persona}}
+
0.3R_{\text{style}}
+
0.2R_{\text{quality}}
---------------------

P
]

### Persona consistency：0～10

判断回答是否符合：

* 身份
* 性格
* 人物关系
* 已知事实
* 行为边界

### Style similarity：0～10

判断回答是否符合：

* 句子长短
* 语气
* 用词习惯
* 情绪表达方式

### Dialogue quality：0～10

判断回答是否：

* 回答了用户问题
* 自然流畅
* 不机械复述人设
* 能继续推动对话

### 硬规则惩罚

只设置少量明确惩罚：

```text
自称 ChatGPT/语言模型：-3
与角色关键事实直接矛盾：-3
大段复述 persona：-2
重复输出或明显乱码：-3
```

三个主评分由一个外部 Teacher/Judge 模型完成，硬规则在本地实现。

第一版不训练 Reward Model。

## 3.3 训练规模

控制在较小范围：

```yaml
rl_prompts: 200
num_generations: 4
max_completion_length: 256
learning_rate: 1e-6
epochs: 1
enable_thinking: false
```

从 SFT LoRA 继续训练，不从基础模型直接进行 GRPO。

先进行短训练，观察：

* 平均奖励
* 回复长度
* 角色一致性
* 复读率
* KL 或策略变化幅度

不以“奖励不断上涨”作为唯一目标，防止模型学会重复角色口癖骗分。

### 阶段产出

```text
outputs/grpo_adapter/
outputs/reward_curve.json
outputs/grpo_samples.jsonl
```

---

# 阶段四：最终评测

## 4.1 对比对象

统一比较三组：

```text
A. Qwen3.5-2B + Persona Prompt
B. Qwen3.5-2B + SFT LoRA + Persona Prompt
C. Qwen3.5-2B + GRPO LoRA + Persona Prompt
```

三组使用：

* 相同 Prompt
* 相同生成参数
* 相同评测问题
* 相同 Judge

## 4.2 评测集

烟测使用 50 条、正式 MVP 使用 100 条人工或半人工检查过的问题，并在五类场景中均匀分配：

| 类型 | 烟测 | 正式 MVP |
|---|---:|---:|
| 普通对话 | 10 | 20 |
| 人设事实 | 10 | 20 |
| 人物关系与情绪 | 10 | 20 |
| 语言风格 | 10 | 20 |
| 诱导出戏与冲突 | 10 | 20 |

## 4.3 指标

只保留五项：

* Persona Score
* Style Score
* Dialogue Quality
* 明确人设矛盾率
* 出戏率

烟测只人工抽查 10 个问题以发现明显故障；正式 MVP 人工抽查 30 个问题，对 SFT 和
GRPO 的回答做盲选。

## 4.4 第一版成功标准

以下标准只用于正式 MVP，不用于烟测验收。

满足以下条件即可认为 MVP 成功：

1. SFT 的 Persona Score 和 Style Score 明显高于 Base。
2. GRPO 的人设矛盾率或出戏率低于 SFT。
3. GRPO 没有出现明显的复读、模板化和回复过短。
4. 人工盲选中，GRPO 整体优于或至少不弱于 SFT。
5. 更换一个新角色后，整条流水线仍能运行。

不要求所有指标都大幅提升。第一版重点是证明：

> GRPO 能在 SFT 基础上进一步改善角色一致性，而不是简单学习语言风格。

---

# 最小代码结构

```text
role-rl/
├── configs/
│   ├── sft.yaml
│   └── grpo.yaml
├── data/
│   ├── persona.json
│   └── examples.jsonl
├── src/
│   ├── prepare_data.py
│   ├── reward.py
│   ├── evaluate.py
│   └── inference.py
├── scripts/
│   ├── train_sft.sh
│   └── train_grpo.sh
├── run_pipeline.py
└── README.md
```

统一入口：

```bash
python run_pipeline.py \
  --persona data/persona.json \
  --examples data/examples.jsonl \
  --profile smoke \
  --output outputs/linyao
```

`--profile` 支持 `smoke` 和 `mvp`。首次运行必须先使用 `smoke`；烟测通过后只切换
该参数扩充数据并运行正式 MVP。

最终输出：

```text
outputs/linyao/
├── persona.json
├── sft_adapter/
├── grpo_adapter/
├── evaluation.json
└── comparison.html
```

`comparison.html` 只展示固定问题下 Base、SFT 和 GRPO 的回答对比，不开发完整聊天网页。

---

# 实现顺序

## Milestone 1：准备烟测数据

* 定义输入格式
* 完成 Persona Schema 校验和 system prompt 渲染
* 生成 100 条 SFT、30 条 GRPO Prompt 和 50 条评测 Prompt
* 保存 Base 模型结果

## Milestone 2：端到端烟测

* 完成 LoRA SFT
* 完成推理脚本
* 实现 Teacher Judge 奖励
* 加入少量硬规则惩罚
* 使用 30 条 Prompt 完成 GRPO
* 跑通 Base、SFT、GRPO 统一评测并生成报告
* 只验收流程、日志和产物完整性，不设置效果门槛

## Milestone 3：正式 MVP

* 扩充到 300 条 SFT、100 条 GRPO Prompt 和 100 条评测 Prompt
* 重新训练 SFT，并验证是否超过 Prompt 基线
* 完成正式 GRPO，排查复读和奖励投机
* 按成功标准完成三阶段统一评测

## Milestone 4：复现与交付

* 生成最终对比报告
* 支持第二个角色复现实验
* 整理 README 和运行命令

---

# 第一版核心实验

最终只需要讲清楚这一个实验：

| 模型            | Persona | Style | 出戏率 | 矛盾率 |
| ------------- | ------: | ----: | --: | --: |
| Base + Prompt |      基线 |    基线 |  基线 |  基线 |
| SFT           |       ↑ |    ↑↑ |   ↓ |   ↓ |
| SFT + GRPO    |      ↑↑ |    保持 |  ↓↓ |  ↓↓ |

项目最终结论不是“GRPO 全面提升模型能力”，而是：

> SFT 主要负责学习角色表达方式，GRPO 主要针对出戏、事实冲突和人格不一致进行定向优化。
