# morgana-v2 执行规约

本文承接 `PLAN.md` 中不影响理解核心逻辑、但执行时必须明确的固定细节。这里记录预定配置与
验收口径；实际运行及偏差写入 `RUNLOG.md`，发现的问题写入 `ISSUES.md`。

`RUNLOG.md` 只记录 v2 的实际情况；`ISSUES.md` 中的 v1 内容保持封存，v2 新问题必须明确标注
版本。

## 1. 运行边界与技术基线

- Run 名称：`morgana-v2`。
- 数据目录：`data/runs/morgana-v2/`。
- 模型目录：`output/morgana-v2/`。
- 训练与正式评测基座：`Qwen/Qwen3.5-2B`，固定并记录实际 revision。
- QLoRA：ms-swift + bitsandbytes 4-bit，Colab Tesla T4。
- Teacher/Judge：`deepseek-v4-flash`，开启 thinking，并记录实际模型和请求配置。
- Student、SFT、GRPO 生成关闭 thinking。
- Base、SFT、GRPO 比较固定使用同一模型 revision、聊天模板和生成参数；学习项目允许使用
  同一可用推理链路，不要求为此额外搭建 Transformers 专用环境。

v1 数据和模型产物保持只读。角色源设定可以复用，但必须重新保存 v2 输入快照和哈希；v1 的
system prompt、Teacher 标签、Base 输出和 SFT adapter 不进入 v2 训练或正式评测。

每次运行记录输入哈希、代码 commit、模型 revision、环境版本、实际命令、配置、日志和输出。
训练集、Dev 和 Eval 必须隔离。

## 2. 数据与 Base

### 2.1 冻结输入

检查并冻结：

- `persona.json`：核心身份、性格、语言风格、关系、事实和边界。
- `style_examples.jsonl`：10～20 组代表性对话，只表达风格，不把示例经历当作事实。
- v2 system prompt：优先保证自然可读；动作描写可选，不规定括号、emoji 或固定开头。

风格样例应包含自然变化，避免所有样例使用同一种动作模板。输入通过结构检查和人工抽查后保存
快照与哈希。

### 2.2 Prompt 数据

使用五类场景：日常对话、背景与关系、情绪与选择、语言风格、出戏与冲突。

| 数据 | 数量 | 每类场景 | 用途 |
|---|---:|---:|---|
| Pilot | 5 | 1 | 验证 Student/Teacher 链路 |
| SFT | 50 | 10 | 监督训练 |
| GRPO | 20 | 4 | 强化学习 |
| Dev | 10 | 2 | 阶段选择与观察 |
| Eval | 20 | 4 | 最终统一评测 |

先生成 100 条共享候选池，再按固定 seed 切分为 SFT/GRPO/Dev/Eval。四个 split 必须数量正确、
场景均衡、全局无规范化精确重复。记录只包含 `id`、`scenario`、`target_goals` 和 `user`。
Eval 在最终统一评测前不得用于修改数据、提示词或训练配置。

### 2.3 Teacher-corrected SFT

```text
Persona + SFT Prompt → Student baseline
Persona + style examples + Prompt + baseline → Teacher 最小充分纠错
```

Teacher 依次检查生成稳定性、角色一致性和对话质量。合格回答原样保留，不合格回答只做必要
改写。动作括号、emoji、固定开头或精确复述 Persona 均不是通过条件。

先运行五类场景各一条的 Pilot 并人工复核，再生成 50 条正式标签。正式产物必须结构正确、逐条
对齐、回答非空，且没有乱码和明显复读；保留 Student baseline、Teacher audit 和最终训练标签。

### 2.4 Base Dev

为 10 条 Dev 用一个固定 seed 生成 10 条 Base 输出。首次推理时记录实际模型/revision、推理
链路、seed、聊天模板和生成参数；后续 SFT 与 GRPO 沿用这些条件即可。无需为基线额外搭建
Transformers 环境，也不要求多 seed 重复采样。

进入 SFT 前必须具备：冻结输入、四个有效 split、通过人工复核的 Pilot、50 条有效 SFT 标签和
10 条有效 Base Dev。

## 3. QLoRA SFT

只运行一组主配置：

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
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
enable_thinking: false
```

该配置用于第二次 SFT。第一次 SFT 的配置和结果已原样归档；第二次把有效 batch size 从 16
降到 4，使 50 条数据、3 epochs 下的预期 optimizer step 从 12 增加到 39。物理 batch size
设为 2、梯度累积设为 2，以利用 T4 的剩余显存；数据、学习率、epoch、LoRA 和推理参数保持
不变。训练前还必须确认 50 条 assistant 标签全部包含“吾辈”，且不包含“本大爷”或“本喵”。

只对 assistant 回复计算 loss，不使用 Dev 搜索 epoch、学习率或 LoRA 参数。技术失败可以修复明确
的实现问题后重跑；行为失败不得在同一轮静默调参。

### 3.1 技术门槛

- 3 epochs 正常结束，optimizer step 数与预期一致。
- 所有记录的 `grad_norm` 有限且为正。
- 所有 LoRA-B 张量出现非零更新，adapter 张量全部有限。
- adapter 可重新加载并生成非空回答。

### 3.2 生成稳定性门槛

使用阶段一冻结的固定 seed 和推理条件重新生成 Base/SFT 各 10 条，并按 `id` 对齐：

- 两侧输出完整、非空且数量正确。
- SFT 的 `stop` 数不得低于 Base，截断数不得高于 Base。
- SFT 不得出现乱码或严重复读。

严格格式、emoji、括号、自称和长度只统计，不影响通过。

### 3.3 人工门槛

生成稳定性通过后，对主 seed 的 10 对回答做匿名 A/B 复核，分别按生成稳定性、角色一致性和
对话质量评分：

- SFT 至少胜 6 对。
- SFT 明显落后不超过 2 对。
- SFT 无不可读、角色崩坏或视角错位等严重问题。
- 三个维度的 SFT 平均分均不得低于 Base。

仅当技术门槛、生成稳定性门槛和人工门槛全部通过时进入 GRPO。

## 4. GRPO

从通过 SFT 门槛的 adapter 继续训练，使用冻结的 20 条 GRPO Prompt：

```yaml
num_generations: 4
max_completion_length: 256
learning_rate: 1e-6
num_train_epochs: 1
enable_thinking: false
```

奖励定义为：

```text
R = Readable × (RoleConsistency + DialogueQuality) / 2 - ExploitPenalty
```

- `Readable`：0/1；乱码、严重复读或破坏性截断为 0。
- `RoleConsistency`：0～10；检查核心 Persona 和对话视角，不得擅自建立用户个人经历或重大
  共同关系。
- `DialogueQuality`：0～10；检查相关性、自然度和内容价值。
- `ExploitPenalty`：只惩罚大段照抄 Persona/示例等明显奖励投机。

训练前人工检查至少 5 组候选和分数。训练后保存实际配置、adapter、奖励曲线、回答样本、长度、
复读和奖励投机观察。GRPO 必须通过技术门槛以及与 SFT 相同的三层 Dev 评估，才进入统一评测；
失败时保留结果，不根据 Eval 反复调整奖励。

## 5. 统一评测

使用冻结的 20 条 Eval，在相同后端、模型 revision、生成参数和一个固定主 seed 下生成 Base、
SFT、GRPO 各 20 条输出。Eval 只用于最终比较。

逐层报告：

1. **生成稳定性**：非空、正常结束、截断、乱码、严重复读。
2. **角色一致性**：核心身份、性格、关系、边界、视角和整体风格。
3. **对话质量**：相关性、自然度、连贯性、内容价值和可继续对话性。

对 10 条 Eval 做匿名比较并保留具体样本。格式、自称、emoji、创造性细节、模板化和长度只作为
解释性诊断。结论必须区分 SFT 相对 Base、GRPO 相对 SFT 的变化，并注明小样本、单角色、单次
训练和 Judge 偏差的限制。

## 6. 执行清单

### Milestone 1：数据与 Base

- [ ] 冻结 v2 Persona、风格样例、system prompt 和输入哈希。
- [ ] 生成并验证隔离的 SFT/GRPO/Dev/Eval。
- [ ] 完成 Pilot、50 条 Teacher-corrected SFT 和人工抽查。
- [ ] 用固定 seed 生成 10 条可读的 Base Dev，并记录推理条件。

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
