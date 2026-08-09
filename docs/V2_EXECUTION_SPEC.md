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
- QLoRA：ms-swift + bitsandbytes 4-bit，AutoDL 单张 RTX 3090 24GB。实例基础镜像为
  PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8，并直接作为正式训练环境。
  SFT 和 DPO 阶段不安装 flash-linear-attention 和 causal-conv1d，Qwen3.5 使用 Transformers
  的 PyTorch fallback；GRPO 如仍需要变长线性注意力，再单独冻结依赖。
- 第二轮 DPO 由 Codex 通过离线产物直接完成 Judge/Teacher 裁决，不调用外部 API，也不增加
  人工数据复核；候选、裁决和淘汰原因必须完整留档。
- Student、SFT、DPO、GRPO 生成关闭 thinking。
- Base、SFT、DPO、GRPO 比较固定使用同一模型 revision、聊天模板和生成参数；学习项目允许使用
  同一可用推理链路，不要求为此额外搭建 Transformers 专用环境。

v1 的结论保留在 `V1_RETROSPECTIVE.md` 和 `ISSUES.md`，raw 数据和模型产物已清理。
角色源设定可以复用，但必须重新保存 v2 输入快照和哈希；v1 的 system prompt、
Teacher 标签、Base 输出和 SFT adapter 不进入 v2 训练或正式评测。

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
| SFT（原始 split） | 50 | 10 | Teacher-corrected 监督训练 |
| SFT（定向补强） | 12 | 不要求均衡 | 针对 Dev bad case 补强 |
| DPO | 20 | 4 | 主观偏好学习 |
| GRPO | 20 | 规则导向 | 后续明确规则约束；尚未生成 |
| Dev | 10 | 2 | 阶段选择与观察 |
| Eval | 20 | 4 | 最终统一评测 |

原 100 条共享候选池按固定 seed 切出的 SFT、旧 GRPO、Dev 和 Eval 中，旧 GRPO 的 20 条主观
质量 Prompt 原样迁移为 DPO，ID 改为 `dpo_NNNN` 并记录来源哈希。旧 `rl_train.jsonl` 只作为
失败 GRPO 的历史输入。未来规则型 GRPO 另行生成 20 条规则压力 Prompt，并与 SFT、DPO、Dev、
Eval 全局去重。Eval 在最终统一评测前不得用于修改数据、提示词或训练配置。

### 2.3 Teacher-corrected SFT

```text
Persona + SFT Prompt → Student baseline
Persona + style examples + Prompt + baseline → Teacher 最小充分纠错
```

Teacher 依次检查生成稳定性、角色一致性和对话质量。合格回答原样保留，不合格回答只做必要
改写。动作括号、emoji、固定开头或精确复述 Persona 均不是通过条件。

先运行五类场景各一条的 Pilot 并人工复核，再生成 50 条 Teacher-corrected 正式标签。根据第三次
SFT 的 Dev bad case，另加 12 条人工复核的定向样本，覆盖主客体识别、事件与情绪识别、身份边界、
问题意图与回答相关性，最终共 62 条。正式产物必须结构正确、逐条对齐、回答非空，且没有乱码和
明显复读。在人工复核结论、关键统计和哈希写入 `RUNLOG.md` 与复核文档后，
Student baseline 和 Teacher audit 全量文件可作为 raw 中间产物删除；保留定向补强源文件、
最终训练标签，并至少保留一组完整的 Student baseline、Teacher 判断与最终回答对照，
作为该阶段的最小可检查产物。

### 2.4 Base Dev

为 10 条 Dev 用一个固定 seed 生成 10 条 Base 输出。首次推理时记录实际模型/revision、推理
链路、seed、聊天模板和生成参数；后续 SFT、DPO 与 GRPO 沿用这些条件即可。无需为基线额外搭建
Transformers 环境，也不要求多 seed 重复采样。

进入下一次 SFT 前必须具备：冻结输入、四个有效 split、通过人工复核的 Pilot、50 条有效
Teacher-corrected 标签、12 条有效定向标签和 10 条有效 Base Dev。

## 3. QLoRA SFT

只运行一组主配置：

```yaml
model: Qwen/Qwen3.5-2B
tuner_type: lora
target_modules: all-linear
torch_dtype: float32
fp16: false
bf16: false
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

该配置从第二次 SFT 起保持不变。第一次 SFT 的配置和结果已写入 `RUNLOG.md`；
其 raw 产物在阶段 2 收尾后删除。第二、三次使用 50 条数据，
把有效 batch size 从 16 降到 4，使 3 epochs 下的预期 optimizer step 从 12 增加到 39。下一次
训练只把数据增加到 62 条，对应 48 个 optimizer steps；物理 batch size 仍为 2、梯度累积仍为 2，
不改学习率、epoch、LoRA 或推理参数。训练前还必须确认 62 条 assistant 标签全部包含“吾辈”，
且不包含“本大爷”或“本喵”。

第二次 SFT 的 AutoDL 首次运行虽然完成了 39/39 个记录步，但 ms-swift 将未显式指定的
混合精度解析为 `fp16=true`，开头连续 6 个 `grad_norm` 为 `NaN`，因此技术门槛失败。
修复后显式关闭 FP16 和 BF16，保持 model、BNB compute 与 LoRA 均为 FP32；训练结束后还需
读取 ms-swift 的 `args.json`，确认实际参数没有重新启用混合精度。失败 checkpoint 只作证据，
不得续训或进入 DPO。

只对 assistant 回复计算 loss，不使用 Dev 搜索 epoch、学习率或 LoRA 参数。技术失败可以修复明确
的实现问题后重跑；行为失败不得在同一轮静默调参。

### 3.1 技术门槛

- 3 epochs 正常结束，optimizer step 数与预期一致。
- 实际训练参数为 `fp16=false`、`bf16=false`，三个 dtype 均为 `float32`。
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

仅当技术门槛、生成稳定性门槛和人工门槛全部通过时进入 DPO。

## 4. DPO

从通过 SFT 门槛的 adapter 继续训练，学习可以稳定比较、但难以写成客观规则的主观偏好。数据
准备以 [`STAGE3_DPO_PLAN.md`](STAGE3_DPO_PLAN.md) 为准：

- 使用 40 条独立 DPO Prompt，每条由 SFT adapter 生成 2 个固定 seed 候选，不补采样。
- 本地稳定性过滤后，由 Codex 离线匿名裁决，不调用外部 Judge API，也不增加人工复核。
- 两条都不足时，Codex 只对较好候选做最小修改；平局和实质权衡直接排除。
- 至少冻结 30 对，Teacher 修改参与比例不超过三分之一。
- 训练文件使用 ms-swift 标准 `messages + rejected_response` 格式。

DPO 训练只运行一组主配置，从 SFT policy 建立训练 policy，并以同一冻结 SFT 状态作为 reference。
冻结配置为 FP32 QLoRA、`beta=0.1`、sigmoid loss、`learning_rate=1e-6`、3 epochs、物理
batch size 1、梯度累积 4，共 24 个 optimizer steps。训练后验证实际 FP32 参数、有限且为正的
loss/grad norm、完整 step 数、adapter 非零更新和可重新加载生成。

随后在相同推理条件下与 SFT 对齐生成 10 条 Dev。自动稳定性门槛沿用阶段二；匿名人工复核要求
DPO 至少胜 6 对、明显落后不超过 2 对、无严重问题，并且生成稳定性、角色一致性和对话质量三项
均分不低于 SFT。全部通过后状态为 `ready_for_grpo`，否则记录为 `dpo_failed`。

## 5. 规则型 GRPO

旧版从 SFT 直接训练、使用在线主观 Judge 奖励的 GRPO 已执行失败并封存，不进入当前链路。未来
GRPO 从通过验收的 DPO adapter 开始，只学习能够稳定、自动验证的明确规则，并使用独立、规则
导向且与其他 split 去重的 Prompt。主观角色感、自然度和情绪价值不再作为在线奖励。

## 6. 统一评测

使用冻结的 20 条 Eval，在相同后端、模型 revision、生成参数和一个固定主 seed 下生成 Base、
SFT、DPO、GRPO 各 20 条输出。Eval 只用于最终比较。

逐层报告：

1. **生成稳定性**：非空、正常结束、截断、乱码、严重复读。
2. **角色一致性**：核心身份、性格、关系、边界、视角和整体风格。
3. **对话质量**：相关性、自然度、连贯性、内容价值和可继续对话性。

对 10 条 Eval 做匿名比较并保留具体样本。格式、自称、emoji、创造性细节、模板化和长度只作为
解释性诊断。结论必须区分 SFT 相对 Base、DPO 相对 SFT、GRPO 相对 DPO 的变化，并注明小样本、
单角色、单次训练和偏好标注偏差的限制。

## 7. 执行清单

### Milestone 1：数据与 Base

- [ ] 冻结 v2 Persona、风格样例、system prompt 和输入哈希。
- [ ] 生成并验证隔离的 SFT/DPO/Dev/Eval；未来 GRPO 另行生成。
- [ ] 完成 Pilot、50 条 Teacher-corrected SFT、12 条定向补强和人工抽查。
- [ ] 用固定 seed 生成 10 条可读的 Base Dev，并记录推理条件。

### Milestone 2：SFT

- [ ] 完成一次技术有效的 FP32 QLoRA SFT。
- [ ] 通过生成稳定性门槛和三目标匿名人工复核。
- [ ] 记录实际配置、产物、偏差和进入 DPO 的决定。

### Milestone 3：DPO

- [ ] 生成、校准并人工复核至少 16 对 DPO 偏好数据。
- [ ] 完成一次有效 DPO，并完成 DPO 相对 SFT 的三层 Dev 评估。

### Milestone 4：规则型 GRPO

- [ ] 生成独立规则 Prompt，冻结规则奖励并完成一次有效 GRPO。
- [ ] 完成 GRPO 相对 DPO 的规则与质量回归评估。

### Milestone 5：统一评测与复盘

- [ ] 生成对齐的 Base/SFT/DPO/GRPO Eval 输出。
- [ ] 完成自动统计、匿名人工比较和诊断观察。
- [ ] 写出 v2 复盘，明确结果、失败点和结论边界。
