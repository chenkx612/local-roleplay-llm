# 实践运行记录

本文记录每次实践的实际输入、配置、产物、观察和决策，是最终复盘报告的事实来源。
预定流程与验收标准见 `PLAN.md`，问题处理详情见 `ISSUES.md`。后续阶段应在运行完成后更新本文，
不要把运行日志和结果重新堆回计划。

# Run：morgana-v1

## 基本信息

- 角色：摩尔加纳
- 阶段一完成日期：2026-08-07
- 状态：阶段一已完成；阶段二首次运行无效，等待重训；阶段三、四未开始
- 阶段一产物目录：`data/runs/morgana-v1/`
- 阶段二产物目录：`output/morgana-v1/stage2-sft/20260807T170226Z-ad78fb8f/`
- 历史产物：`data/archive/pre-runs-legacy/`，不参与本轮训练或评测

模型与框架决策：

- Prompt、Teacher 纠错：`deepseek-v4-flash`
- Student/Base 本地推理：`mlx-community/Qwen3.5-2B-4bit`
- Student revision：`674aaa7240b91e8012fcad5d791b7dfe5ba90207`
- 后续训练基座：`Qwen/Qwen3.5-2B`，使用 ms-swift/BNB 4-bit QLoRA
- 不将 MLX 转换权重直接作为训练 checkpoint；本地推理与训练使用同源模型的不同加载方式

## 阶段一：输入与数据

冻结输入：

- `inputs/persona.json`
- `inputs/style_examples.jsonl`：10 条
- `system_prompt.txt`
- `input_manifest.json`：输入哈希、数据规模、随机种子和 Prompt 生成配置

Prompt 生成实际配置：

```yaml
model: deepseek-v4-flash
thinking.type: enabled
reasoning_effort: high
temperature: null
top_p: null
max_tokens: 8192
split_seed: 20260806
```

生成策略为五类场景各请求 25 条候选，过滤后保留 20 条；不足时以至少 5 条一批定向补齐。
最终形成 100 条共享候选池，再按场景和固定种子切分：

| Split | 数量 | 每类场景数量 | 文件 |
|---|---:|---:|---|
| SFT | 50 | 10 | `sft_train_prompts.jsonl` |
| GRPO | 20 | 4 | `rl_train.jsonl` |
| Dev | 10 | 2 | `dev.jsonl` |
| Eval | 20 | 4 | `eval.jsonl` |

检查结果：100 条规范化 Prompt 全局唯一，未发现精确重复泄漏；记录结构、场景分布、说话视角
和目标元数据均通过自动检查。

## 阶段一：Pilot 与 SFT 数据

Teacher-corrected SFT 链路：

```text
Persona + SFT Prompt → Student baseline
Persona + style examples + Prompt + baseline → Teacher 最小充分纠错
```

Pilot：

- 5 条，五类场景各 1 条，与 Dev/Eval 隔离。
- 冻结 Student baseline 后使用 rubric v5 重跑 Teacher。
- 5/5 通过自动检查和人工语义复核。
- 证据：`pilot/pilot_report.json`、`pilot/pilot_review.md` 及同目录三份 SFT 产物。

正式 SFT：

- 50 条 Student baseline、Teacher edits 和训练标签逐条对齐。
- Student baseline：44 条以 `stop` 结束；6 条达到 512 output token 上限，作为 bad case 保留并
  交由 Teacher 改写。
- Teacher 决策：48 条 `rewrite`，2 条 `light_rewrite`。
- Teacher 使用 `deepseek-v4-flash`、thinking、`reasoning_effort=high`、`max_tokens=4096`、
  rubric v5，不设置 `temperature` 或 `top_p`。
- 最终标签全部非空、符合格式契约，Teacher final checks 全部通过。
- 人工分层抽检 10 条；其中 1 条对共同经历确认过于武断，已修正。

产物：

- `sft_baseline_outputs.jsonl`
- `sft_teacher_edits.jsonl`
- `sft_train.jsonl`
- `sft_generation_meta.json`

## 阶段一：Base

Dev 10 条和 Eval 20 条均使用冻结 Persona 和同一 Base 模型。实际配置以
`base_generation_meta.json` 为准：

```yaml
model: mlx-community/Qwen3.5-2B-4bit
revision: 674aaa7240b91e8012fcad5d791b7dfe5ba90207
max_tokens: 256
temperature: 0.6
top_p: 0.8
top_k: 20
presence_penalty: 0.4
presence_context_size: 128
repetition_penalty: 1.45
repetition_context_size: 128
enable_thinking: false
max_attempts: 3
```

默认重复控制曾导致部分 Prompt 复读并达到长度上限，因此最终运行使用了更强的重复控制。该差异
已记录，后续 SFT/GRPO 统一评测必须以这里的实际参数为可比基准，或明确说明变更理由。

结果：

- `base_dev_outputs.jsonl`：10/10 非空、`finish_reason=stop`，全部首次成功。
- `base_eval_outputs.jsonl`：20/20 非空、`finish_reason=stop`，全部首次成功。
- 两份输出与冻结输入逐条对齐，输入输出哈希已保存。

## 阶段一：问题、决策与验收

阶段内发现并处理了四类关键问题：数据 split 泄漏、数据生成不足仍写盘、Base 截断/退化、推理
异常被吞掉。最终实现全局去重、数量不足失败、有限重试、结束原因校验和整批原子写入。完整问题
经过、代码位置和修复方式见 `ISSUES.md`。

2026-08-07 收口检查：

- Persona、风格样例和 system prompt 通过加载与结构检查；保存的 system prompt 仅多一个末尾
  换行，正文与运行时渲染一致。
- 四个 split 数量正确、场景均衡，100 条 Prompt 全局唯一。
- 50 条 SFT 的四类关联产物对齐，训练数据为 ms-swift 可读取的 `messages` 结构。
- Pilot 未进入 Dev/Eval；Dev/Eval Base 共 30 条，均通过非空、结束原因和复读检查。
- `python -m unittest discover -s tests -v`：71 项测试全部通过。
- `ISSUES.md` 当前无阶段一待解决问题，可以进入阶段二。

## 阶段二：LoRA SFT

**状态：未完成。** 2026-08-08 本地复核确认首次运行没有产生有效 LoRA 参数更新；以下保留
首次运行事实和诊断，不能作为阶段二完成证据或 GRPO 起点。

运行与环境：

- Run ID：`20260807T170226Z-ad78fb8f`；仓库 commit：
  `fce07400cc1443bb19eefb059fcea84cec484d9b`。
- UTC 创建时间为 `2026-08-07T17:02:27Z`，训练及 Dev 归档于约 23 分钟后完成。
- Colab 2026.04、Tesla T4 14.56 GiB、Python 3.12.13、PyTorch 2.10.0+cu128、
  ms-swift 4.4.1、Transformers 5.12.1、PEFT 0.19.1、bitsandbytes 0.49.2。
- 训练依赖闭包检查无问题；全局 `pip check` 仅报告 Colab 自带 IPython 缺少可选的 `jedi`，
  未影响本次训练或推理。
- 正式命令等价于：

```bash
swift sft configs/morgana_v1_sft_t4.yaml \
  --output_dir /content/roleplay-stage2-runs/20260807T170226Z-ad78fb8f/full \
  --add_version false
```

实际配置：

```yaml
model: Qwen/Qwen3.5-2B
model_revision: 965dcc54bc9c0591873df0e9869c056a54d323d1
tuner_type: lora
target_modules: all-linear
torch_dtype: float16
quant_method: bnb
quant_bits: 4
bnb_4bit_compute_dtype: float16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
max_length: 1024
loss_scale: last_round+ignore_empty_think
add_non_thinking_prefix: true
enable_thinking: false
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 5e-5
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
seed: 20260807
data_seed: 20260807
```

相对当时 `PLAN.md` 的主要偏差是 T4 运行实际采用 `float16`，而非计划中的 `bf16`；参数名使用
ms-swift 的 `tuner_type`/`torch_dtype`，语义仍为 4-bit BNB QLoRA。另显式启用 NF4、double
quant、仅最后一轮 assistant loss、gradient checkpointing 和 non-thinking prefix。训练前没有
对“optimizer step 是否真的更新 LoRA-B”设置阻断检查，这是首次 notebook 的关键验收缺口。

冻结输入校验：SFT 50 条、Dev 10 条；SFT、Dev、Base Dev 哈希分别为
`277323c097305ebbee7bfa93cb27c34247800f2dec835f6d052ede7c2b178a7a`、
`cbce0b38bb6f8b8cbef0bc45fd52b5a8212a66445569dfe0a8c7e8e88f63ddc6`、
`840ce46d346cd01c9209174a8be1325415bd61c9da6567d21b85227f23dfd832`；最长训练样本
816 tokens，小于 `max_length=1024`。

训练结果：

- 1 epoch、4 个 optimizer step，训练 94.929 秒，峰值记录显存 3.09 GiB；没有 OOM 或中断。
- step loss 为 `2.6794 → 2.6063 → 2.6522 → 2.4434`，平均 train loss `2.5953`；
  token accuracy 从 `0.4608` 波动至 `0.4759`。
- 16.8192M 名义可训练参数，占所加载模型参数的 1.0895%。最终 checkpoint 位于
  `full/checkpoint-4/`，权重 SHA-256 为
  `e09c85f3342dd837d0d794cb6ce5d7ef9fe036eab020958fc4ed11dbba18d440`。
- 冒烟和正式 checkpoint 都能被 PEFT 加载并生成非空回答，但“可加载”只证明序列化链路，
  不证明参数发生更新。
- 四个 step 的 `grad_norm` 均为 `NaN`。进一步直接读取 `adapter_model.safetensors`：372 个
  adapter 张量全部有限，其中 186 个 LoRA-A 张量保留随机初始化值，186 个 LoRA-B 张量全部
  精确为零，LoRA-B 非零元素总数为 0。由于 LoRA 的初始 B 即为零，这证明四个 optimizer step
  都没有形成有效 adapter 更新；结合 FP16 AMP 和非有限梯度，最可能是梯度溢出后 step 被跳过。
- 因此 loss 曲线只是 forward/logging 证据，不能证明本次训练有效；该 checkpoint 等价于无效
  adapter，不得作为 GRPO 起点。

Dev 可比性：两次使用相同 Persona、system prompt、10 条冻结 Dev、`max_tokens=256`、
`temperature=0.6`、`top_p=0.8`、`top_k=20`、`repetition_penalty=1.45` 和
`enable_thinking=false`。仍存在以下已记录差异，因此这里只做初步行为观察，不把全部变化因果
归于 LoRA：

- Base 是 MLX 转换模型 revision `674aaa...` 加 OpenAI-compatible server；SFT 是上游 HF
  revision `965dcc...` 加 ms-swift TransformersEngine。
- Base 使用 `presence_penalty=0.4` 和 128-token presence/repetition context；
  TransformersEngine 没有等价 presence penalty，repetition penalty 的上下文语义也不完全相同。
- SFT 输出统一带 `<think>\n\n</think>` non-thinking prefix；Base 输出没有。虽然没有泄漏隐藏
  思考内容，该额外标签本身违反输出格式。

首次运行 Dev 诊断（不能解释为 SFT 效果）：

| 观察 | Base | SFT |
|---|---:|---:|
| 非空 | 10/10 | 10/10 |
| `finish_reason=stop` | 10/10 | 5/10 |
| 长度截断 | 0/10 | 5/10 |
| 严格全角动作括号开头 | 0/10 | 0/10 |
| `<think>` 额外标签 | 0/10 | 10/10 |
| 机械复读嫌疑 | 0/10 | 0/10 |

定性观察：

- **角色一致性：** 两套后端输出都有事实幻觉。首次 Transformers 输出在 `dev_0005` 编造“来自未来都市”，在
  `dev_0004` 把普通宠物描述成供人吃或逗弄的东西，并多次以“本喵/本大爷”取代标志性“吾辈”；
  但因 adapter 未更新，不能据此判断 SFT 改变了角色行为。
- **格式一致性：** MLX Base 因半角括号或缺少动作括号而 0/10 严格通过；首次 Transformers
  输出另有 10/10 `<think>` 标签、过量 emoji、多层括号和长旁白。这主要暴露后端/模板差异，
  不是有效 SFT 的前后对照。
- **对话质量：** 首次 Transformers 输出在 `dev_0003`、`dev_0006`、`dev_0009` 等回答中明显冗长、转题或句中
  截断；`dev_0007`、`dev_0010` 对负面情绪的回应偏攻击或错把自己当作被道歉者。Base 也有
  幻觉和误解，但 MLX Base 10/10 完整结束且整体更短。不同后端使该差异不能归因于 LoRA。
- **其他退化：** 没有空回答、统一拒答、机械复读或过短回答；出现明显模板化的“哼/本大爷”、
  冗长失控、格式投机和虚构共同情境。

阶段产物包括 `training_config.yaml`、`full/console.log`、`full/logging.jsonl`、
`full/checkpoint-4/`、`dev_outputs.jsonl`、`dev_generation_meta.json`、环境清单和阶段摘要。

诊断与重训决定：

- 本次运行判定为**技术无效**而非“有效 SFT 的负向效果”，撤销阶段二完成和进入 GRPO 的决定。
- 重训保持数据、模型 revision、LoRA 结构、学习率和 1 epoch 不变，只把模型计算、BNB compute
  和 LoRA 参数统一改为 `float32`，先解决 FP16 数值更新失败；这是基于故障机制的最小修复，
  不使用 Dev 搜索超参数。
- 新 notebook 在 1-step 冒烟和正式训练后都要求：loss 有限、`grad_norm` 有限且为正、所有
  LoRA-B 张量含非零元素且所有 adapter 张量有限；任一失败立即停止。
- 推理同时生成同一 Transformers 后端、同一 revision 的无 adapter Base 和 SFT，移除协议层的
  空 `<think></think>` wrapper 后再检查。SFT 必须 10/10 非空、10/10 `stop`、无机械复读、
  严格格式至少 8/10 且不差于同后端 Base，才成为人工复核候选。
- 机械门槛通过后仍需人工检查角色一致性和对话质量；只有人工确认有实际改善且无严重退化，
  才能再次把阶段二标为完成并进入 GRPO。

## 阶段三：GRPO

**状态：未开始。** 完成后记录：

- SFT 起点、奖励实现版本、实际训练配置和命令。
- 至少 5 组训练前奖励核查样本及由此做出的修改。
- 奖励、回答长度、复读率曲线，LoRA 路径和加载验证。
- 奖励投机、退化行为、失败与继续评测的决定。

## 阶段四：统一评测

**状态：未开始。** 完成后记录：

- Base/SFT/GRPO checkpoint、生成参数、Judge 配置和 Eval 哈希。
- 三个主指标、宏平均、诊断指标及逐模型样本观察。
- 10 条人工检查或匿名比较结果，以及小型能力保持集结果。
- 自动、规则和人工证据冲突时的判断与结论边界。

## 最终复盘素材清单

撰写最终复盘前确认以下证据齐全：

- [x] 阶段一输入、数据、SFT 数据构建、Base 配置与验收证据。
- [x] 阶段一问题与关键工程决策。
- [ ] 有效 SFT 的实际配置、日志、产物、Dev 结果和进入 GRPO 决策；首次运行已判无效。
- [ ] GRPO 奖励设计验证、训练日志、产物和异常行为。
- [ ] Base/SFT/GRPO 统一评测与能力保持观察。
- [ ] 计划偏差、失败点、结论限制和下一轮唯一优先改进方向。
