# 实践运行记录

本文记录每次实践的实际输入、配置、产物、观察和决策，是最终复盘报告的事实来源。
预定流程与验收标准见 `PLAN.md`，问题处理详情见 `ISSUES.md`。后续阶段应在运行完成后更新本文，
不要把运行日志和结果重新堆回计划。

# Run：morgana-v1

## 基本信息

- 角色：摩尔加纳
- 阶段一完成日期：2026-08-07
- 状态：阶段一已完成；阶段二至四未开始
- 主产物目录：`data/runs/morgana-v1/`
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

**状态：未开始。** 完成后记录：

- 环境、依赖版本、硬件、开始/结束时间和实际命令。
- 模型 revision、数据哈希及相对 `PLAN.md` 的全部配置差异。
- loss 曲线、训练耗时、显存/内存问题、checkpoint 与 LoRA 路径。
- LoRA 加载验证、Dev 输出、相对 Base 的样本观察和进入 GRPO 的决定。
- 失败、重试、临时修复及未解决限制。

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
- [ ] SFT 实际配置、日志、产物、Dev 结果和阶段决策。
- [ ] GRPO 奖励设计验证、训练日志、产物和异常行为。
- [ ] Base/SFT/GRPO 统一评测与能力保持观察。
- [ ] 计划偏差、失败点、结论限制和下一轮唯一优先改进方向。
