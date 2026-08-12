# Stage 4：连续规则型 GRPO 奖励规约 v2

Stage 4 奖励完全在本地确定性计算，不调用在线 Judge。v2 的目标是让同组候选即使普遍较差，仍能
按“离明确要求有多远”稳定排序；规则分不代表整体角色质量。

## Prompt 约束

20条训练 Prompt 使用同一奖励函数，只携带以下目标范围：

```json
{
  "min_actions": 1,
  "max_actions": 1,
  "min_sentences": 1,
  "max_sentences": 2,
  "min_chars": 20,
  "max_chars": 70,
  "min_signatures": 1,
  "max_signatures": 1
}
```

未明确要求动作时，动作范围为 `null/null`；明确禁止动作时为 `0/0`。没有要求标志性自称的
Prompt 使用 `0/1`，避免强迫每个回答机械重复“吾辈”。

## 连续映射

实际值到允许闭区间的距离为：

```text
distance(x, min, max) = max(min - x, 0) + max(x - max, 0)
score(v) = (1 - v) / (1 + v)
```

`v=0/1/3/10` 分别映射为 `1/0/-0.5/-0.818`。因此禁止动作时，0段优于1段，1段优于3段，
不会像旧奖励一样把所有违规候选压到同一个 `-1`。

## 三个有效回答分量

```text
InstructionViolation =
    1.0 × ActionDistance
  + 0.7 × SentenceDistance
  + 0.3 × CharacterDistance / 30

PersonaViolation =
    0.8 × SignatureDistance
  + 0.4 × WrongWoCount
  + 0.8 × WrongAliasCount

StyleViolation =
    0.4 × RelativeLengthDistanceFromTargetMidpoint
  + 0.5 × RepetitionRatio
  + 0.3 × ExcessActionLength / 30
  + 0.3 × ExcessActionRatio / 0.5
  + 0.2 × ActionGapShortfallRatio
  + 0.3 × FormatIssueCount
  + 0.3 × InvalidActionContentCount
  + 0.2 × NestedActionCount
```

三个违规量分别通过 `score(v)` 映射为 `InstructionScore`、`PersonaScore` 和 `StyleScore`，最终：

```text
R_valid = 5 × InstructionScore + 2 × PersonaScore + StyleScore
```

有效回答范围为 `(-8, 8]`。指令符合度权重最高；长度、格式或角色项不能轻易抵消明显的指令
违背。“吾辈”和“吾輩”视为同一标志性自称；引号外的独立“我”及“本大爷”“本喵”“本猫”
“俺”按实际出现次数累计违规，不再一次触发后饱和。

## 硬失败

空回复、截断、乱码、严重复读或括号未闭合进入独立低分区：

```text
Recoverability = mean(
    Nonempty,
    Complete,
    Readable,
    Nonrepeated,
    ParenthesesBalanced
)

R_invalid = -12 + 2 × Recoverability
```

硬失败范围为 `[-12, -10]`，一定低于有效回答，但多个硬失败仍能按可恢复程度排序。

每个候选在 `reward_samples.jsonl` 中保存约束、三个违规量及分数、动作分析、硬失败原因、可恢复度
和总分。完整冻结定义同时写入每个 run 的 `reward_spec.json`。

## Dev 门槛

Dev 使用统一的宽松目标：0～2段动作、1～4句、30～90字、0～1次标志性自称。GRPO 必须完成
10条对齐 Dev，规则均分比 SFT 至少高0.3且至少胜出6条；同时无硬失败，退化和错误自称数量不得
高于 SFT。人工复核只阻断 GRPO 新增的严重问题。
