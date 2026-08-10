# morgana-v2 运行日志

## 2026-08-11：17 条最小编辑 pair DPO 配置

- 从完成的 `20260811-dpo-editability-1` 报告导出 17 条 ms-swift DPO 数据，只包含通过来源
  单缺陷、chosen 8/8/8、相似度和长度比合同的 pair；不混入旧 31 条数据。训练集为
  `dpo_train_editability17.jsonl`，SHA-256 为
  `2836a41969ae250b3ddea692cf441c5c95179723cc5a81bb07c5c0892c39e922`。
- 从同一冻结 SFT adapter 开始，保持 `beta=0.1`、sigmoid loss、LoRA、seed、推理参数和 Dev
  门槛不变；将 `learning_rate` 从 `1e-6` 降到 `5e-7`，`rpo_alpha` 从 `1.0` 降到 `0.3`，
  梯度累积从 4 降到 2。1 epoch 共 9 个 optimizer steps。
- 目标不是证明 17 条数据覆盖充分，而是受控验证：高质量、低分布偏移 pair 能否让 DPO 从
  主观退化转为微弱但稳定的改善。最终评测门槛保持不变。

## 2026-08-11：SFT chosen 最小编辑判定实验

- 冻结 `20260810-dpo-data-3` 的 40 条 Prompt、146 条稳定候选并匿名打乱；Codex 对每条
  Prompt 选择最接近优秀回答的候选，判断能否只修改一个主观偏好维度得到 8/8/8 chosen。
- 合格来源必须为稳定性至少 8 分、目标主观维度恰为 7 分、另一主观维度至少 8 分；编辑后
  字符相似度至少 `0.65`、长度比为 `0.80～1.20`。事实错误、视角错误、明确指令违背和多缺陷
  回答不允许通过编辑伪装成合格偏好对。
- 结果：17/40 可最小编辑，23/40 不可编辑，未达到可用基础门槛 20 条，结论为
  `insufficient_sft_foundation`。成功样本平均相似度 `0.772`，平均长度比 `0.965`；最小相似度
  `0.657`，所有 17 对均通过冻结合同。
- 成功目标分布：角色自然度 6、表达自然度 6、情绪承接 3、对话延续 2。不可编辑样本中，
  19 条至少包含“没有合格来源”，9 条包含“多缺陷”；另有 3 条已经达到 8/8/8，无法制造
  清晰且诚实的主观偏好差。
- 判定：当前 SFT 输出分布附近确实存在少量可用于 DPO 的高质量局部偏好对，但覆盖率不足；
  主要瓶颈是多数候选没有先达到“仅剩一个主观弱点”的 rejected 门槛，而不是 chosen 的
  编辑幅度不够小。
- 命令：`roleplay-dpo-editability prepare` 生成冻结评审包；填写
  `editability_results.json` 后，用 `roleplay-dpo-editability finalize --experiment-dir <dir>`
  校验并生成 `editability_report.json` 与 `editability_pairs.jsonl`；通过
  `roleplay-dpo-editability export-training` 导出正式训练集。

## 2026-08-11：RPO 1.0 DPO run `20260811-0024`

- 使用同一 31 对 run3 数据和训练配置，仅将 `rpo_alpha` 从 `0.3` 提高到 `1.0`。技术训练及
  自动稳定性门通过，adapter 最大更新为 `4.498288035392761e-6`。
- 平均 chosen/rejected reward 从上一 run 的 `-0.02251/-0.02547` 改善到
  `-0.01314/-0.01495`，末步 chosen reward 从 `-0.02529` 改善到 `-0.00903`，但 chosen
  绝对似然仍未转为上升。
- SFT、DPO Dev 输出及匿名复核包与 `20260811-0008` 逐字节相同，因此人工结果仍为 DPO 胜
  `2/10`、明显落后 `1/10`，三项均分仍从 `7.7/5.3/5.5` 降到 `7.5/5.2/5.2`。run 状态为
  `dpo_failed`；继续单纯提高 RPO 权重的边际收益不足。

## 2026-08-11：DPO 将 RPO 权重提高至 1.0

- 高质量数据 run `20260811-0008` 已通过自动稳定性门，但匿名复核仅胜 `2/10`；训练期间平均
  chosen/rejected reward 为 `-0.02251/-0.02547`，说明正 margin 仍主要来自压低 rejected，
  chosen 绝对似然也在下降。
- 下一次受控实验只将 `rpo_alpha` 从 `0.3` 提高到 ms-swift/RPO 推荐值 `1.0`；31 对 run3
  数据、1 epoch、8 steps、学习率、beta、seed、采样参数和评测口径全部保持不变。

## 2026-08-11：高质量偏好数据 DPO run `20260811-0008`

- 使用 31 对 run3 数据、`rpo_alpha=0.3`、1 epoch、8 steps，从同一冻结 SFT adapter 开始；
  技术训练有效，adapter 最大更新为 `4.501183866523206e-6`。
- 自动门首次通过：DPO 退化数由 SFT 的 `1/10` 降为 `0/10`，异常符号由 1 降为 0，括号未闭合
  保持 0，错误自称保持 `1/10`。说明绝对质量门和偏好分差约束有效消除了此前的稳定性退化。
- 匿名人工复核仍失败：前 4 对逐字相同，DPO 胜 `2/10`、明显落后 `1/10`；生成稳定性、角色
  一致性、对话质量均分由 SFT 的 `7.7/5.3/5.5` 变为 `7.5/5.2/5.2`。
- 训练末步 chosen/rejected reward 均为负（`-0.02529/-0.04053`），margin 为正 `0.01525`。
  当前主要问题已从数据噪声转为目标优化：模型仍可通过同时降低 chosen 和 rejected 概率、更多
  压低 rejected 来获得偏好 margin；`rpo_alpha=0.3` 尚不足以让 chosen 绝对似然上升。

## 2026-08-10：DPO 偏好数据加入绝对质量门和最小分差

- 为隔离偏好对噪声，训练配置保持 `rpo_alpha=0.3`、1 epoch 和 8 steps；仅重建偏好数据。
- 数据 run `20260810-dpo-data-3` 为 40 条 Prompt 各生成 4 个固定 seed 候选，共 160 条；146 条
  通过机械稳定性检查，39 组进入匿名 Codex 离线裁决，未调用 DeepSeek 或其他外部 Judge API。
- chosen 必须满足稳定性 `>=8`、角色一致性 `>=7`、对话质量 `>=7`、无问题代码；rejected
  稳定性 `>=7`，且角色一致性或对话质量至少落后 2 分。最终保留 25 对直接偏好和 6 对局部
  编辑偏好，排除 8 组共同缺陷/无清晰偏好，另有 1 组在评审前过滤。
- 冻结训练集 `dpo_train_run3.jsonl` 共 31 对，Teacher 占 `6/31`，SHA-256 为
  `55d6b0efd31dce48fca2009323dd2b7a9f841e74b206187b081a9c6e8e3c47c3`；完整评分、问题代码、
  修改说明和淘汰原因保存在 `dpo_train_audit_run3.json`。

## 2026-08-10：DPO 将 RPO 权重提高至 0.3

- `rpo_alpha=0.1` 的 run `20260810-2315` 已确认 RPO 正常生效，平均 chosen reward 由无 RPO
  run 的 `-0.001468` 改善为 `0.000383`；但固定 seed 下的 10 条 DPO Dev 输出与无 RPO run
  逐字节相同，自动评测仍失败。
- 下一次受控实验仅将 `rpo_alpha` 从 `0.1` 提高到 `0.3`；同一份 30 对数据、1 epoch、8 steps、
  学习率、beta、采样参数和评测全部保持不变。

## 2026-08-10：DPO 加入小权重 chosen NLL/RPO

- 单 epoch 诊断 run `20260810-2304` 技术训练正常，adapter 最大更新降至
  `4.49642539024353e-6`，但自动评测仍失败：总退化由 SFT 的 `1/10` 升至 `3/10`，括号未闭合
  由 0 升至 1，异常符号由 1 升至 2，错误自称由 1 升至 3。减少 epoch 只能缓解退化，不能解决
  偏好目标对 chosen 绝对质量保护不足的问题。
- 下一次受控实验保持同一份 30 对数据、1 epoch、8 steps、学习率、beta、采样参数和评测不变，
  仅加入 `rpo_alpha=0.1`，用小权重 chosen NLL 约束模型提高 chosen 似然，而不是只通过压低
  rejected 获得偏好 margin。

## 2026-08-10：DPO 单 epoch 诊断配置

- 第二次 DPO run `20260810-2213` 相对 SFT 出现明显退化；为优先验证是否由后续 epoch 放大，
  下一次 DPO 继续从同一冻结 SFT adapter 开始并使用同一份 30 对偏好数据，仅将训练从
  3 epochs 改为 1 epoch，预期 optimizer steps 从 24 降为 8。
- 自动稳定性门槛增加总退化数、括号未闭合数、异常符号数和错误自称数不得高于 SFT 的检查；
  训练参数、生成参数、Dev、固定 seed 和人工门槛均保持不变。

## 2026-08-10：第二次 DPO run 扩量并改用 Codex 离线裁决

- 第一次 DPO run `20260809-2350` 技术训练与自动稳定性检查完成，但人工复核仅胜 5/10，
  有 2 次明显落后，并在 `dev_0009` 出现把当前用户与“莲”拆开的严重视角错误；生成稳定性、
  角色一致性、对话质量均分由 SFT 的 `8.0/5.2/5.7` 降至 `7.4/5.1/5.2`，不进入后续阶段。
- 第二次 DPO run 不回退追加 SFT，也不沿用失败 DPO adapter；从同一份已验收 SFT adapter
  重新开始，只改变偏好数据数量和质量。新建 `dpo_prompts_run2.jsonl` 共 40 条，五类场景各 8 条；每条仅
  生成 2 个固定 seed 候选，不补采样。
- 数据 run `20260810-dpo-data-2` 共生成 80 条候选，73 条通过本地稳定性检查，34 组进入匿名
  Codex 裁决；6 组因候选未全部稳定而提前过滤。Codex 确认 21 对直接偏好、完成 9 对最小修改，
  并排除 4 对无清晰偏好的样本。
- `finalize` 为第二次 DPO run 冻结 30 对 ms-swift 数据，Teacher 修改占 `9/30`，未超过三分之一。
  `dpo_train_run2.jsonl` SHA-256 为
  `89dd2030fab814454943b312fd65e619f0c807d93076aee7f6878c72fad8bb82`；完整采用、修改和淘汰
  理由保存在 `dpo_train_audit_run2.json`。
- DPO 超参数保持 `beta=0.1`、`learning_rate=1e-6`、3 epochs、batch size 1、梯度累积 4；
  数据量增加后计划 optimizer steps 从 15 增至 24，训练仍从原 SFT adapter 开始。

## 2026-08-09：旧版主观 Judge GRPO 失败并迁移到 DPO 流程

- 旧 Stage 3 GRPO run `20260809-2031` 技术训练和自动稳定性检查完成，但匿名人工复核失败：
  GRPO 只胜 2/10、明显落后 4/10；生成稳定性、角色一致性、对话质量均分由 SFT 的
  `8.0/5.2/5.7` 降至 `6.8/4.3/4.8`。run 状态为 `grpo_failed`，adapter 不进入后续训练。
- 失败表明在线主观 Judge 奖励没有在这次最小 GRPO 中可靠保护整体对话质量。该 run、奖励规约
  和 20 条原始 `rl_train.jsonl` 仅作为负向历史证据保留。
- 当前流程调整为 `SFT → DPO → 规则型 GRPO`。原 20 条主观质量 Prompt 内容和顺序不变，迁移
  为 `dpo_prompts.jsonl`，ID 从 `grpo_NNNN` 改为 `dpo_NNNN`；未来规则型 GRPO 重新生成独立且
  去重的规则压力 Prompt。
- DPO Judge 与必要时的 Teacher 统一使用 `deepseek-v4-pro`、thinking enabled、
  `reasoning_effort=max`。每条 Prompt 先由冻结 SFT adapter 生成 3 个候选，最终 chosen/rejected
  必须经匿名人工复核，单次模型判断不作为训练真值。

### DPO 偏好数据准备 run `20260809-dpo-data-1`

- 已使用冻结 SFT adapter 和迁移后的 20 条 Prompt 完成真实 `prepare`：首轮生成 60 个候选；
  `dpo_0005` 因稳定候选不足执行唯一一次补采样，新增 3 个候选。最终 63 个候选中 58 个通过
  本地稳定性检查。
- Judge 共处理 20 条 Prompt：19 条为 `clear_preference`，1 条为 `all_inadequate`；Teacher 仅对
  后者选定的原候选完成一次最小修改。最终形成 20 个可复核 A/B 项，0 条 unresolved。
- 匿名复核包 SHA-256 为
  `abb2cb5671068e24f28680b3bd1d009518a5b07ade13ffef9e9346532a1fb7ca`。复核时未查看 Judge
  推荐、seed 或 Teacher 来源；20 项中确认 17 个有效偏好，排除 3 个 tie/实质权衡项。
- `finalize` 已冻结 17 对 ms-swift DPO 数据，其中 Teacher 修改参与 1 对（`1/17`），满足至少
  16 对和不超过三分之一的门槛。`dpo_train.jsonl` SHA-256 为
  `f1db4c30506fa704ac2366ec945a9f0b1302910bf41b667921a2d7f1ce9ae4f9`，run 状态为
  `ready_for_dpo`；来源映射、人工理由和排除记录保存在独立审计文件，不混入训练字段。

## 2026-08-09：历史 raw 产物清理

- v1 实验结论、关键配置、指标、失败原因和结论边界已汇总到
  `V1_RETROSPECTIVE.md` 和 `ISSUES.md`；删除 `data/runs/morgana-v1/`、
  `output/morgana-v1/` 及旧配置文件；第三次的完整主配置已迁入
  `V1_RETROSPECTIVE.md`，不再保留 v1 raw 数据、日志和 adapter。
- 删除 `data/archive/pre-runs-legacy/`，其内容已被 v1/v2 的隔离数据链路取代。
- v2 删除已通过的 Pilot raw，保留一份含前后对照的最小人工复核文档；删除可由最终
  `sft_train.jsonl` 和复核文档代替的 Student baseline 与 Teacher audit。保留冻结输入、
  四个数据 split、最终训练集、Base Dev 基线和生成
  metadata。
- v2 SFT 只保留第二次失败的最小 `run_summary.json` 和当前第四次最终
  run；第一、三次被替代 run 的 raw 日志、输出与 adapter 已删除。第四次 adapter
  是阶段 3 的当前输入，不属于本次历史清理。

## 2026-08-09：阶段 2 SFT 收尾与复盘

### 最终候选与训练有效性

- 阶段 2 最终采用第四次运行 `output/morgana-v2/stage2-sft/4/`，run id
  `20260809-1250`，代码 commit `c9407c6d88ab3cad2d7a8bddac43f4aa3b8dd0ec`。
- 基座为固定 revision 的 `Qwen/Qwen3.5-2B`；训练集 62 条，包含 50 条 Teacher v8 标签和
  12 条第三次 Dev bad case 驱动的定向补充。训练仍使用唯一主配置：4-bit NF4 QLoRA、LoRA
  rank 16、alpha 32、3 epochs、物理 batch 2、梯度累积 2、学习率 `5e-5`、纯 FP32。
- 48/48 optimizer steps 完成，训练耗时 185.4 秒。三个 epoch 的平均 loss 为
  `2.7107 → 2.2956 → 2.1124`；全部梯度有限，186/186 个 LoRA-B 张量产生非零更新，adapter
  重新加载后完成 Base/SFT 对齐 Dev 推理。技术门槛全部通过。
- 正式 adapter SHA-256 为
  `617e6e00535fa356272d32fb16d8fe8d451a9c3cfd2f766f56af02cdf2f9b78d`。第三次及更早
  adapter 的实验结论保留在本日志，raw 权重已清理，不进入阶段 3。

### Dev、人工复核与阶段决策

| 指标 | 同后端 Base | 第四次 SFT |
|---|---:|---:|
| 完整输出 / 正常结束 | 10/10 | 10/10 |
| 截断 | 0 | 0 |
| 退化输出 | 9/10 | 1/10 |
| 乱码 | 1/10 | 0/10 |
| 错误自称 | 10/10 | 1/10 |
| 标志性“吾辈” | 0/10 | 0/10 |

- 生成稳定性门槛通过。相较第三次 SFT，退化输出由 3/10 降至 1/10，未闭合括号由 2 条降为
  0 条，平均回答长度由 170.9 降至 133.9 字符；错误自称仍为 1 条，标志性“吾辈”仍未迁移。
- 10 对 A/B 语义复核已写入 `manual_review_results.json` 并由正式 `review` 命令汇总：SFT
  10 胜、0 次明显落后、无阻断性 severe issue。Base/SFT 三项均分分别为生成稳定性
  `3.3/8.0`、角色一致性 `1.4/5.2`、对话质量 `1.5/5.7`，人工门槛通过。
- `run_summary.json` 最终状态为 `ready_for_grpo`，技术、稳定性和人工门槛均通过；阶段 2
  正式收尾，阶段 3 从第四次 adapter 开始。

### 结果解释与遗留问题

- SFT 已完成本阶段最重要的目标：把不可用的 Base 输出修正为基本稳定、简洁且多数相关的角色
  对话。第四次相对第三次的定向补充对长度控制和局部退化有效，因此不再继续追加 SFT 轮次。
- 角色质量仍只是最低可用：`dev_0003` 使用“本大爷”并有局部逻辑冲突；`dev_0005` 没有守住
  “不是普通宠物”的边界；`dev_0009` 编造“阿波”“小田头儿”等重要人物并堆叠 emoji；部分
  情绪回应和问题意图承接仍偏弱。62/62 训练标签含“吾辈”而 Dev 为 0/10，也说明小规模 SFT
  学到了泛化的傲娇猫系风格，却没有稳定学到核心语言标识。
- `strict_format_rate=0` 只作诊断，不作为质量失败：当前检测器要求只能有一个开头动作括号且
  后文不能再有括号，与 system prompt 允许灵活穿插动作的规范不一致。阶段 3 不围绕该指标
  优化。
- 10/10 胜率不能解释为模型已经成熟。同后端 Base 本身有 9/10 退化和 1 条乱码，导致相对比较
  容易；本轮只有单 seed、10 条 Dev，且复核由项目内单一执行者完成，没有独立多评审。角色
  一致性绝对分仅 5.2，必须在最终 Eval 中保留这一结论边界。
- 按学习项目“缩小每阶段规模但走完整闭环”的优先级，上述局部问题不再触发第五次 SFT。
  阶段 3 使用最小 GRPO 配置，奖励重点覆盖标志性自称、身份边界、禁止无依据人物/经历和回答
  相关性，同时将无乱码、无截断、无复读作为保护项。Dev 只用于阶段对比，不反复据此调奖励。

## 2026-08-09：增加四类定向 SFT 样本

- 根据第三次 SFT 的 Dev bad case，新增 12 条人工样本：主客体识别、事件与情绪识别、身份边界、
  问题意图与回答相关性各 3 条。新增 Prompt 不复刻 Dev，与既有训练 Prompt 和 Dev 的字符
  相似度均低于 0.6。
- 新增明细保存在 `data/runs/morgana-v2/sft_targeted_additions.jsonl`，并合并到
  `sft_train.jsonl`；Teacher v8 的 50 条原始审计保持不变。
- 当前训练集为 62 条，62/62 使用“吾辈”、0/62 使用错误自称，平均 113.2 字符、最长
  246 字符；训练集 SHA-256 为
  `c1ec8824db45db98f0e82547938a67e652fe75b759278d98aef5d0552daab142`。
- 训练参数保持不变；按 batch size 2、gradient accumulation 2、3 epochs 计算，下一轮预期
  48 个 optimizer steps。Stage 2 冻结数量、哈希和 notebook 校验已同步。

## 2026-08-09：第三次 SFT 后定向清洗两条标签

- 复查第三次 SFT 的 Dev 输出后，仅确认 `sft_0003` 和 `sft_0029` 存在明确标签质量问题；
  `sft_0031` 的不确定推测和 `sft_0049` 的其他猫语言设定均作为合理创作保留。
- `sft_0003` 删除过长、重复、无依据能力和威胁式表达；`sft_0029` 修复拒绝学狗叫后的
  自相矛盾和低姿态赔罪。Teacher v8 原始审计不改，人工裁决记录在
  `data/runs/morgana-v2/sft_label_review.md`。
- 清洗后的 `sft_train.jsonl` 仍为 50 条，50/50 使用“吾辈”、0/50 使用错误自称，平均
  125.2 字符、最长 246 字符；新 SHA-256 为
  `2c05d7618f433d3ddf972c8563ae8c3e5662c9ab227fbe83fe01f0282c4f720d`。
- Stage 2 冻结输入哈希已同步；历史 SFT 运行摘要继续保留各次实际使用的旧哈希。

## 2026-08-09：第二次 SFT 数值失败与纯 FP32 修复

- AutoDL run `20260808T170831Z-0b8290d2` 完成了 39/39 个记录步并保存
  `checkpoint-39`，但前 6 个 `grad_norm` 为 `NaN`；流水线按技术门槛标记为
  `training_failed`，未归档 adapter、未执行 Dev 推理，也未进入 GRPO。
- 本次不是 OOM、下载失败或数据校验失败。虽然冻结配置的 model、BNB compute 和 LoRA dtype
  均为 FP32，ms-swift 的实际 `SftArguments` 却为 `fp16=true`、`bf16=false`，触发了 FP16
  梯度溢出。
- 修复采用显式纯 FP32：配置增加 `fp16=false`、`bf16=false`；训练前校验全部精度字段和
  39 steps，训练后读取 `args.json` 再次确认实际精度，并把审计值写入 run summary。
- 失败 run 和 retained work 只作诊断证据。下一次从固定 Qwen 基座与 seed 全新运行，不加载
  v2-2 checkpoint；数据、batch、学习率、epoch、LoRA 和评测门槛保持不变。若纯 FP32 OOM，
  保留失败证据并停止，不自动切换 BF16。

## 2026-08-09：持久化 AutoDL Hugging Face 配置

- `roleplay-stage2-sft run` 默认使用 `https://hf-mirror.com`，并将 Hugging Face 缓存放在
  `/root/autodl-tmp/huggingface`，避免新 shell 或 tmux 会话遗漏环境变量而无法连接模型仓库。
- 用户显式设置的 `HF_ENDPOINT` 和 `HF_HOME` 仍具有优先级；实际值写入 run summary，便于复盘。

## 2026-08-09：直接使用 AutoDL 基础环境

- 独立虚拟环境的 PyTorch 2.10.0+cu128 方案在训练前取消。原因是它会重复安装基础镜像已有的
  PyTorch/CUDA 依赖，且绑定 Torch 2.10 的 causal-conv1d GitHub wheel 在当前网络下无法以
  合理时间下载。
- 正式训练改为直接使用 `/root/miniconda3/bin/python`：PyTorch 2.8.0+cu128、Python 3.12、
  CUDA 12.8、CXX11 ABI True。项目不再创建 `.venv`，只安装缺少的固定直接依赖。
- 基础镜像没有 nvcc；为避免继续下载或编译 CUDA 扩展，移除 flash-linear-attention 和
  causal-conv1d，Qwen3.5 使用 Transformers 的 PyTorch fallback。训练配置和验收门槛不变；
  若首次正式运行 OOM，先保留失败证据，再另行决定是否调整 micro-batch。

## 2026-08-08：第二次 SFT 使用 AutoDL

- Colab 免费额度耗尽且本轮不购买 Colab Pro，第二次 SFT 改在 AutoDL 单张 RTX 3090 24GB
  上执行；不修改数据、训练配置、模型 revision、seed、推理参数或验收门槛。
- 已选实例：NVIDIA GeForce RTX 3090，24576MiB，驱动 570.124.04；基础镜像为
  PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8。
- 原计划使用独立虚拟环境，固定 PyTorch 2.10.0+cu128 和已验证的直接依赖版本。
- 新增 `roleplay-stage2-sft run` 和 `review` 命令；产物保存到
  `output/morgana-v2/stage2-sft/<run-id>/`。
- 训练配置不包含需要随 GPU 型号改变的参数。训练尚未开始，实际 commit、环境、时长、日志
  和结果由 run summary 记录。

## 2026-08-08：阶段 1 基线范围简化

- 为符合本项目“简单快速”的学习目标，取消三 seed、30 条和专用 Transformers 后端的要求。
- Base Dev 改为：固定一个 seed，对 10 条 Dev 各生成一次；记录实际模型/revision、推理链路、
  聊天模板和生成参数。
- SFT、GRPO 的 Dev 对照沿用该固定条件、按 `id` 对齐；多 seed 重复采样和后端一致性不作为
  阶段门槛。

## 2026-08-08：阶段 1.4 Base Dev 基线

- Base：`mlx-community/Qwen3.5-2B-4bit`，revision
  `674aaa7240b91e8012fcad5d791b7dfe5ba90207`；本机 MLX OpenAI-compatible 服务，Apple M4。
- 固定 seed `20260807`，关闭 thinking；其余参数为 `max_tokens=512`、`temperature=0.6`、
  `top_p=0.8`、`top_k=20`、`presence_penalty=0.4`、`repetition_penalty=1.45`，两个 context
  size 均为 `128`。
- 输入：冻结的 v2 Persona、system prompt 和 10 条 Dev；完整哈希及环境版本见
  `data/runs/morgana-v2/base_generation_meta.json`。
- 输出：`base_dev_outputs.jsonl` 共 10 条，按 `id` 与 Dev 对齐；全部非空、首次生成成功、以
  `stop` 结束，未命中连续复读检测。输出 SHA-256 为
  `e3e1f00d904f8f5f27da7c8f86d68c43900a2fab498f668fe9c33b6f4b8335ac`。
- 首轮 `max_tokens=256` 时 `dev_0010` 截断；只将上限提升为 `512` 后完成基线，不再调参。

## 2026-08-08：阶段 1.2 Prompt 数据生成与切分

- 代码 commit：`cd30910a6891312e90a7e5b34f7ad4dc147ceaf4`。
- Teacher：`deepseek-v4-flash`，thinking enabled，reasoning effort high，
  `max_tokens=8192`，未显式设置 temperature。
- 输入：`data/persona.json`、`data/style_examples.jsonl`。
- 输出：`data/runs/morgana-v2/`。
- 共享候选池：100 条，五类场景各 20 条；首轮请求目标含每类 5 条过采样候选。
- 切分 seed：`20260806`。
- 实际命令：

  ```bash
  .venv/bin/roleplay-datagen \
    --persona data/persona.json \
    --style-examples data/style_examples.jsonl \
    --output-dir data/runs/morgana-v2 \
    --split-seed 20260806
  ```

日常场景首次响应没有产生可解析的有效 Prompt；生成器自动重试后成功。全部候选通过过滤后一次性
发布，未留下残缺正式产物。

### 产物与检查

| Split | 数量 | 每类场景 | SHA-256 |
|---|---:|---:|---|
| SFT | 50 | 10 | `ab3ef1bfb0f6e18094acff3ad4dace386d0991786ae7f71a80ae33bf2322f64e` |
| GRPO | 20 | 4 | `b36b4f01f232901ab0b5f6011fa64b66f48e02c75b6b0050035e4caf703e7231` |
| Dev | 10 | 2 | `74cf6d05921155cec5c070ca8a611c7a8e6751b00ca0b77a6f4e9085aeeecb22` |
| Eval | 20 | 4 | `f15f72f5011f2d61bc3235407f0029d32dcd714e290ebf47870c2213eed319a8` |

- 四个 split 合计 100 条，规范化后全局唯一，ID 和数量均正确。
- 每条记录仅含 `id`、`scenario`、`target_goals` 和 `user`。
- `target_goals` 仅包含 `generation_stability`、`character_consistency` 和
  `dialogue_quality`。
- 本地质量过滤复查未发现目标元数据泄漏、用户/角色视角颠倒或依赖缺失上文。
- 与 v1 的 SFT、GRPO、Dev、Eval 和 Pilot 共 105 条 Prompt 交叉检查：规范化精确重复 0 条，
  字符相似度阈值 0.88 的高相似重复 0 条。
- Persona、风格样例和 system prompt 的快照哈希均与 `input_manifest.json` 一致。
- 生成前使用仓库 `.venv` 运行完整测试：86 项通过。

## 2026-08-08：阶段 1.3 Student/Teacher Pilot

- Student：`mlx-community/Qwen3.5-2B-4bit`，revision
  `674aaa7240b91e8012fcad5d791b7dfe5ba90207`，MLX OpenAI-compatible server。
- Teacher：`deepseek-v4-flash`，thinking enabled，reasoning effort high。
- 输入：SFT 五类场景各第一条，共 5 条；Pilot raw 在验收后已清理，保留
  `data/runs/morgana-v2/pilot/pilot_review.md` 作为最小可检查样例。
- 5/5 Student 和 Teacher 均首次调用成功，无重试、截断或残缺产物；三份 JSONL 逐条对齐。
- Teacher 决策为 5 `rewrite` / 0 `light_rewrite` / 0 `keep`。Student baseline 存在明显重复、
  逻辑错乱、角色偏离或未直接回答，全部重写有充分理由。
- 最终回答长度为 baseline 的 25%～34%；自动检查和 5/5 语义复核通过。三条旧格式
  诊断未通过，但动作括号不是 v2 训练目标，因此不阻断 Pilot。
- 正式 50 条的人工抽查需重点关注两点：Teacher 是否引入 Persona 未显式记录的角色背景
  事实；`issues` 与 `improved_assistant` 是否逐项对应。
- Pilot 人工复核结论：通过，可进入 50 条正式 SFT 标签生成。
- 运行后使用仓库 `.venv` 运行完整测试：86 项通过。

## 2026-08-08：阶段 1.3 正式 SFT 标签生成

- 沿用通过 Pilot 的冻结输入、Student revision、生成参数和 Teacher 配置生成 50 条。
- 三份正式 JSONL 均为 50 条，五类场景各 10 条，与 Prompt 和 system prompt 逐条对齐；
  输入哈希与 `sft_generation_meta.json` 一致。
- Teacher 决策：46 `rewrite`、4 `light_rewrite`、0 `keep`；Student 平均 667.6 字符，
  最终标签平均 131.1 字符。
- Student baseline 中 27/50 以 `length` 结束，33/50 命中明显复读；Teacher 最终标签
  0 空回答、0 乱码、0 明显复读。
- 第 46 条暴露链路缺陷：SFT 已保留截断 baseline 供 Teacher 纠错，却在 Student 层拒绝
  复读 baseline。已修改为仅 Teacher-corrected SFT 路径保留复读 bad case；普通推理
  仍拒绝复读，Teacher 最终回答仍必须无复读。
- 全量语义浏览发现 `sft_0016`、`sft_0029`、`sft_0031`、`sft_0036`、`sft_0043`、
  `sft_0049` 至少 6 条仍有事实编造、角色边界或对话质量问题。复核证据见
  `data/runs/morgana-v2/sft_label_review.md`。
- 阶段决策：生成链路完成，但正式标签人工门槛未通过；当前 `sft_train.jsonl`
  不得用于训练。冻结 Student baseline，加强 Teacher 约束后重跑 Teacher 和 QA。
- 产物 SHA-256：baseline `8e69b5de4012e55e76692e4f597c69499f351c4499b28c9176faa0264c564108`；
  audit `96f4d91a4a6d19ac811ac161485339e545023a42fc7cf34465be5f6341297391`；
  train `7d669be254455e622f307945469533b001dee6cfc3057d48b1e7dbdd764410e0`。
- 修复后使用仓库 `.venv` 运行完整测试：88 项通过。

## 2026-08-08：Teacher v7/v8 约束落地与重跑

- Teacher prompt 新增：允许高置信正典知识、禁止猜测未知事实、出戏提问保持角色内
  视角、收紧 `light_rewrite`、要求 issues 逐项闭环，并明确动作长度不是门槛。
- `--teacher-only` 已支持正式数据：校验完整 baseline 和 Student 配置，冻结 Student 输出，
  50 条 Teacher 全部成功后才原子替换 audit、train 和 metadata。
- v7 修复了原有主客体和出戏问题，但仍漏检名字归因和长回答中的身份词冲突；
  最终由 v8 替代，不保留重复产物。
- v8 增加名字来源、共同回忆反例和逐段身份词检查。50 条全部判为 `rewrite`，
  0 空回答、0 乱码、0 明显复读，平均 136.8 字符。
- v8 仍有代表性失败：`sft_0029` 内部逻辑冲突；`sft_0031` 承认不确定后继续猜测；
  `sft_0049` 的正典判断在 v7/v8 间相反。
- 结论：提示词规则已落地，但不再继续追加规则重跑；需要冻结的最小正典参考和独立语义
  复核。当前 v8 `sft_train.jsonl` 仍不得用于训练。
- v6/v7 负向结论仅保留在本运行日志，重复的 audit、train 和 metadata 副本已删除。
- 后续按学习项目尺度重新裁决：只阻断空回答、乱码/严重复读、破坏性截断、身份崩坏、
  明显答非所问或严重主客体错位。v8 未发现这些问题；`sft_0029`、`sft_0031`、
  `sft_0049` 降为非阻断观察。
- 最终阶段决策：正式 SFT 标签通过学习项目的最小人工门槛，`sft_train.jsonl`
  可用于训练。

## 2026-08-08：第二次 SFT 运行前优化

- 首轮 SFT 技术有效：12/12 optimizer step 完成，LoRA-B 全部产生非零更新；但
  `dev_0001` 截断，稳定性门槛失败，因此未进入匿名人工复核或 GRPO。
- 首轮 SFT 将退化样本从 Base 的 10/10 降至 3/10，将错误自称从 10/10 降至 3/10；但
  10 条 Dev 均未出现“吾辈”。训练标签则是 50/50 含“吾辈”、0/50 含“本大爷/本喵”，说明
  数据中已有目标信号，优先验证训练强度而不是改写数据。
- 第二轮将有效 batch size 从 16 降到 4；其余训练和推理参数冻结。具体采用
  `per_device_train_batch_size=2`、`gradient_accumulation_steps=2`，在保持预期 39 个
  optimizer step 的同时利用 GPU 剩余显存，提高训练吞吐。
- 训练前置检查新增训练标签角色信号断言，并把标签统计和预期 optimizer step 写入
  `run_summary.json`。第二轮仍沿用首轮稳定性和匿名人工门槛，不调整 Dev 或验收阈值。
- 第二轮运行前已清空执行环境中的首轮输出；首轮结论保留在本日志，raw 产物
  已于阶段 2 收尾后清理。本地完整测试 92 项通过。
