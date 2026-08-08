# morgana-v2 运行日志

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
- 输入：SFT 五类场景各第一条，共 5 条；输出：`data/runs/morgana-v2/pilot/`。
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
