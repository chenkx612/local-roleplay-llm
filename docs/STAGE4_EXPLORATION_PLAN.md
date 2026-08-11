# Stage 4：GRPO 候选探索实验

## 目的

在不更新模型权重的前提下，从冻结 SFT adapter 采样 Stage 4 的20条规则 Prompt，找到能够稳定
产生合规候选的最小配置。该实验只诊断 GRPO 的探索支持，不修改 SFT、规则 Prompt 或奖励公式。

## 冻结判定

单个候选同时满足以下条件才算完全合规：无硬失败、30～90字、恰好一次“吾辈”、无错误自称、
动作策略满分且无格式问题。

“稳定”按20个 Prompt 组的支持率判断：硬有效和格式支持率100%，奖励有方差100%，简洁和动作
支持率至少80%，标志自称和完全合规支持率至少70%，无错误自称至少80%，两条禁止动作 Prompt
均必须采到无动作候选。门槛在生成新候选前冻结。

采样阶梯按成本和相对原配置的改动从小到大排列：

1. `g4-t06-p08`
2. `g4-t08-p09`
3. `g8-t06-p08`
4. `g8-t08-p09`
5. `g16-t08-p09`

每档都使用 `top_k=20`、`repetition_penalty=1.45` 和最多256 token，并运行两个独立 seed
轮次。总支持率和每一轮支持率都必须达到门槛。程序在首个达标配置停止；若全部失败，则结论是
仅靠扩大采样不足以建立 GRPO 的正确候选支持。

## 执行

在安装 Stage 4 依赖且有24GB GPU 的 AutoDL 环境运行：

```bash
roleplay-stage4-explore run
```

结果保存在 `output/morgana-v2/stage4-exploration/<run-id>/`。`exploration_summary.json` 记录每档
支持率和最终 `selected_config`，每个已执行配置对应一份完整候选 JSONL。

也可以离线分析已有 Stage 4 奖励日志：

```bash
roleplay-stage4-explore analyze \
  --input output/morgana-v2/stage4-grpo/<run-id>/reward_samples.jsonl
```
