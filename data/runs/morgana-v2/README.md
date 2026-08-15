# morgana-v2 Artifact Index

本目录是已完成 v2 实验的冻结、可审计记录。文件路径和内容哈希属于训练合同；为了保持历史运行可复现，
不再按新目录移动这些文件。新版本应创建新的 `data/runs/<experiment>/`，不要向这里追加状态。

## 输入与公共 split

- `inputs/`：Persona 与风格样例的输入快照。
- `input_manifest.json`、`system_prompt.txt`：输入清单和渲染后的系统提示词。
- `dev.jsonl`、`eval.jsonl`：统一 Dev 与最终 Eval split。
- `base_dev_outputs.jsonl`、`base_generation_meta.json`：Base 对照输出及生成配置。

## SFT

- `sft_train_prompts.jsonl`：冻结训练 Prompt。
- `sft_train.jsonl`、`sft_targeted_additions.jsonl`：主训练集与定向补充集。
- `sft_generation_meta.json`、`sft_label_review.md`：生成元数据与标注复核。
- `pilot/`：最小 Pilot 的人工复核记录。

## DPO 历史探索

- `dpo_prompts*.jsonl`：不同轮次的偏好 Prompt。
- `dpo_train*.jsonl`、`dpo_train_audit*.json`：候选训练集及审计。
- `dpo_train_editability17.jsonl`：最终前置 DPO 使用的 17 条最小编辑数据。

带 `run2`、`run3` 的文件是保留的负向实验，不是当前默认输入。

## GRPO

- `rl_train.jsonl`：历史主观 Judge GRPO Prompt；该路线已判定失败。
- `rule_grpo_train.jsonl`：规则型 GRPO 的冻结 Prompt 与约束。

## post-GRPO DPO

- `post_grpo_dpo_prompts.jsonl`、`post_grpo_dpo_holdout.jsonl`：原始训练与 holdout Prompt。
- `post_grpo_dpo_prompts_expansion.jsonl`：扩展 Prompt。
- `post_grpo_dpo_train*.jsonl`：原始、扩展及合并训练输入。
- `post_grpo_dpo_*manifest.json`、`post_grpo_dpo_*audit.json`：采样和偏好对审计合同。

模型 adapter、训练日志和可下载 release 属于 `output/`，不进入本目录或 Git。

