# Training Configurations

这些 YAML 是训练器直接消费的冻结参数，不承担流水线编排职责。跨阶段公共模型与数据合同位于
`src/roleplay/experiments/morgana_v2.py`。

- `morgana_v2_sft.yaml`：Stage 2 SFT。
- `morgana_v2_dpo.yaml`：前置 DPO。
- `morgana_v2_grpo.yaml`：历史主观 Judge GRPO，作为失败实验保留。
- `morgana_v2_stage4_grpo.yaml`：最终规则型 GRPO。
- `morgana_v2_post_grpo_dpo.yaml`：最终 post-GRPO DPO。

新增实验应创建带新实验名的配置，不修改 v2 文件。

