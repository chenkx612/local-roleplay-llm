# Stage Dependencies

每个文件对应一个可独立复现的 AutoDL 阶段环境。公共版本合同由
`src/roleplay/core/runtime.py` 校验，阶段文件只声明新增或禁用的可选加速依赖。

- `stage2_sft_autodl.txt`：SFT 基础环境。
- `stage3_dpo_autodl.txt`：前置 DPO。
- `stage3_grpo_autodl.txt`：历史主观 Judge GRPO。
- `stage4_grpo_autodl.txt`：规则型 GRPO；post-GRPO DPO 复用其基础依赖。
