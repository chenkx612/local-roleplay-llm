# 阶段二：Colab T4 LoRA SFT 执行计划

## 1. 目标

在 Google Colab 单张 NVIDIA T4 上，用一组固定配置完成最小且可审计的 SFT 链路：

```text
环境与冻结输入校验
→ 完整 3 epochs QLoRA SFT
→ 检查有效梯度和 LoRA 更新
→ 三个 seed 的同后端 Base/SFT Dev 推理
→ 自动相对行为门槛
→ 主 seed 匿名 A/B 人工复核
```

本阶段不搜索超参数，也不单独运行 smoke 训练。相对前次计划只把训练从 1 epoch 提升为
3 epochs，预计产生 12 个 optimizer step；模型、50 条训练数据、LoRA 结构、学习率和精度配置
保持不变。一次运行只训练一次，不根据 Dev 自动重训。技术、自动行为和匿名人工三层门槛必须
全部通过才允许进入阶段三；行为失败仍保留完整产物供复盘。

## 2. 固定决策

- 运行环境：Google Colab `2026.04`、Python 3.12、PyTorch 2.10、单张 NVIDIA T4。
- 仓库从公开 GitHub `stage2-sft-dev` clone，并记录实际 commit。合并阶段二改动后再将
  notebook 和本说明中的分支一并切换为 `main`。
- 训练基座：`Qwen/Qwen3.5-2B`，revision 固定为
  `965dcc54bc9c0591873df0e9869c056a54d323d1`。
- ms-swift Transformers 后端使用 BNB 4-bit QLoRA；模型计算、BNB compute 和 LoRA 参数均为
  `float32`，避免首次 FP16 运行的非有限梯度和空更新。
- 使用 `last_round+ignore_empty_think`，只对最后一轮 assistant 回复计算 loss；thinking 关闭并
  加入 non-thinking prefix。
- 训练和中间文件放在 Colab 临时盘；Drive 只保存最终核心产物。

固定依赖版本：

- `ms-swift==4.4.1`
- `datasets==4.8.4`
- `transformers==5.12.1`
- `peft==0.19.1`
- `bitsandbytes==0.49.2`
- `qwen-vl-utils==0.0.14`
- `flash-linear-attention==0.5.1`
- `ninja==1.13.0`
- `causal-conv1d==1.6.2.post1`

不安装 FlashAttention、DeepSpeed、vLLM 或在线实验追踪服务。固定依赖安装失败或实际版本不一致
时停止，不自动升级或切换框架。

## 3. 输入与主配置

冻结输入为 50 条 `sft_train.jsonl`、10 条 `dev.jsonl`、Persona 和 system prompt。运行前校验
文件 SHA-256、记录结构、模型 revision，以及训练模板下的最大 token 长度。

主配置保存在 `configs/morgana_v1_sft_t4.yaml`，关键参数为：

```yaml
model: Qwen/Qwen3.5-2B
model_revision: 965dcc54bc9c0591873df0e9869c056a54d323d1
tuner_type: lora
target_modules: [all-linear]
torch_dtype: float32
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
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
loss_scale: last_round+ignore_empty_think
enable_thinking: false
save_strategy: epoch
save_only_model: true
report_to: none
```

`save_only_model` 避免写出后续不需要的 optimizer、scheduler、scaler 和 RNG 断点状态。

## 4. 执行与验收

1. 验证 Colab runtime、T4、Python、PyTorch/CUDA，挂载 Drive，clone 仓库并记录 commit。
2. 安装固定依赖并核对顶层包实际版本。
3. 校验冻结输入和训练配置，用固定 revision 的 tokenizer 检查长度及标签格式。
4. 在独立子进程运行一次完整的 3-epoch SFT，同时保存 `train.log`。
5. 检查约 12 个 `grad_norm` 均有限且为正、所有 LoRA-B 张量非零，并计算 adapter SHA-256。
6. 在 notebook 进程加载同一 HF Base；对 `20260807/08/09` 每个 seed，在 Base 和 SFT 推理前
   分别重置 Python、Torch 和 CUDA RNG，生成各 30 条输出并按 `(seed, id)` 对齐。
7. 用仓库内纯 Python 逻辑规范化空 thinking wrapper，汇总严格格式、截断、复读、乱码、括号、
   自称等问题，并执行相对行为门槛。
8. 为主 seed 的 10 对输出生成固定匿名顺序的复核包和独立答案映射；提交人工结果后更新最终
   GRPO 决策。最后只复制核心产物到 Drive 并删除 Colab 临时训练目录。

技术门槛要求训练正常结束、LoRA 确实更新、adapter 能重新加载，以及 Base/SFT 各 30 条完整、
非空和对齐。自动相对行为门槛要求：SFT 的 `stop` 数不低于 Base，截断和退化数不高于 Base，
严格格式率至少提高 20 个百分点，“吾辈”比例提高，且“本大爷/本喵”错误别称比例降低。
任一项失败时状态为 `behavior_failed`，产物照常归档但禁止进入 GRPO。

自动门槛通过后，匿名人工复核按角色一致性、事实依据、风格、格式自然度和对话质量评分，并记录
幻觉、视角错位和乱码等严重问题。SFT 必须至少胜出 6 对、明显落后不超过 2 对、没有严重问题；
两条 emotion 样本不得落后，也不得出现视角错位或乱码。

## 5. 产物目录

每次运行使用 UTC 时间戳加随机后缀作为 `run-id`，Drive 目录固定为：

```text
roleplay/morgana-v1/stage2-sft/<run-id>/
├── run_summary.json
├── training_config.yaml
├── train.log
├── adapter/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── additional_config.json
├── hf_base_dev_outputs.jsonl
├── dev_outputs.jsonl
├── manual_review_packet.json
├── manual_review_answer_key.json
└── manual_review_results.json
```

`run_summary.json` 统一保存 run/commit、环境版本、输入哈希、训练命令与指标、adapter 更新统计、
`evaluation_seeds`、两组 Dev 检查、`relative_behavior_gate`、`manual_review`、
`ready_for_grpo` 和文件哈希。规范化输出与 `raw_assistant` 同行保存，不另建 raw 文件。临时
messages、推理脚本、optimizer 状态以及拆分的 context/validation/metadata/notes/command 文件
均不归档。训练技术失败时最多保存摘要和已有训练日志，不复制半成品 checkpoint。

## 6. 完成标准

- [ ] 环境、仓库 commit、模型 revision、冻结输入和实际配置已记录。
- [ ] 3 epochs 正常结束，约 12 个有限正梯度，LoRA-B 已全量更新且 adapter 可重载。
- [ ] 同后端 Base/SFT Dev 各 30 条完整非空并按 `(seed, id)` 对齐；同 seed 重跑可复现。
- [ ] 自动相对行为门槛通过；否则状态明确为 `behavior_failed` 且未进入 GRPO。
- [ ] 主 seed 匿名人工门槛通过，最终 `ready_for_grpo` 等于三层门槛的逻辑与。
- [ ] Drive 目录只包含约定的 11 个核心文件，不含临时或断点续训产物。
- [ ] 实际结果和是否进入 GRPO 的决定已写入 `RUNLOG.md`。
