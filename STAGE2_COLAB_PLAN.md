# 阶段二：Colab T4 LoRA SFT 执行计划

## 1. 目标

在 Google Colab 单张 NVIDIA T4 上，用一组固定配置完成最小且可审计的 SFT 链路：

```text
环境与冻结输入校验
→ 完整 1 epoch QLoRA SFT
→ 检查有效梯度和 LoRA 更新
→ 同后端 Base/SFT Dev 推理
→ 保存核心产物并人工复核
```

本阶段不搜索超参数，也不单独运行 smoke 训练。完整训练只有 4 个 optimizer step，训练后直接
验证 loss、梯度、adapter 更新和重新加载，避免重复加载模型及生成第二套 checkpoint。训练效果
没有超过 Base 仍可进入阶段三，前提是技术链路有效且行为变化得到记录。

## 2. 固定决策

- 运行环境：Google Colab `2026.04`、Python 3.12、PyTorch 2.10、单张 NVIDIA T4。
- 仓库从公开 GitHub `main` clone，并记录实际 commit。
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
num_train_epochs: 1
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
4. 在独立子进程运行完整 SFT，同时保存一份 `train.log`。
5. 检查 loss 和 `grad_norm` 有限且为正、所有 LoRA-B 张量非零，并计算 adapter SHA-256。
6. 在 notebook 进程加载同一 HF Base，先生成 Base Dev，再加载保存的 adapter 生成 SFT Dev。
7. 汇总技术检查和行为指标，只复制核心产物到 Drive，最后删除 Colab 临时训练目录。

技术有效性是唯一阻断门槛：训练正常结束、LoRA 确实更新、adapter 能重新加载，以及 Base/SFT
Dev 各 10 条逐条对齐且非空。格式通过率、截断、finish reason 和复读只进入摘要供人工判断，
不把负向训练效果误判为技术失败。

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
└── dev_outputs.jsonl
```

`run_summary.json` 统一保存 run/commit、环境版本、输入哈希、训练命令与指标、adapter 更新统计、
生成配置、两组 Dev 检查和文件哈希。临时 messages、raw 输出、推理脚本、optimizer 状态以及拆分的
context/validation/metadata/notes/command 文件均不归档。失败时最多保存摘要和已有训练日志，不复制
半成品 checkpoint。

## 6. 完成标准

- [ ] 环境、仓库 commit、模型 revision、冻结输入和实际配置已记录。
- [ ] 完整训练有效，LoRA-B 已更新，adapter 可重新加载。
- [ ] 同后端 Base/SFT Dev 各 10 条完整非空并逐条对齐。
- [ ] 格式、截断、复读和其他行为变化已进入统一摘要并完成人工复核。
- [ ] Drive 目录只包含约定的 8 个核心文件，不含临时或断点续训产物。
- [ ] 实际结果和是否进入 GRPO 的决定已写入 `RUNLOG.md`。
