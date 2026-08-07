# 阶段二：Colab T4 LoRA SFT 执行计划

## 1. 目标

在 Google Colab 的单张 NVIDIA T4（16 GB）上，用一组固定配置完成：

```text
环境与输入校验
→ 1 个 optimizer step 冒烟训练
→ 完整 1 epoch QLoRA SFT
→ 重新加载 LoRA
→ Dev 推理并与 Base 初步比较
→ 保存产物并更新 RUNLOG.md
```

本阶段不搜索超参数。训练效果没有超过 Base 也可以进入阶段三，前提是训练、重载、推理和记录链路有效。

## 2. 固定决策

- 运行环境：Google Colab `2026.04`，Python 3.12、PyTorch 2.10，单张 NVIDIA T4（16 GB）。
- 仓库版本：从公开 GitHub `main` clone，运行时记录实际
  `git rev-parse HEAD`。该 commit 必须同时包含本 notebook 和训练 YAML。
- 训练基座：`Qwen/Qwen3.5-2B`，Hugging Face revision 固定为
  `965dcc54bc9c0591873df0e9869c056a54d323d1`。
- 训练框架：ms-swift，Transformers 后端，BNB 4-bit QLoRA。
- T4 不使用 `bf16`；统一改为 `float16`，并将 BNB compute dtype 设为 `float16`。
- 使用 `last_round+ignore_empty_think`，只对最后一轮 assistant 回复计算 loss，并加入
  non-thinking prefix。
- 冻结 ViT 和 aligner，只向 LLM 的 `all-linear` 层注入 LoRA。
- 训练和中间文件放在 Colab 临时盘；完成每一步后同步到 Google Drive。

相对 `PLAN.md` 的预定配置，本阶段已知计划偏差只有 `bf16 → float16`，原因是 T4 不支持
bf16。实际运行产生的其他偏差必须写入运行摘要和 `RUNLOG.md`。

固定依赖版本：

- `ms-swift==4.4.1`
- `transformers==5.12.1`
- `peft==0.19.1`
- `bitsandbytes==0.49.2`
- `qwen-vl-utils==0.0.14`
- `flash-linear-attention==0.5.1`
- `causal-conv1d==1.6.2.post1`

不安装 FlashAttention、DeepSpeed、vLLM 或在线实验追踪服务。严格版本不兼容时停止并保留
日志，不自动升级或切换框架。

## 3. 输入与主配置

冻结输入：

- `data/runs/morgana-v1/sft_train.jsonl`：50 条训练样本。
- `data/runs/morgana-v1/dev.jsonl`：10 条 Dev Prompt。
- `data/runs/morgana-v1/base_dev_outputs.jsonl`：10 条 Base 对照。
- `data/runs/morgana-v1/inputs/persona.json` 和 `system_prompt.txt`。

主配置：

```yaml
model: Qwen/Qwen3.5-2B
model_revision: 965dcc54bc9c0591873df0e9869c056a54d323d1
use_hf: true
tuner_type: lora
target_modules: [all-linear]
freeze_llm: false
freeze_vit: true
freeze_aligner: true
torch_dtype: float16
quant_method: bnb
quant_bits: 4
bnb_4bit_compute_dtype: float16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
max_length: 1024
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 5e-5
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
loss_scale: last_round+ignore_empty_think
add_non_thinking_prefix: true
enable_thinking: false
packing: false
padding_free: false
split_dataset_ratio: 0.0
seed: 20260807
data_seed: 20260807
logging_steps: 1
save_strategy: epoch
report_to: none
```

仓库中的完整主配置为 `configs/morgana_v1_sft_t4.yaml`。安装成功后立即保存 Python、
CUDA、GPU、PyTorch 和所有固定依赖的实际版本，不使用未记录的浮动环境作为最终运行证据。

## 4. 执行步骤

### 4.1 Colab 预检

1. 选择 `2026.04` runtime 和单张 T4 GPU，检查 Python 3.12、PyTorch 2.10、CUDA 和显存。
2. 挂载 Google Drive，从公开 GitHub `main` clone 仓库并记录实际 commit。
3. 严格安装固定依赖，运行 `pip check` 并验证每个实际版本。
4. 记录 `nvidia-smi`、Python 和依赖版本。
5. 校验 Hugging Face 模型 revision；校验训练/Dev/Base 数据条数、结构和冻结 SHA256。
6. 用训练时 tokenizer 复核所有样本均不超过 1024 tokens。

任一输入校验失败时停止训练，不自动修改或重建冻结数据。

### 4.2 冒烟训练

使用主 YAML，仅覆盖 `max_steps=1` 和冒烟输出路径，输出到独立的 `smoke/` 目录。一次 step
指一个 optimizer update；在当前梯度累积配置下会包含 16 个 micro-batch。

通过条件：

- 模型以 BNB 4-bit 和 `float16` 正常加载。
- 完成一次 optimizer update，无 OOM、NaN 或无限 loss。
- loss 非零且可解释。
- 产生可读取的 LoRA adapter，并能生成一条非空回答。

冒烟失败时先记录完整错误和环境，再修复技术问题；不要通过反复调整学习率、LoRA rank 等方式试配置。

### 4.3 完整训练

冒烟和完整训练都由独立操作系统子进程执行。冒烟通过后从干净进程启动完整 1 epoch 训练，
输出到 `full/` 目录。只运行这一组主配置，并保存：

- 最终训练 YAML 和完整命令。
- `args.json`、训练日志、trainer state 和 loss 记录。
- adapter 配置、权重和 tokenizer/template 相关文件。
- 开始/结束时间、训练耗时和峰值显存信息。

### 4.4 LoRA 重载与 Dev 推理

从保存的 adapter 启动全新推理进程，先验证单条回答，再处理全部 10 条 Dev。

先将冻结 Dev 转为 10 条 `messages` 临时数据，再通过 ms-swift `TransformersEngine` 进行推理。
生成配置为 `max_new_tokens=256`、`temperature=0.6`、`top_p=0.8`、`top_k=20`、
`repetition_penalty=1.45`、thinking 关闭。`RequestConfig` 中对应字段名为 `max_tokens`。

Base 使用的 `presence_penalty=0.4` 在 Transformers backend 中没有等价支持，因此不应用，并在
generation metadata 和运行摘要中明确记录。其他后端差异也必须记录，不能静默忽略。

保存逐条对齐的 Dev 输出，并检查：

- 10/10 回答非空，无明显截断或连续复读。
- 统一格式是否稳定。
- 角色、事实、边界和语言风格是否相对 Base 改变。
- 是否出现模板化、统一拒答、过短回答或格式投机。

## 5. 产物目录

每次运行先生成 UTC 时间戳加随机后缀的 `run-id`，Google Drive 持久目录为：

```text
roleplay/morgana-v1/stage2-sft/<run-id>/
├── environment/
├── smoke/
├── full/
├── run_context.json
├── training_config.yaml
├── data_validation.json
├── dev_messages.jsonl
├── dev_outputs.jsonl
├── dev_generation_meta.json
├── stage2_summary.json
└── stage2_notes.md
```

`<run-id>` 目录必须以“不存在”为前提创建，绝不覆盖旧尝试。依赖/数据校验、冒烟、完整训练和
Dev 推理各阶段结束时同步现有产物；失败也尽可能先同步日志。大型基座权重不复制到 Drive。

训练完成后，将必要的小型产物同步回仓库的 `data/runs/morgana-v1/stage2-sft/`。大型重复模型权重不进入 Git；LoRA adapter 是否提交应在检查文件大小和敏感内容后决定。

## 6. 完成标准

- [ ] Colab T4 环境、依赖版本、仓库 commit 和模型 revision 已保存。
- [ ] 冒烟训练通过，loss 有效，LoRA 可重新加载。
- [ ] 完整 1 epoch 训练结束，配置、日志和 adapter 齐全。
- [ ] 10/10 条 Dev 输出非空且逐条对齐；格式、截断和复读异常已进入摘要，不静默重跑。
- [ ] 已记录 Transformers backend 不支持 Base `presence_penalty` 的差异。
- [ ] Dev 已与 Base 做三项目标和退化行为的初步比较。
- [ ] 实际配置、偏差、错误、产物路径和阶段观察已写入 `RUNLOG.md`。
- [ ] 已形成是否进入阶段三 GRPO 的明确决定。
