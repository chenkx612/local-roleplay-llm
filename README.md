# Roleplay Post-Training

一个用于学习角色扮演模型后训练流程的轻量项目。目标是用最小规模的数据和配置，完整走通：

```text
Persona 与数据准备 → Base 评测 → SFT → DPO → 规则型 GRPO → 统一评测 → 复盘
```

项目强调流程完整、结果可检查，不面向生产环境或大规模训练。

## 快速开始

要求 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

## 常用命令

查看各工具的参数：

```bash
roleplay-chat --help
roleplay-datagen --help
roleplay-inference --help
roleplay-sft-data --help
roleplay-stage2-sft --help
roleplay-dpo-data --help
roleplay-stage3-dpo --help
```

连接本地 OpenAI 兼容服务后，可进行单轮或交互式对话：

```bash
roleplay-chat --message "你是谁？"
roleplay-chat
```

默认服务地址为 `http://127.0.0.1:8080/v1`。数据生成使用 DeepSeek API，密钥通过环境变量提供：

```bash
export DEEPSEEK_API_KEY="your-api-key"
roleplay-datagen --output-dir data/runs/<run-name>
```

请勿将 API Key 提交到仓库。

## 项目结构

```text
src/roleplay/   核心代码与命令行工具
tests/          unittest 测试
data/           Persona、样例及实验数据
configs/        训练配置
requirements/   分阶段依赖
docs/           计划、执行指南、日志与复盘
```

## 文档

- [训练计划](docs/PLAN.md)
- [v2 执行规约](docs/V2_EXECUTION_SPEC.md)
- [AutoDL 训练指南](docs/AUTODL.md)
- [DPO 计划](docs/STAGE3_DPO_PLAN.md)
- [旧版 GRPO 计划（历史失败）](docs/STAGE3_GRPO_PLAN.md)
- [运行记录](docs/RUNLOG.md)
- [已知问题](docs/ISSUES.md)

## 当前状态

项目正在按 `morgana-v2` 流程推进。阶段二 SFT 已通过；旧版主观 Judge GRPO 和第一次 DPO
已失败并封存。第二轮 DPO 使用 40 个 Prompt、每条 2 个 SFT 候选和离线 Codex 裁决，已冻结
30 对偏好数据，下一步从原 SFT adapter 重新训练。后续进展以 `docs/` 中的计划和运行记录为准。
