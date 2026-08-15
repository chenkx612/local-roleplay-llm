# Roleplay Post-Training

一个用于学习角色扮演模型后训练流程的轻量项目。项目以最小规模完整保留：

```text
Persona 与数据准备 → Base 评测 → SFT → DPO → 规则型 GRPO
→ post-GRPO DPO → 统一评测 → 复盘
```

项目不面向生产训练平台；优先保证流程短、依赖少、产物可检查。

## 快速开始

要求 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

## 统一命令

所有阶段都可以从一个入口发现：

```bash
roleplay --help
roleplay data prompts --help
roleplay data sft --help
roleplay train sft --help
roleplay train dpo --help
roleplay train grpo --help
roleplay train post-dpo --help
roleplay eval inspect --help
roleplay eval compare --help
```

例如：

```bash
roleplay chat --message "你是谁？"
roleplay data prompts --output-dir data/runs/<run-name>
roleplay train sft run
roleplay eval compare --baseline base.jsonl --candidate candidate.jsonl
```

数据生成使用 DeepSeek API，密钥通过 `DEEPSEEK_API_KEY` 提供。Chat 和推理默认连接
`http://127.0.0.1:8080/v1` 的 OpenAI 兼容服务。

历史文档中的 `roleplay-stage2-sft` 等命令仍保留兼容；新工作优先使用统一入口。

## 项目结构

```text
src/roleplay/core/          原子 IO、运行环境、adapter 与 release 基础设施
src/roleplay/evaluation/    统一输出检查、模型比较和评测 CLI
src/roleplay/experiments/   跨阶段冻结实验合同
src/roleplay/*.py           明确的数据、奖励、训练阶段及兼容 CLI
tests/                      无网络 unittest 回归测试
data/runs/morgana-v2/       已冻结的 v2 输入、数据集和审计产物
configs/                    各训练器消费的显式配置
requirements/               分阶段 AutoDL 依赖
docs/                       当前入口、历史计划、运行记录与复盘
```

架构边界和依赖规则见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 当前状态

`morgana-v2` 已完成并归档，结论见
[v2 实验收尾报告](docs/V2_RETROSPECTIVE.md)。`morgana-v3` 尚未立项；新的实验目标、预算和
晋升门槛应先写入 [训练计划](docs/PLAN.md)，不得把 v2 的中间状态当作当前任务继续追加。

请勿提交 API Key、大型模型产物或未经检查的敏感对话数据。
