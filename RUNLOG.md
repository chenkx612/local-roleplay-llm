# morgana-v2 运行日志

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
