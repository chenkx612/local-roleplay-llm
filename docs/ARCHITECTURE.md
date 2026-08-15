# Repository Architecture

## 设计目标

仓库只服务一个小而完整的角色扮演后训练学习循环。架构按“稳定机制”和“实验语义”分离：
共享代码负责可靠地产生产物，阶段代码明确写出数据规则、训练参数和晋升门槛。

## 依赖方向

```text
CLI / stage workflows
        │
        ├── experiments/   跨阶段冻结合同
        ├── evaluation/    对齐评测与匿名复核
        ├── rewards        算法特定奖励语义
        └── core/          artifact、runtime、adapter、release
```

规则如下：

1. `core/` 不依赖任何训练阶段，也不包含 Morgana 的业务门槛。
2. `evaluation/` 可依赖 `core/` 和通用评测规则，不依赖 Stage 2/3/4。
3. 一个训练阶段不得从另一个训练阶段导入工具函数或异常类型。
4. 跨模块使用公开名称；以下划线开头的名称只允许模块内部使用。
5. v2 的跨阶段模型、Dev、System Prompt 和 SFT adapter 合同集中在
   `experiments/morgana_v2.py`；算法专属合同仍靠近对应阶段。

## 流水线入口

`roleplay` 是新的可发现入口：

```text
roleplay data ...    构建和冻结训练数据
roleplay train ...   执行、发布、下载和复核训练阶段
roleplay eval ...    统一检查或比较输出网格
roleplay infer ...   批量生成模型输出
roleplay chat ...    交互体验
```

旧 console script 仍作为 v2 文档和既有自动化的兼容层，不再作为新增工作入口。

## Artifact 边界

- `data/runs/morgana-v2/` 是可审计、可版本控制的冻结实验记录。
- `output/` 是可重新生成或通过 release 交换的本地运行目录，不进入 Git。
- 每个训练 run 独占目录并用哈希清单、原子写入和安全解包保护。
- release 机制只存在于 `core/release.py`；阶段通过小型 `ReleaseSpec` 声明差异。

## 扩展原则

新增 v3 时先添加独立实验合同和最小数据，再复用现有机制。只有两个以上阶段出现相同机械逻辑
时才继续下沉到 `core/`；不得为了潜在需求创建通用 Stage 框架。

