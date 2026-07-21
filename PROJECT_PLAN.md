# 本地角色扮演 LLM 计划

## 目标

在 M4、16GB MacBook Air 上部署 `Qwen3.5-2B`，先用 System Prompt 和 few-shot 验证其原生中文角色扮演能力，再决定是否微调。

目标风格来自 `data.json`：简短、口语化的微信聊天，使用换行模拟连续消息，偶尔出现“好滴”“好勒”“蟹蟹”“俺们”“哈哈哈”等表达。

## 第一阶段范围

模型：

```text
mlx-community/Qwen3.5-2B-4bit
```

本阶段只做：

- 本地部署模型；
- 对比裸模型、System Prompt、System Prompt + few-shot；
- 记录回复质量、内存占用和响应速度。

暂不做微调、视觉输入、长期记忆和聊天 UI。

## 部署

复用已有的 `llm` Conda 环境：

```bash
cd /Users/chenkx/roleplay
conda activate llm
python -m pip install -U mlx-lm openai
```

启动服务：

```bash
mlx_lm.server \
  --model mlx-community/Qwen3.5-2B-4bit \
  --port 8080
```

接口地址：

```text
http://127.0.0.1:8080/v1
```

## 当前进度（2026-07-20）

已完成：

- 已复用 `llm` Conda 环境，安装 `mlx-lm 0.31.3`、`MLX 0.32.0` 和 `openai 2.46.0`；
- 已下载 `mlx-community/Qwen3.5-2B-4bit` 权重（1,722,271,785 字节），并通过 SHA-256 校验；
- 已启动并通过本机 `http://127.0.0.1:8080/v1/models` 健康检查；
- 已创建交互脚本 `chat.py`、System Prompt 和虚构且连贯的 few-shot 示例；
- 已确认请求必须传入 `chat_template_kwargs.enable_thinking=false`，否则模型会先生成冗长推理内容。

初步观察：

- 裸模型在关闭 thinking 后仍明显像通用 AI 助手，会生成长回复、表情和解释；
- System Prompt 模式已能生成短回复，但首条测试把“明天下午”错误理解为“今天”，说明还需要用固定测试集评估其可靠性；
- Few-shot 模式的自动验证在上一轮被中断，尚待完成。

下一步：运行 `bare`、`system`、`fewshot` 三种模式的固定测试，记录输出并据此迭代提示词。

## System Prompt

```text
你正在进行中文日常聊天，并扮演示例对话中的女生。

只输出角色发送的聊天内容，不解释、不分析，不使用动作或心理旁白。
使用自然、随意的中文微信聊天语气，回复通常为1到4行，可以用换行模拟连续发送消息。
可以偶尔使用“好滴”“好勒”“俺们”“蟹蟹”“哈哈哈”“不好意思不好意思”等表达，但不要刻意堆砌。
允许少量省略和不完整句子，不要写成正式文章。
先回应当前消息，必要时再追问一句。
不知道的事实不要编造，可以说还没想好、记不清或需要问问别人。
```

## Few-shot

```json
[
  {"role": "user", "content": "明天我们想去烧烤，咱一起去呗"},
  {"role": "assistant", "content": "好滴，我问问他们几个去不去"},
  {"role": "user", "content": "他们都可以，那我们下午出发不"},
  {"role": "assistant", "content": "好滴好滴\n那中午先去买点东西不"},
  {"role": "user", "content": "可以啊，你们怎么来"},
  {"role": "assistant", "content": "我们应该开车下来\n你给我发个定位可以不"},
  {"role": "user", "content": "好，我把定位发群里"},
  {"role": "assistant", "content": "好滴，蟹蟹"},
  {"role": "user", "content": "昨天的照片你发了吗"},
  {"role": "assistant", "content": "还没呢不好意思\n我晚上找找发群里"},
  {"role": "user", "content": "没事，你方便的时候发就行"},
  {"role": "assistant", "content": "好滴好滴"}
]
```

few-shot 只提供风格示范，不应把 `data.json` 全部放入上下文。

## 测试

对每条测试消息分别运行：

1. 裸模型；
2. System Prompt；
3. System Prompt + few-shot。

初始参数：

```text
max_tokens: 96
temperature: 0.8
top_p: 0.9
top_k: 20
```

测试消息使用未在 few-shot 中出现的新场景，例如：

```text
明天下午有空吗，一起喝咖啡？
照片你怎么还没发呀
我们已经到地方了，你到哪了
周末可能下雨，要不改天？
我刚才是不是说话有点重
这周大家都比较忙，聚餐要不要推迟
```

重点检查：

- 是否正确回应当前消息；
- 是否接近目标微信风格；
- 是否保持简短和自然换行；
- 是否过度使用口癖；
- 是否编造事实或复制 few-shot 中的无关内容；
- 是否输出思考过程、旁白或长篇解释；
- 多轮对话中语气是否稳定。

## 验收条件

满足以下条件即可进入数据整理和微调阶段：

- 模型服务能够稳定运行；
- 多任务使用时内存压力可接受；
- System Prompt + few-shot 比裸模型有稳定改善；
- 大多数新场景能够合理回应；
- 回复通常保持1到4行，且没有明显复读和口癖堆砌。

如果2B普遍答非所问，则用同一测试集对比 `Qwen3.5-4B`；如果原生表现已足够好，则优先完善提示词，不急于微调。
