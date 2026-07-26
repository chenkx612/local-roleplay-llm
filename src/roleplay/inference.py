"""Batch inference: send eval questions to a local model and save responses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from .persona import PersonaValidationError, load_persona, render_persona_prompt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "mlx-community/Qwen3.5-2B-4bit"
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
MAX_TOKENS = 256
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 20


def generate(client: OpenAI, model: str, messages: list[dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        extra_body={
            "top_k": TOP_K,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response.choices[0].message.content or ""


def run_baseline(
    persona_path: Path,
    eval_path: Path,
    output_path: Path,
    model: str,
    base_url: str,
) -> None:
    persona = load_persona(persona_path)
    system_prompt = render_persona_prompt(persona)

    questions: list[str] = []
    with eval_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                questions.append(record["user"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"eval 文件第 {line_no} 行格式无效: {exc}") from exc

    client = OpenAI(base_url=base_url, api_key="none")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(questions)
    with output_path.open("w", encoding="utf-8") as out:
        for i, question in enumerate(questions, 1):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            try:
                answer = generate(client, model, messages)
            except Exception as exc:
                print(f"[{i}/{total}] 失败: {exc}", file=sys.stderr)
                answer = ""

            record = {"user": question, "assistant": answer}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{total}] {question[:30]}...")

    print(f"完成，共 {total} 条，输出: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量推理，生成基线或模型输出")
    parser.add_argument(
        "--persona", type=Path, default=ROOT / "data/persona.json", help="角色设定文件"
    )
    parser.add_argument(
        "--eval", type=Path, default=ROOT / "data/eval.jsonl", help="评测问题文件"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/baseline_outputs.jsonl", help="输出文件"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI 兼容 API 地址")
    args = parser.parse_args()

    try:
        run_baseline(args.persona, args.eval, args.output, args.model, args.base_url)
    except (PersonaValidationError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
