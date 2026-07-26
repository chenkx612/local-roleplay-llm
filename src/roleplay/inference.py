"""Batch inference: send eval questions to a local model and save responses."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

from .persona import PersonaValidationError, load_persona, render_persona_prompt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "mlx-community/Qwen3.5-2B-4bit"
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
PRESENCE_PENALTY = 1.5
PRESENCE_CONTEXT_SIZE = 64
REPETITION_PENALTY = 1.1
REPETITION_CONTEXT_SIZE = 64
MAX_ATTEMPTS = 3
REPETITION_PATTERN = re.compile(r"(.{1,20}?)\1{3,}")


class BaselineGenerationError(RuntimeError):
    """Raised when a baseline answer remains invalid after all attempts."""


def generate(
    client: OpenAI, model: str, messages: list[dict[str, str]]
) -> tuple[str, str]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        presence_penalty=PRESENCE_PENALTY,
        extra_body={
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "repetition_context_size": REPETITION_CONTEXT_SIZE,
            "presence_context_size": PRESENCE_CONTEXT_SIZE,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    choice = response.choices[0]
    return choice.message.content or "", choice.finish_reason or ""


def validate_answer(answer: str, finish_reason: str) -> str | None:
    """Return an error message for an invalid answer, otherwise None."""
    if not answer.strip():
        return "回答为空"
    if finish_reason != "stop":
        return f"finish_reason={finish_reason or 'missing'}"

    compact = re.sub(r"\s+", "", answer)
    match = REPETITION_PATTERN.search(compact)
    if match:
        return f"检测到连续复读: {match.group(1)!r}"
    return None


def generate_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    *,
    item_label: str,
) -> tuple[str, str, int]:
    """Generate one valid answer, retrying API and validation failures."""
    last_error = "未知错误"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            answer, finish_reason = generate(client, model, messages)
            validation_error = validate_answer(answer, finish_reason)
            if validation_error is None:
                return answer, finish_reason, attempt
            last_error = validation_error
        except Exception as exc:
            last_error = f"API 调用失败: {exc}"

        print(
            f"{item_label} 第 {attempt}/{MAX_ATTEMPTS} 次尝试失败: {last_error}",
            file=sys.stderr,
        )

    raise BaselineGenerationError(
        f"{item_label} 连续 {MAX_ATTEMPTS} 次生成失败: {last_error}"
    )


def run_baseline(
    persona_path: Path,
    eval_path: Path,
    output_path: Path,
    model: str,
    base_url: str,
    client: OpenAI | None = None,
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

    if client is None:
        client = OpenAI(base_url=base_url, api_key="none")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(questions)
    records: list[dict[str, object]] = []
    for i, question in enumerate(questions, 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        answer, finish_reason, attempts = generate_with_retry(
            client,
            model,
            messages,
            item_label=f"[{i}/{total}]",
        )
        records.append(
            {
                "user": question,
                "assistant": answer,
                "finish_reason": finish_reason,
                "attempts": attempts,
            }
        )
        print(f"[{i}/{total}] {question[:30]}...")

    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

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
    except (PersonaValidationError, ValueError, BaselineGenerationError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
