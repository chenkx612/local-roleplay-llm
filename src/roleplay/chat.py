import argparse
import json
from pathlib import Path

from openai import OpenAI

from .persona import PersonaValidationError, load_persona, render_persona_prompt


ROOT = Path(__file__).resolve().parents[2]
MODEL = "mlx-community/Qwen3.5-2B-4bit"


def initial_messages(mode: str, persona_path: Path) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    if mode in {"system", "fewshot"}:
        persona = load_persona(persona_path)
        messages.append({"role": "system", "content": render_persona_prompt(persona)})

    if mode == "fewshot":
        examples_path = persona_path.parent / "style_examples.jsonl"
        with examples_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                    user, assistant = example["user"], example["assistant"]
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"无效样例 {examples_path}:{line_number}") from exc
                messages.extend(
                    ({"role": "user", "content": user}, {"role": "assistant", "content": assistant})
                )

    return messages


def reply(client: OpenAI, messages: list[dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=96,
        temperature=0.8,
        top_p=0.9,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("bare", "system", "fewshot"),
        default="fewshot",
    )
    parser.add_argument("--message", help="发送一条消息后退出")
    parser.add_argument(
        "--persona", type=Path, default=ROOT / "data/persona.json", help="角色设定 JSON 文件"
    )
    args = parser.parse_args()

    client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
    try:
        messages = initial_messages(args.mode, args.persona)
    except (PersonaValidationError, ValueError) as exc:
        parser.error(str(exc))

    if args.message:
        messages.append({"role": "user", "content": args.message})
        print(reply(client, messages))
        return

    print(f"模式：{args.mode}（输入 /quit 退出，/reset 清空对话）")
    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/reset":
            messages = initial_messages(args.mode, args.persona)
            print("对话已清空")
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            assistant_reply = reply(client, messages)
        except Exception as exc:
            messages.pop()
            print(f"请求失败：{exc}")
            continue

        print(f"角色：{assistant_reply}")
        messages.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()
