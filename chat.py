import argparse
import json
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parent
MODEL = "mlx-community/Qwen3.5-2B-4bit"


def initial_messages(mode: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    if mode in {"system", "fewshot"}:
        system_prompt = (ROOT / "prompts/system.txt").read_text().strip()
        messages.append({"role": "system", "content": system_prompt})

    if mode == "fewshot":
        few_shot = json.loads((ROOT / "prompts/few_shot.json").read_text())
        messages.extend(few_shot)

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
    args = parser.parse_args()

    client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
    messages = initial_messages(args.mode)

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
            messages = initial_messages(args.mode)
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
