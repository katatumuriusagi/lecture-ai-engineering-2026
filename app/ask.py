"""第1回: 会話履歴を保ちながら素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 既定の質問から会話を始める

続けて質問すると、同じ会話の履歴を使って回答する。
/exit または Ctrl+D で終了する。
ツールを持たない「素の」呼び出しなので、講座に固有の質問には答えられないことがある。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai.chats import Chat
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(chat: Chat, question: str) -> str:
    response = chat.send_message(question)

    usage = response.usage_metadata
    if usage is None:
        raise RuntimeError("トークン使用量を取得できませんでした")
    if usage.prompt_token_count is None or usage.candidates_token_count is None:
        raise RuntimeError("入力または出力のトークン数を取得できませんでした")
    print(f"[入力トークン数] {usage.prompt_token_count}")
    print(f"[出力トークン数] {usage.candidates_token_count}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)

    print(f"[model] {MODEL}")
    question = args.question
    while True:
        if question is None:
            try:
                question = input("\nあなた: ").strip()
            except EOFError:
                print("\n終了します。")
                break

        if question in {"/exit", "/quit"}:
            print("終了します。")
            break
        if not question:
            question = None
            continue

        print(f"[質問] {question}\n")
        try:
            print(ask(chat, question))
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)
        question = None


if __name__ == "__main__":
    main()
