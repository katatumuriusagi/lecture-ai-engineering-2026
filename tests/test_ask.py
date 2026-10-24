"""質問スクリプトのテスト（LLMもAPIキーも使わない）。"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

from app import ask as ask_module
from app.config import MODEL


def test_main_keeps_chat_history_across_questions(monkeypatch, capsys):
    responses = [
        SimpleNamespace(
            text="最初の回答",
            usage_metadata=SimpleNamespace(
                prompt_token_count=2,
                candidates_token_count=3,
            ),
        ),
        SimpleNamespace(
            text="続きの回答",
            usage_metadata=SimpleNamespace(
                prompt_token_count=4,
                candidates_token_count=5,
            ),
        ),
    ]
    chat = SimpleNamespace(send_message=Mock(side_effect=responses))
    client = SimpleNamespace(
        chats=SimpleNamespace(create=Mock(return_value=chat)),
    )
    monkeypatch.setattr(ask_module.genai, "Client", Mock(return_value=client))
    monkeypatch.setattr(ask_module, "get_api_key", Mock(return_value="test-key"))
    monkeypatch.setattr(sys, "argv", ["app.ask", "最初の質問"])
    questions = iter(["続けて質問", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    ask_module.main()

    client.chats.create.assert_called_once_with(model=MODEL)
    assert chat.send_message.call_args_list == [
        call("最初の質問"),
        call("続けて質問"),
    ]
    output = capsys.readouterr().out
    assert "最初の回答" in output
    assert "続きの回答" in output
    assert "[入力トークン数] 2" in output
    assert "[出力トークン数] 5" in output
