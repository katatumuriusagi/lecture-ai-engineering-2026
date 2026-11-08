
import json

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent


@pytest.mark.parametrize("approved", [False, True])
def test_learning_log_approval(
    tmp_path, monkeypatch, approved
):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LOG_DIR", log_dir)

    model = TestModel(call_tools=["save_learning_log"])

    with agent.override(model=model):
        first = agent.run_sync("学習内容を保存して")

        # 承認待ちになっていること
        assert isinstance(first.output, DeferredToolRequests)
        assert len(first.output.approvals) == 1

        call = first.output.approvals[0]
        assert call.tool_name == "save_learning_log"

        log_file = log_dir / "learning_log.jsonl"

        # 承認前に保存されていないこと
        assert not log_file.exists()

        decisions = first.output.build_results(
            approvals={call.tool_call_id: approved}
        )

        second = agent.run_sync(
            message_history=first.all_messages(),
            deferred_tool_results=decisions,
        )

    assert not isinstance(second.output, DeferredToolRequests)

    if approved:
        assert log_file.exists()
        records = [
            json.loads(line)
            for line in log_file.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert len(records) == 1
        assert records[0]["content"]
    else:
        # 拒否された場合は保存されない
        assert not log_file.exists()
