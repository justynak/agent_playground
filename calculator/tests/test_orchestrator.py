import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("DEEPSEEK_API_KEY", "test")  # prevent import-time crash when key is absent

from unittest.mock import patch, MagicMock
import orchestrator


def _llm_response(tool_name, tool_args, call_id="tc1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(tool_args)
    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.content = None
    return MagicMock(choices=[MagicMock(message=msg)])


def test_bail_after_3_consecutive_errors():
    # LLM keeps passing a modified expression to prefilter_syntax —
    # dispatch_tool returns an error each time, triggering the consecutive-error bail.
    bad = _llm_response("prefilter_syntax", {"expression": "MODIFIED"})
    with patch.object(
        orchestrator.client.chat.completions, "create", side_effect=[bad, bad, bad]
    ) as mock_create:
        result = orchestrator.run("2+3")

    assert mock_create.call_count == 3  # bailed before the 20-iteration limit
    assert result.startswith("Aborted:")
