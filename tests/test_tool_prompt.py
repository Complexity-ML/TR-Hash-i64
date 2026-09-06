import json

from tr_hash_i64.api._completions import CompletionsMixin


def test_tool_system_message_matches_native_training_contract():
    message = CompletionsMixin._tool_system_message(
        [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate arithmetic.",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                    "return": {"type": "string"},
                },
            }
        ]
    )

    assert message is not None
    assert message["role"] == "system"
    prefix = "Available tools:\n"
    assert message["content"].startswith(prefix)
    payload = json.loads(message["content"][len(prefix) :])
    assert payload == [
        {
            "function": {
                "description": "Evaluate arithmetic.",
                "name": "calculator",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
            "type": "function",
        }
    ]


def test_tool_system_message_ignores_empty_or_invalid_tools():
    assert CompletionsMixin._tool_system_message(None) is None
    assert CompletionsMixin._tool_system_message([]) is None
    assert CompletionsMixin._tool_system_message([{"type": "function"}]) is None
