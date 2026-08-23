"""Tests for deterministic rolling chat context management."""

import pytest

from tr_hash_i64.api.tracking import ContextMetricsTracker
from tr_hash_i64.api.types import CompletionRequest
from tr_hash_i64.api._helpers import HelpersMixin
from tr_hash_i64.core.context_manager import (
    ContextManager,
    ContextWindowError,
    sanitize_assistant_reasoning,
)


def _encode(text):
    return list(text.encode("utf-8"))


def _decode(token_ids):
    return bytes(token_ids).decode("utf-8", errors="ignore")


def _render(messages):
    return (
        "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        + "<assistant>"
    )


def _manager(
    max_seq_len=512, recent_turns=2, max_summary_tokens=96, compact_at_tokens=None
):
    return ContextManager(
        encode=_encode,
        decode=_decode,
        render=_render,
        max_seq_len=max_seq_len,
        recent_turns=recent_turns,
        max_summary_tokens=max_summary_tokens,
        compact_at_tokens=compact_at_tokens,
    )


def test_context_under_budget_is_unchanged():
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    plan = _manager().fit(messages, max_output_tokens=64)

    assert plan.messages == messages
    assert plan.compressed is False
    assert plan.summarized_messages == 0
    assert plan.dropped_messages == 0
    assert plan.prompt_tokens + plan.reserved_output_tokens <= plan.max_seq_len


def test_complete_reasoning_history_keeps_only_final_answer():
    content = (
        "<|think_start|>old private chain<|think_end|>\n"
        "<|final_start|>Four.<|final_end|>"
    )

    assert sanitize_assistant_reasoning(content) == "Four."


def test_truncated_reasoning_history_is_not_replayed():
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {
            "role": "assistant",
            "content": "<|think_start|>unfinished old chain",
        },
        {"role": "user", "content": "What is 3 + 3?"},
    ]

    plan = _manager().fit(messages, max_output_tokens=64)

    assert "unfinished old chain" not in plan.prompt
    assert "What is 3 + 3?" in plan.prompt
    assert all(message["role"] != "assistant" for message in plan.messages)


def test_api_normalization_sanitizes_reasoning_before_template_rendering():
    helper = HelpersMixin()
    helper._extract_content_text = lambda content: content

    normalized = helper._normalize_chat_messages(
        [
            {"role": "user", "content": "First question"},
            {
                "role": "assistant",
                "content": (
                    "<|think_start|>private stale chain<|think_end|>"
                    "<|final_start|>First answer<|final_end|>"
                ),
            },
            {"role": "user", "content": "Second question"},
        ]
    )

    assert normalized == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]


def test_old_turns_are_summarized_and_recent_turns_are_kept():
    messages = [{"role": "system", "content": "Always preserve this instruction."}]
    for index in range(8):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Question {index}: " + ("detail " * 10),
                },
                {
                    "role": "assistant",
                    "content": f"Answer {index}: " + ("result " * 10),
                },
            ]
        )

    plan = _manager(max_seq_len=500).fit(messages, max_output_tokens=80)

    assert plan.compressed is True
    assert plan.prompt_tokens + plan.reserved_output_tokens <= plan.max_seq_len
    assert plan.messages[0]["content"] == "Always preserve this instruction."
    assert any(
        "Conversation summary of older turns:" in message["content"]
        for message in plan.messages
    )
    assert any("Question 7:" in message["content"] for message in plan.messages)
    assert any("Answer 7:" in message["content"] for message in plan.messages)
    assert plan.summarized_messages > 0
    assert plan.tokens_saved > 0


def test_context_compacts_at_configured_prompt_threshold():
    messages = []
    for index in range(8):
        messages.extend(
            [
                {"role": "user", "content": f"Question {index}: " + ("detail " * 12)},
                {
                    "role": "assistant",
                    "content": f"Answer {index}: " + ("result " * 12),
                },
            ]
        )

    plan = _manager(max_seq_len=2048, compact_at_tokens=1024).fit(
        messages,
        max_output_tokens=512,
    )

    assert plan.original_tokens > 1024
    assert plan.prompt_tokens + plan.reserved_output_tokens <= plan.max_seq_len
    assert plan.compressed is True
    assert plan.prompt_tokens <= 1024
    assert plan.to_metrics()["compact_at_tokens"] == 1024
    assert plan.to_metrics()["available_prompt_tokens"] == 1024


def test_oversized_latest_message_is_head_tail_truncated():
    content = "BEGIN-" + ("x" * 900) + "-END"
    messages = [
        {"role": "system", "content": "Keep safe."},
        {"role": "user", "content": content},
    ]

    plan = _manager(max_seq_len=240).fit(messages, max_output_tokens=40)

    assert plan.compressed is True
    assert plan.prompt_tokens + plan.reserved_output_tokens <= plan.max_seq_len
    fitted_user = plan.messages[-1]["content"]
    assert fitted_user.startswith("BEGIN-")
    assert fitted_user.endswith("-END")
    assert "[…]" in fitted_user


def test_impossible_template_budget_raises():
    manager = ContextManager(
        encode=_encode,
        decode=_decode,
        render=lambda messages: "TEMPLATE-OVERHEAD",
        max_seq_len=16,
    )

    with pytest.raises(ContextWindowError, match="essential chat template"):
        manager.fit([{"role": "user", "content": ""}], max_output_tokens=4)


def test_context_metrics_tracker_aggregates_compression():
    tracker = ContextMetricsTracker()
    tracker.record(
        {
            "compressed": True,
            "original_tokens": 100,
            "prompt_tokens": 60,
            "summary_tokens": 12,
            "tokens_saved": 40,
            "summarized_messages": 3,
            "dropped_messages": 1,
        }
    )
    tracker.record(
        {
            "compressed": False,
            "original_tokens": 20,
            "prompt_tokens": 20,
        }
    )

    snapshot = tracker.snapshot()
    assert snapshot["requests"] == 2
    assert snapshot["compressed_requests"] == 1
    assert snapshot["original_tokens"] == 120
    assert snapshot["prompt_tokens"] == 80
    assert snapshot["tokens_saved"] == 40
    assert snapshot["compression_ratio"] == pytest.approx(0.6667)


def test_completion_request_validates_total_context_budget():
    request = CompletionRequest(prompt="hello", max_tokens=128)

    assert request.validate(max_seq_len=256, prompt_tokens=128) is None
    error = request.validate(max_seq_len=256, prompt_tokens=129)
    assert "prompt_tokens (129) + max_tokens (128)" in error
