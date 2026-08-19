"""
tr-hash-i64 :: Context Window Manager

Token-aware rolling context for OpenAI-format chat messages.

The manager is deliberately deterministic and local.  It never makes a
recursive model call: older turns are reduced to a compact extractive summary,
recent turns stay verbatim, and the final rendered prompt is measured against
the real tokenizer before generation starts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_FINAL_BLOCK_RE = re.compile(r"<final>(.*?)</final>", re.IGNORECASE | re.DOTALL)
_FINAL_OPEN_RE = re.compile(r"<final>", re.IGNORECASE)


def sanitize_assistant_reasoning(content: str) -> str:
    """Keep only model-facing final content from an assistant history turn.

    Hidden reasoning is useful while producing one answer but must not become
    evidence for the next turn.  A truncated ``<think>`` is especially unsafe:
    replaying it makes the next generation continue the old chain instead of
    answering the newest user message.
    """

    final_blocks = _FINAL_BLOCK_RE.findall(content)
    if final_blocks:
        return "\n".join(block.strip() for block in final_blocks if block.strip())

    final_open = _FINAL_OPEN_RE.search(content)
    if final_open:
        return content[final_open.end():].replace("</final>", "").strip()

    # Remove complete thought blocks.  If a thought was truncated, discard the
    # unfinished suffix rather than feeding it back as conversational context.
    visible = _THINK_BLOCK_RE.sub("", content)
    think_open = _THINK_OPEN_RE.search(visible)
    if think_open:
        visible = visible[:think_open.start()]
    return visible.replace("</think>", "").strip()


class ContextWindowError(ValueError):
    """Raised when even the essential chat context cannot fit."""


@dataclass(frozen=True)
class ContextPlan:
    """A rendered chat prompt and the measurements used to produce it."""

    messages: List[Dict[str, str]]
    prompt: str
    prompt_token_ids: List[int]
    max_seq_len: int
    reserved_output_tokens: int
    original_messages: int
    retained_messages: int
    summarized_messages: int
    dropped_messages: int
    original_tokens: int
    summary_tokens: int

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.prompt_tokens)

    @property
    def compressed(self) -> bool:
        return self.original_tokens != self.prompt_tokens or self.dropped_messages > 0

    def to_metrics(self) -> dict:
        return {
            "compressed": self.compressed,
            "max_seq_len": self.max_seq_len,
            "reserved_output_tokens": self.reserved_output_tokens,
            "available_prompt_tokens": self.max_seq_len - self.reserved_output_tokens,
            "original_messages": self.original_messages,
            "retained_messages": self.retained_messages,
            "summarized_messages": self.summarized_messages,
            "dropped_messages": self.dropped_messages,
            "original_tokens": self.original_tokens,
            "prompt_tokens": self.prompt_tokens,
            "summary_tokens": self.summary_tokens,
            "tokens_saved": self.tokens_saved,
        }


class ContextManager:
    """
    Fit chat messages inside a model context window.

    Policy:
      1. Keep system messages.
      2. Keep the two newest user turns verbatim when possible.
      3. Convert older turns into a bounded extractive summary.
      4. Drop the oldest retained turn only when the exact rendered prompt is
         still too large.
      5. As a last resort, token-truncate essential content while preserving
         both its beginning and end.
    """

    SUMMARY_PREFIX = "Conversation summary of older turns:"
    TRUNCATION_MARKER = "\n[…]\n"

    def __init__(
        self,
        *,
        encode: Callable[[str], List[int]],
        decode: Callable[[List[int]], str],
        render: Callable[[List[Dict[str, str]]], str],
        max_seq_len: int,
        recent_turns: int = 2,
        max_summary_tokens: int = 256,
    ):
        self.encode = encode
        self.decode = decode
        self.render = render
        self.max_seq_len = int(max_seq_len)
        self.recent_turns = max(1, int(recent_turns))
        self.max_summary_tokens = max(16, int(max_summary_tokens))

    def fit(
        self,
        messages: Sequence[Dict],
        *,
        max_output_tokens: int,
    ) -> ContextPlan:
        if max_output_tokens < 1:
            raise ContextWindowError("max_tokens must be >= 1")
        prompt_budget = self.max_seq_len - max_output_tokens
        if prompt_budget < 1:
            raise ContextWindowError(
                f"max_tokens leaves no room for a prompt in the {self.max_seq_len}-token context window"
            )

        normalized = self._normalize(messages)
        if not normalized:
            raise ContextWindowError("messages must not be empty")

        original_prompt, original_ids = self._render_and_encode(normalized)
        if len(original_ids) <= prompt_budget:
            return ContextPlan(
                messages=normalized,
                prompt=original_prompt,
                prompt_token_ids=original_ids,
                max_seq_len=self.max_seq_len,
                reserved_output_tokens=max_output_tokens,
                original_messages=len(normalized),
                retained_messages=len(normalized),
                summarized_messages=0,
                dropped_messages=0,
                original_tokens=len(original_ids),
                summary_tokens=0,
            )

        systems = [message for message in normalized if message["role"] == "system"]
        conversation = [message for message in normalized if message["role"] != "system"]
        turns = self._group_turns(conversation)
        recent_turns = turns[-self.recent_turns:] if turns else []
        older_turns = turns[:-len(recent_turns)] if recent_turns else turns

        # Always preserve at least the newest turn. Older retained turns may be
        # rolled into the summary if the exact template overhead is too large.
        while True:
            recent_messages = [message for turn in recent_turns for message in turn]
            older_messages = [message for turn in older_turns for message in turn]
            summary_budget = min(
                self.max_summary_tokens,
                max(16, prompt_budget // 4),
            )
            summary, summarized_count, summary_dropped = self._build_summary(
                older_messages,
                summary_budget,
            )
            candidate = systems + ([summary] if summary else []) + recent_messages
            prompt, prompt_ids = self._render_and_encode(candidate)

            if len(prompt_ids) <= prompt_budget:
                summary_tokens = len(self.encode(summary["content"])) if summary else 0
                return ContextPlan(
                    messages=candidate,
                    prompt=prompt,
                    prompt_token_ids=prompt_ids,
                    max_seq_len=self.max_seq_len,
                    reserved_output_tokens=max_output_tokens,
                    original_messages=len(normalized),
                    retained_messages=len(systems) + len(recent_messages),
                    summarized_messages=summarized_count,
                    dropped_messages=summary_dropped,
                    original_tokens=len(original_ids),
                    summary_tokens=summary_tokens,
                )

            if summary:
                summary, summarized_count, summary_dropped = self._shrink_summary_to_fit(
                    systems,
                    recent_messages,
                    older_messages,
                    prompt_budget,
                    summary_budget,
                )
                candidate = systems + ([summary] if summary else []) + recent_messages
                prompt, prompt_ids = self._render_and_encode(candidate)
                if len(prompt_ids) <= prompt_budget:
                    summary_tokens = len(self.encode(summary["content"])) if summary else 0
                    return ContextPlan(
                        messages=candidate,
                        prompt=prompt,
                        prompt_token_ids=prompt_ids,
                        max_seq_len=self.max_seq_len,
                        reserved_output_tokens=max_output_tokens,
                        original_messages=len(normalized),
                        retained_messages=len(systems) + len(recent_messages),
                        summarized_messages=summarized_count,
                        dropped_messages=summary_dropped,
                        original_tokens=len(original_ids),
                        summary_tokens=summary_tokens,
                    )

            if len(recent_turns) > 1:
                older_turns.append(recent_turns.pop(0))
                continue
            break

        # The newest turn and system instructions are essential. If they are
        # individually too large, shrink their contents with a head+tail view.
        essential = systems + ([message for turn in recent_turns for message in turn])
        fitted, prompt, prompt_ids = self._fit_essential_messages(
            essential,
            prompt_budget,
        )
        represented = len(fitted)
        older_count = max(0, len(normalized) - represented)
        return ContextPlan(
            messages=fitted,
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            max_seq_len=self.max_seq_len,
            reserved_output_tokens=max_output_tokens,
            original_messages=len(normalized),
            retained_messages=represented,
            summarized_messages=0,
            dropped_messages=older_count,
            original_tokens=len(original_ids),
            summary_tokens=0,
        )

    @staticmethod
    def _normalize(messages: Sequence[Dict]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "user") or "user")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = str(content) if content is not None else ""
            if role == "assistant":
                content = sanitize_assistant_reasoning(content)
                if not content:
                    continue
            normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _group_turns(messages: Sequence[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        turns: List[List[Dict[str, str]]] = []
        current: List[Dict[str, str]] = []
        for message in messages:
            if message["role"] == "user" and current:
                turns.append(current)
                current = []
            current.append(message)
        if current:
            turns.append(current)
        return turns

    def _render_and_encode(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, List[int]]:
        prompt = self.render(messages)
        return prompt, self.encode(prompt)

    @staticmethod
    def _snippet(text: str, limit: int = 240) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        head = max(1, int(limit * 0.68))
        tail = max(1, limit - head - 5)
        return f"{compact[:head].rstrip()} […] {compact[-tail:].lstrip()}"

    def _build_summary(
        self,
        messages: Sequence[Dict[str, str]],
        token_budget: int,
    ) -> Tuple[Dict[str, str] | None, int, int]:
        if not messages or token_budget < 8:
            return None, 0, len(messages)

        # Prefer the newest old messages: they are the bridge into the recent
        # verbatim turns. Lines are restored to chronological order.
        selected: List[str] = []
        included = 0
        for message in reversed(messages):
            role = message["role"].capitalize()
            compact = re.sub(r"\s+", " ", message["content"]).strip()
            if not compact:
                continue

            # Find the longest extractive head+tail snippet that lets this
            # message fit. This avoids degrading to an "N messages omitted"
            # marker merely because every original turn is long.
            best_line = None
            low, high = 1, min(240, len(compact))
            while low <= high:
                limit = (low + high) // 2
                line = f"- {role}: {self._snippet(compact, limit)}"
                trial = [line] + selected
                omitted = len(messages) - (included + 1)
                suffix = (
                    f"\n- [{omitted} earlier message{'s' if omitted != 1 else ''} omitted.]"
                    if omitted
                    else ""
                )
                text = f"{self.SUMMARY_PREFIX}\n" + "\n".join(trial) + suffix
                if len(self.encode(text)) <= token_budget:
                    best_line = line
                    low = limit + 1
                else:
                    high = limit - 1
            if best_line is None:
                continue
            selected = [best_line] + selected
            included += 1

        dropped = max(0, len(messages) - included)
        if not selected:
            marker = f"{self.SUMMARY_PREFIX}\n- [{len(messages)} earlier messages omitted.]"
            marker_ids = self.encode(marker)
            if len(marker_ids) > token_budget:
                marker = self.decode(marker_ids[:token_budget])
            return {"role": "system", "content": marker}, 0, len(messages)

        suffix = f"\n- [{dropped} earlier message{'s' if dropped != 1 else ''} omitted.]" if dropped else ""
        content = f"{self.SUMMARY_PREFIX}\n" + "\n".join(selected) + suffix
        return {"role": "system", "content": content}, included, dropped

    def _shrink_summary_to_fit(
        self,
        systems: List[Dict[str, str]],
        recent_messages: List[Dict[str, str]],
        older_messages: List[Dict[str, str]],
        prompt_budget: int,
        initial_budget: int,
    ) -> Tuple[Dict[str, str] | None, int, int]:
        budget = initial_budget
        while budget >= 8:
            summary, included, dropped = self._build_summary(older_messages, budget)
            candidate = systems + ([summary] if summary else []) + recent_messages
            _, prompt_ids = self._render_and_encode(candidate)
            if len(prompt_ids) <= prompt_budget:
                return summary, included, dropped
            overflow = len(prompt_ids) - prompt_budget
            budget -= max(8, overflow)
        return None, 0, len(older_messages)

    def _fit_essential_messages(
        self,
        messages: List[Dict[str, str]],
        prompt_budget: int,
    ) -> Tuple[List[Dict[str, str]], str, List[int]]:
        fitted = [dict(message) for message in messages]
        for _ in range(64):
            prompt, prompt_ids = self._render_and_encode(fitted)
            if len(prompt_ids) <= prompt_budget:
                return fitted, prompt, prompt_ids

            overflow = len(prompt_ids) - prompt_budget
            candidates = [
                (index, len(self.encode(message["content"])))
                for index, message in enumerate(fitted)
                if message["content"]
            ]
            if not candidates:
                break
            index, content_tokens = max(candidates, key=lambda item: item[1])
            if content_tokens <= 4:
                break
            target = max(4, content_tokens - overflow - 4)
            fitted[index]["content"] = self._head_tail_tokens(
                fitted[index]["content"],
                target,
            )

        prompt, prompt_ids = self._render_and_encode(fitted)
        if len(prompt_ids) > prompt_budget:
            raise ContextWindowError(
                f"essential chat template requires {len(prompt_ids)} prompt tokens but only "
                f"{prompt_budget} are available"
            )
        return fitted, prompt, prompt_ids

    def _head_tail_tokens(self, text: str, token_budget: int) -> str:
        ids = self.encode(text)
        if len(ids) <= token_budget:
            return text
        marker_ids = self.encode(self.TRUNCATION_MARKER)
        content_budget = max(1, token_budget - len(marker_ids))
        head_count = max(1, int(content_budget * 0.65))
        tail_count = max(0, content_budget - head_count)
        kept = ids[:head_count]
        if tail_count:
            kept = kept + marker_ids + ids[-tail_count:]
        return self.decode(kept)
