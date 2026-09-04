"""
tr-hash-i64 :: Tool/Function Call Parser

Parses function calls from model-generated text.
Supports OpenAI-compatible tool_calls format.

Extraction strategies:
  1. Native Agentic <|tool_call_start|>...<|tool_call_end|> tags
  2. <tool_call> XML-style tags (common in fine-tuned models)
  3. JSON object with "name" and "arguments" fields
"""

import json
import re
import uuid
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class ToolCall:
    """A parsed function/tool call from model output."""
    id: str
    type: str = "function"
    function_name: str = ""
    function_arguments: str = ""  # JSON string


class ToolCallParser:
    """
    Parse tool/function calls from generated text.

    Tries multiple extraction patterns:
      1. Native Agentic tags: <|tool_call_start|>{...}<|tool_call_end|>
      2. XML tags: <tool_call>{"name": "func", ...}</tool_call>
      3. JSON: {"name": "func", "arguments": {...}}
    """

    def __init__(self, tools: List[Dict]):
        self.tools = tools
        self.function_names = set()
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                self.function_names.add(t["function"]["name"])

    def parse(self, text: str) -> Optional[List[ToolCall]]:
        """
        Try to extract function calls from generated text.

        Returns list of ToolCall objects, or None if no calls found.
        """
        calls = []

        # Strategy 1: native Agentic tool-call delimiters. The Agentic SFT
        # emits arguments before name, so parse JSON rather than relying on
        # field order.
        native_pattern = re.compile(
            r'<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>',
            re.DOTALL,
        )
        for match in native_pattern.finditer(text):
            call = self._parse_json_call(match.group(1))
            if call:
                calls.append(call)

        if calls:
            return calls

        # Strategy 2: <tool_call>...</tool_call> tags
        tag_pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
        for match in tag_pattern.finditer(text):
            call = self._parse_json_call(match.group(1))
            if call:
                calls.append(call)

        if calls:
            return calls

        # Strategy 3: bare JSON objects. JSONDecoder handles nested arguments
        # and accepts either field order.
        decoder = json.JSONDecoder()
        position = 0
        while True:
            start = text.find("{", position)
            if start < 0:
                break
            try:
                data, consumed = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                position = start + 1
                continue
            call = self._tool_call_from_data(data)
            if call:
                calls.append(call)
            position = start + consumed

        return calls if calls else None

    def _parse_json_call(self, text: str) -> Optional[ToolCall]:
        """Try to parse a single JSON function call."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        return self._tool_call_from_data(data)

    def _tool_call_from_data(self, data: object) -> Optional[ToolCall]:
        """Validate decoded JSON and convert it to a tool call."""
        if not isinstance(data, dict):
            return None

        name = data.get("name", "")
        if name not in self.function_names:
            return None

        arguments = data.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            function_name=name,
            function_arguments=arguments,
        )
