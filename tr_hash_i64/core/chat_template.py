"""
tr-hash-i64 :: Chat Template

Apply chat templates to messages for conversational models.
Loads Jinja2 templates from checkpoint directories.

INL - 2025
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger("tr_hash_i64.chat_template")


class ChatTemplate:
    """
    Chat template renderer.

    Loads a Jinja2 template (like the one in pacific-prime-chat)
    and renders messages into a prompt string.
    """

    def __init__(self, template_str: str):
        from jinja2 import Template
        self.template = Template(template_str)

    def apply(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Render messages into a prompt string.

        Args:
            messages: [{"role": "user", "content": "..."}, ...]
            add_generation_prompt: append assistant turn marker

        Returns:
            formatted prompt string
        """
        return self.template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def from_file(path: str) -> "ChatTemplate":
        """Load template from a .jinja file."""
        with open(path, "r", encoding="utf-8") as f:
            return ChatTemplate(f.read())


def find_chat_template(checkpoint_path: str) -> Optional[str]:
    """Find the chat template shipped with a model checkpoint.

    Hugging Face commonly stores the template inside
    ``tokenizer_config.json`` rather than as a standalone Jinja file.  Both
    layouts are model artifacts and must be honored by the chat endpoint.
    """

    source = Path(checkpoint_path).expanduser()
    search_dir = source.parent if source.is_file() else source
    for _ in range(4):
        for name in ("chat_template.jinja", "chat_template.j2", "template.jinja"):
            path = search_dir / name
            if path.is_file():
                logger.info("chat_template: %s", path)
                return path.read_text(encoding="utf-8")

        tokenizer_config = search_dir / "tokenizer_config.json"
        if tokenizer_config.is_file():
            try:
                template = json.loads(
                    tokenizer_config.read_text(encoding="utf-8")
                ).get("chat_template")
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Invalid tokenizer config %s: %s", tokenizer_config, exc)
            else:
                if isinstance(template, str) and template.strip():
                    logger.info("chat_template: %s#chat_template", tokenizer_config)
                    return template

        parent = search_dir.parent
        if parent == search_dir:
            break
        search_dir = parent
    return None


def load_chat_template(model_name: str) -> Optional[ChatTemplate]:
    """
    Load chat template for a registered model.

    Looks for chat_template.jinja next to config.json.
    """
    from tr_hash_i64.core.registry import get_model_entry

    entry = get_model_entry(model_name)
    if not entry.config_path:
        return None

    config_dir = os.path.dirname(entry.config_path)

    # Look for chat_template.jinja
    for name in ["chat_template.jinja", "chat_template.j2", "template.jinja"]:
        path = os.path.join(config_dir, name)
        if os.path.exists(path):
            logger.info("chat_template: %s", path)
            return ChatTemplate.from_file(path)

    # Try parent directory
    parent_dir = os.path.dirname(config_dir)
    for name in ["chat_template.jinja", "chat_template.j2"]:
        path = os.path.join(parent_dir, name)
        if os.path.exists(path):
            logger.info("chat_template: %s", path)
            return ChatTemplate.from_file(path)

    return None
