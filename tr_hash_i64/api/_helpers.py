"""
tr-hash-i64 :: Server Helpers Mixin

Tokenize/detokenize, chat template, image processing, response building.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional

from aiohttp import web

from tr_hash_i64.core.logging import get_logger
from tr_hash_i64.core.context_manager import (
    ContextManager,
    ContextPlan,
    sanitize_assistant_reasoning,
)
from tr_hash_i64.api.types import CompletionResponse
from tr_hash_i64.engine.i64_engine import GenerationResult

logger = get_logger("tr_hash_i64.server")


class HelpersMixin:
    """Shared helpers: tokenization, chat template, image pre-processing, response building."""

    _NATIVE_CHAT_MARKERS = (
        "<|think_start|>",
        "<|think_end|>",
        "<|final_start|>",
        "<|final_end|>",
    )

    # ------------------------------------------------------------------
    # Request ID
    # ------------------------------------------------------------------

    def _next_request_id(self) -> str:
        n = next(self._request_counter)
        self.request_counter = n
        return f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[int]:
        if self.tokenizer:
            return self.tokenizer.encode(text)
        return [int(b) for b in text.encode("utf-8")]

    async def _tokenize_async(self, text: str) -> List[int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._tokenize_pool, self._tokenize, text)

    def _detokenize(
        self,
        token_ids: List[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        if self.tokenizer:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
            )
        safe_ids = [t & 0xFF for t in token_ids]
        return bytes(safe_ids).decode("utf-8", errors="replace")

    def _native_chat_marker_ids(self) -> Dict[int, str]:
        """Return only markers that are exact registered tokenizer tokens."""
        if self.tokenizer is None:
            return {}
        token_to_id = getattr(self.tokenizer, "token_to_id", None)
        if token_to_id is None:
            backend = getattr(self.tokenizer, "tokenizer", None)
            token_to_id = getattr(backend, "token_to_id", None)
        if token_to_id is None:
            return {}

        markers: Dict[int, str] = {}
        for token in self._NATIVE_CHAT_MARKERS:
            token_id = token_to_id(token)
            if token_id is not None:
                markers[int(token_id)] = token
        return markers

    def _chat_stop_token_ids(self) -> List[int]:
        marker_ids = self._native_chat_marker_ids()
        return [
            token_id
            for token_id, token in marker_ids.items()
            if token == "<|final_end|>"
        ]

    def _detokenize_chat(self, token_ids: List[int]) -> str:
        """Decode chat text while preserving registered envelope markers.

        All other registered special tokens remain hidden. Segment-wise
        decoding avoids exposing BOS/EOS/PAD while keeping the four chat
        control tokens available to the UI parser.
        """
        markers = self._native_chat_marker_ids()
        if not markers:
            return self._detokenize(token_ids)

        output: List[str] = []
        text_ids: List[int] = []

        def flush_text() -> None:
            if text_ids:
                output.append(self._detokenize(text_ids))
                text_ids.clear()

        for token_id in token_ids:
            marker = markers.get(int(token_id))
            if marker is None:
                text_ids.append(int(token_id))
                continue
            flush_text()
            output.append(marker)
        flush_text()
        return "".join(output)

    async def _detokenize_async(self, token_ids: List[int]) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._tokenize_pool, self._detokenize, token_ids)

    def _stream_text_delta(
        self,
        token_ids: List[int],
        emitted_text: str,
        *,
        final: bool = False,
    ) -> tuple[str, str]:
        """Decode a stable streaming suffix without leaking partial UTF-8."""
        decoded = self._detokenize(token_ids)
        stable_text = decoded if final else decoded.rstrip("\ufffd")
        return stable_text[len(emitted_text):], stable_text

    def _stream_chat_text_delta(
        self,
        token_ids: List[int],
        emitted_text: str,
        *,
        final: bool = False,
    ) -> tuple[str, str]:
        decoded = self._detokenize_chat(token_ids)
        stable_text = decoded if final else decoded.rstrip("\ufffd")
        return stable_text[len(emitted_text):], stable_text

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    parts.append("<image>")
            return "\n".join(parts) if parts else ""
        return str(content) if content else ""

    @staticmethod
    def _extract_images_from_messages(messages: List[Dict]) -> list:
        images = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") != "image_url":
                    continue
                image_url = item.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                if not url:
                    continue
                try:
                    if url.startswith("data:"):
                        import base64
                        import io
                        from PIL import Image
                        _, b64_data = url.split(",", 1)
                        image = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
                        images.append(image)
                    else:
                        logger.warning("Non-base64 image URLs not supported: %s...", url[:60])
                except Exception as e:
                    logger.error("Failed to decode image: %s", e)
        return images

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def _normalize_chat_messages(self, messages: List[Dict]) -> List[Dict[str, str]]:
        normalized = []
        for message in messages:
            role = message.get("role", "user")
            content = self._extract_content_text(message.get("content", ""))
            if role == "assistant":
                content = sanitize_assistant_reasoning(content)
                if not content:
                    continue
            normalized.append({"role": role, "content": content})
        return normalized

    def _render_chat_template(self, normalized: List[Dict[str, str]]) -> str:
        if self.chat_template:
            from jinja2 import Template
            eos_token = ""
            bos_token = ""
            pad_token = ""
            if self.tokenizer is not None:
                backend = getattr(self.tokenizer, "tokenizer", None)
                if backend is not None:
                    eos_token = backend.id_to_token(self.tokenizer.eos_token_id) or ""
                    bos_token = backend.id_to_token(self.tokenizer.bos_token_id) or ""
                    pad_token = backend.id_to_token(self.tokenizer.pad_token_id) or ""
            prompt = Template(self.chat_template).render(
                messages=normalized,
                add_generation_prompt=True,
                eos_token=eos_token,
                bos_token=bos_token,
                pad_token=pad_token,
            )
            logger.info("[CHAT] Jinja template applied")
            return prompt

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    normalized, tokenize=False, add_generation_prompt=True,
                )
                logger.info("[CHAT] HF tokenizer template applied")
                return prompt
            except Exception as e:
                logger.debug("[CHAT] HF apply_chat_template failed: %s", e)

        # Pre-train model — preserve message boundaries without introducing an
        # instruction/chat template the checkpoint was not trained to follow.
        return "\n\n".join(
            content
            for message in normalized
            if (content := message.get("content", "").strip())
        )

    def _apply_chat_template(self, messages: List[Dict]) -> str:
        return self._render_chat_template(self._normalize_chat_messages(messages))

    def _prepare_chat_context_sync(
        self,
        messages: List[Dict],
        max_output_tokens: int,
        max_seq_len: int,
    ) -> ContextPlan:
        manager = ContextManager(
            encode=self._tokenize,
            decode=self._detokenize,
            render=self._render_chat_template,
            max_seq_len=max_seq_len,
            compact_at_tokens=self.context_compact_tokens,
        )
        return manager.fit(
            self._normalize_chat_messages(messages),
            max_output_tokens=max_output_tokens,
        )

    async def _prepare_chat_context(
        self,
        messages: List[Dict],
        max_output_tokens: int,
        max_seq_len: int,
    ) -> ContextPlan:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._tokenize_pool,
            self._prepare_chat_context_sync,
            messages,
            max_output_tokens,
            max_seq_len,
        )

    @staticmethod
    def _chat_stop_sequences(user_stop: Optional[List[str]] = None) -> List[str]:
        return list(user_stop) if user_stop else []

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def _preprocess_images(self, images: list):
        import torch
        model = getattr(self.sync_engine, 'model', None)
        if model is not None and hasattr(model, 'vision_encoder') and model.vision_encoder is not None:
            return torch.cat([model.vision_encoder.preprocess_image(img) for img in images], dim=0)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        return torch.cat([transform(img).unsqueeze(0) for img in images], dim=0)

    # ------------------------------------------------------------------
    # Response building
    # ------------------------------------------------------------------

    def _build_response(
        self,
        result: GenerationResult,
        prompt_ids: List[int],
        *,
        chat_response: bool = False,
    ) -> CompletionResponse:
        output_text = (
            self._detokenize_chat(result.output_tokens)
            if chat_response
            else self._detokenize(result.output_tokens)
        )
        choice = {"text": output_text, "index": 0, "finish_reason": result.finish_reason}
        if result.token_logprobs:
            choice["logprobs"] = {
                "tokens": [self._detokenize([lp.token_id]) for lp in result.token_logprobs],
                "token_logprobs": [lp.logprob for lp in result.token_logprobs],
                "top_logprobs": [lp.top_logprobs for lp in result.token_logprobs],
            }
        resp = CompletionResponse(
            id=self._next_request_id(),
            created=int(time.time()),
            model=self.model_name,
            choices=[choice],
        )
        resp._usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(result.output_tokens),
            "total_tokens": len(prompt_ids) + len(result.output_tokens),
        }
        resp._engine_metrics = {
            "engine_steps": result.num_steps,
            "elapsed_ms": round(result.elapsed_ms, 2),
        }
        return resp

    # ------------------------------------------------------------------
    # Admin auth
    # ------------------------------------------------------------------

    def _require_admin(self, request: web.Request) -> Optional[web.Response]:
        if not self.api_key:
            return None
        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else None
        if token != self.api_key:
            return web.json_response(
                {"error": {"message": "Admin endpoint requires valid API key", "type": "auth_error"}},
                status=403,
            )
        return None
