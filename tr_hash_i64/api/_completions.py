"""
tr-hash-i64 :: Completions Mixin

/v1/completions and /v1/chat/completions handlers + async streaming.
"""

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import replace
from typing import AsyncGenerator, Dict, List, Optional

from aiohttp import web

from tr_hash_i64.core.logging import get_logger
from tr_hash_i64.core.context_manager import ContextWindowError
from tr_hash_i64.api.types import CompletionRequest

logger = get_logger("tr_hash_i64.server")


class CompletionsMixin:

    _THINK_RESPONSE_PREFILL = "<think>\n"
    _THINK_FINAL_TRANSITION = "\n</think>\n<final>\n"

    @classmethod
    def _chat_response_prefill(cls, prompt: str) -> str:
        """Restore a generation prefill in the assistant API content.

        A chat template may end the prompt with ``<think>`` so the model starts
        directly inside its reasoning envelope.  Because those tokens belong
        to the prompt, generation APIs would otherwise omit the opening tag
        while still returning ``</think>`` later in the response.
        """

        return (
            cls._THINK_RESPONSE_PREFILL
            if prompt.endswith(cls._THINK_RESPONSE_PREFILL)
            else ""
        )

    # ------------------------------------------------------------------
    # Core async generation
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_namespace(
        api_key: Optional[str],
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[bytes]:
        """Derive a 16-byte KV cache namespace from tenant and conversation.

        ``conversation_id`` is preferred when supplied. ``user_id`` remains
        supported for OpenAI-compatible clients that use that field as their
        conversation/session identifier. The namespace is deliberately kept
        separate from the rendered prompt: two new chats can have the same
        system/template prefix without sharing KV blocks.
        """
        scope = conversation_id or user_id
        if not api_key and not scope:
            return None
        material = f"{api_key or ''}\0{scope or ''}".encode()
        return hashlib.sha256(material).digest()[:16]

    @staticmethod
    def _chat_conversation_id(request: web.Request, body: dict) -> str:
        """Return the cache scope for a chat request.

        The UI may provide a stable ``user`` field or one of the explicit
        conversation headers. If it provides neither, mint a request-local
        scope. This makes a fresh anonymous chat strictly isolated instead of
        reusing the common system/template prefix from another chat.
        """
        return (
            body.get("conversation_id")
            or body.get("user")
            or request.headers.get("X-Conversation-Id")
            or request.headers.get("X-Session-Id")
            or f"anonymous-chat-{uuid.uuid4().hex}"
        )

    async def _async_complete(
        self,
        request: CompletionRequest,
        api_key: Optional[str] = None,
        endpoint: str = "/v1/completions",
    ):
        t0 = time.monotonic()
        prompt_ids = getattr(request, "_prompt_token_ids", None)
        if prompt_ids is None:
            prompt_ids = await self._tokenize_async(request.prompt)
        pixel_values = getattr(request, '_pixel_values', None)
        ns = self._cache_namespace(api_key, request.user)

        result = await self.async_engine.generate(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
            pixel_values=pixel_values,
            cache_namespace=ns,
        )

        if len(result.output_tokens) <= 3:
            eos_cfg = getattr(self.async_engine.engine.model, 'config', None)
            eos_id = getattr(eos_cfg, 'eos_token_id', '?') if eos_cfg else '?'
            logger.warning(
                "[DEBUG] Short generation: prompt_ids=%s output_tokens=%s eos_config=%s finish=%s",
                prompt_ids[-5:], result.output_tokens, eos_id, result.finish_reason,
            )

        resp = self._build_response(result, prompt_ids)
        context_metrics = getattr(request, "_context_metrics", None)
        if context_metrics is not None:
            resp._context_metrics = context_metrics
        latency_ms = (time.monotonic() - t0) * 1000
        from tr_hash_i64.api.types import compute_partition
        partition = compute_partition(api_key, getattr(request, "user", None))
        self._usage_tracker.record(api_key or "", len(prompt_ids), len(result.output_tokens))
        self._latency_tracker.record(endpoint, latency_ms)
        self._request_logger.log_request(
            endpoint=endpoint, status=200, latency_ms=latency_ms,
            prompt_tokens=len(prompt_ids), completion_tokens=len(result.output_tokens),
            api_key=api_key, request_id=resp.id, partition=partition,
            context_metrics=context_metrics,
        )
        return resp

    async def _async_stream(self, request: CompletionRequest, api_key: Optional[str] = None) -> AsyncGenerator[str, None]:
        prompt_ids = getattr(request, "_prompt_token_ids", None)
        if prompt_ids is None:
            prompt_ids = await self._tokenize_async(request.prompt)
        stream_id = self._next_request_id()
        created = int(time.time())
        output_ids: List[int] = []
        prev_text = ""
        finish_reason = "length"
        ns = self._cache_namespace(api_key, request.user)
        async for item in self.async_engine.generate_stream(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
            cache_namespace=ns,
        ):
            # Engine sends ("__done__", finish_reason) as final sentinel
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__done__":
                finish_reason = item[1]
                break
            output_ids.append(item)
            token_text, prev_text = self._stream_text_delta(output_ids, prev_text)
            if not token_text:
                continue
            yield f"data: {json.dumps({'id': stream_id, 'object': 'text_completion', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'text': token_text, 'finish_reason': None}]})}\n\n"

        token_text, prev_text = self._stream_text_delta(output_ids, prev_text, final=True)
        if token_text:
            yield f"data: {json.dumps({'id': stream_id, 'object': 'text_completion', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'text': token_text, 'finish_reason': None}]})}\n\n"

        yield f"data: {json.dumps({'id': stream_id, 'object': 'text_completion', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'text': '', 'finish_reason': finish_reason}]})}\n\n"
        yield "data: [DONE]\n\n"

    async def _async_chat_stream(
        self, request: CompletionRequest, tools: Optional[list] = None, api_key: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        prompt_ids = getattr(request, "_prompt_token_ids", None)
        if prompt_ids is None:
            prompt_ids = await self._tokenize_async(request.prompt)
        stream_id = self._next_request_id()
        created = int(time.time())
        ns = self._cache_namespace(api_key, request.user)

        initial_chunk = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": getattr(
                            request,
                            "_chat_response_prefill",
                            "",
                        ),
                    },
                    "finish_reason": None,
                }
            ],
        }
        context_metrics = getattr(request, "_context_metrics", None)
        if context_metrics is not None:
            initial_chunk["context_metrics"] = context_metrics
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        output_ids: List[int] = []
        prev_text = ""
        finish_reason = "length"
        pixel_values = getattr(request, '_pixel_values', None)
        async for item in self.async_engine.generate_stream(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
            pixel_values=pixel_values,
            cache_namespace=ns,
        ):
            # Engine sends ("__done__", finish_reason) as final sentinel
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__done__":
                finish_reason = item[1]
                break
            output_ids.append(item)
            token_text, prev_text = self._stream_text_delta(output_ids, prev_text)
            if not token_text:
                continue
            yield f"data: {json.dumps({'id': stream_id, 'object': 'chat.completion.chunk', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': token_text}, 'finish_reason': None}]})}\n\n"

        token_text, prev_text = self._stream_text_delta(output_ids, prev_text, final=True)
        if token_text:
            yield f"data: {json.dumps({'id': stream_id, 'object': 'chat.completion.chunk', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': token_text}, 'finish_reason': None}]})}\n\n"

        response_prefill = getattr(request, "_chat_response_prefill", "")
        if response_prefill and "</think>" not in prev_text:
            transition = self._THINK_FINAL_TRANSITION
            yield f"data: {json.dumps({'id': stream_id, 'object': 'chat.completion.chunk', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': transition}, 'finish_reason': None}]})}\n\n"

            transition_ids = await self._tokenize_async(transition)
            total_budget = getattr(request, "_chat_total_max_tokens", request.max_tokens)
            remaining = total_budget - len(output_ids) - len(transition_ids)
            final_text = ""
            if remaining > 0:
                continuation_prompt = request.prompt + prev_text + transition
                continuation_ids = await self._tokenize_async(continuation_prompt)
                available = self.sync_engine.scheduler.max_seq_len - len(continuation_ids)
                final_budget = min(remaining, max(0, available))
                if final_budget > 0:
                    final_ids: List[int] = []
                    async for item in self.async_engine.generate_stream(
                        prompt_token_ids=continuation_ids,
                        max_new_tokens=final_budget,
                        sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
                        pixel_values=pixel_values,
                        cache_namespace=ns,
                    ):
                        if (
                            isinstance(item, tuple)
                            and len(item) == 2
                            and item[0] == "__done__"
                        ):
                            finish_reason = item[1]
                            break
                        final_ids.append(item)
                        token_text, final_text = self._stream_text_delta(final_ids, final_text)
                        if token_text:
                            yield f"data: {json.dumps({'id': stream_id, 'object': 'chat.completion.chunk', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': token_text}, 'finish_reason': None}]})}\n\n"

                    token_text, final_text = self._stream_text_delta(final_ids, final_text, final=True)
                    if token_text:
                        yield f"data: {json.dumps({'id': stream_id, 'object': 'chat.completion.chunk', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': token_text}, 'finish_reason': None}]})}\n\n"

            if "</final>" not in final_text:
                closing = "\n</final>"
                yield f"data: {json.dumps({'id': stream_id, 'object': 'chat.completion.chunk', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': closing}, 'finish_reason': None}]})}\n\n"

        yield f"data: {json.dumps({'id': stream_id, 'object': 'chat.completion.chunk', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def handle_completions(self, request: web.Request) -> web.Response:
        """POST /v1/completions"""
        if self.async_engine is None:
            return web.json_response({"error": {"message": "No model loaded", "type": "server_error"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)

        prompt = body.get("prompt")
        if not prompt:
            return web.json_response({"error": {"message": "Missing 'prompt'", "type": "invalid_request_error"}}, status=400)

        req = CompletionRequest(
            prompt=prompt,
            max_tokens=body.get("max_tokens", 256),
            temperature=body.get("temperature", 0.8),
            top_k=body.get("top_k", 50),
            top_p=body.get("top_p", 0.9),
            min_p=body.get("min_p", 0.0),
            typical_p=body.get("typical_p", 1.0),
            repetition_penalty=body.get("repetition_penalty", 1.1),
            min_tokens=body.get("min_tokens", 0),
            stream=body.get("stream", False),
            response_format=body.get("response_format"),
            stop=body.get("stop"),
            n=body.get("n", 1),
            best_of=body.get("best_of", 1),
            logprobs=body.get("logprobs"),
            seed=body.get("seed"),
            logit_bias=body.get("logit_bias"),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            priority=body.get("priority", 0),
            suppress_first_tokens=self._space_suppress_ids,
            user=body.get("user"),
        )
        max_seq_len = self.sync_engine.scheduler.max_seq_len
        error = req.validate(max_seq_len=max_seq_len)
        if error:
            return web.json_response({"error": {"message": error, "type": "invalid_request_error"}}, status=400)
        prompt_ids = await self._tokenize_async(req.prompt)
        error = req.validate(max_seq_len=max_seq_len, prompt_tokens=len(prompt_ids))
        if error:
            return web.json_response({"error": {"message": error, "type": "invalid_request_error"}}, status=400)
        req._prompt_token_ids = prompt_ids

        auth = request.headers.get("Authorization", "")
        req_api_key = auth[7:] if auth.startswith("Bearer ") else None

        try:
            if req.stream:
                response = web.StreamResponse()
                response.content_type = "text/event-stream"
                response.headers["Cache-Control"] = "no-cache"
                await response.prepare(request)
                gen = self._async_stream(req, api_key=req_api_key)
                try:
                    async for chunk in gen:
                        await response.write(chunk.encode())
                except (ConnectionResetError, ConnectionError):
                    await gen.aclose()
                return response

            cache_kwargs = dict(
                temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
                min_p=req.min_p, typical_p=req.typical_p,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                seed=req.seed,
            )
            cached = self._request_cache.get(req.prompt, req.max_tokens, **cache_kwargs)
            if cached is not None:
                return web.json_response(cached)

            result = await self._async_complete(req, api_key=req_api_key)
            result_dict = result.to_dict()
            self._request_cache.put(req.prompt, req.max_tokens, result_dict, **cache_kwargs)
            return web.json_response(result_dict)
        except (ConnectionResetError, ConnectionError):
            return web.Response(status=499, text="Client disconnected")
        except Exception as e:
            logger.error("Completion error: %s", e, exc_info=True)
            return web.json_response({"error": {"message": "Internal server error", "type": "server_error"}}, status=500)

    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """POST /v1/chat/completions"""
        if self.async_engine is None:
            return web.json_response({"error": {"message": "No model loaded", "type": "server_error"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)

        messages = body.get("messages")
        if not messages:
            return web.json_response({"error": {"message": "Missing 'messages'", "type": "invalid_request_error"}}, status=400)

        images = self._extract_images_from_messages(messages)
        pixel_values = None
        if images:
            pixel_values = self._preprocess_images(images)

        context_messages = list(messages)
        if body.get("rag") and self.rag_enabled and self.retriever is not None:
            user_query = messages[-1].get("content", "")
            if isinstance(user_query, list):
                user_query = self._extract_content_text(user_query)
            if user_query:
                context = self.retriever.get_context(user_query, k=body.get("rag_k", 3))
                if context:
                    context_messages = [
                        {
                            "role": "system",
                            "content": f"Retrieved context for this request:\n{context}",
                        },
                        *context_messages,
                    ]

        max_tokens = body.get("max_tokens", 256)
        max_seq_len = self.sync_engine.scheduler.max_seq_len
        context_management = body.get("context_management", "auto")
        context_enabled = context_management not in (False, None, "disabled", "off", "none")
        try:
            if context_enabled:
                context_plan = await self._prepare_chat_context(
                    context_messages,
                    max_output_tokens=max_tokens,
                    max_seq_len=max_seq_len,
                )
                prompt = context_plan.prompt
                prompt_ids = context_plan.prompt_token_ids
                context_metrics = context_plan.to_metrics()
                context_metrics["policy"] = "rolling_summary"
            else:
                prompt = self._apply_chat_template(context_messages)
                prompt_ids = await self._tokenize_async(prompt)
                context_metrics = {
                    "compressed": False,
                    "policy": "disabled",
                    "max_seq_len": max_seq_len,
                    "reserved_output_tokens": max_tokens,
                    "available_prompt_tokens": max_seq_len - max_tokens,
                    "original_messages": len(context_messages),
                    "retained_messages": len(context_messages),
                    "summarized_messages": 0,
                    "dropped_messages": 0,
                    "original_tokens": len(prompt_ids),
                    "prompt_tokens": len(prompt_ids),
                    "summary_tokens": 0,
                    "tokens_saved": 0,
                }
        except ContextWindowError as exc:
            return web.json_response(
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                status=400,
            )

        req = CompletionRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=body.get("temperature", 0.8),
            top_k=body.get("top_k", 50),
            top_p=body.get("top_p", 0.9),
            min_p=body.get("min_p", 0.0),
            typical_p=body.get("typical_p", 1.0),
            repetition_penalty=body.get("repetition_penalty", 1.1),
            min_tokens=body.get("min_tokens", 0),
            stream=body.get("stream", False),
            response_format=body.get("response_format"),
            stop=self._chat_stop_sequences(body.get("stop")),
            n=body.get("n", 1),
            best_of=body.get("best_of", 1),
            logprobs=body.get("logprobs"),
            seed=body.get("seed"),
            logit_bias=body.get("logit_bias"),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            priority=body.get("priority", 0),
            suppress_first_tokens=self._space_suppress_ids,
            user=self._chat_conversation_id(request, body),
        )
        req._pixel_values = pixel_values
        req._prompt_token_ids = prompt_ids
        req._context_metrics = context_metrics
        req._chat_response_prefill = self._chat_response_prefill(prompt)

        error = req.validate(max_seq_len=max_seq_len, prompt_tokens=len(prompt_ids))
        if error:
            return web.json_response({"error": {"message": error, "type": "invalid_request_error"}}, status=400)
        req._chat_total_max_tokens = req.max_tokens
        if req._chat_response_prefill:
            default_thinking_budget = min(
                192,
                max(32, req.max_tokens // 2),
            )
            try:
                thinking_budget = int(
                    body.get("thinking_budget", default_thinking_budget)
                )
            except (TypeError, ValueError):
                return web.json_response(
                    {
                        "error": {
                            "message": "thinking_budget must be an integer",
                            "type": "invalid_request_error",
                        }
                    },
                    status=400,
                )
            if thinking_budget < 1:
                return web.json_response(
                    {
                        "error": {
                            "message": "thinking_budget must be >= 1",
                            "type": "invalid_request_error",
                        }
                    },
                    status=400,
                )
            req.max_tokens = min(req.max_tokens, thinking_budget)
        self._context_tracker.record(context_metrics)

        auth = request.headers.get("Authorization", "")
        req_api_key = auth[7:] if auth.startswith("Bearer ") else None

        try:
            if req.stream:
                response = web.StreamResponse()
                response.content_type = "text/event-stream"
                response.headers["Cache-Control"] = "no-cache"
                await response.prepare(request)
                gen = self._async_chat_stream(req, body.get("tools"), api_key=req_api_key)
                try:
                    async for chunk in gen:
                        await response.write(chunk.encode())
                except (ConnectionResetError, ConnectionError):
                    await gen.aclose()
                return response

            result = await self._async_complete(req, api_key=req_api_key, endpoint="/v1/chat/completions")
            result_dict = result.to_dict()
            if result_dict["choices"]:
                raw_text = result_dict["choices"][0]["text"]
                response_prefill = getattr(req, "_chat_response_prefill", "")
                text = raw_text
                if response_prefill and not raw_text.startswith(response_prefill):
                    text = response_prefill + raw_text

                if response_prefill and "</think>" not in raw_text:
                    transition = self._THINK_FINAL_TRANSITION
                    transition_ids = await self._tokenize_async(transition)
                    used = int(
                        result_dict.get("usage", {}).get(
                            "completion_tokens",
                            0,
                        )
                    )
                    total_budget = getattr(
                        req,
                        "_chat_total_max_tokens",
                        req.max_tokens,
                    )
                    remaining = total_budget - used - len(transition_ids)
                    final_text = ""
                    follow_dict = None
                    if remaining > 0:
                        continuation_prompt = req.prompt + raw_text + transition
                        continuation_ids = await self._tokenize_async(
                            continuation_prompt
                        )
                        available = max_seq_len - len(continuation_ids)
                        final_budget = min(remaining, max(0, available))
                        if final_budget > 0:
                            follow_request = replace(
                                req,
                                prompt=continuation_prompt,
                                max_tokens=final_budget,
                                stream=False,
                            )
                            follow_request._prompt_token_ids = continuation_ids
                            follow_request._pixel_values = pixel_values
                            follow_result = await self._async_complete(
                                follow_request,
                                api_key=req_api_key,
                                endpoint="/v1/chat/completions/final",
                            )
                            follow_dict = follow_result.to_dict()
                            if follow_dict["choices"]:
                                final_text = follow_dict["choices"][0]["text"]

                    text += transition + final_text
                    if "</final>" not in final_text:
                        text += "\n</final>"

                    if follow_dict is not None:
                        first_usage = result_dict.get("usage", {})
                        follow_usage = follow_dict.get("usage", {})
                        completion_tokens = (
                            int(first_usage.get("completion_tokens", 0))
                            + len(transition_ids)
                            + int(follow_usage.get("completion_tokens", 0))
                        )
                        first_usage["completion_tokens"] = completion_tokens
                        first_usage["total_tokens"] = (
                            int(first_usage.get("prompt_tokens", len(prompt_ids)))
                            + completion_tokens
                        )
                finish_reason = result_dict["choices"][0].get("finish_reason", "length")
                message = {"role": "assistant", "content": text}
                tools = body.get("tools")
                if tools:
                    from tr_hash_i64.core.tool_parser import ToolCallParser
                    tool_calls = ToolCallParser(tools).parse(text)
                    if tool_calls:
                        message["tool_calls"] = [
                            {"id": tc.id, "type": tc.type, "function": {"name": tc.function_name, "arguments": tc.function_arguments}}
                            for tc in tool_calls
                        ]
                        finish_reason = "tool_calls"
                chat_choice = {"message": message, "index": 0, "finish_reason": finish_reason}
                if "logprobs" in result_dict["choices"][0]:
                    chat_choice["logprobs"] = result_dict["choices"][0]["logprobs"]
                result_dict["choices"][0] = chat_choice
            result_dict["object"] = "chat.completion"
            return web.json_response(result_dict)
        except (ConnectionResetError, ConnectionError):
            return web.Response(status=499, text="Client disconnected")
        except Exception as e:
            logger.error("Chat completion error: %s", e, exc_info=True)
            return web.json_response({"error": {"message": "Internal server error", "type": "server_error"}}, status=500)
