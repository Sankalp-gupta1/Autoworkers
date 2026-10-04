"""Model-agnostic LLM layer.

Supports:
- Anthropic
- OpenAI
- Gemini through Google's OpenAI-compatible endpoint
- Groq / Together / Ollama / other OpenAI-compatible providers

Gemini 3 tool calling requires thought signatures to be preserved
between turns. This module captures and restores those signatures.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:12])
    thought_signature: str | None = None


@dataclass
class LLMResponse:
    text: str | None
    call: ToolCall | None
    usage: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


def _with_retries(fn, attempts: int = 4):
    delay = 2.0

    for i in range(attempts):
        try:
            return fn()

        except Exception as exc:
            status = getattr(exc, "status_code", None)

            permanent = (
                status is not None
                and 400 <= status < 500
                and status not in (408, 409, 429)
            )

            if permanent or i == attempts - 1:
                raise LLMError(
                    f"LLM call failed after {attempts} attempts: {exc}"
                ) from exc

            time.sleep(delay)
            delay *= 2


class AnthropicLLM:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
    ):
        import anthropic

        self.model = model

        self.client = (
            anthropic.Anthropic(api_key=api_key)
            if api_key
            else anthropic.Anthropic()
        )

    def complete(
        self,
        system: str,
        history: list[dict],
        tools: list[dict],
        max_tokens: int = 1500,
    ) -> LLMResponse:

        messages: list[dict] = []

        def push(role, block):
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].append(block)
            else:
                messages.append(
                    {
                        "role": role,
                        "content": [block],
                    }
                )

        for h in history:

            if h["role"] == "user":
                push(
                    "user",
                    {
                        "type": "text",
                        "text": h["text"],
                    },
                )

            elif h["role"] == "assistant":

                if h.get("text"):
                    push(
                        "assistant",
                        {
                            "type": "text",
                            "text": h["text"],
                        },
                    )

                if h.get("call"):
                    c = h["call"]

                    push(
                        "assistant",
                        {
                            "type": "tool_use",
                            "id": c["id"],
                            "name": c["name"],
                            "input": c["args"],
                        },
                    )

            elif h["role"] == "tool":

                push(
                    "user",
                    {
                        "type": "tool_result",
                        "tool_use_id": h["id"],
                        "content": h["text"],
                    },
                )

        anth_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

        resp = _with_retries(
            lambda: self.client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                tools=anth_tools,
                tool_choice={
                    "type": "any",
                    "disable_parallel_tool_use": True,
                },
                max_tokens=max_tokens,
            )
        )

        text = (
            "\n".join(
                b.text
                for b in resp.content
                if b.type == "text"
            ).strip()
            or None
        )

        call = next(
            (
                ToolCall(
                    b.name,
                    dict(b.input),
                    b.id,
                )
                for b in resp.content
                if b.type == "tool_use"
            ),
            None,
        )

        usage = {
            "input": resp.usage.input_tokens,
            "output": resp.usage.output_tokens,
        }

        return LLMResponse(
            text=text,
            call=call,
            usage=usage,
        )


class OpenAICompatLLM:
    """OpenAI-compatible provider.

    Works with:
    - OpenAI
    - Gemini
    - Groq
    - Together
    - Ollama
    - other compatible APIs

    Gemini 3 requires its thought_signature to be returned inside
    assistant tool_calls on subsequent requests.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        import openai

        self.model = model

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.tool_choice = "required"
        self.token_param = "max_tokens"

    def _create(
        self,
        messages,
        tools,
        max_tokens,
    ):
        for _ in range(3):

            kwargs = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                self.token_param: max_tokens,
            }

            if self.tool_choice:
                kwargs["tool_choice"] = self.tool_choice

            try:
                return _with_retries(
                    lambda: self.client.chat.completions.create(
                        **kwargs
                    )
                )

            except LLMError as exc:

                msg = str(exc.__cause__ or exc)

                if (
                    "max_tokens" in msg
                    and self.token_param == "max_tokens"
                ):
                    self.token_param = "max_completion_tokens"

                elif (
                    "tool_choice" in msg
                    and self.tool_choice
                ):
                    self.tool_choice = None

                else:
                    raise

        raise LLMError(
            "Could not find request parameters accepted by this provider."
        )

    @staticmethod
    def _extract_thought_signature(tool_call) -> str | None:
        """Extract Gemini's thought signature from an OpenAI tool call."""

        # The OpenAI SDK normally exposes extra provider fields through
        # model_dump(). Gemini places the signature here:
        #
        # extra_content.google.thought_signature

        try:
            data = tool_call.model_dump(
                exclude_none=True
            )
        except Exception:
            data = {}

        extra_content = data.get("extra_content") or {}

        google = extra_content.get("google") or {}

        signature = google.get("thought_signature")

        if signature:
            return signature

        # Some SDK/provider versions may expose provider-specific
        # information under a slightly different field.

        provider_fields = (
            data.get("provider_specific_fields")
            or data.get("extra_fields")
            or {}
        )

        google = provider_fields.get("google") or {}

        signature = google.get("thought_signature")

        if signature:
            return signature

        # Last fallback: inspect the raw object attributes.

        try:
            extra = getattr(
                tool_call,
                "extra_content",
                None,
            )

            if extra:

                google = getattr(
                    extra,
                    "google",
                    None,
                )

                if google:

                    signature = getattr(
                        google,
                        "thought_signature",
                        None,
                    )

                    if signature:
                        return signature

        except Exception:
            pass

        return None

    def complete(
        self,
        system: str,
        history: list[dict],
        tools: list[dict],
        max_tokens: int = 1500,
    ) -> LLMResponse:

        messages: list[dict] = [
            {
                "role": "system",
                "content": system,
            }
        ]

        for h in history:

            if h["role"] == "user":

                messages.append(
                    {
                        "role": "user",
                        "content": h["text"],
                    }
                )

            elif h["role"] == "assistant":

                m = {
                    "role": "assistant",
                    "content": h.get("text") or "",
                }

                if h.get("call"):

                    c = h["call"]

                    tool_call = {
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(
                                c["args"]
                            ),
                        },
                    }

                    # -------------------------------------------------
                    # GEMINI 3 THOUGHT SIGNATURE
                    # -------------------------------------------------
                    #
                    # This MUST be returned exactly where Gemini
                    # originally supplied it.
                    #
                    signature = c.get(
                        "thought_signature"
                    )

                    if signature:

                        tool_call["extra_content"] = {
                            "google": {
                                "thought_signature": signature
                            }
                        }

                    m["tool_calls"] = [
                        tool_call
                    ]

                messages.append(m)

            elif h["role"] == "tool":

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": h["id"],
                        "content": h["text"],
                    }
                )

        oa_tools = [
            {
                "type": "function",
                "function": t,
            }
            for t in tools
        ]

        resp = self._create(
            messages,
            oa_tools,
            max_tokens,
        )

        msg = resp.choices[0].message

        call = None

        if msg.tool_calls:

            tc = msg.tool_calls[0]

            try:
                args = json.loads(
                    tc.function.arguments or "{}"
                )
            except json.JSONDecodeError:

                args = {
                    "_malformed_arguments":
                        tc.function.arguments
                }

            signature = self._extract_thought_signature(
                tc
            )

            call = ToolCall(
                name=tc.function.name,
                args=args,
                id=(
                    tc.id
                    or (
                        "call_"
                        + uuid.uuid4().hex[:12]
                    )
                ),
                thought_signature=signature,
            )

        usage = {}

        if resp.usage:

            usage = {
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            }

        return LLMResponse(
            text=(msg.content or "").strip() or None,
            call=call,
            usage=usage,
        )


def make_llm(settings):

    if settings.provider == "anthropic":
        return AnthropicLLM(
            settings.model,
            settings.api_key,
        )

    if settings.provider in (
        "openai",
        "gemini",
        "groq",
        "openai-compatible",
    ):
        return OpenAICompatLLM(
            settings.model,
            settings.api_key,
            settings.base_url,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER {settings.provider!r}"
    )