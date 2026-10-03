"""Checks that the provider adapters build well-formed API requests from the
neutral history format, using fake clients (no network, no API key)."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

from agent.llm import AnthropicLLM, OpenAICompatLLM
from agent.tools import TOOLS

HISTORY = [
    {"role": "user", "text": "TASK: do it"},
    {"role": "assistant", "text": "Planning.", "call": {"id": "t1", "name": "navigate",
                                                       "args": {"url": "http://x/mail", "reason": "r"}}},
    {"role": "tool", "id": "t1", "name": "navigate", "text": "URL: http://x/mail"},
    {"role": "assistant", "text": None, "call": {"id": "t2", "name": "click",
                                                 "args": {"element_id": 3, "reason": "r"}}},
    {"role": "tool", "id": "t2", "name": "click", "text": "ok"},
]


def test_anthropic_request_shape():
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.model = "m"
    captured = {}

    def create(**kw):
        captured.update(kw)
        return NS(content=[NS(type="text", text="hm"),
                           NS(type="tool_use", id="t3", name="observe", input={"reason": "r"})],
                  usage=NS(input_tokens=1, output_tokens=2))

    llm.client = NS(messages=NS(create=create))
    out = llm.complete("sys", HISTORY, TOOLS)
    msgs = captured["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user", "assistant", "user"]   # strictly alternating
    assert msgs[1]["content"][1]["type"] == "tool_use"
    assert msgs[2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1",
                                     "content": "URL: http://x/mail"}
    assert captured["tool_choice"]["type"] == "any"
    assert all("input_schema" in t for t in captured["tools"])
    assert out.call.name == "observe" and out.text == "hm"


def test_openai_request_shape_and_param_fallback():
    llm = OpenAICompatLLM.__new__(OpenAICompatLLM)
    llm.model, llm.tool_choice, llm.token_param = "m", "required", "max_tokens"
    calls = []

    class BadParam(Exception):
        status_code = 400

    def create(**kw):
        calls.append(kw)
        if "max_tokens" in kw:
            raise BadParam("Unsupported parameter: 'max_tokens'")
        tc = NS(id="c9", function=NS(name="click", arguments=json.dumps({"element_id": 2, "reason": "r"})))
        return NS(choices=[NS(message=NS(content="", tool_calls=[tc]))],
                  usage=NS(prompt_tokens=1, completion_tokens=1))

    llm.client = NS(chat=NS(completions=NS(create=create)))
    out = llm.complete("sys", HISTORY, TOOLS)
    assert "max_completion_tokens" in calls[-1]
    msgs = calls[-1]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "navigate"
    assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": "URL: http://x/mail"}
    assert out.call.args["element_id"] == 2
