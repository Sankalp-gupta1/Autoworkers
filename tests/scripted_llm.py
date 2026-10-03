"""A fake, deterministic "model" for tests.

It lets us test everything *around* the LLM - real browser, real sandbox,
tools, approval gate, secrets, retries, context handling, verification and
reporting - without an API key and without flakiness. A "brain" is a generator
that yields tool calls and receives each tool result back.

This is test scaffolding only. Real runs use a real model (see agent/llm.py).
"""
from __future__ import annotations

import re

from agent.llm import LLMResponse, ToolCall


class ScriptedLLM:
    model = "scripted-test-brain"

    def __init__(self, brain_fn):
        self.brain_fn = brain_fn
        self.gen = None
        self.calls = 0

    def complete(self, system, history, tools, max_tokens=1500):
        self.calls += 1
        last_tool = next((h["text"] for h in reversed(history) if h["role"] == "tool"), "")
        try:
            if self.gen is None:
                self.gen = self.brain_fn()
                name, args = next(self.gen)
            else:
                name, args = self.gen.send(last_tool)
        except StopIteration:
            name, args = "finish", {"status": "failed", "summary": "script ended"}
        args.setdefault("reason", f"scripted step {self.calls}")
        return LLMResponse(None, ToolCall(name, args))


def eid(obs: str, pattern: str) -> int:
    """Find the element number of the first observation line matching `pattern`."""
    for line in obs.splitlines():
        m = re.match(r"\[(\d+)\] (.*)", line)
        if m and re.search(pattern, m.group(2), re.I):
            return int(m.group(1))
    raise AssertionError(f"no element matching {pattern!r} in observation:\n{obs[:1500]}")


class KeywordVerifierLLM:
    """Fake auditor: passes only if every required string appears in the fresh evidence."""
    model = "scripted-auditor"

    def __init__(self, required: list[str]):
        self.required = required
        self.prompts: list[str] = []

    def complete(self, system, history, tools, max_tokens=1500):
        prompt = history[-1]["text"]
        self.prompts.append(prompt)
        evidence = prompt.split("FRESH EVIDENCE", 1)[-1]
        checks = [{"requirement": f"evidence shows {r}", "satisfied": r in evidence,
                   "evidence": "found in fresh page" if r in evidence else "missing"}
                  for r in self.required]
        ok = all(c["satisfied"] for c in checks)
        return LLMResponse(None, ToolCall("submit_verdict", {
            "checks": checks, "verified": ok,
            "explanation": "All requirements visible on evidence pages." if ok else "Missing evidence."}))
