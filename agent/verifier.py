"""Independent verification of the agent's claimed outcome.

The agent saying "done" is not proof. When it finishes with status=success, the
verifier:
  1. re-opens every evidence URL itself and takes a fresh observation + screenshot;
  2. re-reads the source documents the agent's facts came from;
  3. asks a separate model call - with an auditor prompt and none of the agent's
     reasoning - to check every requirement of the original task against that
     fresh evidence.
If verification fails, the reasons go back to the agent, which keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .browser import ActionError, Browser

VERIFIER_SYSTEM = """You are a strict auditor checking whether an AI worker really completed a task.
You get: the user's original task, what the worker claims, the facts it extracted
(with sources), the raw text of those source documents, and FRESH observations of
the evidence pages that you loaded yourself.

Rules:
- Trust only the fresh evidence and source documents, not the worker's claims.
- Split the task into concrete requirements and check each one.
- Check values exactly (amounts, dates, identifiers) against the source documents.
  Formatting differences are fine (184500.00 == 1,84,500.00; 30-10-2026 == 30 October 2026).
- If a requirement cannot be confirmed from the evidence, it is NOT satisfied.
- The final "tell the user" part of a task is satisfied by the worker's summary itself.
Call submit_verdict exactly once."""

VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Report whether the task outcome is verified.",
    "parameters": {"type": "object", "properties": {
        "checks": {"type": "array", "items": {"type": "object", "properties": {
            "requirement": {"type": "string"}, "satisfied": {"type": "boolean"},
            "evidence": {"type": "string"}}, "required": ["requirement", "satisfied", "evidence"]}},
        "verified": {"type": "boolean"},
        "explanation": {"type": "string"}},
        "required": ["checks", "verified", "explanation"]},
}


@dataclass
class Verdict:
    verified: bool
    explanation: str
    checks: list[dict] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)


class Verifier:
    def __init__(self, llm, browser: Browser, runner, trace):
        self.llm = llm
        self.browser = browser
        self.runner = runner
        self.trace = trace

    def verify(self, task: str, claim: dict) -> Verdict:
        urls = [u for u in claim.get("evidence_urls", []) if u][:3]
        if not urls:
            return Verdict(False, "No evidence_urls were provided. Open a page that shows the "
                                  "completed result and pass its URL in evidence_urls.")
        evidence_blocks, shots = [], []
        for i, url in enumerate(urls):
            try:
                self.browser.navigate(url)
                obs = self.browser.observe()
                evidence_blocks.append(f"### Evidence page {url}\n{obs.render(max_text=6000)}")
                shot = self.browser.screenshot(self.trace.path(f"verify_{i + 1}.png"))
                if shot:
                    shots.append(shot)
            except ActionError as exc:
                evidence_blocks.append(f"### Evidence page {url}\nCOULD NOT LOAD: {exc}")

        docs = "\n\n".join(f"### Source document {u}\n{t[:3000]}"
                           for u, t in list(self.runner.documents_read.items())[-4:])
        facts = self.runner.memory.render()
        prompt = (f"ORIGINAL TASK:\n{task}\n\nWORKER'S CLAIMED SUMMARY:\n{claim.get('summary')}\n\n"
                  f"WORKER'S CLAIMED RESULT DATA:\n{claim.get('result')}\n\n"
                  f"FACTS THE WORKER SAVED:\n{facts}\n\n"
                  f"SOURCE DOCUMENTS:\n{docs or '(none)'}\n\n"
                  f"FRESH EVIDENCE (loaded by you just now):\n" + "\n\n".join(evidence_blocks))
        resp = self.llm.complete(VERIFIER_SYSTEM, [{"role": "user", "text": prompt}],
                                 [VERDICT_TOOL], max_tokens=1500)
        if not resp.call or resp.call.name != "submit_verdict":
            return Verdict(False, "Verifier did not return a verdict.", screenshots=shots)
        a = resp.call.args
        checks = a.get("checks") or []
        verified = bool(a.get("verified")) and all(c.get("satisfied") for c in checks)
        return Verdict(verified, a.get("explanation", ""), checks, shots)
