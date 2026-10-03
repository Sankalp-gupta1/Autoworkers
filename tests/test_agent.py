"""End-to-end tests: real Chromium + real sandbox + real agent loop, scripted model."""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import pytest

from agent.config import Settings
from agent.human import AutoHuman
from agent.policy import Policy
from run import build_agent, ensure_sandbox
from tests.scripted_llm import KeywordVerifierLLM, ScriptedLLM, eid

URL = "http://127.0.0.1:5056"


@pytest.fixture()
def settings(tmp_path):
    s = Settings.from_env()
    s.sandbox_url = URL
    s.runs_dir = tmp_path / "runs"
    s.notes_path = tmp_path / "notes.json"
    s.max_steps = 60
    ensure_sandbox(URL, reset=True, chaos=True)
    return s


def state():
    return json.loads(urllib.request.urlopen(URL + "/__state").read())


def globex_brain():
    """Mimics a sensible agent, including mistakes a real model makes."""
    yield "update_plan", {"steps": [{"description": "find latest Globex invoice", "status": "in_progress"},
                                    {"description": "enter in ERP", "status": "pending"},
                                    {"description": "verify", "status": "pending"}]}
    obs = yield "navigate", {"url": f"{URL}/mail"}
    assert "COVERED" in obs                      # the policy modal blocks the inbox
    obs = yield "click", {"element_id": eid(obs, r'link "Invoice GLX-2026-0871')}
    assert "ACTION FAILED" in obs                # click intercepted by overlay -> must recover
    obs = yield "click", {"element_id": eid(obs, r'button "Acknowledge"')}
    obs = yield "click", {"element_id": eid(obs, r'link "Invoice GLX-2026-0871')}
    pdf = re.search(r'-> (/mail/attachments/\S+\.pdf)', obs).group(1)
    doc = yield "read_document", {"url": URL + pdf}
    assert "1,84,500.00" in doc and "30 October 2026" in doc
    yield "remember", {"key": "amount", "value": "184500.00", "source": URL + pdf}
    yield "remember", {"key": "due_date", "value": "30-10-2026", "source": URL + pdf}
    obs = yield "navigate", {"url": f"{URL}/erp/invoices/new"}
    assert "Sign in" in obs
    yield "fill", {"element_id": eid(obs, "name=username"), "text": "{{secret:erp_username}}"}
    yield "fill", {"element_id": eid(obs, "name=password"), "text": "{{secret:erp_password}}"}
    obs = yield "click", {"element_id": eid(obs, 'button "Sign in"')}
    assert "New vendor invoice" in obs

    def fill_form(obs, amount):
        yield "select_option", {"element_id": eid(obs, "name=vendor"), "option": "Globex Supplies"}
        yield "fill", {"element_id": eid(obs, "name=number"), "text": "GLX-2026-0871"}
        yield "fill", {"element_id": eid(obs, "name=amount"), "text": amount}
        yield "select_option", {"element_id": eid(obs, "name=currency"), "option": "INR"}
        yield "fill", {"element_id": eid(obs, "name=due_date"), "text": "30-10-2026"}
        return (yield "click", {"element_id": eid(obs, 'button "Save invoice"')})

    obs = yield from fill_form(obs, "1,84,500.00")       # 1st try: simulated ERP outage
    assert "HTTP 503" in obs
    obs = yield "navigate", {"url": f"{URL}/erp/invoices/new"}
    obs = yield from fill_form(obs, "1,84,500.00")       # 2nd try: format rejected
    assert "plain number" in obs
    yield "save_site_note", {"site": f"{URL}/erp", "note": "Amounts must be plain numbers like 184500.00"}
    yield "fill", {"element_id": eid(obs, "name=amount"), "text": "184500.00"}
    obs = yield "click", {"element_id": eid(obs, 'button "Save invoice"')}
    assert "saved successfully" in obs
    # First finish forgets evidence -> verifier must push back.
    obs = yield "finish", {"status": "success", "summary": "Done"}
    assert "VERIFICATION FAILED" in obs
    yield "finish", {"status": "success",
                     "summary": "Entered Globex invoice GLX-2026-0871 (INR 184500.00, due 30-10-2026).",
                     "result": {"invoice": "GLX-2026-0871", "amount": 184500.0, "due": "30-10-2026"},
                     "evidence_urls": [f"{URL}/erp/invoices"]}


def test_invoice_task_end_to_end(settings):
    human = AutoHuman(approve_all=True)
    auditor = KeywordVerifierLLM(["GLX-2026-0871", "184,500.00", "30-10-2026"])
    agent, browser = build_agent(settings, human, llm=ScriptedLLM(globex_brain),
                                 verifier_llm=auditor, on_event=lambda *a: None)
    try:
        res = agent.run("Find the latest invoice from Globex, enter it into the ERP.")
    finally:
        browser.close()

    assert res.status == "success" and res.verdict.verified
    new = [i for i in state()["invoices"] if i["number"] == "GLX-2026-0871"]
    assert len(new) == 1 and new[0]["amount"] == 184500.0 and new[0]["due_date"] == "2026-10-30"
    # every "Save invoice" click went through the human approval gate
    assert len(human.approvals) == 3 and all("GLX-2026-0871" in a for a in human.approvals)
    # secrets never leak into the trace or report
    trace = (Path(res.trace_dir) / "trace.jsonl").read_text()
    report = Path(res.report_path).read_text()
    assert "demo-pass-123" not in trace and "demo-pass-123" not in report
    assert "{{secret:erp_password}}" in trace
    # verifier received the source document, not just the agent's claim
    assert "Total Amount Due" in auditor.prompts[-1]
    assert "plain numbers" in (settings.notes_path.read_text())


def test_rejected_approval_is_not_executed(settings):
    def brain():
        obs = yield "navigate", {"url": f"{URL}/erp/vendors"}
        yield "fill", {"element_id": eid(obs, "name=username"), "text": "{{secret:erp_username}}"}
        yield "fill", {"element_id": eid(obs, "name=password"), "text": "{{secret:erp_password}}"}
        obs = yield "click", {"element_id": eid(obs, 'button "Sign in"')}
        obs = yield "click", {"element_id": eid(obs, 'link "Edit Globex Supplies"')}
        yield "fill", {"element_id": eid(obs, "name=contact_email"), "text": "pay@evil.example"}
        obs = yield "click", {"element_id": eid(obs, 'button "Save changes"')}
        assert "NOT EXECUTED" in obs
        yield "finish", {"status": "needs_user", "summary": "User rejected the change."}

    human = AutoHuman(approve_all=False)
    agent, browser = build_agent(settings, human, llm=ScriptedLLM(brain), on_event=lambda *a: None)
    try:
        res = agent.run("change globex email")
    finally:
        browser.close()
    assert res.status == "needs_user"
    globex = next(v for v in state()["vendors"] if v["name"] == "Globex Supplies")
    assert globex["contact_email"] == "billing@globex.example"


def test_stuck_agent_is_stopped(settings):
    def brain():
        yield "navigate", {"url": f"{URL}/mail"}
        while True:
            yield "click", {"element_id": 999}

    agent, browser = build_agent(settings, AutoHuman(), llm=ScriptedLLM(brain),
                                 on_event=lambda *a: None)
    try:
        res = agent.run("anything")
    finally:
        browser.close()
    assert res.status == "failed" and "stuck" in res.summary
    assert res.steps < 10


def test_navigation_outside_allowlist_blocked(settings):
    def brain():
        obs = yield "navigate", {"url": "https://example.com/"}
        assert "blocked by policy" in obs
        obs = yield "navigate", {"url": f"{URL}/__state"}
        assert "blocked by policy" in obs
        yield "finish", {"status": "failed", "summary": "blocked"}

    agent, browser = build_agent(settings, AutoHuman(), llm=ScriptedLLM(brain),
                                 on_event=lambda *a: None)
    try:
        assert agent.run("x").status == "failed"
    finally:
        browser.close()


def test_policy_rules():
    p = Policy("risky", {"erp_password": "demo-pass-123"})
    assert p.approval_reason("click", {"tag": "button", "type": "submit", "text": "Save invoice"})
    assert not p.approval_reason("click", {"tag": "a", "type": "", "text": "Invoices"})
    assert not p.approval_reason("fill", None)
    assert p.resolve("{{secret:erp_password}}") == "demo-pass-123"
    assert p.redact("pw is demo-pass-123") == "pw is {{secret:erp_password}}"
    with pytest.raises(KeyError):
        p.resolve("{{secret:bank_pin}}")
