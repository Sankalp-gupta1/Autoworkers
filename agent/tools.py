"""Tool definitions (what the model can do) and their implementations.

Every tool takes a required `reason` argument. It costs a few tokens but gives us
a per-step explanation of *why* the agent acted, which shows up in the live log
and the HTML report - important for debugging and for trusting the agent.

Tools are generic (navigate / click / fill / read_document ...). Nothing here
knows about invoices or ERPs, which is what lets the same agent take on
different tasks without code changes.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

from .browser import ActionError, Browser
from .memory import SiteNotes, WorkingMemory
from .policy import Policy

REASON = {"type": "string", "description": "One short sentence: why this is the right next action."}


def _tool(name, description, props=None, required=None):
    props = dict(props or {})
    props["reason"] = REASON
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props,
                           "required": list(required or []) + ["reason"]}}


TOOLS = [
    _tool("update_plan", "Create or update your step-by-step plan. Call it first, and again "
          "whenever you learn something that changes the plan.",
          {"steps": {"type": "array", "items": {"type": "object", "properties": {
              "description": {"type": "string"},
              "status": {"type": "string", "enum": ["pending", "in_progress", "done", "blocked"]}},
              "required": ["description", "status"]}}}, ["steps"]),
    _tool("navigate", "Open a URL in the browser.", {"url": {"type": "string"}}, ["url"]),
    _tool("click", "Click an element by its number from the latest observation.",
          {"element_id": {"type": "integer"}}, ["element_id"]),
    _tool("fill", "Replace the contents of an input/textarea. To type a stored credential use a "
          "placeholder like {{secret:erp_password}} - never ask for or invent passwords.",
          {"element_id": {"type": "integer"}, "text": {"type": "string"}}, ["element_id", "text"]),
    _tool("select_option", "Choose an option in a <select> by its visible text.",
          {"element_id": {"type": "integer"}, "option": {"type": "string"}},
          ["element_id", "option"]),
    _tool("press_key", "Press a keyboard key such as Enter, Escape or Tab.",
          {"key": {"type": "string"}}, ["key"]),
    _tool("observe", "Re-read the current page. Use full_text=true to read long pages completely.",
          {"full_text": {"type": "boolean"}}),
    _tool("read_document", "Download a document (PDF, text, CSV, JSON) by URL using the browser "
          "session and return its text. Use this for attachments instead of clicking them.",
          {"url": {"type": "string"}}, ["url"]),
    _tool("remember", "Save an important fact you discovered (with where it came from) so it "
          "is not lost. Saved facts are shown to you on every step.",
          {"key": {"type": "string"}, "value": {"type": "string"},
           "source": {"type": "string", "description": "URL or document the fact came from"}},
          ["key", "value", "source"]),
    _tool("save_site_note", "Save a durable lesson about how a site behaves (e.g. required "
          "formats) so future runs avoid the same mistake.",
          {"site": {"type": "string", "description": "URL of the site/app"},
           "note": {"type": "string"}}, ["site", "note"]),
    _tool("ask_human", "Ask the user a question when the task is ambiguous, information is "
          "missing, or something looks wrong/unsafe. Do not ask about things you can find out "
          "yourself.", {"question": {"type": "string"}}, ["question"]),
    _tool("finish", "End the task. Use status=success only after you have checked the outcome "
          "on a page that proves it; list those page URLs in evidence_urls - an independent "
          "verifier will open them and check your claim.",
          {"status": {"type": "string", "enum": ["success", "failed", "needs_user"]},
           "summary": {"type": "string", "description": "Concise summary for the user."},
           "result": {"type": "string", "description": "Key data produced, as short "
                      "'key: value' lines (e.g. invoice: GLX-1; amount: 1200.00)."},
           "evidence_urls": {"type": "array", "items": {"type": "string"}}},
          ["status", "summary"]),
]
TOOL_NAMES = {t["name"] for t in TOOLS}


@dataclass
class ToolResult:
    text: str
    ok: bool = True
    observation_changed: bool = False
    finish: dict | None = None
    approval: dict | None = None        # {"summary", "approved", "feedback"} if asked
    notes: list[str] = field(default_factory=list)


class ToolRunner:
    def __init__(self, browser: Browser, policy: Policy, human, memory: WorkingMemory,
                 site_notes: SiteNotes):
        self.browser = browser
        self.policy = policy
        self.human = human
        self.memory = memory
        self.site_notes = site_notes
        self.plan: list[dict] = []
        self.documents_read: dict[str, str] = {}

    # ------------------------------------------------------------------
    def _obs_text(self, full: bool = False) -> str:
        obs = self.browser.observe()
        return obs.render(max_text=12000 if full else 3000)

    def _form_snapshot(self) -> str:
        obs = self.browser.last_obs
        if not obs:
            return ""
        rows = [f"    {e['label'] or e['name']}: {e['value']}" for e in obs.elements
                if e["tag"] in ("input", "select", "textarea") and e.get("value")]
        return "\n".join(rows)

    def _gate(self, reason: str, action_desc: str, step_reason: str) -> dict:
        obs = self.browser.last_obs
        summary = (f"  Action : {action_desc}\n  Why    : {reason}\n"
                   f"  Agent's reason: {step_reason}\n  Page   : {obs.url if obs else '?'}")
        form = self._form_snapshot()
        if form:
            summary += f"\n  Form values that will be submitted:\n{form}"
        approved, feedback = self.human.approve(summary)
        return {"summary": summary, "approved": approved, "feedback": feedback}

    def execute(self, name: str, args: dict, step: int) -> ToolResult:
        if name not in TOOL_NAMES:
            return ToolResult(f"Unknown tool {name!r}. Available: {sorted(TOOL_NAMES)}", ok=False)
        reason = args.get("reason", "")
        try:
            return getattr(self, f"_t_{name}")(args, reason, step)
        except ActionError as exc:
            # Failure is information: return it with a fresh view of the page.
            try:
                page = self._obs_text()
            except Exception:
                page = "(could not observe page)"
            return ToolResult(f"ACTION FAILED: {exc}\n\nCurrent page:\n{page}", ok=False,
                              observation_changed=True)
        except (KeyError, ValueError, TypeError) as exc:
            return ToolResult(f"ACTION FAILED: bad arguments for {name}: {exc}", ok=False)

    # ------------------------------------------------------------------ tools
    def _t_update_plan(self, a, reason, step):
        self.plan = [{"description": s.get("description", ""), "status": s.get("status", "pending")}
                     for s in a["steps"]]
        return ToolResult("Plan updated.")

    def _t_navigate(self, a, reason, step):
        self.browser.navigate(a["url"])
        return ToolResult(self._obs_text(), observation_changed=True)

    def _t_click(self, a, reason, step):
        eid = int(a["element_id"])
        obs = self.browser.last_obs or self.browser.observe()
        el = obs.element(eid)
        approval = None
        why = self.policy.approval_reason("click", el)
        if why:
            approval = self._gate(why, f"click [{eid}] {obs.describe(el) if el else ''}", reason)
            if not approval["approved"]:
                return ToolResult(f"NOT EXECUTED - {approval['feedback']} Re-check your values, "
                                  f"fix anything wrong, or ask_human what to change.",
                                  ok=False, approval=approval)
        self.browser.click(eid)
        return ToolResult(self._obs_text(), observation_changed=True, approval=approval)

    def _t_fill(self, a, reason, step):
        eid = int(a["element_id"])
        real = self.policy.resolve(str(a["text"]))
        self.browser.fill(eid, real)
        obs = self.browser.observe()
        el = obs.element(eid)
        shown = el.get("value") if el else ""
        return ToolResult(f'Filled [{eid}]. Field now contains: "{shown}"', observation_changed=True)

    def _t_select_option(self, a, reason, step):
        eid = int(a["element_id"])
        self.browser.select(eid, a["option"])
        obs = self.browser.observe()
        el = obs.element(eid)
        return ToolResult(f'Selected in [{eid}]: "{el.get("value") if el else a["option"]}"',
                          observation_changed=True)

    def _t_press_key(self, a, reason, step):
        approval = None
        if a["key"].lower() == "enter" and self.policy.mode != "never":
            approval = self._gate("pressing Enter may submit a form", "press Enter", reason)
            if not approval["approved"]:
                return ToolResult(f"NOT EXECUTED - {approval['feedback']}", ok=False,
                                  approval=approval)
        self.browser.press(a["key"])
        return ToolResult(self._obs_text(), observation_changed=True, approval=approval)

    def _t_observe(self, a, reason, step):
        return ToolResult(self._obs_text(full=bool(a.get("full_text"))), observation_changed=True)

    def _t_read_document(self, a, reason, step):
        body, ctype, status = self.browser.fetch(a["url"])
        if status >= 400:
            raise ActionError(f"Download failed with HTTP {status} for {a['url']}")
        url = a["url"]
        if "pdf" in ctype or body[:4] == b"%PDF":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(body))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            if not text.strip():
                text = "(PDF contains no extractable text - it may be a scanned image.)"
        elif "json" in ctype:
            text = json.dumps(json.loads(body), indent=2)
        else:
            text = body.decode("utf-8", errors="replace")
        self.documents_read[url] = text
        clipped = text[:8000] + ("\n...[truncated]" if len(text) > 8000 else "")
        return ToolResult(f"DOCUMENT {url} ({ctype or 'unknown type'}):\n{clipped}")

    def _t_remember(self, a, reason, step):
        self.memory.remember(a["key"], str(a["value"]), a["source"], step)
        return ToolResult(f"Saved {a['key']} = {a['value']}")

    def _t_save_site_note(self, a, reason, step):
        self.site_notes.add(a["site"], a["note"])
        return ToolResult("Site note saved for future runs.")

    def _t_ask_human(self, a, reason, step):
        answer = self.human.ask(a["question"])
        return ToolResult(f"USER ANSWERED: {answer}")

    def _t_finish(self, a, reason, step):
        return ToolResult("Finish requested.", finish={
            "status": a["status"], "summary": a["summary"], "result": a.get("result") or {},
            "evidence_urls": a.get("evidence_urls") or []})
