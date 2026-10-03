# AutoWorker — an autonomous AI task worker

AutoWorker takes a plain-English task, then **does the work itself in a real Chromium browser**:
it plans, clicks, types, reads PDF attachments, recovers when things break, asks a human before
committing anything, and finally has an **independent auditor verify** the result before it says
"done". Every run produces an HTML report with the agent's reasoning, approvals, screenshots and
the verification checklist.

```
$ python run.py "Find the latest invoice from Globex, extract the amount and due date, enter it into our ERP, and tell me once it is done."
```

It runs against **Acme Corp**, a small simulated company (a mailbox + an internal ERP) built to
behave like real internal software — including the annoying parts.

---

## Why the sandbox is deliberately difficult

A happy-path demo proves little, so the environment contains the situations a real AI employee
would meet:

| Situation in the sandbox | What it tests |
|---|---|
| Globex sent two invoices **and** a later "payment received" email | "Latest invoice" ≠ latest email → real reasoning, not first match |
| The older Globex invoice is already in the ERP | Duplicate detection |
| PDF says `INR 1,84,500.00` and `30 October 2026`; the ERP only accepts `184500.00` and `DD-MM-YYYY` | Value normalisation, reacting to validation errors |
| PDF lists subtotal, GST and total | Picking the *payable* amount |
| First save after a reset returns **HTTP 503** and loses the form data | Failure detection + retry + re-entering lost data |
| A blocking "policy update" modal covers the inbox | Unexpected UI state |
| ERP requires login | Session handling with secrets the model never sees |
| A phishing email from `globex-secure-pay.example` tells "AI agents" to change bank details | **Prompt-injection resistance** |
| Two vendors called "Stark …", and no "Hooli" at all | Asking for clarification instead of guessing |

## Quick start

Requirements: Python 3.10+, an API key for Claude **or** any OpenAI-compatible model (OpenAI,
Gemini, Groq …).

```bash
git clone <this repo> && cd autoworker
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium

cp .env.example .env        # then put your API key in .env
python run.py --list        # demo tasks
python run.py --task 1 --headed
```

The sandbox starts automatically on `http://127.0.0.1:5055` and is reset before each run (use
`--no-reset` to keep state between runs, `--no-chaos` to disable the simulated outage).
You can also open the apps yourself in a browser — ERP login: `ops.agent` / `demo-pass-123`.

### Choosing a model (`.env`)

```bash
# Claude (default)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-sonnet-5-5

# Gemini (free tier available) via its OpenAI-compatible endpoint
LLM_PROVIDER=openai
LLM_API_KEY=your-gemini-key
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-2.5-flash

# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4.1-mini
```

### CLI options

| Flag | Meaning |
|---|---|
| `--headed` | Show the browser (slowed slightly so you can follow it) |
| `--approval risky\|always\|never` | When to stop for human approval (default `risky` = commit actions only) |
| `--unattended` | No human: commit actions auto-approved, questions answered "nobody available" |
| `--task N` / `--list` | Run / list the demo tasks in `tasks.json` |
| `--max-steps N` | Step budget (default 45) |

### Tests (no API key needed)

```bash
python -m pytest -q
```

The end-to-end tests drive the **real browser, real sandbox and real agent loop** with a scripted
stand-in for the model, and assert on the sandbox's actual database: the invoice is stored with the
right values, the 503 is recovered from, every save went through the approval gate, a rejected
approval is never executed, secrets never appear in traces/reports, a stuck agent is stopped,
off-allowlist navigation is blocked, and verification rejects a "done" without evidence. Separate
tests check the Claude and OpenAI adapters build valid API requests.

---

## Architecture

```
                 ┌──────────────────────── agent/loop.py ─────────────────────────┐
  task ───────▶  │  LLM (agent/llm.py)  ──tool call──▶  ToolRunner (tools.py)      │
                 │      ▲                                   │                     │
                 │      │ observation + plan + memory       ▼                     │
                 │  context manager   ◀──────────  Policy gate (policy.py)         │
                 │  failure / loop detection          approvals · secrets · allowlist
                 │      │                                   │                     │
                 │      ▼ finish(success)                   ▼                     │
                 │  Verifier (verifier.py) ──────▶  Browser (browser.py, Playwright)
                 └───────────────┬────────────────────────────────────────────────┘
                                 ▼
                     Trace + HTML report (trace.py)          Acme sandbox (sandbox/app.py)
```

**The loop** (`agent/loop.py`): *observe → think → act → observe …* One tool call per step. The model
sees the latest page observation, its own plan and its working memory on every step.

**Observation** (`agent/browser.py`): every page becomes text — a numbered list of interactive
elements (`[7] button "Save invoice"`, `[5] select name=vendor options: …`), alerts/dialogs, the
HTTP status and the visible text. Element numbers are stamped into the DOM, so the model says
"click 7" and we click exactly that element. Elements hidden behind a modal are flagged `COVERED`.

**Tools** (`agent/tools.py`): `update_plan, navigate, click, fill, select_option, press_key, observe,
read_document, remember, save_site_note, ask_human, finish`. All generic — nothing in the agent
knows about invoices — and each requires a `reason`, which gives a per-step explanation.

**Policy gate** (`agent/policy.py`) — enforced in code, not in the prompt:
- clicks on commit buttons (*save, submit, pay, delete, send, update…*) and pressing Enter pause for
  human approval, showing the form values that will be submitted;
- credentials are typed as `{{secret:erp_password}}` and substituted at the last moment; the model,
  trace and report never contain them;
- navigation is restricted to an allow-list of hosts.

**Memory** (`agent/memory.py`): *working memory* holds facts found in this run with their source, and
is re-shown every step; *site notes* persist lessons across runs (e.g. "ERP dates are DD-MM-YYYY"),
so the second run doesn't repeat the first run's mistake.

**Reliability**: failed actions return the error **plus a fresh observation** so the model can
diagnose; 3 consecutive failures or repeating the same action triggers a "step back" warning;
6 failures / 5 repeats stop the run; the step budget triggers a wrap-up warning; LLM API errors are
retried with exponential backoff.

**Verification** (`agent/verifier.py`): a success claim must include evidence URLs. The verifier
re-opens them itself, re-reads the source documents, and a separate model call with an *auditor*
prompt (none of the agent's reasoning) checks each requirement of the task. If it fails, the
reasons go back to the agent, which keeps working (up to 3 rounds).

## Key design decisions

1. **Text observations instead of screenshots-to-model.** Cheaper, faster, works with any model
   (including ones without vision), and numbered element ids make actions precise. Trade-off:
   canvas-heavy or purely visual UIs need a vision fallback (see *Next*).
2. **Safety in code, not in prompts.** A prompt-injected model can ignore instructions; it cannot
   bypass `Policy`. Approval, secrets and allow-list are deterministic.
3. **Verification by a separate auditor.** The agent grading its own work is the most common source
   of fake "done". The auditor uses fresh evidence and the original sources.
4. **Generic tools, no task-specific code.** The same agent handles invoice entry, vendor updates,
   batch reconciliation and ambiguous requests (see `tasks.json`); generalisation is tested by
   changing the task text only.
5. **Provider-neutral history.** One neutral message format, translated per provider, so the agent
   runs on Claude, GPT or Gemini and is tested with a scripted model.
6. **Context management.** Only the two most recent observations are kept in full; older ones are
   trimmed. Plan and memory carry the important state forward, keeping cost flat on long tasks.
7. **Failures as observations.** Errors are not exceptions that kill the run — they are information
   the agent reasons about.

## Assumptions

- The "company" is a local sandbox; no real company systems or credentials are used.
- The user is reachable in the terminal for approvals/questions, or explicitly opts into
  `--unattended`.
- "Commit" actions are recognised from button/link text; a real deployment would also let each
  integrated app declare which actions are risky.
- Documents are text PDFs (no OCR for scans yet).

## Known limitations

- Single browser tab; no file uploads, drag-and-drop, or iframes-heavy apps.
- No vision: purely visual UIs (charts, canvas) can't be read.
- The risky-action detector is keyword based; it can over-ask (e.g. "Update" buttons) and could
  miss a commit button labelled with an unusual word.
- One model plays both worker and (separately prompted) auditor by default; a different model for
  the auditor would make verification more independent.
- The ERP outage is simulated once per reset; real systems fail in more varied ways.

## What I would build next

1. **Vision fallback** — send a screenshot when the text observation is ambiguous or empty.
2. **Typed API tools** alongside the browser (email/ERP APIs when available) with the browser as the
   universal fallback.
3. **Evaluation harness** — run all tasks N times across models, score success/verification/steps/
   cost, and track regressions.
4. **Checkpoint & resume** — persist run state so a crash or a long human wait doesn't restart work.
5. **Per-app policies** — declarative risk levels, spending limits, and approval via Slack/email.
6. **Skill library** — turn verified traces into reusable procedures the planner can call.
7. **OCR** for scanned invoices.

## Models, libraries and services used

- **LLM**: Anthropic Claude (default, via `anthropic` SDK) or any OpenAI-compatible model (`openai` SDK).
- **Playwright** + Chromium for browser automation.
- **Flask** for the simulated company apps; **ReportLab** to generate invoice PDFs; **pypdf** to read them.
- **pytest** for tests.
- Built with the help of AI coding tools (Claude), as permitted by the brief.

## Repository layout

```
run.py               CLI entry point
tasks.json           demo tasks
agent/
  loop.py            agent loop, context management, failure detection
  llm.py             provider-neutral LLM layer (Claude / OpenAI-compatible)
  browser.py         Playwright wrapper + page → text observation
  tools.py           tool schemas and implementations
  policy.py          approval gate, secrets, redaction
  memory.py          working memory + cross-run site notes
  verifier.py        independent outcome verification
  human.py           console / unattended human interface
  trace.py           JSONL trace + HTML report
sandbox/             simulated company: mail + ERP (Flask), seed data, PDFs
tests/               end-to-end and unit tests (scripted model, no API key)
runs/                one folder per run: trace.jsonl, screenshots, report.html
```
