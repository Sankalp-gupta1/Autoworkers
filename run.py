"""AutoWorker command line.

Examples
  python run.py "Find the latest invoice from Globex, extract the amount and due date, enter it into our ERP, and tell me once it is done."
  python run.py --headed --task 1          # run demo task 1 from tasks.json with a visible browser
  python run.py --list                     # show demo tasks
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

from agent.browser import Browser
from agent.config import Settings
from agent.human import AutoHuman, ConsoleHuman
from agent.llm import make_llm
from agent.loop import Agent
from agent.memory import SiteNotes, WorkingMemory
from agent.policy import Policy
from agent.trace import Trace

ROOT = Path(__file__).parent
C = {"dim": "\033[2m", "b": "\033[1m", "g": "\033[92m", "r": "\033[91m", "y": "\033[93m",
     "c": "\033[96m", "x": "\033[0m"}


def sandbox_up(url: str) -> bool:
    try:
        urllib.request.urlopen(url + "/", timeout=2)
        return True
    except Exception:
        return False


def ensure_sandbox(url: str, reset: bool, chaos: bool) -> None:
    if not sandbox_up(url):
        import logging
        from werkzeug.serving import make_server
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        from sandbox.app import app
        port = int(url.rsplit(":", 1)[-1])
        server = make_server("127.0.0.1", port, app, threaded=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        for _ in range(50):
            if sandbox_up(url):
                break
            time.sleep(0.1)
        print(f"{C['dim']}Started sandbox company apps at {url}{C['x']}")
    if reset:
        req = urllib.request.Request(f"{url}/__reset?chaos={'1' if chaos else '0'}", method="POST")
        urllib.request.urlopen(req, timeout=5)


def printer(kind: str, data: dict) -> None:
    if kind == "action":
        args = {k: v for k, v in data["args"].items() if k != "reason"}
        if data["tool"] == "update_plan":
            args = {"steps": [f"[{s.get('status')}] {s.get('description')}" for s in args.get("steps", [])]}
        print(f"\n{C['b']}Step {data['step']:>2} ▸ {data['tool']}{C['x']} "
              f"{C['c']}{json.dumps(args, ensure_ascii=False)[:220]}{C['x']}")
        if data["args"].get("reason"):
            print(f"   {C['dim']}💭 {data['args']['reason']}{C['x']}")
    elif kind == "result":
        first = data["text"].strip().splitlines()[0] if data["text"].strip() else ""
        col = C["g"] if data["ok"] else C["r"]
        print(f"   {col}{'✓' if data['ok'] else '✗'} {first[:200]}{C['x']}")
    elif kind == "verify":
        print(f"\n{C['y']}🔎 Verifying independently using {data['urls']}{C['x']}")
    elif kind == "verdict":
        v = data["verdict"]
        col = C["g"] if v.verified else C["r"]
        print(f"   {col}{'VERIFIED' if v.verified else 'NOT VERIFIED'}: {v.explanation}{C['x']}")
        for c in v.checks:
            print(f"     {'✔' if c.get('satisfied') else '✘'} {c.get('requirement')}")


def build_agent(settings: Settings, human, llm=None, verifier_llm=None, on_event=printer):
    llm = llm or make_llm(settings)
    policy = Policy(settings.approval, settings.secrets)
    browser = Browser(headless=settings.headless, allowed_hosts=settings.allowed_hosts,
                      slow_mo=0 if settings.headless else 250).start()
    trace = Trace(settings.runs_dir, policy.redact)
    agent = Agent(llm, browser, policy, human, WorkingMemory(), SiteNotes(settings.notes_path),
                  trace, settings, verifier_llm=verifier_llm, on_event=on_event)
    return agent, browser


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="AutoWorker - autonomous AI task worker")
    p.add_argument("task", nargs="?", help="Natural-language task")
    p.add_argument("--task", dest="task_no", type=int, help="Run demo task N from tasks.json")
    p.add_argument("--list", action="store_true", help="List demo tasks")
    p.add_argument("--headed", action="store_true", help="Show the browser window")
    p.add_argument("--approval", choices=["risky", "always", "never"], help="Human approval mode")
    p.add_argument("--unattended", action="store_true",
                   help="No human: auto-approve commit actions, unanswered questions")
    p.add_argument("--no-reset", action="store_true", help="Keep sandbox state from earlier runs")
    p.add_argument("--no-chaos", action="store_true", help="Disable the simulated ERP outage")
    p.add_argument("--max-steps", type=int)
    args = p.parse_args(argv)

    demo = json.loads((ROOT / "tasks.json").read_text())
    if args.list:
        for i, t in enumerate(demo, 1):
            print(f"{i}. [{t['name']}] {t['task']}")
        return 0
    task = args.task or (demo[args.task_no - 1]["task"] if args.task_no else None)
    if not task:
        p.error("give a task in quotes, or --task N (see --list)")

    settings = Settings.from_env()
    if args.headed:
        settings.headless = False
    if args.approval:
        settings.approval = args.approval
    if args.max_steps:
        settings.max_steps = args.max_steps
    if not settings.api_key:
        print(f"{C['r']}No API key. Set ANTHROPIC_API_KEY (or LLM_PROVIDER=openai + LLM_API_KEY). "
              f"See README.{C['x']}")
        return 2

    ensure_sandbox(settings.sandbox_url, reset=not args.no_reset, chaos=not args.no_chaos)
    human = AutoHuman() if args.unattended else ConsoleHuman()
    print(f"{C['b']}🤖 AutoWorker{C['x']} · {settings.provider}/{settings.model} · "
          f"approval={settings.approval}\n{C['b']}Task:{C['x']} {task}")
    agent, browser = build_agent(settings, human)
    try:
        res = agent.run(task)
    finally:
        browser.close()

    col = C["g"] if res.status == "success" else (C["y"] if res.status == "needs_user" else C["r"])
    print(f"\n{'─' * 70}\n{col}{C['b']}{res.status.upper()}{C['x']} in {res.steps} steps, "
          f"{res.seconds}s\n\n{res.summary}")
    if res.result:
        shown = res.result if isinstance(res.result, str) else json.dumps(res.result, indent=2)
        print(f"\n{C['dim']}{shown}{C['x']}")
    print(f"\n📄 Report: {res.report_path}")
    return 0 if res.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
