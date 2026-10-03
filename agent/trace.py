"""Run recording: JSONL trace, per-step screenshots and a self-contained HTML report.

Every run gets its own folder under runs/. The report is the "evidence of
completion" handed back to the user: what was asked, every step with the agent's
reason, every approval, screenshots, and the verifier's requirement checklist.
"""
from __future__ import annotations

import base64
import html
import json
import time
from datetime import datetime
from pathlib import Path


class Trace:
    def __init__(self, runs_dir: Path, redact):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(runs_dir) / stamp
        n = 1
        while self.dir.exists():
            n += 1
            self.dir = Path(runs_dir) / f"{stamp}-{n}"
        self.dir.mkdir(parents=True)
        self.redact = redact
        self.events: list[dict] = []
        self.started = time.time()
        self._fh = open(self.dir / "trace.jsonl", "w", encoding="utf-8")

    def path(self, name: str) -> Path:
        return self.dir / name

    def log(self, kind: str, **data) -> dict:
        event = {"t": round(time.time() - self.started, 2), "kind": kind, **data}
        clean = json.loads(self.redact(json.dumps(event, default=str)))
        self.events.append(clean)
        self._fh.write(json.dumps(clean) + "\n")
        self._fh.flush()
        return clean

    def close(self) -> None:
        self._fh.close()

    # ------------------------------------------------------------------ report
    def write_report(self, task: str, outcome: dict) -> Path:
        def img(path):
            if not path or not Path(path).exists():
                return ""
            b64 = base64.b64encode(Path(path).read_bytes()).decode()
            return (f'<a href="data:image/png;base64,{b64}" target="_blank">'
                    f'<img src="data:image/png;base64,{b64}"></a>')

        esc = lambda s: html.escape(str(s if s is not None else ""))
        rows = []
        for e in self.events:
            if e["kind"] != "step":
                continue
            badge = "ok" if e.get("ok") else "fail"
            appr = ""
            if e.get("approval"):
                a = e["approval"]
                appr = (f'<div class="appr">{"✅ Approved" if a["approved"] else "⛔ Rejected"} '
                        f'by human<pre>{esc(a["summary"])}</pre></div>')
            rows.append(
                f'<tr><td class="n">{e["step"]}</td><td><b>{esc(e["tool"])}</b> '
                f'<code>{esc(json.dumps({k: v for k, v in e["args"].items() if k != "reason"}))}</code>'
                f'<div class="why">💭 {esc(e["args"].get("reason", ""))}</div>{appr}'
                f'<details><summary class="{badge}">{"result" if e.get("ok") else "FAILED"}</summary>'
                f'<pre>{esc(e["result"][:2500])}</pre></details></td>'
                f'<td class="shot">{img(e.get("screenshot"))}</td></tr>')

        v = outcome.get("verdict") or {}
        checks = "".join(
            f'<li class="{"ok" if c.get("satisfied") else "fail"}">{"✔" if c.get("satisfied") else "✘"} '
            f'{esc(c.get("requirement"))} <span class="muted">— {esc(c.get("evidence"))}</span></li>'
            for c in v.get("checks", []))
        vshots = "".join(img(s) for s in v.get("screenshots", []))
        status = outcome.get("status", "unknown")
        res = outcome.get("result") or ""
        verified = v.get("verified")
        color = "#15803d" if status == "success" and verified else (
            "#b45309" if status in ("needs_user",) else "#b91c1c")
        html_doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>AutoWorker run report</title>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;color:#1f2937}}
.status{{color:#fff;background:{color};display:inline-block;padding:4px 12px;border-radius:999px;font-weight:600}}
table{{border-collapse:collapse;width:100%}} td{{border-top:1px solid #e5e7eb;padding:10px;vertical-align:top}}
td.n{{width:28px;color:#6b7280}} td.shot{{width:260px}} img{{width:250px;border:1px solid #d1d5db;border-radius:4px;margin:4px}}
pre{{white-space:pre-wrap;background:#f9fafb;padding:8px;font-size:12px;max-height:320px;overflow:auto}}
.why{{color:#4b5563;margin:4px 0}} .fail{{color:#b91c1c}} .ok{{color:#15803d}} .muted{{color:#6b7280}}
.appr{{background:#fffbeb;border:1px solid #fcd34d;padding:6px;margin:6px 0;border-radius:4px}} code{{font-size:12px}}
.box{{background:#f3f4f6;padding:14px 18px;border-radius:8px;margin:14px 0}}</style></head><body>
<h1>AutoWorker run report</h1>
<div class="box"><b>Task:</b> {esc(task)}<br><br>
<span class="status">{esc(status.upper())}{" · VERIFIED" if verified else (" · NOT VERIFIED" if verified is False else "")}</span>
&nbsp; {outcome.get("steps", "?")} steps · {outcome.get("seconds", "?")}s · model {esc(outcome.get("model", ""))}</div>
<h2>Summary</h2><p>{esc(outcome.get("summary", ""))}</p>
<h3>Result data</h3><pre>{esc(res if isinstance(res, str) else json.dumps(res, indent=2))}</pre>
<h2>Independent verification</h2><p>{esc(v.get("explanation", "Not run."))}</p><ul>{checks}</ul>{vshots}
<h2>Working memory at the end</h2><pre>{esc(outcome.get("memory", ""))}</pre>
<h2>Step-by-step trace</h2><table>{"".join(rows)}</table></body></html>"""
        out = self.dir / "report.html"
        out.write_text(self.redact(html_doc), encoding="utf-8")
        return out
