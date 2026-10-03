"""Acme Corp sandbox: a fake company mailbox and a fake internal ERP.

Run standalone:  python -m sandbox.app   (serves on http://127.0.0.1:5055)

The ERP intentionally behaves like real internal software:
  * it needs a login and keeps a session cookie;
  * it validates input strictly (amounts must be plain numbers, dates DD-MM-YYYY);
  * it rejects duplicate invoice numbers;
  * with CHAOS enabled, the first save after a reset fails with HTTP 503 and the
    form data is lost (a realistic transient outage the agent must recover from);
  * the mailbox shows a blocking "policy update" modal on first visit.

Test-only endpoints (/__reset, /__state, /__chaos) exist so demos and automated
tests can start from a clean state. The agent is never told about them and the
URL policy blocks the agent from visiting them.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from functools import wraps

from flask import (Flask, abort, jsonify, redirect, render_template_string, request,
                   send_from_directory, session, url_for)

from .fixtures import EMAILS, ERP_USERS, GENERATED_DIR, generate_pdfs, initial_state

app = Flask(__name__)
app.secret_key = "sandbox-only-not-a-secret"

STATE = initial_state()
CHAOS = {"enabled": os.environ.get("SANDBOX_CHAOS", "1") != "0", "save_failures_left": 1}


def reset_state(chaos: bool | None = None) -> None:
    global STATE
    STATE = initial_state()
    if chaos is not None:
        CHAOS["enabled"] = chaos
    CHAOS["save_failures_left"] = 1 if CHAOS["enabled"] else 0


reset_state()

# ---------------------------------------------------------------- templates

BASE = """<!doctype html><html><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1d2330}
 header{background:{{ color }};color:#fff;padding:12px 24px;display:flex;gap:24px;align-items:center}
 header a{color:#fff;text-decoration:none;opacity:.9} header b{font-size:18px}
 main{max-width:960px;margin:24px auto;background:#fff;padding:24px;border-radius:8px;box-shadow:0 1px 3px #0002}
 table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid #e3e5ea;padding:8px;text-align:left;font-size:14px}
 .alert{padding:10px 14px;border-radius:6px;margin-bottom:14px}
 .alert-error{background:#fde8e8;color:#9b1c1c;border:1px solid #f5b5b5}
 .alert-success{background:#e6f6ec;color:#1d6b3a;border:1px solid #a9dcbb}
 label{display:block;margin:12px 0 4px;font-weight:600;font-size:14px}
 input,select{padding:8px;width:320px;border:1px solid #c5c9d3;border-radius:4px;font-size:14px}
 button{margin-top:16px;padding:9px 18px;background:{{ color }};color:#fff;border:0;border-radius:4px;font-size:14px;cursor:pointer}
 .muted{color:#6b7280;font-size:13px} pre{white-space:pre-wrap;font-family:inherit}
 .overlay{position:fixed;inset:0;background:#0009;display:flex;align-items:center;justify-content:center}
 .modal{background:#fff;padding:28px;border-radius:8px;max-width:420px}
</style></head><body>
<header><b>{{ brand }}</b>{% for href, label in nav %}<a href="{{ href }}">{{ label }}</a>{% endfor %}</header>
<main>{{ body|safe }}</main>{{ extra|safe }}</body></html>"""

MAIL_NAV = [("/mail", "Inbox")]
ERP_NAV = [("/erp", "Dashboard"), ("/erp/invoices", "Invoices"),
           ("/erp/invoices/new", "New invoice"), ("/erp/vendors", "Vendors"),
           ("/erp/logout", "Sign out")]


def page(title, body, *, app_name="erp", extra="", status=200):
    if app_name == "mail":
        brand, color, nav = "Acme Mail", "#2563eb", MAIL_NAV
    else:
        brand, color, nav = "Acme ERP", "#0f766e", ERP_NAV if session.get("user") else []
    html = render_template_string(BASE, title=title, body=body, brand=brand, color=color,
                                  nav=nav, extra=extra)
    return html, status


def fmt_amount(v: float) -> str:
    return f"{v:,.2f}"


# ---------------------------------------------------------------- mailbox

@app.route("/")
def home():
    return page("Acme Corp", '<h2>Acme Corp intranet</h2><p><a href="/mail">Mail</a> · '
                '<a href="/erp">ERP</a></p>', app_name="mail")


def mail_modal() -> str:
    if request.cookies.get("policy_ack"):
        return ""
    return ('<div class="overlay" id="policy-modal"><div class="modal" role="dialog">'
            '<h3>We updated our acceptable-use policy</h3>'
            '<p>Please acknowledge the updated mailbox policy to continue.</p>'
            '<form method="post" action="/mail/ack"><input type="hidden" name="next" value="'
            + request.path + '"><button type="submit">Acknowledge</button></form>'
            '</div></div>')


@app.route("/mail/ack", methods=["POST"])
def mail_ack():
    resp = redirect(request.form.get("next") or "/mail")
    resp.set_cookie("policy_ack", "1")
    return resp


@app.route("/mail")
def inbox():
    rows = "".join(
        f'<tr><td>{e["from_name"]}<div class="muted">{e["from_addr"]}</div></td>'
        f'<td><a href="/mail/{e["id"]}">{e["subject"]}</a>'
        f'{" 📎" if e["attachment"] else ""}</td><td>{e["date"]}</td></tr>'
        for e in sorted(EMAILS, key=lambda e: e["date"], reverse=True))
    body = (f"<h2>Inbox</h2><p class='muted'>accounts@acme.example · {len(EMAILS)} messages</p>"
            f"<table><tr><th>From</th><th>Subject</th><th>Received</th></tr>{rows}</table>")
    return page("Inbox · Acme Mail", body, app_name="mail", extra=mail_modal())


@app.route("/mail/<int:mid>")
def message(mid):
    e = next((e for e in EMAILS if e["id"] == mid), None)
    if not e:
        abort(404)
    att = ""
    if e["attachment"]:
        att = (f'<p><b>Attachment:</b> <a href="/mail/attachments/{e["attachment"]}">'
               f'{e["attachment"]}</a></p>')
    body = (f'<p><a href="/mail">← Back to inbox</a></p><h2>{e["subject"]}</h2>'
            f'<p class="muted">From: {e["from_name"]} &lt;{e["from_addr"]}&gt; · {e["date"]}</p>'
            f'<pre>{e["body"]}</pre>{att}')
    return page(e["subject"] + " · Acme Mail", body, app_name="mail", extra=mail_modal())


@app.route("/mail/attachments/<path:name>")
def attachment(name):
    return send_from_directory(GENERATED_DIR, name)


# ---------------------------------------------------------------- ERP

def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not session.get("user"):
            return redirect(url_for("erp_login", next=request.path))
        return fn(*a, **kw)
    return wrapper


@app.route("/erp/login", methods=["GET", "POST"])
def erp_login():
    error = ""
    if request.method == "POST":
        u, p = request.form.get("username", ""), request.form.get("password", "")
        if ERP_USERS.get(u) == p:
            session["user"] = u
            return redirect(request.args.get("next") or "/erp")
        error = '<div class="alert alert-error" role="alert">Invalid username or password.</div>'
    body = (f"<h2>Sign in to Acme ERP</h2>{error}<form method='post'>"
            "<label for='username'>Username</label><input id='username' name='username'>"
            "<label for='password'>Password</label><input id='password' name='password' type='password'>"
            "<br><button type='submit'>Sign in</button></form>")
    return page("Sign in · Acme ERP", body)


@app.route("/erp/logout")
def erp_logout():
    session.clear()
    return redirect("/erp/login")


@app.route("/erp")
@login_required
def erp_home():
    open_count = sum(1 for i in STATE["invoices"] if i["status"] == "Open")
    body = (f"<h2>Welcome, {session['user']}</h2><p>{open_count} open invoices.</p>"
            "<p><a href='/erp/invoices/new'>Enter a new vendor invoice</a></p>")
    return page("Dashboard · Acme ERP", body)


@app.route("/erp/invoices")
@login_required
def invoice_list():
    flash = ""
    if request.args.get("saved"):
        flash = (f'<div class="alert alert-success" role="alert">Invoice '
                 f'{request.args["saved"]} saved successfully.</div>')
    rows = "".join(
        f'<tr><td>{i["number"]}</td><td>{i["vendor"]}</td><td>{fmt_amount(i["amount"])}</td>'
        f'<td>{i["currency"]}</td><td>{datetime.strptime(i["due_date"], "%Y-%m-%d"):%d-%m-%Y}</td>'
        f'<td>{i["status"]}</td><td>{i["entered_by"]}</td></tr>'
        for i in sorted(STATE["invoices"], key=lambda i: i["id"], reverse=True))
    body = (f"<h2>Vendor invoices</h2>{flash}<table><tr><th>Invoice no.</th><th>Vendor</th>"
            f"<th>Amount</th><th>Currency</th><th>Due date</th><th>Status</th><th>Entered by</th>"
            f"</tr>{rows}</table>")
    return page("Invoices · Acme ERP", body)


def invoice_form(values: dict, errors: list[str]) -> str:
    err = "".join(f'<div class="alert alert-error" role="alert">{e}</div>' for e in errors)
    vopts = "".join(
        f'<option{" selected" if values.get("vendor") == v["name"] else ""}>{v["name"]}</option>'
        for v in STATE["vendors"])
    copts = "".join(f'<option{" selected" if values.get("currency") == c else ""}>{c}</option>'
                    for c in ("INR", "USD", "EUR"))
    esc = lambda k: (values.get(k) or "").replace('"', "&quot;")
    return (f"<h2>New vendor invoice</h2>{err}<form method='post'>"
            f"<label for='vendor'>Vendor</label><select id='vendor' name='vendor'>"
            f"<option value=''>— choose vendor —</option>{vopts}</select>"
            f"<label for='number'>Invoice number</label><input id='number' name='number' value=\"{esc('number')}\">"
            f"<label for='amount'>Amount (total payable, incl. tax)</label>"
            f"<input id='amount' name='amount' placeholder='e.g. 12500.00' value=\"{esc('amount')}\">"
            f"<label for='currency'>Currency</label><select id='currency' name='currency'>{copts}</select>"
            f"<label for='due_date'>Due date</label>"
            f"<input id='due_date' name='due_date' placeholder='DD-MM-YYYY' value=\"{esc('due_date')}\">"
            f"<br><button type='submit'>Save invoice</button></form>")


@app.route("/erp/invoices/new", methods=["GET", "POST"])
@login_required
def invoice_new():
    if request.method == "GET":
        return page("New invoice · Acme ERP", invoice_form({}, []))

    v = {k: (request.form.get(k) or "").strip() for k in ("vendor", "number", "amount",
                                                         "currency", "due_date")}
    if CHAOS["enabled"] and CHAOS["save_failures_left"] > 0:
        CHAOS["save_failures_left"] -= 1
        body = ("<h2>503 Service Temporarily Unavailable</h2>"
                '<div class="alert alert-error" role="alert">The ERP is under heavy load and could '
                "not process your request. Your changes were NOT saved. Please try again.</div>"
                "<p><a href='/erp/invoices/new'>Back to the invoice form</a></p>")
        return page("503 · Acme ERP", body, status=503)

    errors = []
    if v["vendor"] not in {x["name"] for x in STATE["vendors"]}:
        errors.append("Please choose a vendor.")
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9\-/]{2,30}", v["number"] or ""):
        errors.append("Invoice number is required (letters, digits and dashes only).")
    elif any(i["number"] == v["number"] for i in STATE["invoices"]):
        dup = next(i for i in STATE["invoices"] if i["number"] == v["number"])
        errors.append(f"Invoice {v['number']} already exists for {dup['vendor']} "
                      f"(status: {dup['status']}). Duplicate invoices are not allowed.")
    if not re.fullmatch(r"\d+(\.\d{1,2})?", v["amount"]):
        errors.append("Amount must be a plain number without currency symbols or commas "
                      "(e.g. 184500.00).")
    due = None
    if not re.fullmatch(r"\d{2}-\d{2}-\d{4}", v["due_date"]):
        errors.append("Due date must be in DD-MM-YYYY format.")
    else:
        try:
            due = datetime.strptime(v["due_date"], "%d-%m-%Y")
        except ValueError:
            errors.append("Due date is not a valid calendar date.")
    if errors:
        return page("New invoice · Acme ERP", invoice_form(v, errors), status=422)

    inv = {"id": STATE["next_invoice_id"], "vendor": v["vendor"], "number": v["number"],
           "amount": float(v["amount"]), "currency": v["currency"] or "INR",
           "due_date": due.strftime("%Y-%m-%d"), "status": "Open", "entered_by": session["user"]}
    STATE["next_invoice_id"] += 1
    STATE["invoices"].append(inv)
    STATE["audit_log"].append({"action": "create_invoice", "number": inv["number"],
                               "user": session["user"]})
    return redirect(f"/erp/invoices?saved={inv['number']}")


@app.route("/erp/vendors")
@login_required
def vendor_list():
    flash = ""
    if request.args.get("saved"):
        flash = '<div class="alert alert-success" role="alert">Vendor details updated.</div>'
    rows = "".join(f'<tr><td>{v["name"]}</td><td>{v["contact_email"]}</td>'
                   f'<td><a href="/erp/vendors/{v["id"]}/edit">Edit {v["name"]}</a></td></tr>'
                   for v in STATE["vendors"])
    body = (f"<h2>Vendors</h2>{flash}<table><tr><th>Vendor</th><th>Accounts contact</th><th></th>"
            f"</tr>{rows}</table>")
    return page("Vendors · Acme ERP", body)


@app.route("/erp/vendors/<int:vid>/edit", methods=["GET", "POST"])
@login_required
def vendor_edit(vid):
    vendor = next((v for v in STATE["vendors"] if v["id"] == vid), None)
    if not vendor:
        abort(404)
    error = ""
    if request.method == "POST":
        email = (request.form.get("contact_email") or "").strip()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", email):
            error = '<div class="alert alert-error" role="alert">Enter a valid email address.</div>'
        else:
            vendor["contact_email"] = email
            STATE["audit_log"].append({"action": "update_vendor", "vendor": vendor["name"],
                                       "email": email, "user": session["user"]})
            return redirect("/erp/vendors?saved=1")
    body = (f"<h2>Edit vendor: {vendor['name']}</h2>{error}<form method='post'>"
            f"<label for='contact_email'>Accounts contact email</label>"
            f"<input id='contact_email' name='contact_email' value='{vendor['contact_email']}'>"
            f"<br><button type='submit'>Save changes</button></form>")
    return page(f"Edit {vendor['name']} · Acme ERP", body)


# ---------------------------------------------------------------- test hooks

@app.route("/__reset", methods=["POST"])
def __reset():
    chaos = request.args.get("chaos")
    reset_state(None if chaos is None else chaos != "0")
    return jsonify(ok=True, chaos=CHAOS["enabled"])


@app.route("/__state")
def __state():
    return jsonify(STATE)


generate_pdfs()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("SANDBOX_PORT", 5055)), debug=False)
