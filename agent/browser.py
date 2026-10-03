"""Browser layer: drives a real Chromium via Playwright and turns pages into text
observations the model can reason about.

Design choice: instead of sending screenshots to the model, every page is
converted into a compact, numbered list of interactive elements plus the visible
text. Element ids are stamped into the DOM (data-aw-id) so the model says
"click 7" and we click exactly that element - no fragile CSS selectors invented
by the model, and it works on any website without site-specific code.
Screenshots are still captured on every step, but as evidence for humans.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from playwright.sync_api import Error as PWError
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

OBSERVE_JS = r"""
() => {
  document.querySelectorAll('[data-aw-id]').forEach(e => e.removeAttribute('data-aw-id'));
  const visible = el => {
    const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const labelFor = el => {
    if (el.id) { const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`); if (l) return l.innerText.trim(); }
    const p = el.closest('label'); if (p) return p.innerText.trim();
    return el.getAttribute('aria-label') || '';
  };
  const sel = 'a[href], button, input, select, textarea, [role=button], [role=link], [onclick]';
  const out = []; let n = 0;
  for (const el of document.querySelectorAll(sel)) {
    if (el.type === 'hidden' || !visible(el)) continue;
    n += 1; el.setAttribute('data-aw-id', String(n));
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    let covered = false;
    if (cy >= 0 && cy <= innerHeight && cx >= 0 && cx <= innerWidth) {
      const top = document.elementFromPoint(cx, cy);
      covered = !!top && !(top === el || el.contains(top) || top.contains(el));
    }
    const tag = el.tagName.toLowerCase();
    const item = {id: n, tag, type: el.type || '', name: el.name || '',
      label: labelFor(el).slice(0, 80), placeholder: el.placeholder || '',
      href: el.getAttribute('href') || '', disabled: !!el.disabled, covered, text: ''};
    if (tag === 'input' || tag === 'textarea') {
      item.value = el.type === 'password' ? (el.value ? '********' : '') : (el.value || '');
    } else if (tag === 'select') {
      item.options = [...el.options].map(o => o.text.trim());
      item.value = el.options[el.selectedIndex] ? el.options[el.selectedIndex].text.trim() : '';
    } else {
      item.text = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().replace(/\s+/g, ' ').slice(0, 100);
    }
    out.push(item);
  }
  const alerts = [...document.querySelectorAll('[role=alert], [role=dialog], .alert, .error')]
    .filter(visible).map(a => a.innerText.trim().replace(/\s+/g, ' ').slice(0, 300));
  return {url: location.href, title: document.title, elements: out, alerts,
          text: document.body ? document.body.innerText : ''};
}
"""


class ActionError(Exception):
    """A browser action failed in a way the agent should see and react to."""


@dataclass
class Observation:
    url: str
    title: str
    status: int | None
    elements: list[dict] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    text: str = ""

    def element(self, eid: int) -> dict | None:
        return next((e for e in self.elements if e["id"] == eid), None)

    @staticmethod
    def describe(e: dict) -> str:
        tag = e["tag"]
        if tag == "a":
            s = f'link "{e["text"]}" -> {e["href"]}'
        elif tag == "button" or e["type"] in ("submit", "button"):
            s = f'button "{e["text"] or e.get("value", "")}"'
        elif tag == "select":
            opts = " | ".join(e.get("options", [])[:25])
            s = f'select name={e["name"]} label="{e["label"]}" selected="{e["value"]}" options: {opts}'
        elif tag in ("input", "textarea"):
            s = (f'{tag}[{e["type"] or "text"}] name={e["name"]} label="{e["label"]}" '
                 f'value="{e["value"]}"')
            if e["placeholder"]:
                s += f' placeholder="{e["placeholder"]}"'
        else:
            s = f'{tag} "{e["text"]}"'
        if e.get("disabled"):
            s += " (disabled)"
        if e.get("covered"):
            s += " (COVERED by another element, e.g. a modal - not clickable right now)"
        return s

    def render(self, max_text: int = 3000, max_elements: int = 90) -> str:
        status = f"HTTP {self.status}" if self.status else "HTTP ?"
        lines = [f"URL: {self.url}  ({status})", f"TITLE: {self.title}"]
        if self.status and self.status >= 400:
            lines.append(f"!! PAGE ERROR: the last page load returned HTTP {self.status}.")
        for a in self.alerts:
            lines.append(f"ALERT/DIALOG: {a}")
        lines.append("INTERACTIVE ELEMENTS (use the number as element_id):")
        for e in self.elements[:max_elements]:
            lines.append(f"[{e['id']}] {self.describe(e)}")
        if len(self.elements) > max_elements:
            lines.append(f"... {len(self.elements) - max_elements} more elements not shown")
        text = re.sub(r"\n{3,}", "\n\n", self.text).strip()
        if len(text) > max_text:
            text = text[:max_text] + f"\n... [page text truncated, {len(text)} chars total; " \
                                     f"call observe with full_text=true to read more]"
        lines.append("PAGE TEXT:")
        lines.append(text or "(empty)")
        return "\n".join(lines)


class Browser:
    def __init__(self, headless: bool = True, allowed_hosts: list[str] | None = None,
                 slow_mo: int = 0):
        self.headless = headless
        self.allowed_hosts = allowed_hosts or []
        self.slow_mo = slow_mo
        self.last_status: int | None = None
        self._pw = self._browser = self.context = self.page = None
        self.last_obs: Observation | None = None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> "Browser":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless, slow_mo=self.slow_mo)
        self.context = self._browser.new_context(viewport={"width": 1280, "height": 860})
        self.page = self.context.new_page()
        self.page.on("response", self._on_response)
        return self

    def close(self) -> None:
        for obj in (self.context, self._browser):
            try:
                obj and obj.close()
            except Exception:
                pass
        if self._pw:
            self._pw.stop()

    def _on_response(self, resp) -> None:
        try:
            if resp.request.resource_type == "document" and resp.frame == self.page.main_frame:
                self.last_status = resp.status
        except Exception:
            pass

    # ------------------------------------------------------------- helpers
    def check_url(self, url: str) -> str:
        url = urljoin(self.page.url if self.page.url.startswith("http") else "", url)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ActionError(f"Only http(s) URLs are allowed, got {url!r}.")
        if self.allowed_hosts and parsed.hostname not in self.allowed_hosts:
            raise ActionError(f"Navigation to {parsed.hostname} is blocked by policy. "
                              f"Allowed hosts: {', '.join(self.allowed_hosts)}.")
        if parsed.path.startswith("/__"):
            raise ActionError("That URL is an internal test endpoint and is blocked by policy.")
        return url

    def _settle(self) -> None:
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=8000)
            self.page.wait_for_load_state("networkidle", timeout=3000)
        except PWTimeout:
            pass

    def _locate(self, element_id: int):
        loc = self.page.locator(f'[data-aw-id="{int(element_id)}"]')
        if loc.count() == 0:
            raise ActionError(f"Element [{element_id}] does not exist on the current page. "
                              "Element ids change after every page load - use ids from the "
                              "latest observation.")
        return loc.first

    @staticmethod
    def _clean_error(exc: Exception) -> str:
        msg = str(exc).split("\n")
        first = msg[0]
        if "intercepts pointer events" in str(exc):
            return ("The click was blocked because another element (probably a modal or "
                    "overlay) is covering the target. Deal with the overlay first.")
        return first[:300]

    # ------------------------------------------------------------- actions
    def observe(self) -> Observation:
        data = self.page.evaluate(OBSERVE_JS)
        self.last_obs = Observation(data["url"], data["title"], self.last_status,
                                    data["elements"], data["alerts"], data["text"])
        return self.last_obs

    def navigate(self, url: str) -> None:
        url = self.check_url(url)
        try:
            resp = self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
            if resp:
                self.last_status = resp.status
        except PWError as exc:
            raise ActionError(f"Could not load {url}: {self._clean_error(exc)}")
        self._settle()

    def click(self, element_id: int) -> None:
        loc = self._locate(element_id)
        href = loc.get_attribute("href")
        if href:
            self.check_url(href)
        try:
            loc.click(timeout=4000)
        except PWError as exc:
            raise ActionError(f"Click on [{element_id}] failed: {self._clean_error(exc)}")
        self._settle()

    def fill(self, element_id: int, text: str) -> None:
        loc = self._locate(element_id)
        try:
            loc.fill(text, timeout=4000)
        except PWError as exc:
            raise ActionError(f"Typing into [{element_id}] failed: {self._clean_error(exc)}")

    def select(self, element_id: int, option: str) -> str:
        loc = self._locate(element_id)
        try:
            chosen = loc.select_option(label=option, timeout=4000)
        except PWError:
            try:
                chosen = loc.select_option(value=option, timeout=4000)
            except PWError as exc:
                options = loc.evaluate("el => [...el.options].map(o => o.text.trim())")
                raise ActionError(f"Option {option!r} not found in [{element_id}]. "
                                  f"Available options: {options}") from exc
        return ", ".join(chosen)

    def press(self, key: str) -> None:
        try:
            self.page.keyboard.press(key)
        except PWError as exc:
            raise ActionError(f"Key press failed: {self._clean_error(exc)}")
        self._settle()

    def fetch(self, url: str) -> tuple[bytes, str, int]:
        """Download a URL using the browser's cookies (so logged-in files work)."""
        url = self.check_url(url)
        resp = self.context.request.get(url, timeout=15000)
        return resp.body(), resp.headers.get("content-type", ""), resp.status

    def screenshot(self, path) -> str | None:
        try:
            self.page.screenshot(path=str(path), full_page=False)
            return str(path)
        except Exception:
            return None
