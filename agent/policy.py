"""Safety layer that sits between the model and the real world.

These rules are enforced in code, not in the prompt, so a confused or
prompt-injected model cannot talk its way around them:

* Human approval before "commit" actions (save / submit / pay / delete ...).
* Secrets are referenced by name ({{secret:erp_password}}) and substituted only
  at the moment of typing. The model, the trace and the report never see them.
* Navigation is restricted to an allow-list of hosts (enforced in browser.py).
"""
from __future__ import annotations

import re

COMMIT_WORDS = re.compile(
    r"\b(save|submit|pay|payment|delete|remove|approve|send|confirm|update|transfer|"
    r"place order|purchase|sign out|log ?out)\b", re.I)
SECRET_RE = re.compile(r"\{\{secret:([a-zA-Z0-9_]+)\}\}")


class Policy:
    def __init__(self, approval_mode: str = "risky", secrets: dict[str, str] | None = None):
        if approval_mode not in ("risky", "always", "never"):
            raise ValueError("approval_mode must be risky, always or never")
        self.mode = approval_mode
        self.secrets = secrets or {}

    # ---------------------------------------------------------- approvals
    def approval_reason(self, tool: str, element: dict | None) -> str | None:
        """Return why this action needs a human OK, or None if it can run freely."""
        if self.mode == "never":
            return None
        if tool not in ("click", "press_key"):
            return None
        if self.mode == "always":
            return "approval mode is 'always'"
        if tool == "press_key":
            return None  # Enter-to-submit is handled by checking the key in tools.py
        if not element:
            return None
        label = " ".join(str(element.get(k, "")) for k in ("text", "value", "label"))
        is_submit = element["tag"] == "button" or element.get("type") in ("submit", "button")
        if element.get("tag") == "a" and COMMIT_WORDS.search(label):
            return f'link "{label.strip()}" looks like it commits a change'
        if is_submit and COMMIT_WORDS.search(label):
            return f'button "{label.strip()}" commits data to a system'
        return None

    # ---------------------------------------------------------- secrets
    def resolve(self, text: str) -> str:
        def sub(m):
            name = m.group(1)
            if name not in self.secrets:
                raise KeyError(f"Unknown secret {name!r}. Available: {sorted(self.secrets)}")
            return self.secrets[name]
        return SECRET_RE.sub(sub, text)

    def redact(self, text: str) -> str:
        for name, value in self.secrets.items():
            if value and len(value) >= 4:
                text = text.replace(value, f"{{{{secret:{name}}}}}")
        return text
