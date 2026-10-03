"""Runtime settings, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader so we don't need python-dotenv. Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    provider: str = "anthropic"          # "anthropic" or "openai" (any OpenAI-compatible API)
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None          # for OpenAI-compatible providers (Gemini, Groq, ...)
    sandbox_url: str = "http://127.0.0.1:5055"
    max_steps: int = 45
    approval: str = "risky"              # "risky" | "always" | "never"
    headless: bool = True
    allowed_hosts: list[str] = field(default_factory=lambda: ["127.0.0.1", "localhost"])
    runs_dir: Path = Path("runs")
    notes_path: Path = Path("memory/site_notes.json")
    secrets: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        default_model = "claude-sonnet-5-5" if provider == "anthropic" else "gpt-4.1-mini"
        api_key = os.environ.get("LLM_API_KEY") or (
            os.environ.get("ANTHROPIC_API_KEY") if provider == "anthropic"
            else os.environ.get("OPENAI_API_KEY"))
        # Credentials the agent may *use* but never *see*. The model only knows the
        # secret names and types placeholders such as {{secret:erp_password}}.
        secrets = {
            "erp_username": os.environ.get("ERP_USERNAME", "ops.agent"),
            "erp_password": os.environ.get("ERP_PASSWORD", "demo-pass-123"),
        }
        return cls(
            provider=provider,
            model=os.environ.get("LLM_MODEL", default_model),
            api_key=api_key,
            base_url=os.environ.get("LLM_BASE_URL") or None,
            sandbox_url=os.environ.get("SANDBOX_URL", "http://127.0.0.1:5055").rstrip("/"),
            max_steps=int(os.environ.get("MAX_STEPS", 45)),
            approval=os.environ.get("APPROVAL_MODE", "risky"),
            headless=os.environ.get("HEADLESS", "1") != "0",
            secrets=secrets,
        )
