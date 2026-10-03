"""How the agent talks to a person: clarifying questions and approvals."""
from __future__ import annotations


class ConsoleHuman:
    """Interactive terminal human."""

    def ask(self, question: str) -> str:
        print(f"\n\033[95m🙋 Agent asks:\033[0m {question}")
        answer = input("   Your answer (leave empty to tell it to stop): ").strip()
        return answer or "The user did not answer. Do not guess; finish with status needs_user."

    def approve(self, summary: str) -> tuple[bool, str]:
        print(f"\n\033[93m⚠  Approval needed:\033[0m\n{summary}")
        ans = input("   Approve? [y]es / [n]o / or type feedback: ").strip()
        if ans.lower() in ("y", "yes", ""):
            return True, ""
        if ans.lower() in ("n", "no"):
            return False, "The user rejected this action."
        return False, f"The user rejected this action with feedback: {ans}"


class AutoHuman:
    """Non-interactive human for tests, CI and unattended demos.

    approve_all: auto-approve commit actions (still logged in the trace).
    answers: queued answers for ask_human; when empty, the agent is told nobody
    is available, which is the correct unattended behaviour.
    """

    def __init__(self, approve_all: bool = True, answers: list[str] | None = None,
                 reject_first: int = 0):
        self.approve_all = approve_all
        self.answers = list(answers or [])
        self.reject_first = reject_first
        self.approvals: list[str] = []
        self.questions: list[str] = []

    def ask(self, question: str) -> str:
        self.questions.append(question)
        if self.answers:
            return self.answers.pop(0)
        return ("No human is available right now (unattended run). Do not guess - finish with "
                "status needs_user and explain exactly what you need.")

    def approve(self, summary: str) -> tuple[bool, str]:
        self.approvals.append(summary)
        if self.reject_first > 0:
            self.reject_first -= 1
            return False, "The user rejected this action: please double-check the values first."
        return (True, "") if self.approve_all else (False, "Auto-rejected (approve_all=False).")
