"""Pass protocol shared by every optimisation pass.

A pass plans first, returning Decision records for everything it considered, including
skipped and rejected ones, each with a reason; then it applies the applied decisions.
"""

from __future__ import annotations

from typing import Protocol

from synapse.ir import Decision, Profile, Workflow

__all__ = ["Decision", "Pass"]


class Pass(Protocol):
    name: str

    def plan(self, wf: Workflow, profile: Profile | None) -> list[Decision]: ...

    def apply(self, wf: Workflow, decisions: list[Decision]) -> Workflow: ...
