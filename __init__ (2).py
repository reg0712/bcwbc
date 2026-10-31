"""Mock bank services and the test suites that drive them: loan applicants and complaints.
Synthetic data only: obviously fake names, PANs and accounts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mock_services import complaints, loan, research


@dataclass(frozen=True)
class Suite:
    """A workflow's test set: its cases, the output field that carries the decision, and the
    decision each case should reach under the mock rule."""

    workflow: str
    cases: Callable[[], list[dict[str, Any]]]
    key_field: str
    expected: Callable[[str], Any]

    def case(self, case_id: str) -> dict[str, Any]:
        """One case by id; LookupError names the valid ids."""
        by_id = {case["case_id"]: case for case in self.cases()}
        if case_id not in by_id:
            raise LookupError(f"no case {case_id!r} for {self.workflow}; "
                              f"cases are {', '.join(by_id)}")
        return by_id[case_id]


SUITES = {"loan_prescreen": Suite("loan_prescreen", loan.cases, "risk.decision",
                                  loan.expected_decision),
          "complaint_triage": Suite("complaint_triage", complaints.cases, "triage.category",
                                    complaints.expected_category)}

# Deterministic stand-ins for LLM prompts, keyed by prompt file name (prompts/<name>.txt).
# Each workflow's prompts have their own names, so no task replaces another.
MOCK_TASKS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {**loan.MOCK_TASKS,
                                                                     **complaints.MOCK_TASKS,
                                                                     **research.MOCK_TASKS}


def suite_for(workflow: str) -> Suite:
    """The test suite of a workflow; LookupError if it has none."""
    if workflow not in SUITES:
        raise LookupError(f"no test cases for workflow {workflow!r}")
    return SUITES[workflow]
