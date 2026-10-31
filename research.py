"""A deterministic evidence-only stand-in; it does not score HotpotQA accuracy."""

from typing import Any


def research_answer(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep source attribution explicit without pretending to solve the question."""
    evidence = [inputs["evidence_one"], inputs["evidence_two"]]
    return {"answer": "Evidence for review: " + " ".join(p["text"][:180] for p in evidence),
            "sources": [p["title"] for p in evidence], "supported": False}


MOCK_TASKS = {"research_answer": research_answer}
