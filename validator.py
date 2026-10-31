"""Compare candidate outputs with baseline traces and reject bad rewrites."""

from __future__ import annotations

from typing import Any, Literal
from collections.abc import Sequence
from difflib import SequenceMatcher
from functools import lru_cache

from pydantic import BaseModel

from synapse.ir import Decision, ExecPlan, RunRecord, Workflow
from synapse.compiler import compile_workflow
from synapse.executor import execute, final_outputs, resolve
from synapse.backends import Backends
from synapse.backends.llm import output_schema
from synapse.ir import NodeKind


class Check(BaseModel):
    case_id: str
    node_id: str
    kind: Literal["exact", "schema", "decision", "similarity", "assertion"]
    passed: bool
    detail: str = ""


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer
    from synapse.backends.retrieval import embedding_model_name
    return SentenceTransformer(embedding_model_name(), device="cpu", local_files_only=True)


def _similarity(first: str, second: str) -> float:
    return _text_similarity(first, second)[0]


def _text_similarity(first: str, second: str) -> tuple[float, str]:
    """Text similarity with an offline fallback when the embedding model is unavailable."""
    if first == second:
        return 1.0, "exact text"
    try:
        vectors = _embedding_model().encode([first, second], normalize_embeddings=True)
        return float(vectors[0] @ vectors[1]), "embedding"
    except Exception:
        return SequenceMatcher(None, first, second).ratio(), "difflib fallback"


def _similar_enough(first: str, second: str, floor: float) -> tuple[bool, float, str]:
    score, method = _text_similarity(first, second)
    threshold = max(0.85, 1 - floor - 0.03)
    return score + 1e-6 >= threshold, score, method


def _text_leaves(value: Any, prefix: str = "") -> dict[str, str]:
    """String leaves of a structured result, for measuring baseline variation."""
    if isinstance(value, str):
        return {prefix: value}
    if isinstance(value, dict):
        return {key: text for name, child in value.items()
                for key, text in _text_leaves(child, f"{prefix}.{name}").items()}
    if isinstance(value, list):
        return {key: text for index, child in enumerate(value)
                for key, text in _text_leaves(child, f"{prefix}[{index}]").items()}
    return {}


def validate(golden: list[RunRecord], candidate: list[RunRecord],
             noise_floor: float, *, wf: Workflow | None = None,
             key_fields: Sequence[str] = ()) -> list[Check]:
    """Compare candidate runs with golden runs and return one Check per comparison."""
    candidates = {record.case_id: record for record in candidate}
    definitions = {node.id: node for node in wf.nodes} if wf is not None else {}
    checks: list[Check] = []
    for original in golden:
        other = candidates.get(original.case_id)
        if other is None:
            checks.append(Check(case_id=original.case_id, node_id="run", kind="exact",
                                passed=False, detail="candidate run is missing"))
            continue
        original_nodes = {node.node_id: node for node in original.nodes}
        new_nodes = {node.node_id: node for node in other.nodes}
        for reference in key_fields:
            try:
                expected_key = resolve(reference, final_outputs(original))
                actual_key = resolve(reference, final_outputs(other))
                passed = expected_key == actual_key
                detail = f"expected {expected_key}, got {actual_key}"
            except KeyError as exc:
                passed, detail = False, str(exc)
            checks.append(Check(case_id=original.case_id, node_id=reference,
                                kind="decision", passed=passed, detail=detail))
        for node_id, old in original_nodes.items():
            if node_id not in new_nodes:
                continue  # removed or merged nodes are checked through their consumers
            new = new_nodes[node_id]
            if old.error or new.error:
                checks.append(Check(case_id=original.case_id, node_id=node_id, kind="exact",
                                    passed=False, detail=f"errors: {old.error}; {new.error}"))
                continue
            definition = definitions.get(node_id)
            schema = output_schema(definition) if definition and definition.kind is NodeKind.LLM else None
            for field, expected in old.output.items():
                actual = new.output.get(field)
                if schema is not None:
                    try:
                        schema.model_validate(actual)
                        schema_ok = True
                    except Exception:
                        schema_ok = False
                    checks.append(Check(case_id=original.case_id, node_id=node_id,
                                        kind="schema", passed=schema_ok,
                                        detail=schema.__name__ if schema_ok else "invalid schema"))
                    if not schema_ok:
                        continue
                    for key, value in expected.items():
                        got = actual.get(key) if isinstance(actual, dict) else None
                        is_enum = hasattr(schema, "model_fields") and bool(
                            schema.model_fields.get(key) and
                            schema.model_fields[key].annotation and
                            getattr(schema.model_fields[key].annotation, "__origin__", None)
                            is Literal)
                        if isinstance(value, str) and isinstance(got, str) and not is_enum:
                            passed, score, method = _similar_enough(value, got, noise_floor)
                            kind = "similarity"
                            detail = f"{field}.{key}: {method} similarity {score:.3f}"
                        else:
                            passed, kind = got == value, "exact"
                            detail = f"{field}.{key}" if passed else f"{field}.{key} differs"
                        checks.append(Check(case_id=original.case_id, node_id=node_id,
                                            kind=kind, passed=passed, detail=detail))
                elif definition and definition.kind is NodeKind.LLM and isinstance(expected, str) and isinstance(actual, str):
                    passed, similarity, method = _similar_enough(expected, actual, noise_floor)
                    checks.append(Check(case_id=original.case_id, node_id=node_id,
                                        kind="similarity", passed=passed,
                                        detail=f"{method} similarity {similarity:.3f}"))
                else:
                    checks.append(Check(case_id=original.case_id, node_id=node_id,
                                        kind="exact", passed=actual == expected,
                                        detail=field if actual == expected else f"{field} differs"))
        for node_id in new_nodes.keys() - original_nodes.keys():
            checks.append(Check(case_id=original.case_id, node_id=node_id, kind="exact",
                                passed=False, detail="unexpected candidate node"))
        if wf is not None:
            for reference in wf.outputs:
                try:
                    resolve(reference, final_outputs(original))
                    resolve(reference, final_outputs(other))
                    passed = True
                except KeyError:
                    passed = False
                checks.append(Check(case_id=original.case_id, node_id=reference,
                                    kind="assertion", passed=passed,
                                    detail=f"workflow output {reference} is present"))
    return checks


def noise_floor(first: Sequence[RunRecord], second: Sequence[RunRecord]) -> float:
    """Largest observed baseline text variation, as a fraction of changed characters."""
    peers = {record.case_id: record for record in second}
    variation = 0.0
    for record in first:
        peer = peers.get(record.case_id)
        if peer is None:
            continue
        old_nodes = {n.node_id: n for n in record.nodes}
        for new in peer.nodes:
            old = old_nodes.get(new.node_id)
            if old is None:
                continue
            old_text = _text_leaves(old.output)
            new_text = _text_leaves(new.output)
            for field, value in old_text.items():
                if field in new_text:
                    variation = max(variation, 1 - _similarity(value, new_text[field]))
    return variation


async def validate_workflow(wf: Workflow, cases: Sequence[dict], backends: Backends,
                            unsafe_route_risk: bool = False,
                            live: bool = False) -> tuple[ExecPlan, list[Check], float]:
    """Apply passes incrementally and retain only those that pass all cases."""
    if unsafe_route_risk and wf.name != "loan_prescreen":
        raise ValueError("the unsafe risk routing demo is only defined for loan_prescreen")
    golden = [await execute(wf, case, "baseline", backends, live=live) for case in cases]
    repeat = [await execute(wf, case, "baseline", backends, live=live) for case in cases]
    floor = noise_floor(golden, repeat)
    try:
        from mock_services import suite_for
        key_fields = [suite_for(wf.name).key_field]
    except LookupError:
        key_fields = []
    accepted: list[str] = []
    rejected: list[Decision] = []
    final_checks: list[Check] = []
    for name in ("dead_nodes", "cse", "compact_inputs", "parallelise", "memoise", "routing"):
        proposed = compile_workflow(wf, passes=tuple([*accepted, name]),
                                    routing_probe=name == "routing")
        candidate = [await execute(wf, case, "compiled", backends, plan=proposed, live=live)
                     for case in cases]
        checks = validate(golden, candidate, floor, wf=wf, key_fields=key_fields)
        if all(c.passed for c in checks):
            accepted.append(name)
            final_checks = checks
        else:
            first = next(c for c in checks if not c.passed)
            rejected.append(Decision(pass_name=name, nodes=[first.node_id], status="rejected",
                                     reason=f"case {first.case_id}: {first.detail}"))
    routing_evidence = None
    if "routing" in accepted:
        routed = [node.id for node in proposed.workflow.nodes if node.fallback_model]
        routing_evidence = {node_id: None for node_id in routed}
    plan = compile_workflow(wf, passes=tuple(accepted), routing_evidence=routing_evidence)
    if unsafe_route_risk:
        unsafe = plan.model_copy(deep=True)
        from synapse.passes.routing import final_steps
        final = final_steps(unsafe.workflow)
        decision = next((node for node in unsafe.workflow.nodes
                         if node.id in final and node.kind is NodeKind.LLM), None)
        if decision is None:
            raise ValueError("unsafe routing demo needs a final LLM decision step")
        decision.model = "small"
        candidate = [await execute(wf, case, "compiled", backends, plan=unsafe, live=live)
                     for case in cases]
        checks = validate(golden, candidate, floor, wf=wf, key_fields=key_fields)
        if any(not c.passed for c in checks):
            first = next(c for c in checks if not c.passed)
            rejected.append(Decision(pass_name="unsafe_route_risk", nodes=[decision.id],
                                     status="rejected", reason=f"case {first.case_id}: {first.detail}"))
        else:
            plan = unsafe
            final_checks = checks
    plan.decisions.extend(rejected)
    return plan, final_checks, floor
