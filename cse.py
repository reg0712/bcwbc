"""Common-subexpression elimination: merge nodes that compute the same thing.

Two nodes are identical when their kind, effect, inputs (in order), model and fallback, prompt,
tool, index, query, expiry and tests all match. A later duplicate is merged into the first
copy and its readers read the kept outputs instead (policy_b.x becomes policy_a.x). Nodes are
compared after earlier merges are applied, so duplicates of duplicates merge too.

A merge is legal only when both nodes are pure or read (never a write, and an untagged node
counts as a write), deterministic or an LLM call (every LLM call runs at temperature 0), not
required checks, the duplicate produces no workflow output, no write is listed between two
identical reads, and every reader of the duplicate is sent the same arguments afterwards.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from synapse.graph import build_graph, input_root
from synapse.ir import Decision, Effect, Node, NodeKind, Profile, Workflow


def _rewire(inputs: list[str], aliases: Mapping[str, str]) -> list[str]:
    """Inputs with each merged output replaced by the kept one; a field read twice is read
    once, since the executor passes a node one value per reference."""
    rewired = []
    for field in inputs:
        root, dot, rest = field.partition(".")
        rewired.append(aliases.get(root, root) + dot + rest)
    return list(dict.fromkeys(rewired))


def _signature(node: Node, inputs: list[str]) -> tuple:
    """What a node computes. Output names count when they pick fields out of the result (a
    transform, or several outputs); a single output holds the whole result whatever its name."""
    names: object = (tuple(node.outputs) if node.kind is NodeKind.TRANSFORM
                     or len(node.outputs) > 1 else len(node.outputs))
    return (node.kind, node.effect, tuple(inputs), node.model, node.fallback_model, node.prompt,
            node.tool, node.index, node.query, node.ttl_s,
            json.dumps(node.tests, sort_keys=True, default=str), names)


def _same_arguments(reader: Node, reference: str) -> bool:
    """Whether a reader is sent the same arguments when reference names the kept node's output.
    Tools name each argument by the last segment of its path, so only a dotted path keeps its
    name; transforms receive the whole reference as a key, so no rename is safe; LLM and
    retrieval nodes receive the values, with a merged value shown once under the kept name."""
    if reader.kind is NodeKind.TOOL:
        return "." in reference
    return reader.kind in (NodeKind.LLM, NodeKind.RETRIEVE)


class _Legality:
    """The checks a merge of drop into keep must pass, over the workflow before CSE."""

    def __init__(self, wf: Workflow) -> None:
        self.wf = wf
        self.position = {node.id: index for index, node in enumerate(wf.nodes)}
        self.published = {input_root(field) for field in wf.outputs}

    def blocker(self, keep: Node, drop: Node) -> str | None:
        """Why drop cannot be merged into keep, or None when the merge is legal."""
        if keep.effect is Effect.WRITE or drop.effect is Effect.WRITE:
            return "write effect"
        if keep.required or drop.required:
            return "required check"
        if drop.kind is not NodeKind.LLM and not (keep.deterministic and drop.deterministic):
            return "not deterministic"
        if self.position[keep.id] >= self.position[drop.id]:
            return f"{keep.id} is not listed before {drop.id}"
        published = [field for field in drop.outputs if field in self.published]
        if published:
            return f"produces workflow output {', '.join(published)}"
        if drop.effect is Effect.READ:
            between = self.wf.nodes[self.position[keep.id] + 1:self.position[drop.id]]
            writes = [node.id for node in between if node.effect is Effect.WRITE]
            if writes:
                return f"write {writes[0]} runs between the two reads"
        for reader in self.wf.nodes:
            for field in reader.inputs:
                if input_root(field) in drop.outputs and not _same_arguments(reader, field):
                    return f"{reader.id} would be sent {field} under another name"
        return None


class CSE:
    name = "cse"

    def plan(self, wf: Workflow, profile: Profile | None = None) -> list[Decision]:
        """An applied decision per merge and a skipped one per identical pair that cannot
        merge; each lists the kept node first."""
        build_graph(wf)
        legality = _Legality(wf)
        first: dict[tuple, Node] = {}
        aliases: dict[str, str] = {}
        decisions = []
        for node in wf.nodes:
            key = _signature(node, _rewire(node.inputs, aliases))
            keep = first.get(key)
            if keep is None:
                first[key] = node
                continue
            blocker = legality.blocker(keep, node)
            if blocker is None:
                aliases.update(zip(node.outputs, keep.outputs))
                decisions.append(Decision(
                    pass_name=self.name, nodes=[keep.id, node.id], status="applied",
                    reason=f"{node.id} merged into {keep.id}: identical {node.kind.value} "
                           f"step with the same inputs"))
            else:
                first[key] = node  # later copies compare with the nearest one
                decisions.append(Decision(
                    pass_name=self.name, nodes=[keep.id, node.id], status="skipped",
                    reason=f"{node.id} not merged into {keep.id}: {blocker}"))
        return decisions

    def apply(self, wf: Workflow, decisions: list[Decision]) -> Workflow:
        """Drop each applied duplicate and point its readers at the kept node, checking every
        merge again; raises ValueError on one that is not identical or not legal."""
        merges: dict[str, str] = {}
        for decision in decisions:
            if decision.status != "applied":
                continue
            if len(decision.nodes) != 2:
                raise ValueError("a CSE decision names the kept node, then the duplicate")
            keep_id, drop_id = decision.nodes
            merges[drop_id] = keep_id
        legality = _Legality(wf)
        result = wf.model_copy(deep=True)
        kept: dict[str, Node] = {}
        aliases: dict[str, str] = {}
        for node in result.nodes:
            node.inputs = _rewire(node.inputs, aliases)
            keep_id = merges.pop(node.id, None)
            if keep_id is None:
                kept[node.id] = node
                continue
            keep = kept.get(keep_id)
            if keep is None:
                raise ValueError(f"{node.id} can merge only into an earlier node that stays, "
                                 f"not {keep_id}")
            if _signature(keep, keep.inputs) != _signature(node, node.inputs):
                raise ValueError(f"{keep_id} and {node.id} are not identical")
            blocker = legality.blocker(keep, node)
            if blocker:
                raise ValueError(f"cannot merge {node.id} into {keep_id}: {blocker}")
            aliases.update(zip(node.outputs, keep.outputs))
        if merges:
            raise ValueError(f"cannot merge unknown node {next(iter(merges))}")
        result.nodes = list(kept.values())
        build_graph(result)
        return result
