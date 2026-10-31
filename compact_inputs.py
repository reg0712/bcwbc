"""Lossless prompt serialization; never remove evidence or change a write."""

from synapse.ir import Decision, Effect, NodeKind, Profile, Workflow


class CompactInputs:
    name = "compact_inputs"

    def plan(self, wf: Workflow, profile: Profile | None = None) -> list[Decision]:
        """Enable compact input rendering on model nodes without write effects."""
        return [Decision(pass_name=self.name, nodes=[node.id],
                         status="applied" if node.effect is not Effect.WRITE else "skipped",
                         reason="lossless JSON and repeated-key tables; all values retained"
                         if node.effect is not Effect.WRITE else "write effect: unchanged")
                for node in wf.nodes if node.kind is NodeKind.LLM]

    def apply(self, wf: Workflow, decisions: list[Decision]) -> Workflow:
        """Keep templates and schemas intact; only select the input representation."""
        chosen = {id for d in decisions if d.status == "applied" for id in d.nodes}
        result = wf.model_copy(deep=True)
        for node in result.nodes:
            if node.id in chosen:
                node.compact_inputs = True
        return result
