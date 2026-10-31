"""Token cost estimates; mock units and live USD are deliberately separate."""

from __future__ import annotations

from synapse.config import load_config
from synapse.ir import Usage


def estimate_cost(usage: Usage, tier: str | None, *, provider: str = "mock") -> float:
    """Price the model actually called; workflow tiers only price mock calls."""
    if not usage.model or not tier:
        return 0.0
    prices = load_config("prices.yaml")
    if provider == "mock":
        rate = prices["cost_per_1k_tokens"][tier]
        return (usage.tokens_in + usage.tokens_out) * rate / 1000
    models = load_config("models.yaml")["providers"][provider]
    actual_tier = next((name for name in ("small", "large")
                        if models[name] == usage.model), None)
    if actual_tier is None:
        raise ValueError(f"live price for {provider} model {usage.model!r} is not configured")
    rates = prices["live_usd_per_1k_tokens"][provider][actual_tier]
    if rates["in"] is None or rates["out"] is None:
        raise ValueError(f"live prices for {provider}/{tier} are not configured")
    return (usage.tokens_in * rates["in"] + usage.tokens_out * rates["out"]) / 1000
