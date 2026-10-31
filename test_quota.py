"""Local free-tier budgets are shared and survive a new guard instance."""

from __future__ import annotations

import pytest

from synapse.backends import quota


def test_quota_reservations_enforce_minute_day_and_provider_blocks(tmp_path, monkeypatch) -> None:
    now = [1_000_000.0]
    monkeypatch.setitem(quota.LIMITS, "groq", quota.Limit(rpm=2, rpd=3, tpm=2_000,
                                                       tpd=4_000))
    guard = quota.QuotaGuard(tmp_path / "quota.db", clock=lambda: now[0])
    first = guard.reserve("groq", "test-model", "hello")
    guard.settle(first, 10)
    guard.reserve("groq", "test-model", "hello")
    with pytest.raises(quota.QuotaExceeded, match="minute budget") as caught:
        guard.reserve("groq", "test-model", "hello")
    assert caught.value.retry_after_s == pytest.approx(60.1)
    now[0] += 61
    another = quota.QuotaGuard(tmp_path / "quota.db", clock=lambda: now[0])
    another.reserve("groq", "test-model", "hello")
    with pytest.raises(quota.QuotaExceeded, match="daily budget") as daily:
        another.reserve("groq", "test-model", "hello")
    assert daily.value.retry_after_s is None
    assert another.status("groq", "test-model")["rpd_remaining"] == 0
    now[0] += 86_401
    assert another.status("groq", "test-model")["rpd_remaining"] == 3
    another.block("groq", "test-model")
    with pytest.raises(quota.QuotaExceeded, match="daily quota is exhausted"):
        another.reserve("groq", "test-model", "hello")


def test_advisory_daily_quota_allows_user_to_try_again(tmp_path, monkeypatch) -> None:
    now = [1_000_000.0]
    monkeypatch.setitem(quota.LIMITS, "groq", quota.Limit(rpm=2, rpd=1, tpm=2_000, tpd=100))
    guard = quota.QuotaGuard(tmp_path / "quota.db", clock=lambda: now[0], advisory_daily=True)
    guard.block("groq", "test")
    assert guard.status("groq", "test")["blocked"]
    ticket = guard.reserve("groq", "test", "hello")
    guard.settle(ticket, 50)
    assert not guard.status("groq", "test")["blocked"]
    assert guard.status("groq", "test")["rpd_remaining"] == 0
    now[0] += 61
    assert guard.reserve("groq", "test", "hello") > ticket
