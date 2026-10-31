"""Conservative local budget for live model calls on free-tier accounts.

Groq's published free limits for GPT-OSS 20B/120B are 30 RPM, 1,000 RPD,
8,000 TPM and 200,000 TPD. Google says Gemini limits vary by project; the
Gemini caps here are deliberately lower than the 20 RPD observed for Flash on
this project's free account. This ledger counts only SYNAPSE calls, so calls
made elsewhere can still exhaust a provider's real quota.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

from synapse import ROOT


@dataclass(frozen=True)
class Limit:
    rpm: int
    rpd: int
    tpm: int
    tpd: int


LIMITS = {
    "gemini": Limit(rpm=5, rpd=16, tpm=4_000, tpd=80_000),
    "groq": Limit(rpm=25, rpd=900, tpm=7_000, tpd=180_000),
}
OUTPUT_RESERVE = 1_024


class QuotaExceeded(RuntimeError):
    """A new call would exceed SYNAPSE's local free-tier budget."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class QuotaGuard:
    """SQLite-backed reservations shared across server reloads and CLI processes."""

    def __init__(self, path: str | Path = ROOT / "data" / "quota.db",
                 clock=time.time, *, advisory_daily: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.advisory_daily = advisory_daily
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, "
                       "provider TEXT, model TEXT, at REAL, tokens INTEGER)")
            db.execute("CREATE TABLE IF NOT EXISTS blocks (provider TEXT, model TEXT, "
                       "until REAL, PRIMARY KEY(provider, model))")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA busy_timeout=10000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, provider: str, model: str, prompt: str, *,
                output_tokens: int = OUTPUT_RESERVE) -> int:
        """Reserve one request and conservatively estimated input/output tokens."""
        limit = LIMITS[provider]
        tokens = max(64, len(prompt.encode("utf-8")) // 2 + 1) + output_tokens
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM calls WHERE at < ?", (now - 86_400,))
            blocked = db.execute("SELECT until FROM blocks WHERE provider=? AND model=?",
                                 (provider, model)).fetchone()
            if blocked and blocked[0] > now and not self.advisory_daily:
                raise QuotaExceeded(f"{provider} {model} daily quota is exhausted; "
                                    "choose another model or Mock")
            minute = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM calls "
                "WHERE provider=? AND model=? AND at>=?", (provider, model, now - 60)).fetchone()
            day = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM calls "
                "WHERE provider=? AND model=? AND at>=?", (provider, model, now - 86_400)).fetchone()
            if not self.advisory_daily and (day[0] + 1 > limit.rpd or day[1] + tokens > limit.tpd):
                raise QuotaExceeded(f"{provider} {model} local daily budget reached; "
                                    "choose another model or Mock")
            if tokens > limit.tpm and not self.advisory_daily:
                raise QuotaExceeded(f"{provider} {model} prompt exceeds the local minute "
                                    "token budget; choose another model or shorten the prompt")
            if minute[0] > 0 and (minute[0] + 1 > limit.rpm or minute[1] + tokens > limit.tpm):
                rows = db.execute("SELECT at, tokens FROM calls WHERE provider=? AND model=? "
                                  "AND at>=? ORDER BY at", (provider, model, now - 60)).fetchall()
                count, used = minute
                retry_after = 60.0
                for at, spent in rows:
                    count -= 1
                    used -= spent
                    if count == 0 or (count + 1 <= limit.rpm and used + tokens <= limit.tpm):
                        retry_after = max(0.1, at + 60 - now + 0.1)
                        break
                raise QuotaExceeded(f"{provider} {model} local minute budget reached; "
                                    "waiting for the minute window", retry_after)
            return db.execute("INSERT INTO calls (provider, model, at, tokens) "
                              "VALUES (?, ?, ?, ?)", (provider, model, now, tokens)).lastrowid

    def settle(self, call_id: int, tokens: int) -> None:
        """Replace the reservation with actual token usage after a successful reply."""
        with self._connect() as db:
            db.execute("UPDATE calls SET tokens=? WHERE id=?", (tokens, call_id))
            db.execute("DELETE FROM blocks WHERE (provider, model) IN "
                       "(SELECT provider, model FROM calls WHERE id=?)", (call_id,))

    def block(self, provider: str, model: str, seconds: float = 86_400) -> None:
        """Remember provider exhaustion; live mode displays it without disabling retries."""
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO blocks VALUES (?, ?, ?)",
                       (provider, model, self.clock() + seconds))

    def status(self, provider: str, model: str) -> dict[str, int | bool]:
        """Remaining local requests and whether a provider-reported block is active."""
        limit = LIMITS[provider]
        now = self.clock()
        with self._connect() as db:
            minute = db.execute("SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM calls "
                                "WHERE provider=? AND model=? AND at>=?",
                                (provider, model, now - 60)).fetchone()
            day = db.execute("SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM calls "
                             "WHERE provider=? AND model=? AND at>=?",
                             (provider, model, now - 86_400)).fetchone()
            blocked = db.execute("SELECT until FROM blocks WHERE provider=? AND model=?",
                                 (provider, model)).fetchone()
        return {"rpm_remaining": max(0, limit.rpm - minute[0]),
                "rpd_remaining": max(0, limit.rpd - day[0]),
                "tpm_remaining": max(0, limit.tpm - minute[1]),
                "tpd_remaining": max(0, limit.tpd - day[1]),
                "blocked": bool(blocked and blocked[0] > now)}
