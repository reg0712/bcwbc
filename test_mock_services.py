"""Mock bank services and synthetic applicants (M2): obviously fake data, the decision rule's
8/7/5 split and its boundaries, and each endpoint's data, errors, logging and delay, including
the complaint workflow's /transactions and /tickets."""

from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from mock_services import loan
from mock_services.bank_api import create_app
from synapse.config import load_config

SCALE = 0.001
CASE_IDS = [f"A{n:02d}" for n in range(1, 21)]


def post(path: str, payload: dict[str, Any], app: FastAPI | None = None) -> httpx.Response:
    """POST to an in-process mock bank and return the response."""
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app if app is not None else create_app(scale=SCALE))
        async with httpx.AsyncClient(transport=transport, base_url="http://bank.mock") as client:
            return await client.post(path, json=payload)
    return asyncio.run(send())


def test_twenty_obviously_fake_applicants() -> None:
    applicants = loan.load_applicants()
    assert [a["case_id"] for a in applicants] == CASE_IDS
    assert all(re.fullmatch(r"FAKEP\d{4}[A-Z]", a["pan"]) for a in applicants)
    assert len({a["pan"] for a in applicants}) == len({a["customer_id"] for a in applicants}) == 20


def test_rule_splits_the_cases_8_7_5() -> None:
    assert Counter(map(loan.expected_decision, CASE_IDS)) == {"approve": 8, "refer": 7,
                                                              "decline": 5}


@pytest.mark.parametrize(("case_id", "decision"), [
    pytest.param("A17", "decline", id="score-645"),
    pytest.param("A10", "refer", id="score-650"),
    pytest.param("A16", "refer", id="score-720"),
    pytest.param("A15", "approve", id="score-721"),
    pytest.param("A07", "approve", id="emis-exactly-half-of-income"),
    pytest.param("A13", "refer", id="emis-just-over-half"),
    pytest.param("A08", "decline", id="kyc-fails-despite-good-score"),
])
def test_rule_boundaries(case_id: str, decision: str) -> None:
    assert loan.expected_decision(case_id) == decision


def test_application_text_parses_back_to_the_applicant() -> None:
    a = loan.applicant("A18")
    parsed = loan.parse_application({"application_pdf": loan.application_text(a)})
    assert parsed == {"name": a["name"], "pan": a["pan"], "income": a["income"],
                      "employment": a["employment"], "requested_amount": a["requested_amount"]}


def test_kyc_verifies_only_passing_records_with_matching_names() -> None:
    a01, a08 = loan.applicant("A01"), loan.applicant("A08")
    assert post("/kyc", {"pan": a01["pan"], "name": a01["name"]}).json()["status"] == "verified"
    assert post("/kyc", {"pan": a01["pan"], "name": "Someone Else"}).json()["status"] == "failed"
    assert post("/kyc", {"pan": a08["pan"], "name": a08["name"]}).json()["status"] == "failed"


def test_bureau_and_statement_serve_the_applicants_figures() -> None:
    a = loan.applicant("A06")
    bureau = post("/bureau", {"pan": a["pan"]}).json()
    assert (bureau["score"], bureau["existing_emi"]) == (790, 70000)
    statement = post("/statement", {"customer_id": a["customer_id"]}).json()
    credits = [t["amount"] for t in statement["transactions"] if t["type"] == "credit"]
    assert credits == [120000] * 6


def test_unknown_pan_and_customer_are_404() -> None:
    assert post("/bureau", {"pan": "FAKEP9999Z"}).status_code == 404
    assert post("/offers", {"customer_id": "CUST-NOPE"}).status_code == 404


def test_sms_ids_are_stable_and_every_send_is_logged() -> None:
    app = create_app(scale=SCALE)
    first = post("/sms", {"customer_id": "CUST-A01", "decision": "approve"}, app).json()
    second = post("/sms", {"customer_id": "CUST-A01", "decision": "approve"}, app).json()
    assert first["sms_id"] == second["sms_id"]
    assert first["status"] == "queued"
    assert [sent["customer_id"] for sent in app.state.sms_log] == ["CUST-A01", "CUST-A01"]


def test_endpoints_wait_their_configured_latency() -> None:
    app = create_app(scale=0.05)
    started = time.perf_counter()
    post("/statement", {"customer_id": "CUST-A01"}, app)
    assert time.perf_counter() - started >= 1.5 * 0.05  # statement: 1,500 ms at scale 0.05


def test_transactions_serve_the_complaint_rows_newest_first() -> None:
    history = post("/transactions", {"customer_id": "CUST-C06"}).json()
    assert (history["account"], history["account_holder"]) == ("TEST-ACC-1006", "Rehan Testqureshi")
    dates = [t["date"] for t in history["transactions"]]
    assert dates == sorted(dates, reverse=True)
    ids = [t["txn_id"] for t in history["transactions"]]
    assert ids[:2] == ["TEST-TXN-C06-01", "TEST-TXN-C06-02"] and len(set(ids)) == len(ids) == 5
    assert post("/transactions", {"customer_id": "CUST-NOPE"}).status_code == 404


def test_ticket_ids_are_stable_and_every_ticket_is_logged() -> None:
    app = create_app(scale=SCALE)
    ticket = {"customer_id": "CUST-C01", "category": "failed_transfer", "team": "digital_payments",
              "priority": "normal", "subject": "Failed transfer", "body": "Dear Ira, ...",
              "resolution_days": 2}
    first = post("/tickets", ticket, app).json()
    second = post("/tickets", ticket, app).json()
    assert first == second
    assert re.fullmatch(r"TKT-[0-9a-f]{10}", first["ticket_id"])
    assert (first["status"], first["due_in_working_days"]) == ("open", 2)
    other = post("/tickets", {**ticket, "team": "fraud_response"}, app).json()
    assert other["ticket_id"] != first["ticket_id"]
    assert post("/tickets", {**ticket, "customer_id": "CUST-NOPE"}, app).status_code == 404
    assert [t["team"] for t in app.state.ticket_log] == ["digital_payments", "digital_payments",
                                                         "fraud_response"]


def test_complaint_endpoints_wait_their_configured_latency() -> None:
    latency_ms = load_config("mock.yaml")["api_latency_ms"]
    app = create_app(scale=0.05)
    ticket = {"customer_id": "CUST-C01", "category": "other", "team": "customer_care",
              "priority": "normal", "subject": "S", "body": "B", "resolution_days": 7}
    started = time.perf_counter()
    post("/transactions", {"customer_id": "CUST-C01"}, app)
    post("/tickets", ticket, app)
    waited_ms = (time.perf_counter() - started) * 1000
    assert waited_ms >= (latency_ms["transactions"] + latency_ms["tickets"]) * 0.05
