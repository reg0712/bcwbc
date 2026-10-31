"""Mock bank APIs (/kyc /bureau /statement /offers /sms for loans, /transactions /tickets
for complaints) with configured delays and fixed synthetic data for 20 applicants and 17
complainants. Every endpoint takes a JSON POST, so PANs never appear in URLs. `make mock`
serves it on its own; create_app() also runs it in-process."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mock_services import complaints, loan
from synapse.config import load_config


class PanRequest(BaseModel):
    pan: str


class KycRequest(PanRequest):
    name: str


class CustomerRequest(BaseModel):
    customer_id: str


class SmsRequest(CustomerRequest):
    decision: str


class TicketRequest(CustomerRequest):
    category: str
    team: str
    priority: str
    subject: str
    body: str
    resolution_days: int


def create_app(scale: float | None = None,
               latency_ms: Mapping[str, float] | None = None) -> FastAPI:
    """Build the mock bank. Each endpoint waits its config/mock.yaml latency times scale;
    sent messages collect in app.state.sms_log and logged tickets in app.state.ticket_log,
    so tests can count writes."""
    config = load_config("mock.yaml")
    scale = config.get("latency_scale", 1.0) if scale is None else scale
    delays_ms = dict(config["api_latency_ms"] if latency_ms is None else latency_ms)
    applicants = loan.load_applicants()
    by_pan = {a["pan"]: a for a in applicants}
    by_customer = {a["customer_id"]: a for a in applicants}
    complainants = {c["customer_id"]: c for c in complaints.load_complaints()}
    app = FastAPI(title="SYNAPSE mock bank APIs")
    app.state.sms_log = []
    app.state.ticket_log = []

    async def wait(api: str) -> None:
        await asyncio.sleep(delays_ms[api] * scale / 1000)

    def find(table: dict[str, dict[str, Any]], key: str, what: str) -> dict[str, Any]:
        if key not in table:
            raise HTTPException(status_code=404, detail=f"unknown {what}")
        return table[key]

    @app.post("/kyc")
    async def kyc(request: KycRequest) -> dict[str, Any]:
        await wait("kyc")
        return loan.kyc_result(find(by_pan, request.pan, "PAN"), request.name)

    @app.post("/bureau")
    async def bureau(request: PanRequest) -> dict[str, Any]:
        await wait("bureau")
        return loan.bureau_report(find(by_pan, request.pan, "PAN"))

    @app.post("/statement")
    async def statement(request: CustomerRequest) -> dict[str, Any]:
        await wait("statement")
        return loan.bank_statement(find(by_customer, request.customer_id, "customer"))

    @app.post("/offers")
    async def offers(request: CustomerRequest) -> dict[str, Any]:
        await wait("offers")
        return loan.offers(find(by_customer, request.customer_id, "customer"))

    @app.post("/sms")
    async def sms(request: SmsRequest) -> dict[str, Any]:
        await wait("sms")
        find(by_customer, request.customer_id, "customer")
        message = f"Your personal loan application has been pre-screened. Outcome: {request.decision}."
        digest = hashlib.sha256(f"{request.customer_id}|{message}".encode()).hexdigest()
        sent = {"sms_id": f"SMS-{digest[:10]}", "customer_id": request.customer_id,
                "message": message}
        app.state.sms_log.append(sent)
        return {"sms_id": sent["sms_id"], "status": "queued", "message": message}

    @app.post("/transactions")
    async def transactions(request: CustomerRequest) -> dict[str, Any]:
        await wait("transactions")
        return complaints.transaction_history(
            find(complainants, request.customer_id, "customer"))

    @app.post("/tickets")
    async def tickets(request: TicketRequest) -> dict[str, Any]:
        await wait("tickets")
        find(complainants, request.customer_id, "customer")
        fields = request.model_dump()
        digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
        ticket = {"ticket_id": f"TKT-{digest[:10]}", "status": "open", **fields}
        app.state.ticket_log.append(ticket)
        return {"ticket_id": ticket["ticket_id"], "status": "open", "team": request.team,
                "priority": request.priority, "due_in_working_days": request.resolution_days}

    return app


app = create_app()
