"""Loan pre-screening test data: the 20 synthetic applicants, the documents the mock bank
serves for them, the decision rule the mocks follow, and one deterministic stand-in for each
LLM prompt of the loan workflow. Synthetic data only."""

from __future__ import annotations

import json
import re
from typing import Any

from synapse import ROOT

APPLICANTS_FILE = ROOT / "mock_services" / "data" / "applicants.json"
STATEMENT_MONTHS = 6

_CREDIT_LABEL = {"salaried": "SALARY", "self_employed": "BUSINESS RECEIPTS",
                 "contract": "CONTRACT PAYMENT"}


def load_applicants() -> list[dict[str, Any]]:
    """The 20 synthetic applicants, A01 to A20."""
    return json.loads(APPLICANTS_FILE.read_text())


def applicant(case_id: str) -> dict[str, Any]:
    """One applicant by case id."""
    return next(a for a in load_applicants() if a["case_id"] == case_id)


def inr(amount: int) -> str:
    """Rupees with Indian digit grouping: 1500000 -> '15,00,000'."""
    digits = str(amount)
    if len(digits) <= 3:
        return digits
    head, groups = digits[:-3], [digits[-3:]]
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join([head, *groups])


def application_text(a: dict[str, Any]) -> str:
    """An applicant's loan application as text, as if extracted from the PDF they uploaded."""
    return "\n".join([
        "PERSONAL LOAN APPLICATION",
        "Synthetic test record; not a real person.",
        "",
        f"Applicant name: {a['name']}",
        f"PAN: {a['pan']}",
        f"Customer ID: {a['customer_id']}",
        f"Employment: {a['employment']}",
        f"Employer or business: {a['employer']}",
        f"Net monthly income: INR {inr(a['income'])}",
        f"Loan amount requested: INR {inr(a['requested_amount'])}",
        f"Tenure: {a['tenure_months']} months",
        f"Purpose: {a['purpose']}",
    ])


def cases() -> list[dict[str, Any]]:
    """One test case per applicant: the loan workflow's inputs plus a case_id."""
    return [{"case_id": a["case_id"], "application_pdf": application_text(a),
             "customer_id": a["customer_id"]} for a in load_applicants()]


def decide(kyc_verified: bool, score: int, emi: float, income: float) -> tuple[str, list[str]]:
    """The mock decision rule, for testing only: decline if KYC fails or the bureau score is
    below 650; refer if the score is 650 to 720 or EMIs exceed 50% of income; else approve."""
    if not kyc_verified:
        return "decline", ["KYC not verified"]
    if score < 650:
        return "decline", [f"bureau score {score} is below 650"]
    if income <= 0:
        return "refer", ["no income credits found"]
    reasons = []
    if score <= 720:
        reasons.append(f"bureau score {score} is in the 650 to 720 refer band")
    if emi > 0.5 * income:
        reasons.append(f"existing EMIs are {emi / income:.0%} of income, above 50%")
    if reasons:
        return "refer", reasons
    return "approve", [f"bureau score {score} and EMIs at {emi / income:.0%} of income meet policy"]


def expected_decision(case_id: str) -> str:
    """The decision the rule gives an applicant's ground-truth data."""
    a = applicant(case_id)
    return decide(a["kyc_ok"], a["bureau_score"], a["existing_emi"], a["income"])[0]


# Documents the mock bank APIs return.

def kyc_result(a: dict[str, Any], name: str) -> dict[str, Any]:
    """KYC registry check: verified only if the record passes and the name matches."""
    name_match = name.strip().lower() == a["name"].lower()
    status = "verified" if a["kyc_ok"] and name_match else "failed"
    return {"pan": a["pan"], "status": status, "name_match": name_match}


def bureau_report(a: dict[str, Any]) -> dict[str, Any]:
    """Credit bureau report: score and existing monthly EMI obligations."""
    return {"pan": a["pan"], "score": a["bureau_score"], "existing_emi": a["existing_emi"],
            "active_loans": a["active_loans"], "enquiries_last_6_months": a["active_loans"] + 1}


def bank_statement(a: dict[str, Any]) -> dict[str, Any]:
    """Six months of transactions: income credits, EMI debits, spends and any bounce charges."""
    credit = f"{_CREDIT_LABEL[a['employment']]} {a['employer'].upper()}"
    bounce_months = set(range(STATEMENT_MONTHS, STATEMENT_MONTHS - a["bounced_payments"], -1))
    transactions = []
    for month in range(1, STATEMENT_MONTHS + 1):
        transactions.append({"month": month, "type": "credit", "description": credit,
                             "amount": a["income"]})
        if a["existing_emi"]:
            transactions.append({"month": month, "type": "debit",
                                 "description": "EMI EXISTING LOANS", "amount": a["existing_emi"]})
        transactions.append({"month": month, "type": "debit", "description": "UPI AND CARD SPENDS",
                             "amount": round(a["income"] * 0.35)})
        transactions.append({"month": month, "type": "debit", "description": "RENT AND UTILITIES",
                             "amount": round(a["income"] * 0.2)})
        if month in bounce_months:
            transactions.append({"month": month, "type": "debit",
                                 "description": "EMI BOUNCE CHARGE", "amount": 590})
    return {"customer_id": a["customer_id"], "account": a["account"],
            "period_months": STATEMENT_MONTHS, "summary": a["statement_summary"],
            "avg_balance": max(5000, int(round((a["income"] - a["existing_emi"]) * 1.5, -2))),
            "transactions": transactions}


def offers(a: dict[str, Any]) -> dict[str, Any]:
    """Pre-approved cross-sell offers for the customer."""
    return {"customer_id": a["customer_id"], "offers": [
        {"product": "credit card", "limit": int(round(a["income"] * 2, -3))},
        {"product": "fixed deposit", "rate_pct": 7.1}]}


# Deterministic stand-ins for the workflow's LLM prompts, keyed by prompt file name.

def parse_application(inputs: dict[str, Any]) -> dict[str, Any]:
    """prompts/parse_app.txt: pull the applicant's fields out of the application text."""
    text = inputs["application_pdf"]

    def field(label: str) -> str:
        match = re.search(rf"^{label}:\s*(.+)$", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"application has no {label!r} line")
        return match.group(1).strip()

    def rupees(label: str) -> int:
        return int(field(label).removeprefix("INR").replace(",", "").strip())

    return {"name": field("Applicant name"), "pan": field("PAN"),
            "income": rupees("Net monthly income"), "employment": field("Employment"),
            "requested_amount": rupees("Loan amount requested")}


def analyse_statement(inputs: dict[str, Any]) -> dict[str, Any]:
    """prompts/statement_analysis.txt: monthly income, EMI outflow and bounces."""
    statement = inputs["statement"]
    months = statement["period_months"]
    rows = statement["transactions"]
    income = round(sum(t["amount"] for t in rows if t["type"] == "credit") / months)
    emi = round(sum(t["amount"] for t in rows if t["description"] == "EMI EXISTING LOANS") / months)
    bounces = sum(1 for t in rows if "BOUNCE" in t["description"])
    return {"monthly_income": income, "emi_outflow": emi, "avg_balance": statement["avg_balance"],
            "bounced_payments": bounces,
            "summary": (f"Average monthly credits of INR {inr(income)}, EMI outflow of INR "
                        f"{inr(emi)}, {bounces} bounced payment(s) in {months} months.")}


def classify_segment(inputs: dict[str, Any]) -> dict[str, Any]:
    """prompts/classify_segment.txt: customer segment from employment and income."""
    employment, income = inputs["applicant.employment"], inputs["applicant.income"]
    if employment == "salaried":
        return {"label": "salaried_prime" if income >= 100000 else "salaried_mass"}
    return {"label": employment}


def assess_risk(inputs: dict[str, Any]) -> dict[str, Any]:
    """prompts/risk_summary.txt: the pre-screening decision under the mock rule."""
    kyc_verified = inputs["kyc_status"]["status"] == "verified"
    score = inputs["bureau"]["score"]
    emi, income = inputs["bureau"]["existing_emi"], inputs["cashflow"]["monthly_income"]
    decision, reasons = decide(kyc_verified, score, emi, income)
    foir = round(emi / income, 3) if income > 0 else 0.0
    return {"decision": decision, "kyc_verified": kyc_verified, "bureau_score": score,
            "foir": foir, "reasons": reasons,
            "summary": (f"{decision.capitalize()}: {'; '.join(reasons)}. "
                        f"Segment {inputs['segment']['label']}.")}


MOCK_TASKS = {
    "parse_app": parse_application,
    "statement_analysis": analyse_statement,
    "classify_segment": classify_segment,
    "risk_summary": assess_risk,
}
