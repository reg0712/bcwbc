"""Complaint triage test data: the 17 synthetic complaints with their hand-assigned category
and team, the transaction history the mock bank serves for each complainant, and one
deterministic stand-in for each LLM prompt of the complaint workflow. Synthetic data only."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from mock_services.loan import inr
from synapse import ROOT

COMPLAINTS_FILE = ROOT / "mock_services" / "data" / "complaints.json"

# The team for each category; suspected fraud goes to FRAUD_TEAM whatever the category.
TEAMS = {"failed_transfer": "digital_payments", "card_dispute": "card_disputes",
         "atm_cash": "atm_operations", "loan_charges": "loan_servicing",
         "account_access": "account_services", "other": "customer_care"}
FRAUD_TEAM = "fraud_response"
HIGH_PRIORITY_INR = 50000

# Resolution timelines in working days, as data/complaint_policy.json (cp-01) states them.
RESOLUTION_DAYS = {"failed_transfer": 2, "card_dispute": 10, "atm_cash": 5,
                   "loan_charges": 7, "account_access": 3, "other": 7}

# The mock classifier's keyword rules, tried in order. The first match wins, so an ATM
# complaint that mentions a card is still an ATM complaint.
_RULES = [
    ("atm_cash", re.compile(r"\batm\b|dispens", re.IGNORECASE)),
    ("card_dispute", re.compile(r"\bcard\b|charged twice|chargeback", re.IGNORECASE)),
    ("loan_charges", re.compile(r"\bloan\b|\bemi\b|foreclos|prepay", re.IGNORECASE)),
    ("account_access", re.compile(r"\blog ?in\b|\bsign ?in\b|net ?banking|mobile banking|"
                                  r"\bkyc\b|frozen|locked|password", re.IGNORECASE)),
    ("failed_transfer", re.compile(r"\b(upi|imps|neft|rtgs)\b|transfer|beneficiary",
                                   re.IGNORECASE)),
]
_FRAUD = re.compile(r"unauthori[sz]ed|fraud|\bstolen\b|\blost (my|the) card\b|"
                    r"\b(did not|didn't|never) (make|made|authori[sz]e)", re.IGNORECASE)
_AMOUNT = re.compile(r"(?:INR|Rs\.?|₹)\s*(\d[\d,]*)")

_ISSUES = {"failed_transfer": "a payment that has not reached its beneficiary",
           "card_dispute": "a disputed card payment",
           "atm_cash": "cash the ATM did not dispense",
           "loan_charges": "the debits on your loan account",
           "account_access": "trouble accessing your account",
           "other": "your recent experience with us"}
_ACTIONS = {"failed_transfer": "check the payment with the beneficiary's bank and return the "
                               "amount to you if it was not credited",
            "card_dispute": "look into the payment and raise a chargeback with the merchant's "
                            "bank where the policy allows",
            "atm_cash": "check the ATM's cash count and credit back any amount it did not "
                        "dispense",
            "loan_charges": "review the debits against your loan agreement and refund anything "
                            "taken in error",
            "account_access": "verify your details and restore your access",
            "other": "look into what happened and get back to you"}
_TEAM_NAMES = {"digital_payments": "digital payments", "card_disputes": "card disputes",
               "fraud_response": "fraud response", "atm_operations": "ATM operations",
               "loan_servicing": "loan servicing", "account_services": "account services",
               "customer_care": "customer care"}


def load_complaints() -> list[dict[str, Any]]:
    """The 17 synthetic complaints, C01 to C17."""
    return json.loads(COMPLAINTS_FILE.read_text())


def complaint(case_id: str) -> dict[str, Any]:
    """One complaint by case id."""
    return next(c for c in load_complaints() if c["case_id"] == case_id)


def cases() -> list[dict[str, Any]]:
    """One test case per complaint: the complaint workflow's inputs plus a case_id."""
    return [{"case_id": c["case_id"], "complaint": c["complaint"],
             "customer_id": c["customer_id"]} for c in load_complaints()]


def expected_category(case_id: str) -> str:
    """The category the complaint was labelled with."""
    return complaint(case_id)["expected_category"]


def expected_team(case_id: str) -> str:
    """The team the complaint was labelled for."""
    return complaint(case_id)["expected_team"]


def amounts_mentioned(text: str) -> list[int]:
    """Rupee amounts a complaint mentions, in order: 'INR 18,400' -> 18400."""
    return [int(amount.replace(",", "")) for amount in _AMOUNT.findall(text)]


def triage(text: str) -> dict[str, str]:
    """The mock classifier, for testing only: keyword rules pick the category, suspected fraud
    sends it to the fraud team, and fraud or INR 50,000 or more at stake makes it high priority."""
    category = next((name for name, rule in _RULES if rule.search(text)), "other")
    fraud = _FRAUD.search(text) is not None
    high = fraud or max(amounts_mentioned(text), default=0) >= HIGH_PRIORITY_INR
    return {"category": category, "team": FRAUD_TEAM if fraud else TEAMS[category],
            "priority": "high" if high else "normal"}


# Documents the mock bank APIs return.

def transaction_history(c: dict[str, Any]) -> dict[str, Any]:
    """The complainant's recent transactions, newest first: a salary credit and two routine
    payments, then the transactions the complaint is about."""
    n = int(c["case_id"][1:])  # varies the routine amounts from one customer to the next
    routine = [
        _routine(c, 1, "2026-10-01", "credit", "NEFT", "SALARY CREDIT TEST EMPLOYER LTD",
                 40000 + 1575 * n),
        _routine(c, 2, "2026-10-08", "debit", "BILLPAY", "ELECTRICITY BILL TEST POWER CO",
                 1000 + 87 * n),
        _routine(c, 3, "2026-10-14", "debit", "UPI", "UPI PAYMENT TO TEST FRESH MART",
                 1500 + 131 * n),
    ]
    # A reversed sort is still stable, so same-day rows keep the order they are listed in.
    rows = sorted(routine + c["transactions"], key=lambda t: t["date"], reverse=True)
    return {"customer_id": c["customer_id"], "account": c["account"],
            "account_holder": c["name"], "transactions": rows}


def _routine(c: dict[str, Any], number: int, day: str, kind: str, channel: str,
             description: str, amount: int) -> dict[str, Any]:
    """One completed routine transaction for a complainant."""
    return {"txn_id": f"TEST-TXN-{c['case_id']}-R{number}", "date": day, "type": kind,
            "channel": channel, "description": description, "amount": amount,
            "status": "completed"}


# Deterministic stand-ins for the workflow's LLM prompts, keyed by prompt file name.

def classify_complaint(inputs: dict[str, Any]) -> dict[str, Any]:
    """prompts/classify_complaint.txt: category, team and priority from the complaint text."""
    return triage(inputs["complaint"])


def draft_reply(inputs: dict[str, Any]) -> dict[str, Any]:
    """prompts/draft_reply.txt: a plain reply that cites the transactions whose amounts the
    complaint mentions and the policy's resolution timeline for its category."""
    category, team = inputs["triage"]["category"], inputs["triage"]["team"]
    history = inputs["history"]
    amounts = set(amounts_mentioned(inputs["complaint"]))
    cited = [t for t in history["transactions"] if t["amount"] in amounts]
    days = RESOLUTION_DAYS[category]
    sentences = [f"Thank you for telling us about {_ISSUES[category]}."]
    if cited:
        sentences.append(f"We can see {_listing([_describe(t) for t in cited])} "
                         f"on your account.")
    if team == FRAUD_TEAM:
        sentences.append("As you have reported transactions you did not make, we are "
                         "treating this as suspected fraud.")
    sentences += [f"Our {_TEAM_NAMES[team]} team will {_ACTIONS[category]}, and under our "
                  f"complaint policy we will resolve your complaint within {days} working days.",
                  "If you are not satisfied with our reply, you can write to our Grievance "
                  "Redressal Officer.",
                  "We will never ask for your PIN, OTP or password."]
    body = "\n\n".join([f"Dear {history['account_holder'].split()[0]},", " ".join(sentences),
                        "Customer Care"])
    return {"subject": f"Your complaint about {_ISSUES[category]}", "body": body,
            "transaction_ids": [t["txn_id"] for t in cited],
            "policy_ids": [passage["id"] for passage in inputs["policy"]],
            "resolution_days": days}


def _describe(t: dict[str, Any]) -> str:
    """A transaction as a reply cites it: amount, date, description and reference."""
    day = date.fromisoformat(t["date"])
    return (f"a {t['type']} of INR {inr(t['amount'])} on {day.day} {day:%B %Y} "
            f"({t['description']}, reference {t['txn_id']})")


def _listing(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


MOCK_TASKS = {
    "classify_complaint": classify_complaint,
    "draft_reply": draft_reply,
}
