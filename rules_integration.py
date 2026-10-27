"""Adapter between Member 3's canonical result and the review dashboard."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rules_engine import RulesEngineConfig, revalidate_corrected_fields

AMOUNT_FIELDS = {
    "subtotal",
    "tax_amount",
    "tax_rate",
    "discount_amount",
    "rounding_adjustment",
    "total_amount",
}


def is_rules_result(payload: Any) -> bool:
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    if not isinstance(payload, Mapping):
        return False
    fields = payload.get("fields")
    return bool(
        isinstance(fields, Mapping)
        and fields
        and all(isinstance(details, Mapping) and "normalized_value" in details for details in fields.values())
    )


def to_rules_dict(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    return dict(payload)


def review_fields_from_rules(
    payload: Any,
    confidential_fields: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    rules_payload = to_rules_dict(payload)
    confidential = confidential_fields or set()
    review_fields = {}

    document_config = RulesEngineConfig().get_document_config(rules_payload.get("document_type", "generic"))
    required = set(document_config.required_fields)
    for field_name, details in rules_payload["fields"].items():
        absent = details.get("ocr_value") in (None, "") and details.get("ai_value") in (None, "")
        if absent and field_name not in required:
            continue  # optional and not printed on the document: nothing to review
        validation_status = details.get("validation_status", "not_checkable")
        decision = details.get("decision", "human_review")
        issues = []
        if validation_status in {"invalid", "missing", "warning"}:
            issues.extend(details.get("validation_reasons", []))
        if decision != "auto_accept":
            issues.extend(details.get("decision_reasons", []))

        field_type = None
        if field_name == "date":
            field_type = "date"
        elif field_name in AMOUNT_FIELDS:
            field_type = "amount"

        review_fields[field_name] = {
            "value": details.get("normalized_value") or "",
            "confidence": details.get("confidence_score") or 0.0,
            "confidential": field_name in confidential,
            "issues": list(dict.fromkeys(str(issue) for issue in issues)),
            "type": field_type,
            "decision": decision,
            "validation_status": validation_status,
            "evidence_sufficient": details.get("evidence_sufficient", False),
        }

    for field_name in document_config.required_fields:
        if field_name not in review_fields:
            review_fields[field_name] = {
                "value": "",
                "confidence": 0.0,
                "confidential": field_name in confidential,
                "issues": [f"Required field '{field_name}' is missing"],
                "type": "date" if field_name == "date" else ("amount" if field_name in AMOUNT_FIELDS else None),
                "decision": "human_review",
                "validation_status": "missing",
                "evidence_sufficient": False,
            }

    return review_fields


def rules_validation_errors(payload: Any) -> tuple[dict[str, list[str]], list[str]]:
    rules_payload = to_rules_dict(payload)
    errors = {field_name: [] for field_name in rules_payload.get("fields", {})}
    document_errors = []

    for field_name, details in rules_payload.get("fields", {}).items():
        if details.get("validation_status") in {"invalid", "missing"}:
            errors[field_name].extend(str(reason) for reason in details.get("validation_reasons", []))
        if details.get("decision") in {"retry_extraction", "reject"}:
            errors[field_name].extend(str(reason) for reason in details.get("decision_reasons", []))

    check_fields = {
        "total_reconciliation": (
            "subtotal",
            "tax_amount",
            "discount_amount",
            "rounding_adjustment",
            "total_amount",
        ),
        "tax_calculation": ("subtotal", "tax_rate", "tax_amount"),
    }
    for check in rules_payload.get("cross_field_checks", []):
        if check.get("status") != "invalid" and not check.get("is_critical_failure", False):
            continue
        reasons = [str(reason) for reason in check.get("reasons", [])]
        assigned = False
        for field_name in check_fields.get(check.get("check_name"), ()):
            if field_name in errors:
                errors[field_name].extend(reasons)
                assigned = True
        if not assigned:
            document_errors.extend(reasons)

    document_errors.extend(str(reason) for reason in rules_payload.get("critical_failures", []))
    for field_name in errors:
        errors[field_name] = list(dict.fromkeys(errors[field_name]))
    document_errors = list(dict.fromkeys(document_errors))
    return errors, document_errors


def revalidate_review_corrections(
    original_payload: Any,
    current_values: dict[str, str],
    corrected_field_names: set[str],
) -> tuple[dict[str, list[str]], list[str], dict[str, Any]]:
    corrections = {
        field_name: current_values[field_name] for field_name in corrected_field_names if field_name in current_values
    }
    if corrections:
        result = revalidate_corrected_fields(original_payload, corrections)
        payload = result.to_dict()
    else:
        payload = to_rules_dict(original_payload)
    field_errors, document_errors = rules_validation_errors(payload)
    return field_errors, document_errors, payload
