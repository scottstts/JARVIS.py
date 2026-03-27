"""Shared write-time contract helpers for the memory_write tool."""

from __future__ import annotations

from typing import Any

FACT_STATUS_VALUES = ("current", "past", "uncertain", "superseded")
CONFIDENCE_VALUES = ("low", "medium", "high")
RELATION_CARDINALITY_VALUES = ("single", "multi")

FACT_EXAMPLE = (
    '[{"text":"Scott is building a Three.js/WebGPU/TSL shader playground.",'
    '"status":"current"}]'
)
RELATION_EXAMPLE = (
    '[{"subject":"Scott","predicate":"is_building","object":"Visual Pipeline Project",'
    '"status":"current","cardinality":"single"}]'
)
BODY_SECTIONS_EXAMPLE = (
    '{"Overview":"Scott is building a Three.js/WebGPU/TSL shader playground."}'
)
FACTS_USAGE_GUIDANCE = (
    'Prefer real fact objects in facts whenever the user states an explicit durable fact; '
    'use the literal string "None" only when there is genuinely no worthwhile fact to store.'
)
MEMORY_WRITE_EXAMPLE = (
    f"facts={FACT_EXAMPLE}; relations={RELATION_EXAMPLE}; "
    f"body_sections={BODY_SECTIONS_EXAMPLE}"
)

FACT_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "status": {"type": "string", "enum": list(FACT_STATUS_VALUES)},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
        "fact_id": {"type": "string"},
        "first_seen_at": {"type": "string"},
        "last_seen_at": {"type": "string"},
        "valid_from": {"type": "string"},
        "valid_to": {"type": "string"},
        "source_ref_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text"],
    "additionalProperties": True,
}

RELATION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "predicate": {"type": "string"},
        "object": {"type": "string"},
        "status": {"type": "string", "enum": list(FACT_STATUS_VALUES)},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
        "cardinality": {"type": "string", "enum": list(RELATION_CARDINALITY_VALUES)},
        "relation_id": {"type": "string"},
        "first_seen_at": {"type": "string"},
        "last_seen_at": {"type": "string"},
        "valid_from": {"type": "string"},
        "valid_to": {"type": "string"},
        "source_ref_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["subject", "predicate", "object"],
    "additionalProperties": True,
}

BODY_SECTIONS_SCHEMA = {
    "type": "object",
    "additionalProperties": {"type": "string"},
}


def validate_memory_write_contract(
    *,
    operation: str,
    target_kind: str,
    arguments: dict[str, object],
) -> list[str]:
    errors: list[str] = []
    require_truth_decision = operation in {"create", "upsert"} and target_kind in {"core", "ongoing"}
    errors.extend(
        _validate_truth_argument(
            field_name="facts",
            value=arguments.get("facts"),
            require_explicit=require_truth_decision,
        )
    )
    errors.extend(
        _validate_truth_argument(
            field_name="relations",
            value=arguments.get("relations"),
            require_explicit=require_truth_decision,
        )
    )
    errors.extend(_validate_body_sections_argument(arguments.get("body_sections")))
    return errors


def format_memory_write_contract_error(
    *,
    operation: str,
    target_kind: str,
    errors: list[str],
) -> str:
    prefix = "Invalid memory_write payload."
    if operation in {"create", "upsert"} and target_kind in {"core", "ongoing"}:
        prefix = (
            "Invalid memory_write payload. For core/ongoing create and upsert, facts and relations "
            "are explicit-decision fields, and summary is not a substitute. "
            f"{FACTS_USAGE_GUIDANCE}"
        )
    return (
        f"{prefix} Fix all of these before retrying: {'; '.join(errors)}. "
        f"Minimal valid example: {MEMORY_WRITE_EXAMPLE}"
    )


def _validate_truth_argument(
    *,
    field_name: str,
    value: object,
    require_explicit: bool,
) -> list[str]:
    errors: list[str] = []
    if value is None:
        if require_explicit:
            errors.append(
                f'{field_name} is required; pass a non-empty structured array or the literal string "None"'
            )
        return errors
    if isinstance(value, str):
        if value.strip().lower() == "none":
            return errors
        errors.append(
            f'{field_name} must be a structured array or the literal string "None"'
        )
        return errors
    if not isinstance(value, list):
        errors.append(
            f'{field_name} must be a structured array or the literal string "None"'
        )
        return errors
    if not value:
        errors.append(
            f'{field_name} cannot be an empty array; empty arrays are not allowed; use the literal string "None" instead'
        )
        return errors
    for index, item in enumerate(value):
        item_path = f"{field_name}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_path} must be an object; {field_name} array items must be objects")
            continue
        if field_name == "facts":
            errors.extend(_validate_fact_item(item_path, item))
        else:
            errors.extend(_validate_relation_item(item_path, item))
    return errors


def _validate_body_sections_argument(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        return [
            "body_sections must be an object keyed by section name with string values; "
            "Do not pass a list of section objects"
        ]
    errors: list[str] = []
    for section_name, section_value in value.items():
        if not isinstance(section_name, str) or not section_name.strip():
            errors.append("body_sections contains a blank or non-string section name")
            continue
        if not isinstance(section_value, str) or not section_value.strip():
            errors.append(
                f'body_sections["{section_name}"] must be a non-empty string'
            )
    return errors


def _validate_fact_item(path: str, item: dict[str, Any]) -> list[str]:
    errors = _validate_required_string_fields(item, path, ("text",))
    errors.extend(
        _validate_optional_choice_fields(
            item,
            path,
            (
                ("status", FACT_STATUS_VALUES),
                ("confidence", CONFIDENCE_VALUES),
            ),
        )
    )
    errors.extend(
        _validate_optional_string_fields(
            item,
            path,
            ("fact_id", "first_seen_at", "last_seen_at", "valid_from", "valid_to"),
        )
    )
    errors.extend(_validate_string_list_field(item, path, "source_ref_ids"))
    return errors


def _validate_relation_item(path: str, item: dict[str, Any]) -> list[str]:
    errors = _validate_required_string_fields(item, path, ("subject", "predicate", "object"))
    errors.extend(
        _validate_optional_choice_fields(
            item,
            path,
            (
                ("status", FACT_STATUS_VALUES),
                ("confidence", CONFIDENCE_VALUES),
                ("cardinality", RELATION_CARDINALITY_VALUES),
            ),
        )
    )
    errors.extend(
        _validate_optional_string_fields(
            item,
            path,
            ("relation_id", "first_seen_at", "last_seen_at", "valid_from", "valid_to"),
        )
    )
    errors.extend(_validate_string_list_field(item, path, "source_ref_ids"))
    return errors


def _validate_required_string_fields(
    item: dict[str, Any],
    path: str,
    field_names: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    for field_name in field_names:
        value = item.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{path}.{field_name} must be a non-empty string")
    return errors


def _validate_optional_string_fields(
    item: dict[str, Any],
    path: str,
    field_names: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    for field_name in field_names:
        if field_name not in item:
            continue
        value = item.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{path}.{field_name} must be a non-empty string when provided")
    return errors


def _validate_optional_choice_fields(
    item: dict[str, Any],
    path: str,
    field_specs: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[str]:
    errors: list[str] = []
    for field_name, allowed_values in field_specs:
        if field_name not in item:
            continue
        value = item.get(field_name)
        if not isinstance(value, str) or value not in allowed_values:
            allowed_text = ", ".join(allowed_values)
            errors.append(f"{path}.{field_name} must be one of: {allowed_text}")
    return errors


def _validate_string_list_field(
    item: dict[str, Any],
    path: str,
    field_name: str,
) -> list[str]:
    if field_name not in item:
        return []
    value = item.get(field_name)
    if not isinstance(value, list) or any(not isinstance(entry, str) or not entry.strip() for entry in value):
        return [f"{path}.{field_name} must be a list of non-empty strings when provided"]
    return []
