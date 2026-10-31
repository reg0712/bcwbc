"""Lossless table encoding for repeated JSON object keys in model inputs."""

import json
from typing import Any

TABLE_HINT = ('Objects with "$columns" and "$rows" encode lists of records: '
              'zip each row with the columns. All original values are retained.\n')


def compact_json(value: Any) -> str:
    """JSON without insignificant whitespace; string contents remain unchanged."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _has_reserved_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(k in ("$columns", "$rows") for k in value) or any(
            _has_reserved_key(v) for v in value.values())
    return isinstance(value, list) and any(_has_reserved_key(v) for v in value)


def _pack(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _pack(v) for k, v in value.items()}
    if not isinstance(value, list):
        return value
    packed = [_pack(v) for v in value]
    if len(value) >= 2 and all(isinstance(v, dict) for v in value):
        columns = list(value[0])
        if columns and all(set(v) == set(columns) for v in value):
            table = {"$columns": columns, "$rows": [[v[k] for k in columns] for v in packed]}
            if len(compact_json(table)) < len(compact_json(packed)):
                return table
    return packed


def encode_inputs(value: dict[str, Any]) -> tuple[str, bool]:
    """Choose tables only when their savings cover the decoding instructions."""
    plain = compact_json(value)
    if _has_reserved_key(value):
        return plain, False
    packed = compact_json(_pack(value))
    if len(packed) + len(TABLE_HINT) + 20 < len(plain):
        return packed, True
    return plain, False


def unpack(value: Any) -> Any:
    """Decode an explicitly table-encoded block, preserving row and column order."""
    if isinstance(value, dict):
        if set(value) == {"$columns", "$rows"}:
            return [{k: unpack(v) for k, v in zip(value["$columns"], row, strict=True)}
                    for row in value["$rows"]]
        return {k: unpack(v) for k, v in value.items()}
    return [unpack(v) for v in value] if isinstance(value, list) else value
