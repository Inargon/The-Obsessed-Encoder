#!/usr/bin/env python3
"""Validate the experiment-data registry and optionally export numeric values."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
FORBIDDEN_DATA_KEYS = {"claim", "conclusion", "interpretation", "recommendation"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def walk(node: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, (*path, str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk(value, (*path, str(index)))


def collect_protocol_references(node: Any) -> set[str]:
    references: set[str] = set()
    for path, value in walk(node):
        if not path:
            continue
        key = path[-1]
        if key in {"protocol", "evaluation_protocol"} and isinstance(value, str):
            references.add(value)
        elif key == "protocol_ids" and isinstance(value, list):
            references.update(item for item in value if isinstance(item, str))
        elif key == "protocols" and isinstance(value, dict):
            references.update(item for item in value.values() if isinstance(item, str))
    return references


def validate_file(path: Path, data: Any, known_protocols: set[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"{path.name}: root must be an object"]
    if data.get("schema_version") != 1:
        errors.append(f"{path.name}: schema_version must equal 1")

    for record_path, value in walk(data):
        dotted = ".".join(record_path) or "<root>"
        if record_path and record_path[-1].lower() in FORBIDDEN_DATA_KEYS:
            errors.append(f"{path.name}:{dotted}: inference field is not allowed")
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{path.name}:{dotted}: non-finite number")
        if record_path and "success_rate" in record_path[-1] and isinstance(value, (int, float)):
            if not 0 <= float(value) <= 1:
                errors.append(f"{path.name}:{dotted}: success rate outside [0, 1]")
        if record_path and record_path[-1] == "std" and isinstance(value, (int, float)):
            if value < 0:
                errors.append(f"{path.name}:{dotted}: negative standard deviation")

    missing_protocols = collect_protocol_references(data) - known_protocols
    for protocol_id in sorted(missing_protocols):
        errors.append(f"{path.name}: unknown protocol ID {protocol_id!r}")
    return errors


def numeric_rows(file_name: str, data: Any) -> Iterable[dict[str, str | float]]:
    for record_path, value in walk(data):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if record_path and record_path[-1] == "std":
            continue
        std = ""
        parent = data
        for part in record_path[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        if isinstance(parent, dict) and record_path and record_path[-1] == "mean":
            candidate = parent.get("std")
            if isinstance(candidate, (int, float)):
                std = float(candidate)
        yield {
            "file": file_name,
            "record_path": ".".join(record_path),
            "value": float(value),
            "std": std,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, help="write a long-form numeric CSV")
    args = parser.parse_args()

    catalog = load_json(ROOT / "catalog.json")
    protocols_data = load_json(ROOT / "protocols.json")
    known_protocols = set(protocols_data.get("protocols", {}))
    listed = catalog.get("files", [])

    errors: list[str] = []
    loaded: dict[str, Any] = {}
    for file_name in listed:
        path = ROOT / file_name
        if not path.is_file():
            errors.append(f"catalog.json: listed file does not exist: {file_name}")
            continue
        try:
            loaded[file_name] = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{file_name}: cannot load JSON: {exc}")
            continue
        errors.extend(validate_file(path, loaded[file_name], known_protocols))

    if errors:
        raise SystemExit("REGISTRY INVALID\n" + "\n".join(f"- {item}" for item in errors))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["file", "record_path", "value", "std"])
            writer.writeheader()
            for file_name, data in loaded.items():
                writer.writerows(numeric_rows(file_name, data))

    print(f"REGISTRY VALID: {len(loaded)} data files, {len(known_protocols)} protocols")
    if args.csv:
        print(f"CSV WRITTEN: {args.csv}")


if __name__ == "__main__":
    main()
