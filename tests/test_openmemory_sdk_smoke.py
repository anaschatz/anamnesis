from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anamnesis.openmemory_sdk_smoke import (
    load_openmemory_sdk_pin,
    python_source_tree_identity,
)

ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = ROOT / "eval" / "openmemory_sdk_v1.3.0.pin.json"


def test_frozen_openmemory_sdk_pin() -> None:
    pin = load_openmemory_sdk_pin(PIN_PATH)

    assert pin.upstream_revision == "b04bf6e245577d0a024ea37cc02f4187ca7b0ffc"
    assert pin.package_version == "1.3.0"
    assert pin.python_source_tree_sha256 == (
        "434a82caaced7548041a4c2a72a3d626a034af6dc7b1ac7e1c80af5035d3d600"
    )
    assert pin.python_source_file_count == 49
    assert pin.python_source_bytes == 177242
    assert pin.embedding_provider == "synthetic"
    assert pin.database_backend == "sqlite"
    assert pin.network_calls_allowed is False
    assert pin.hypothesis_test_eligible is False


def test_source_tree_identity_is_path_independent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
        (root / "a.py").write_text("A = 1\n", encoding="utf-8")
        (root / "nested" / "b.py").write_text("B = 2\n", encoding="utf-8")
        (root / "ignored.txt").write_text("ignored", encoding="utf-8")

    assert python_source_tree_identity(first) == python_source_tree_identity(second)


def test_source_tree_identity_rejects_byte_drift(tmp_path: Path) -> None:
    root = tmp_path / "package"
    root.mkdir()
    source = root / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    before = python_source_tree_identity(root)
    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert python_source_tree_identity(root) != before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hypothesis_test_eligible", True),
        ("network_calls_allowed", True),
        ("embedding_provider", "openai"),
        ("database_backend", "postgres"),
    ],
)
def test_pin_rejects_unsafe_protocol_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_openmemory_sdk_pin(path)


def test_pin_bytes_are_stable() -> None:
    assert hashlib.sha256(PIN_PATH.read_bytes()).hexdigest() == (
        "9d42b1446e5dab7b85fe13c3ef85a3c4b1e61eba406354a1f451f67be531b91c"
    )
