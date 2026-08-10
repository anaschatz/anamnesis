"""One-attempt Mem0 v2 ``infer=true`` diagnostic over fresh events."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from anamnesis.mem0_inference_diagnostic import (
    Mem0InferenceResult,
    _write_result,
    run_mem0_inference_diagnostic,
)

PROTOCOL_SCHEMA_VERSION = "mem0_inference_protocol.v2"
PROTOCOL_SHA256 = "e0031a0e9044b02b816afacb2ff1ecf4fe96bd4b26d2b6239d72dc496c3f5f7d"


async def run_mem0_inference_v2_diagnostic(
    *,
    protocol_path: Path,
    embedding_snapshot: Path,
    models_root: Path,
    source_commit: str,
) -> Mem0InferenceResult:
    return await run_mem0_inference_diagnostic(
        protocol_path=protocol_path,
        embedding_snapshot=embedding_snapshot,
        models_root=models_root,
        source_commit=source_commit,
        protocol_sha256=PROTOCOL_SHA256,
        protocol_schema_version=PROTOCOL_SCHEMA_VERSION,
        collection_name="anamnesis_mem0_inference_v2",
    )


def mem0_inference_v2_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--embedding-snapshot", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = asyncio.run(
        run_mem0_inference_v2_diagnostic(
            protocol_path=args.protocol,
            embedding_snapshot=args.embedding_snapshot,
            models_root=args.models_root,
            source_commit=args.source_commit,
        )
    )
    _write_result(args.output, result)
    return 0 if result.integrity_passed else 2


if __name__ == "__main__":
    raise SystemExit(mem0_inference_v2_main())


__all__ = [
    "PROTOCOL_SHA256",
    "mem0_inference_v2_main",
    "run_mem0_inference_v2_diagnostic",
]
