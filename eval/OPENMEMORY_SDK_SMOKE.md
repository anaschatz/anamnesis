# OpenMemory v1.3.0 real-SDK contract smoke

This is a local compatibility test, not a benchmark or hypothesis test. It
exercises the real CaviraOSS OpenMemory Python SDK through Anamnesis's
non-authoritative recall boundary. It must not feed OpenMemory identifiers or
retrieval scores into the prospective temporal store or action evidence.

The upstream identity is tag `v1.3.0`, commit
`b04bf6e245577d0a024ea37cc02f4187ca7b0ffc`. The installed 49-file Python/SQL
source tree must hash to
`434a82caaced7548041a4c2a72a3d626a034af6dc7b1ac7e1c80af5035d3d600`.
The run uses a fresh absolute SQLite path and OpenMemory's deterministic
`synthetic` embedding provider. No model or remote embedding call is allowed.

## Known upstream packaging defect

A clean install of the official `openmemory==1.3.0` wheel cannot import or
construct `Memory()`: several imported packages are absent from its declared
dependencies, and migrations use `pkg_resources` without declaring
`setuptools`. The exact compatibility packages and versions are pinned in
`eval/openmemory_sdk_v1.3.0.pin.json`. They belong only in the isolated SDK
environment; they are deliberately not added to the Anamnesis dependency lock.

## One-shot procedure

1. Create a fresh external virtual environment.
2. Install the exact upstream commit and the exact compatibility packages from
   the pin.
3. From a clean Anamnesis source commit, run:

   ```bash
   PYTHONPATH=src /path/to/sdk-venv/bin/python -m anamnesis.openmemory_sdk_smoke \
     --pin eval/openmemory_sdk_v1.3.0.pin.json \
     --database /absolute/path/to/fresh/openmemory.sqlite \
     --output results/runs/local/openmemory_sdk_v1.3.0.json
   ```

The command performs exactly one `add -> search -> get -> delete` lifecycle.
It fails closed on dependency, source, locality, scope, content, deletion, or
output-path drift. The result contains no provider memory identifier or local
absolute path. A failure is retained and reported; it is not repaired or
selectively rerun under the same protocol identity.
