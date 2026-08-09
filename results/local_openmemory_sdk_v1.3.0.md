# Real OpenMemory SDK contract smoke — diagnostic only

The one authorized lifecycle passed against the official CaviraOSS OpenMemory
Python SDK `v1.3.0` at commit
`b04bf6e245577d0a024ea37cc02f4187ca7b0ffc`. The installed Python and migration
source bytes matched the frozen 49-file manifest before `Memory()` was imported.
The run used a fresh local SQLite database and the SDK's deterministic synthetic
embedding provider; it made no model or remote embedding call.

| Contract check | Result |
|---|---:|
| Add verified through scoped provider `get` | Pass |
| Search returned exactly one stored-text match | Pass |
| Opaque-handle `get` returned exact content | Pass |
| Delete verified by provider absence | Pass |
| Deleted opaque handle expired | Pass |
| Remaining scoped memory rows | 0 |
| Authoritative temporal memory | No |
| Can supply action evidence | No |
| Mutates Anamnesis state | No |

This establishes that the Anamnesis recall adapter works with the real upstream
SDK lifecycle, not only with compatible fakes or the earlier FastEmbed-backed
diagnostic index. It does **not** show that OpenMemory improves action quality;
the earlier v6/v7 experiments measured that architecture question separately.

## Upstream compatibility findings

A clean installation of the tagged package was not runnable as published. Its
code imports `python-dotenv`, `PyYAML`, `pypdf`, `mammoth`, `markdownify`, and
`openai` without declaring them in `pyproject.toml`; database migration also
imports `pkg_resources` without declaring `setuptools`. The run therefore used
the exact isolated compatibility versions frozen in the protocol pin. None was
added to the Anamnesis production dependency lock.

The upstream repository also explicitly warns that the project is in a rewrite
with breaking changes. For that reason, Anamnesis keeps OpenMemory optional,
caller-injected, byte-pinned, namespace-scoped, and non-authoritative.

## Reproducibility

- Anamnesis source commit: `cba7a1d9` (full SHA in provenance)
- Protocol: [`eval/OPENMEMORY_SDK_SMOKE.md`](../eval/OPENMEMORY_SDK_SMOKE.md)
- Pin: [`eval/openmemory_sdk_v1.3.0.pin.json`](../eval/openmemory_sdk_v1.3.0.pin.json)
- Machine-readable result: [`local_openmemory_sdk_v1.3.0.json`](local_openmemory_sdk_v1.3.0.json)
- Raw one-shot result SHA-256:
  `c68fc883cb1f79b05f9a7075c6e5476c22962dcfffceae7a8146a1036209222d`

This is a compatibility diagnostic, not a hypothesis test, benchmark, or
promotion of OpenMemory to the authoritative prospective-memory engine.
