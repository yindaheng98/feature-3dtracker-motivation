# Decision Index

| ID | Status | Decision | Evidence or reason | Detail |
| --- | --- | --- | --- | --- |
| DEC-HARNESS-001 | active | Store raw logs/artifacts in `output/harness-runs/`; keep only compact summaries and relative paths in durable memory. | Prevent large generated content from polluting model context. | this index |
| DEC-HARNESS-002 | active | Only the main agent writes canonical records and indexes; subagents return summaries or unique inbox candidates. | Prevent concurrent memory corruption and duplicate claims. | this index |

Create a separate decision card from `harness/templates/DECISION.md` when a
decision needs alternatives, consequences, or supersession history.

