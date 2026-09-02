# Decision Index

| ID | Status | Decision | Evidence or reason | Detail |
| --- | --- | --- | --- | --- |
| DEC-HARNESS-001 | active | Store raw logs/artifacts in `output/harness-runs/`; keep only compact summaries and relative paths in durable memory. | Prevent large generated content from polluting model context. | this index |
| DEC-HARNESS-002 | active | Only the main agent writes canonical records and indexes; subagents return summaries or unique inbox candidates. | Prevent concurrent memory corruption and duplicate claims. | this index |
| DEC-HARNESS-003 | active | If an attempt misses the user's acceptance criteria, restore all attempt-owned code/configuration changes and verify every affected Git repository against its baseline before exit. Leave `rollback_pending` and block later attempts when safe restoration cannot be verified. | Prevent future work from silently building on a failed or partial implementation while preserving pre-existing user changes. | [root instructions](../../../AGENTS.md) |

Create a separate decision card from `harness/templates/DECISION.md` when a
decision needs alternatives, consequences, or supersession history.
