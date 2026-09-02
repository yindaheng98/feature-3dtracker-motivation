# Subagent Memory Inbox

Subagents normally return bounded summaries directly. If a report is too long,
they may create a uniquely named Markdown file under a task/session subdirectory
here. Use the fixed structure:

1. Facts
2. Evidence paths or experiment IDs
3. Inferences and confidence
4. Contradictions or uncertainty
5. Recommended canonical updates

Inbox files are candidates, not durable facts. The main agent verifies and merges
them, then removes or marks the candidate as consumed. Subagents must never edit
canonical memory cards or indexes.

