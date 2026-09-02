# Open-d4rt

## Role

4D reconstruction/tracking with WorldTrack evaluation, demos, and training.

## Bounded evaluation

Run from `Open-d4rt/` and always use a unique output path:

```bash
PATH="../.venv/bin:$PATH" \
LIMIT_SEQS=1 \
SUBSETS=adt_mini \
QUERY_CHUNK_SIZE=1024 \
OUTPUT_DIR=../output/harness-runs/<experiment-id>/worldtrack \
bash run_eval_worldtrack.sh
```

This is a template, not permission to run: first verify the expected checkpoint,
dataset, imports, GPUs, and free memory. Reduce the query chunk further if the
experiment is specifically diagnosing memory use.

## Harness rules

- Default evaluation chunks can OOM; begin at `QUERY_CHUNK_SIZE<=1024` and one
  sequence/subset.
- Training scripts may default to eight GPUs and automatic evaluation while the
  host has four GPUs. Run their documented dry-run first; a one-GPU, few-step
  smoke requires explicit bounded settings.
- Model weights and default WorldTrack data may be absent. Missing inputs are a
  preflight failure, not a reason to auto-download.
- Never reuse a shared result directory when scripts can overwrite summary or
  per-sequence JSON files.

