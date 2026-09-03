#!/usr/bin/env python3
"""Small, dependency-free experiment memory and bounded run helper."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from typing import Iterator


ROOT = Path(__file__).resolve().parents[2]
MEMORY = ROOT / "harness" / "memory"
ACTIVE = MEMORY / "ACTIVE.md"
HOT_INDEX = MEMORY / "INDEX.md"
EXPERIMENT_INDEX = MEMORY / "experiments" / "INDEX.md"
EXPERIMENT_CODE = ROOT / "experiments"
STATE = ROOT / ".harness-state"
LOCK = STATE / "memory.lock"
SUBMODULES = (
    "SpaTrackerV2",
    "Open-d4rt",
    "MV-TAP",
    "TrackerSplat",
    "Look-Around-and-Pay-Attention-LAPA-",
)
VALID_STATUS = {"planned", "running", "successful", "failed", "inconclusive", "aborted"}
RAW_SUFFIXES = {".log", ".jsonl", ".npy", ".npz", ".zip", ".mp4", ".png", ".jpg", ".ply", ".pt", ".pth", ".ckpt"}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso(value: dt.datetime | None = None) -> str:
    return (value or utc_now()).isoformat().replace("+00:00", "Z")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@contextlib.contextmanager
def memory_lock() -> Iterator[None]:
    STATE.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    result: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip('"')
    return result


def set_frontmatter(text: str, updates: dict[str, str]) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("record has no YAML frontmatter")
    end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if end is None:
        raise ValueError("record frontmatter is not closed")
    found: set[str] = set()
    for i in range(1, end):
        if ":" not in lines[i]:
            continue
        key = lines[i].split(":", 1)[0].strip()
        if key in updates:
            lines[i] = f"{key}: {json.dumps(updates[key], ensure_ascii=False)}"
            found.add(key)
    for key, value in updates.items():
        if key not in found:
            lines.insert(end, f"{key}: {json.dumps(value, ensure_ascii=False)}")
            end += 1
    return "\n".join(lines) + "\n"


def active_metadata() -> dict[str, str]:
    return parse_frontmatter(ACTIVE.read_text(encoding="utf-8"))


def active_text(experiment_id: str | None = None, detail: str | None = None,
                output_dir: str | None = None, rollback: str = "not-needed") -> str:
    status = "rollback_pending" if experiment_id and rollback == "pending" else ("running" if experiment_id else "idle")
    note = (
        "Audit the listed repositories, processes, and paths before another attempt."
        if experiment_id
        else "No experiment is active. If this file says `running` or `rollback_pending`, audit\n"
        "the listed repositories, processes, and paths before starting another attempt."
    )
    return f"""---
schema: harness.active/v1
experiment_id: {experiment_id or 'null'}
status: {status}
updated_at: {iso()}
detail: {detail or 'null'}
output_dir: {output_dir or 'null'}
rollback: {rollback}
---

# Active Experiment

{note}
"""


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else f"unavailable ({result.stderr.strip()})"


def git_snapshot() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, repo in [("root", ROOT), *[(name, ROOT / name) for name in SUBMODULES]]:
        if not repo.exists():
            continue
        status = git_value(repo, "status", "--short")
        rows.append({
            "repo": name,
            "branch": git_value(repo, "branch", "--show-current") or "detached",
            "commit": git_value(repo, "rev-parse", "HEAD"),
            "dirty_paths": "; ".join(status.splitlines()) if status else "clean",
        })
    return rows


def sanitize_cell(value: str, limit: int = 140) -> str:
    compact = " ".join(value.split()).replace("|", "\\|")
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def resolve_experiment_code_dir(value: str) -> Path:
    requested = Path(value)
    if requested.is_absolute():
        raise SystemExit("experiment code directory must be repository-relative")
    resolved = (ROOT / requested).resolve()
    code_root = EXPERIMENT_CODE.resolve()
    if resolved == code_root or code_root not in resolved.parents:
        raise SystemExit("experiment code directory must be below experiments/")
    return resolved


def upsert_table_row(path: Path, start: str, end: str, record_id: str, row: str) -> None:
    text = path.read_text(encoding="utf-8")
    before, sep, rest = text.partition(start)
    if not sep:
        raise ValueError(f"missing marker {start} in {path}")
    middle, sep2, after = rest.partition(end)
    if not sep2:
        raise ValueError(f"missing marker {end} in {path}")
    lines = middle.strip("\n").splitlines()
    kept = [line for line in lines if not line.startswith(f"| {record_id} |")]
    insert_at = 2 if len(kept) >= 2 and kept[0].startswith("| ID |") else 0
    kept.insert(insert_at, row)
    rebuilt = "\n".join(kept)
    atomic_write(path, before + start + "\n" + rebuilt + "\n" + end + after)


def find_record(experiment_id: str) -> Path:
    matches = list((MEMORY / "experiments").glob(f"*/*/{experiment_id}.md"))
    if len(matches) != 1:
        raise SystemExit(f"expected one record for {experiment_id}, found {len(matches)}")
    return matches[0]


def render_record(experiment_id: str, title: str, hypothesis: str, scope: str,
                  code_mode: str, code_dir: str, created: dt.datetime,
                  detail: Path, output_dir: Path) -> str:
    snapshots = git_snapshot()
    snapshot_rows = "\n".join(
        f"| `{row['repo']}` | `{row['branch']}` | `{row['commit']}` | {sanitize_cell(row['dirty_paths'], 240)} |"
        for row in snapshots
    )
    return f"""---
schema: harness.experiment/v1
id: {experiment_id}
created_at: {iso(created)}
updated_at: {iso(created)}
status: running
scope: {json.dumps(scope, ensure_ascii=False)}
hypothesis: {json.dumps(hypothesis, ensure_ascii=False)}
code_mode: {code_mode}
code_dir: {code_dir}
output_dir: {output_dir.relative_to(ROOT)}
rollback: not-needed
---

# Experiment: {title}

## Intent

- User request:
- Hypothesis: {hypothesis}
- Success criteria:

## Reproducibility

| Repository | Branch | Commit | Dirty paths before |
| --- | --- | --- | --- |
{snapshot_rows}

- Python executable: `.venv/bin/python`
- Experiment code placement: `{code_mode}`
- Primary experiment code directory: `{code_dir}`
- Experiment scripts/components used:
- Placement rationale:
- Dataset refs and split:
- Config refs:
- Seed:
- Resource envelope: GPU, CPU, RAM, timeout

## Changes

- Allowed paths:
- Changed files and symbols:
- Compact change summary:

## Runs

<!-- run-rows:start -->
| Run | Command/manifest | Exit | Duration | Log/artifact path |
| --- | --- | --- | --- | --- |
<!-- run-rows:end -->

## Result

- Metrics with units and evaluation set:
- Error signature, at most five lines or 500 characters:
- Observation (direct evidence only):

## Interpretation

- Inference:
- Confidence: high / medium / low
- Conclusion: pending
- Follow-up ideas:
- Related experiment IDs:

## Cleanup

- Acceptance outcome: met / not-met
- Working trees after:
- Baseline comparison for root and each affected submodule:
- Rollback: not-needed / restored / pending
- Unresolved attempt-owned paths:
- Retained changes and reason:

<!-- closure:start -->
Experiment is still running.
<!-- closure:end -->
"""


def command_start(args: argparse.Namespace) -> int:
    created = utc_now()
    experiment_id = f"EXP-{created.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"
    detail = MEMORY / "experiments" / created.strftime("%Y") / created.strftime("%m") / f"{experiment_id}.md"
    output_dir = ROOT / "output" / "harness-runs" / experiment_id
    code_dir = resolve_experiment_code_dir(args.code_dir)
    code_dir_rel = code_dir.relative_to(ROOT).as_posix()
    with memory_lock():
        active = active_metadata()
        if active.get("status") not in {None, "idle"} or active.get("experiment_id") not in {None, "null"}:
            raise SystemExit(f"active experiment must be audited first: {active.get('experiment_id')}")
        if args.code_mode == "reuse":
            if not code_dir.is_dir():
                raise SystemExit(f"reuse code directory does not exist: {code_dir_rel}")
        else:
            if code_dir.exists():
                raise SystemExit(f"new code directory already exists: {code_dir_rel}")
            code_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        atomic_write(detail, render_record(
            experiment_id, args.title, args.hypothesis, args.scope,
            args.code_mode, code_dir_rel, created, detail, output_dir
        ))
        rel_detail = detail.relative_to(ROOT).as_posix()
        rel_output = output_dir.relative_to(ROOT).as_posix()
        exp_row = (
            f"| {experiment_id} | {created.date()} | running | {sanitize_cell(args.scope)} | "
            f"{sanitize_cell(args.hypothesis)} | started | [{experiment_id}]({detail.relative_to(EXPERIMENT_INDEX.parent).as_posix()}) |"
        )
        hot_row = (
            f"| {experiment_id} | {created.date()} | running | {sanitize_cell(args.scope)} | "
            f"started | [{experiment_id}]({detail.relative_to(HOT_INDEX.parent).as_posix()}) |"
        )
        upsert_table_row(EXPERIMENT_INDEX, "<!-- experiment-rows:start -->", "<!-- experiment-rows:end -->", experiment_id, exp_row)
        upsert_table_row(HOT_INDEX, "<!-- experiment-rows:start -->", "<!-- experiment-rows:end -->", experiment_id, hot_row)
        atomic_write(ACTIVE, active_text(experiment_id, rel_detail, rel_output))
    print(json.dumps({
        "experiment_id": experiment_id,
        "detail": rel_detail,
        "output_dir": rel_output,
        "code_mode": args.code_mode,
        "code_dir": code_dir_rel,
    }, ensure_ascii=False, indent=2))
    return 0


def tail_text(path: Path, byte_limit: int = 8192, char_limit: int = 4000) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - byte_limit))
        data = handle.read()
    text = data.decode("utf-8", errors="replace")
    return text[-char_limit:]


def command_run(args: argparse.Namespace) -> int:
    record = find_record(args.experiment_id)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("provide a command after --")
    cwd = (ROOT / args.cwd).resolve() if not Path(args.cwd).is_absolute() else Path(args.cwd).resolve()
    if ROOT not in cwd.parents and cwd != ROOT:
        raise SystemExit("run cwd must stay inside the repository")
    if not cwd.is_dir():
        raise SystemExit(f"run cwd does not exist: {cwd}")

    run_stamp = utc_now().strftime("%Y%m%dT%H%M%SZ") + f"-{secrets.token_hex(2)}"
    run_dir = ROOT / "output" / "harness-runs" / args.experiment_id / run_stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    manifest_path = run_dir / "manifest.json"
    started = utc_now()
    timed_out = False
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return_code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()
    ended = utc_now()
    duration = max(0.0, (ended - started).total_seconds())
    manifest = {
        "schema": "harness.run/v1",
        "experiment_id": args.experiment_id,
        "started_at": iso(started),
        "ended_at": iso(ended),
        "duration_seconds": duration,
        "cwd": cwd.relative_to(ROOT).as_posix() or ".",
        "command": command,
        "timeout_seconds": args.timeout,
        "timed_out": timed_out,
        "exit_code": return_code,
        "log": log_path.relative_to(ROOT).as_posix(),
    }
    atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    with memory_lock():
        text = record.read_text(encoding="utf-8")
        command_label = sanitize_cell(" ".join(command), 90)
        rel_manifest = manifest_path.relative_to(ROOT).as_posix()
        rel_log = log_path.relative_to(ROOT).as_posix()
        run_row = (
            f"| {run_stamp} | `{command_label}`; `{rel_manifest}` | {return_code}"
            f"{' (timeout)' if timed_out else ''} | {duration:.1f}s | `{rel_log}` |"
        )
        tmp = record.with_name(record.name + ".run-update")
        atomic_write(tmp, text)
        upsert_table_row(tmp, "<!-- run-rows:start -->", "<!-- run-rows:end -->", run_stamp, run_row)
        os.replace(tmp, record)

    summary = {key: manifest[key] for key in ("experiment_id", "exit_code", "timed_out", "duration_seconds", "log")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("--- bounded log tail ---")
    print(tail_text(log_path))
    return return_code if return_code >= 0 else 128 + abs(return_code)


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    before, sep, rest = text.partition(start)
    if not sep:
        raise ValueError(f"missing marker {start}")
    _, sep2, after = rest.partition(end)
    if not sep2:
        raise ValueError(f"missing marker {end}")
    return before + start + "\n" + replacement.rstrip() + "\n" + end + after


def command_finish(args: argparse.Namespace) -> int:
    if args.status not in VALID_STATUS - {"planned", "running"}:
        raise SystemExit(f"invalid closed status: {args.status}")
    record = find_record(args.experiment_id)
    metadata = parse_frontmatter(record.read_text(encoding="utf-8"))
    rollback = args.rollback or ("not-needed" if args.status == "successful" else "pending")
    with memory_lock():
        text = record.read_text(encoding="utf-8")
        text = set_frontmatter(text, {"updated_at": iso(), "status": args.status, "rollback": rollback})
        closure = (
            f"- Status: {args.status}\n"
            f"- Headline: {args.summary}\n"
            f"- Next: {args.next}\n"
            f"- Rollback: {rollback}"
        )
        text = replace_between(text, "<!-- closure:start -->", "<!-- closure:end -->", closure)
        atomic_write(record, text)

        created = metadata.get("created_at", "unknown")[:10]
        scope = metadata.get("scope", "unknown")
        hypothesis = metadata.get("hypothesis", "")
        rel_exp = record.relative_to(EXPERIMENT_INDEX.parent).as_posix()
        rel_hot = record.relative_to(HOT_INDEX.parent).as_posix()
        exp_row = (
            f"| {args.experiment_id} | {created} | {args.status} | {sanitize_cell(scope)} | "
            f"{sanitize_cell(hypothesis)} | {sanitize_cell(args.summary)} | [{args.experiment_id}]({rel_exp}) |"
        )
        hot_row = (
            f"| {args.experiment_id} | {created} | {args.status} | {sanitize_cell(scope)} | "
            f"{sanitize_cell(args.summary)} | [{args.experiment_id}]({rel_hot}) |"
        )
        upsert_table_row(EXPERIMENT_INDEX, "<!-- experiment-rows:start -->", "<!-- experiment-rows:end -->", args.experiment_id, exp_row)
        upsert_table_row(HOT_INDEX, "<!-- experiment-rows:start -->", "<!-- experiment-rows:end -->", args.experiment_id, hot_row)
        active = active_metadata()
        if active.get("experiment_id") == args.experiment_id:
            if rollback == "pending":
                atomic_write(
                    ACTIVE,
                    active_text(
                        args.experiment_id,
                        record.relative_to(ROOT).as_posix(),
                        metadata.get("output_dir", "null"),
                        rollback,
                    ),
                )
            else:
                atomic_write(ACTIVE, active_text())
    print(json.dumps({"experiment_id": args.experiment_id, "status": args.status, "rollback": rollback}, ensure_ascii=False, indent=2))
    if rollback == "pending":
        print("warning: rollback is pending; do not start another code attempt on this state", file=sys.stderr)
    return 0


def command_check(_: argparse.Namespace) -> int:
    required = [ACTIVE, HOT_INDEX, EXPERIMENT_INDEX, ROOT / "AGENTS.md"]
    errors: list[str] = []
    warnings: list[str] = []
    for path in required:
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")
    if HOT_INDEX.is_file():
        content = HOT_INDEX.read_text(encoding="utf-8")
        if len(content.encode("utf-8")) > 12 * 1024:
            errors.append("harness/memory/INDEX.md exceeds 12 KiB")
        if len(content.splitlines()) > 200:
            errors.append("harness/memory/INDEX.md exceeds 200 lines")
    if ACTIVE.is_file():
        active = active_metadata()
        if active.get("status") not in {"idle", "running", "rollback_pending"}:
            errors.append(f"invalid ACTIVE status: {active.get('status')}")
        active_id = active.get("experiment_id")
        if active_id not in {None, "null"}:
            try:
                find_record(active_id)
            except SystemExit as exc:
                errors.append(str(exc))
    for record in (MEMORY / "experiments").glob("*/*/EXP-*.md"):
        metadata = parse_frontmatter(record.read_text(encoding="utf-8"))
        for key in (
            "schema", "id", "created_at", "updated_at", "status", "scope",
            "hypothesis", "code_mode", "code_dir", "output_dir", "rollback",
        ):
            if not metadata.get(key):
                errors.append(f"{record.relative_to(ROOT)} missing frontmatter key {key}")
        if metadata.get("status") not in VALID_STATUS:
            errors.append(f"{record.relative_to(ROOT)} has invalid status {metadata.get('status')}")
        if metadata.get("code_mode") not in {"reuse", "new"}:
            errors.append(f"{record.relative_to(ROOT)} has invalid code_mode {metadata.get('code_mode')}")
        code_dir = metadata.get("code_dir", "")
        if code_dir:
            try:
                resolved_code_dir = resolve_experiment_code_dir(code_dir)
            except SystemExit as exc:
                errors.append(f"{record.relative_to(ROOT)} invalid code_dir: {exc}")
            else:
                normalized_code_dir = resolved_code_dir.relative_to(ROOT).as_posix()
                if normalized_code_dir != code_dir:
                    errors.append(
                        f"{record.relative_to(ROOT)} code_dir is not normalized: {code_dir}"
                    )
        if record.stat().st_size > 12 * 1024:
            warnings.append(f"large experiment card should be compacted: {record.relative_to(ROOT)}")
    for path in MEMORY.rglob("*"):
        if path.is_file() and path.suffix.lower() in RAW_SUFFIXES:
            errors.append(f"raw artifact found in durable memory: {path.relative_to(ROOT)}")
    result = {"ok": not errors, "errors": errors, "warnings": warnings}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    start = subparsers.add_parser("start", help="create an experiment card and active marker")
    start.add_argument("--title", required=True)
    start.add_argument("--hypothesis", required=True)
    start.add_argument("--scope", required=True)
    start.add_argument("--code-mode", required=True, choices=("reuse", "new"))
    start.add_argument("--code-dir", required=True, help="repository-relative directory below experiments/")
    start.set_defaults(func=command_start)

    run = subparsers.add_parser("run", help="run a command with bounded output capture")
    run.add_argument("experiment_id")
    run.add_argument("--cwd", default=".", help="repository-relative working directory")
    run.add_argument("--timeout", type=float, default=None)
    run.add_argument("command", nargs="*")
    run.set_defaults(func=command_run)

    finish = subparsers.add_parser("finish", help="close an experiment card and update indexes")
    finish.add_argument("experiment_id")
    finish.add_argument("--status", required=True, choices=sorted(VALID_STATUS - {"planned", "running"}))
    finish.add_argument("--summary", required=True)
    finish.add_argument("--next", required=True)
    finish.add_argument("--rollback", choices=("not-needed", "restored", "pending"))
    finish.set_defaults(func=command_finish)

    check = subparsers.add_parser("check", help="validate harness structure and memory budgets")
    check.set_defaults(func=command_check)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
