"""
Non-blocking lifecycle event emitter and merged telemetry reader for Local Agents.

Tracks live `"running"` local executions (`body scout`, `body implement`, `body loop`,
`HybridRouter`, `StateCapsuleBuilder`) alongside completed shadow evaluations in
`~/.whole_body_compute/run/local_agent_runs.jsonl`.

Security & reliability invariants:
- `0o600` permissions on `local_agent_runs.jsonl`, `.lock`, and `.tmp` files.
- `fcntl.flock` exclusive locking on appends and atomic log rotation.
- `PYTEST_CURRENT_TEST` home-write isolation: skips disk writes silently when
  targeting `~/.whole_body_compute`.
- Sanitization via `redact_secrets` + `EpithelialScreenFilter`.
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from whole_body_compute.local.shadow_ledger import (
    _is_inside_live_wbc_dir,
    _resolve_telemetry_paths,
)
from whole_body_compute.organs.immune_vascular import EpithelialScreenFilter
from whole_body_compute.redaction import redact_secrets

logger = logging.getLogger("whole_body_compute.local.run_events")

MAX_RUN_LOG_LINES = 2000
MAX_RUN_LOG_BYTES = 2 * 1024 * 1024
ROTATE_KEEP_LINES = 500
STALE_RUNNING_AGE_SECONDS = 600.0
PROMOTION_TARGET = 20
MAX_DETAIL_FIELD_CHARS = 16000
MAX_LIST_SUMMARY_CHARS = 320

ALLOWED_END_STATUSES = frozenset({"pass", "guardrail_blocked", "error"})


def _resolve_runs_log_path(runs_log_path: Path | None = None) -> Path:
    """Resolve the backing JSONL path for local agent run lifecycle events."""
    if runs_log_path is not None:
        return Path(runs_log_path).expanduser()
    env_path = os.environ.get("WBC_LOCAL_RUNS_LOG", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".whole_body_compute" / "run" / "local_agent_runs.jsonl"


def _lock_path_for(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.name}.lock")


def is_run_events_target_safe_for_test(runs_log_path: Path | None = None) -> bool:
    """
    Return True if writing lifecycle events is safe under pytest (i.e. the
    resolved log file and its `.lock`/`.tmp` sibling paths live outside
    `~/.whole_body_compute`).
    """
    try:
        resolved = _resolve_runs_log_path(runs_log_path)
        lock_path = _lock_path_for(resolved)
        tmp_probe = resolved.with_name(f".{resolved.name}.probe.tmp")
    except (OSError, RuntimeError, ValueError):
        return False
    return not (
        _is_inside_live_wbc_dir(resolved)
        or _is_inside_live_wbc_dir(lock_path)
        or _is_inside_live_wbc_dir(tmp_probe)
    )


def _should_skip_test_write(runs_log_path: Path | None = None) -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) and not is_run_events_target_safe_for_test(
        runs_log_path
    )


def _chmod_0600(path: Path | str) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _coerce_end_status(status: str) -> str:
    raw = str(status or "").strip().lower()
    if raw in ALLOWED_END_STATUSES:
        return raw
    if "guardrail" in raw or "block" in raw:
        return "guardrail_blocked"
    return "error"


def _truncate_str(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _sanitize_text(text: str) -> str:
    """Redact secrets and neutralize prompt injections in telemetry strings."""
    if not isinstance(text, str) or not text:
        return "" if isinstance(text, str) else str(text)
    redacted = redact_secrets(text)
    cleaned, _ = EpithelialScreenFilter.sanitize(
        redacted,
        source="local_agent_telemetry",
        wrap_untrusted_tags=False,
        record_antigen=False,
    )
    return cleaned


def _sanitize_value(value: Any, *, max_str_chars: int | None = None) -> Any:
    """Recursively sanitize all strings in nested dicts/lists."""
    if isinstance(value, str):
        return _truncate_str(_sanitize_text(value), max_str_chars)
    if isinstance(value, dict):
        return {
            (_sanitize_text(k) if isinstance(k, str) else str(k)): _sanitize_value(
                v, max_str_chars=max_str_chars
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_value(item, max_str_chars=max_str_chars) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item, max_str_chars=max_str_chars) for item in value]
    return value


def _rotate_if_needed_locked(resolved_log: Path) -> None:
    """Atomically keep newest `ROTATE_KEEP_LINES` records when thresholds are exceeded."""
    try:
        stat_res = resolved_log.stat()
    except OSError:
        return

    needs_byte_rotate = stat_res.st_size > MAX_RUN_LOG_BYTES
    with open(resolved_log, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]

    if not needs_byte_rotate and len(lines) <= MAX_RUN_LOG_LINES:
        return

    kept = lines[-ROTATE_KEEP_LINES:]
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(resolved_log.parent),
        prefix=f".{resolved_log.name}.",
        suffix=".tmp",
    )
    try:
        _chmod_0600(tmp_path)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_fh:
            if kept:
                tmp_fh.write("\n".join(kept) + "\n")
        _chmod_0600(tmp_path)
        os.replace(tmp_path, resolved_log)
        _chmod_0600(resolved_log)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _append_event(event: dict[str, Any], runs_log_path: Path | None = None) -> bool:
    """Append a JSONL event under `fcntl.flock` with `0o600` permissions and auto-rotation."""
    if _should_skip_test_write(runs_log_path):
        return False

    try:
        resolved_log = _resolve_runs_log_path(runs_log_path)
        resolved_log.parent.mkdir(parents=True, exist_ok=True)
        lock_path = _lock_path_for(resolved_log)

        with open(lock_path, "a+", encoding="utf-8") as lock_fh:
            _chmod_0600(lock_path)
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                with open(resolved_log, "a", encoding="utf-8") as log_fh:
                    _chmod_0600(resolved_log)
                    log_fh.write(json.dumps(event, sort_keys=False) + "\n")
                    log_fh.flush()
                _chmod_0600(resolved_log)
                _rotate_if_needed_locked(resolved_log)
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return True
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("Local run event append skipped: %s", exc)
        return False


def record_local_run_start(
    role: str,
    task_type: str,
    model: str,
    engine: str,
    mode: str = "live",
    *,
    run_id: str | None = None,
    summary: str = "",
    workspace: str = "",
    extra: dict[str, Any] | None = None,
    runs_log_path: Path | None = None,
) -> str:
    """
    Record the start of a local agent run to `local_agent_runs.jsonl`.

    Returns `run_id`, or `""` when skipped under `PYTEST_CURRENT_TEST` home-write isolation.
    Never raises on I/O errors.
    """
    if _should_skip_test_write(runs_log_path):
        return ""

    resolved_id = str(run_id or f"lrun_{uuid.uuid4().hex[:12]}")
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "event": "start",
        "run_id": resolved_id,
        "pid": os.getpid(),
        "started_at": now_iso,
        "role": str(role),
        "task_type": str(task_type),
        "model": str(model),
        "engine": str(engine),
        "mode": str(mode or "live"),
        "summary": str(summary or ""),
        "workspace": str(workspace or ""),
        "extra": dict(extra) if isinstance(extra, dict) else {},
    }
    _append_event(payload, runs_log_path=runs_log_path)
    return resolved_id


def record_local_run_end(
    run_id: str,
    status: str,
    latency_ms: float,
    *,
    valid_guardrail: bool = True,
    tokens_saved_est: int = 0,
    sla_ms: float = 2000.0,
    detail_ref: dict[str, Any] | None = None,
    runs_log_path: Path | None = None,
) -> dict[str, Any]:
    """
    Record the completion of a local agent run to `local_agent_runs.jsonl`.

    Returns the recorded payload, or `{}` when skipped under `PYTEST_CURRENT_TEST`
    home-write isolation. Never raises on I/O errors.
    """
    if _should_skip_test_write(runs_log_path):
        return {}

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        lat_val = round(float(latency_ms), 2)
    except (TypeError, ValueError):
        lat_val = 0.0
    try:
        sla_val = round(float(sla_ms), 2)
    except (TypeError, ValueError):
        sla_val = 2000.0
    try:
        tokens_val = int(tokens_saved_est or 0)
    except (TypeError, ValueError):
        tokens_val = 0

    payload: dict[str, Any] = {
        "event": "end",
        "run_id": str(run_id),
        "ended_at": now_iso,
        "status": _coerce_end_status(status),
        "latency_ms": lat_val,
        "valid_guardrail": bool(valid_guardrail),
        "tokens_saved_est": tokens_val,
        "sla_ms": sla_val,
        "detail_ref": dict(detail_ref) if isinstance(detail_ref, dict) else {},
    }
    _append_event(payload, runs_log_path=runs_log_path)
    return payload


def _parse_iso_ts(value: str | None) -> float | None:
    if not value or not isinstance(value, str):
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _is_pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _read_jsonl(path: Path) -> list[tuple[str, dict[str, Any]]]:
    if not path.is_file():
        return []
    records: list[tuple[str, dict[str, Any]]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    records.append((line, parsed))
    except OSError:
        return []
    return records


def _load_promotion_stats(
    shadow_log_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load per-task promotion metrics from `router_telemetry_ledger.json`."""
    resolved_shadow, default_ledger = _resolve_telemetry_paths(shadow_log_path)
    _ = resolved_shadow
    resolved_ledger = Path(ledger_path).expanduser() if ledger_path is not None else default_ledger
    if not resolved_ledger.is_file():
        return {}
    try:
        data = json.loads(resolved_ledger.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        return {}

    stats: dict[str, dict[str, Any]] = {}
    for task_name, entry in tasks.items():
        if not isinstance(entry, dict):
            continue
        count = int(entry.get("count", 0) or 0)
        pass_rate = float(entry.get("guardrail_pass_rate", 0.0) or 0.0)
        p95_ms = float(entry.get("p95_ms", 0.0) or 0.0)
        sla_ms = float(entry.get("sla_ms", 2000.0) or 2000.0)
        eligible = bool(
            entry.get(
                "promotion_eligible",
                count >= PROMOTION_TARGET and pass_rate >= 0.98 and p95_ms <= sla_ms,
            )
        )
        stats[str(task_name)] = {
            "count": count,
            "guardrail_pass_rate": round(pass_rate, 4),
            "p95_ms": round(p95_ms, 2),
            "sla_ms": round(sla_ms, 2),
            "promotion_eligible": eligible,
            "promotion_target": PROMOTION_TARGET,
        }
    return stats


def _merge_runs_and_shadow(
    *,
    runs_log_path: Path | None = None,
    shadow_log_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Merge start/end lifecycle records and shadow eval records into `(summaries, raw_details)`.
    """
    resolved_runs = _resolve_runs_log_path(runs_log_path)
    resolved_shadow, _ = _resolve_telemetry_paths(shadow_log_path)

    now_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
    by_id: dict[str, dict[str, Any]] = {}
    details_by_id: dict[str, dict[str, Any]] = {}
    order_seq = 0

    for _, ev in _read_jsonl(resolved_runs):
        run_id = str(ev.get("run_id") or "").strip()
        if not run_id:
            continue
        ev_type = str(ev.get("event") or "").strip().lower()
        order_seq += 1

        if run_id not in by_id:
            by_id[run_id] = {
                "run_id": run_id,
                "pid": ev.get("pid"),
                "role": str(ev.get("role") or ev.get("task_type") or "Local Agent"),
                "task_type": str(ev.get("task_type") or "unknown"),
                "model": str(ev.get("model") or "unknown"),
                "engine": str(ev.get("engine") or "unknown"),
                "mode": str(ev.get("mode") or "live"),
                "status": "running",
                "started_at": ev.get("started_at"),
                "ended_at": None,
                "latency_ms": None,
                "sla_ms": float(ev.get("sla_ms", 2000.0) or 2000.0),
                "valid_guardrail": True,
                "tokens_saved_est": int(ev.get("tokens_saved_est", 0) or 0),
                "cost_usd": 0.0,
                "summary": str(ev.get("summary") or ""),
                "workspace": str(ev.get("workspace") or ""),
                "_seq": order_seq,
            }
            details_by_id[run_id] = {
                "run_id": run_id,
                "extra": dict(ev.get("extra") or {}) if isinstance(ev.get("extra"), dict) else {},
                "detail_ref": {},
            }

        row = by_id[run_id]
        detail = details_by_id[run_id]
        row["_seq"] = order_seq

        if ev_type == "start":
            row["pid"] = ev.get("pid", row.get("pid"))
            row["started_at"] = ev.get("started_at") or row.get("started_at")
            row["role"] = str(ev.get("role") or row["role"])
            row["task_type"] = str(ev.get("task_type") or row["task_type"])
            row["model"] = str(ev.get("model") or row["model"])
            row["engine"] = str(ev.get("engine") or row["engine"])
            row["mode"] = str(ev.get("mode") or row["mode"])
            if ev.get("summary"):
                row["summary"] = str(ev["summary"])
            if ev.get("workspace"):
                row["workspace"] = str(ev["workspace"])
            if isinstance(ev.get("extra"), dict):
                detail["extra"].update(ev["extra"])
        elif ev_type == "end":
            row["ended_at"] = ev.get("ended_at") or row.get("ended_at")
            row["status"] = _coerce_end_status(str(ev.get("status") or "error"))
            row["latency_ms"] = ev.get("latency_ms")
            row["sla_ms"] = float(ev.get("sla_ms", row["sla_ms"]) or row["sla_ms"])
            row["valid_guardrail"] = bool(ev.get("valid_guardrail", True))
            row["tokens_saved_est"] = int(
                ev.get("tokens_saved_est", row["tokens_saved_est"]) or 0
            )
            if isinstance(ev.get("detail_ref"), dict):
                detail["detail_ref"].update(ev["detail_ref"])

    # Reconcile stale "running" rows whose PID died or age > 600s.
    for row in by_id.values():
        if row["status"] == "running":
            pid = row.get("pid")
            started_ts = _parse_iso_ts(row.get("started_at"))
            age_s = (now_ts - started_ts) if started_ts is not None else STALE_RUNNING_AGE_SECONDS + 1
            if not _is_pid_alive(pid) or age_s > STALE_RUNNING_AGE_SECONDS:
                row["status"] = "stale_interrupted"

    # Merge shadow evals from specialist_router_shadow.jsonl without duplicating run_ids.
    for raw_line, shadow in _read_jsonl(resolved_shadow):
        explicit_id = str(shadow.get("run_id") or "").strip()
        if explicit_id and explicit_id in by_id:
            continue

        order_seq += 1
        if explicit_id:
            shadow_id = explicit_id
        else:
            digest = hashlib.sha256(raw_line.encode("utf-8", errors="replace")).hexdigest()[:12]
            shadow_id = f"shadow_{digest}"

        if shadow_id in by_id:
            continue

        valid_guardrail = bool(shadow.get("valid_guardrail", True))
        ts = shadow.get("timestamp")
        task_name = str(shadow.get("task_type") or "unknown")
        by_id[shadow_id] = {
            "run_id": shadow_id,
            "pid": None,
            "role": str(shadow.get("role") or task_name),
            "task_type": task_name,
            "model": str(shadow.get("model") or "unknown"),
            "engine": str(shadow.get("engine") or "unknown"),
            "mode": "shadow",
            "status": "pass" if valid_guardrail else "guardrail_blocked",
            "started_at": ts,
            "ended_at": ts,
            "latency_ms": shadow.get("latency_ms"),
            "sla_ms": float(shadow.get("sla_ms", 2000.0) or 2000.0),
            "valid_guardrail": valid_guardrail,
            "tokens_saved_est": int(shadow.get("tokens_saved_est", 0) or 0),
            "cost_usd": 0.0,
            "summary": str(shadow.get("summary") or f"Shadow eval ({shadow.get('baseline_used', 'shadow')})"),
            "workspace": str(shadow.get("workspace") or ""),
            "_seq": order_seq,
        }
        details_by_id[shadow_id] = {
            "run_id": shadow_id,
            "extra": {
                k: v
                for k, v in shadow.items()
                if k not in {"timestamp", "task_type", "engine", "model", "latency_ms", "valid_guardrail"}
            },
            "detail_ref": dict(shadow.get("detail_ref") or {})
            if isinstance(shadow.get("detail_ref"), dict)
            else {},
        }

    return by_id, details_by_id


def _sort_key_for_run(row: dict[str, Any]) -> tuple[float, int]:
    ts = _parse_iso_ts(row.get("ended_at")) or _parse_iso_ts(row.get("started_at")) or 0.0
    return (ts, int(row.get("_seq", 0)))


def list_merged_local_runs(
    *,
    limit: int = 40,
    status: str | None = None,
    task_type: str | None = None,
    mode: str | None = None,
    runs_log_path: Path | None = None,
    shadow_log_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """
    Return merged, sanitized local agent runs and FinOps/promotion summary.
    """
    clamped_limit = max(1, min(100, int(limit or 40)))
    by_id, _ = _merge_runs_and_shadow(
        runs_log_path=runs_log_path,
        shadow_log_path=shadow_log_path,
    )
    promo_by_task = _load_promotion_stats(
        shadow_log_path=shadow_log_path,
        ledger_path=ledger_path,
    )

    all_rows = sorted(by_id.values(), key=_sort_key_for_run, reverse=True)

    status_filter = status.strip().lower() if isinstance(status, str) and status.strip() else None
    task_filter = task_type.strip().lower() if isinstance(task_type, str) and task_type.strip() else None
    mode_filter = mode.strip().lower() if isinstance(mode, str) and mode.strip() else None

    filtered: list[dict[str, Any]] = []
    for row in all_rows:
        if status_filter and str(row.get("status", "")).lower() != status_filter:
            continue
        if task_filter and str(row.get("task_type", "")).lower() != task_filter:
            continue
        if mode_filter and str(row.get("mode", "")).lower() != mode_filter:
            continue
        filtered.append(row)

    selected = filtered[:clamped_limit]

    sanitized_runs: list[dict[str, Any]] = []
    for row in selected:
        task_name = str(row.get("task_type") or "unknown")
        promo = promo_by_task.get(
            task_name,
            {
                "count": 0,
                "guardrail_pass_rate": 0.0,
                "p95_ms": 0.0,
                "sla_ms": float(row.get("sla_ms", 2000.0) or 2000.0),
                "promotion_eligible": False,
                "promotion_target": PROMOTION_TARGET,
            },
        )
        clean_row = {
            "run_id": _sanitize_text(str(row.get("run_id") or "")),
            "pid": row.get("pid"),
            "role": _truncate_str(_sanitize_text(str(row.get("role") or "")), 120),
            "task_type": _truncate_str(_sanitize_text(task_name), 120),
            "model": _truncate_str(_sanitize_text(str(row.get("model") or "")), 120),
            "engine": _truncate_str(_sanitize_text(str(row.get("engine") or "")), 120),
            "mode": _truncate_str(_sanitize_text(str(row.get("mode") or "live")), 32),
            "status": str(row.get("status") or "unknown"),
            "started_at": row.get("started_at"),
            "ended_at": row.get("ended_at"),
            "latency_ms": row.get("latency_ms"),
            "sla_ms": row.get("sla_ms"),
            "valid_guardrail": bool(row.get("valid_guardrail", True)),
            "tokens_saved_est": int(row.get("tokens_saved_est", 0) or 0),
            "cost_usd": 0.0,
            "summary": _truncate_str(_sanitize_text(str(row.get("summary") or "")), MAX_LIST_SUMMARY_CHARS),
            "workspace": _truncate_str(_sanitize_text(str(row.get("workspace") or "")), MAX_LIST_SUMMARY_CHARS),
            "promotion": promo,
        }
        sanitized_runs.append(clean_row)

    running_count = sum(1 for r in all_rows if r.get("status") == "running")
    pass_count = sum(1 for r in all_rows if r.get("status") == "pass")
    blocked_count = sum(1 for r in all_rows if r.get("status") == "guardrail_blocked")
    error_count = sum(1 for r in all_rows if r.get("status") == "error")
    stale_count = sum(1 for r in all_rows if r.get("status") == "stale_interrupted")
    total_tokens_saved = sum(int(r.get("tokens_saved_est", 0) or 0) for r in all_rows)

    return {
        "runs": sanitized_runs,
        "summary": {
            "total_runs": len(all_rows),
            "filtered_runs": len(filtered),
            "returned_runs": len(sanitized_runs),
            "running_count": running_count,
            "pass_count": pass_count,
            "blocked_count": blocked_count,
            "error_count": error_count,
            "stale_count": stale_count,
            "total_tokens_saved_est": total_tokens_saved,
            "total_cost_usd": 0.0,
            "promotion_by_task": _sanitize_value(promo_by_task),
        },
    }


def get_local_run_detail(
    run_id: str,
    *,
    runs_log_path: Path | None = None,
    shadow_log_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any] | None:
    """
    Return sanitized detail for one `run_id`, with preview strings hard-capped
    at `<= 16000` chars per field. Returns `None` if `run_id` is not found.
    """
    target_id = str(run_id or "").strip()
    if not target_id:
        return None

    by_id, details_by_id = _merge_runs_and_shadow(
        runs_log_path=runs_log_path,
        shadow_log_path=shadow_log_path,
    )
    row = by_id.get(target_id)
    if row is None:
        return None

    promo_by_task = _load_promotion_stats(
        shadow_log_path=shadow_log_path,
        ledger_path=ledger_path,
    )
    raw_detail = details_by_id.get(target_id, {})
    task_name = str(row.get("task_type") or "unknown")
    promo = promo_by_task.get(
        task_name,
        {
            "count": 0,
            "guardrail_pass_rate": 0.0,
            "p95_ms": 0.0,
            "sla_ms": float(row.get("sla_ms", 2000.0) or 2000.0),
            "promotion_eligible": False,
            "promotion_target": PROMOTION_TARGET,
        },
    )

    detail_payload: dict[str, Any] = {
        "run_id": row["run_id"],
        "pid": row.get("pid"),
        "role": row.get("role"),
        "task_type": task_name,
        "model": row.get("model"),
        "engine": row.get("engine"),
        "mode": row.get("mode"),
        "status": row.get("status"),
        "started_at": row.get("started_at"),
        "ended_at": row.get("ended_at"),
        "latency_ms": row.get("latency_ms"),
        "sla_ms": row.get("sla_ms"),
        "valid_guardrail": bool(row.get("valid_guardrail", True)),
        "tokens_saved_est": int(row.get("tokens_saved_est", 0) or 0),
        "cost_usd": 0.0,
        "summary": row.get("summary") or "",
        "workspace": row.get("workspace") or "",
        "promotion": promo,
        "extra": raw_detail.get("extra", {}),
        "detail_ref": raw_detail.get("detail_ref", {}),
    }
    return _sanitize_value(detail_payload, max_str_chars=MAX_DETAIL_FIELD_CHARS)
