"""
Unified Shadow Evaluation Ledger & Local Metal Output Hygiene.

Provides:
- `strip_llm_fences_and_thoughts`: Strips `<thought>...</thought>`, `<|thought|>...`,
  and markdown code fences (` ```json ... ``` `) before JSON parsing or display.
- `is_external_models_drive_mounted`: Checks `/Volumes/Local Models` availability with
  `WBC_SIMULATE_UNPLUGGED=1` override for unplugged testing.
- `record_shadow_eval`: Appends flat JSONL telemetry to `specialist_router_shadow.jsonl`
  and atomically updates per-task promotion rollups in `router_telemetry_ledger.json`
  under an `fcntl.flock` lockfile.
"""

from __future__ import annotations

import datetime
import fcntl
import http.client
import json
import logging
import os
import re
import socket
import tempfile
import urllib.error
from pathlib import Path
from typing import Any

logger = logging.getLogger("whole_body_compute.local.shadow_ledger")

_THOUGHT_BLOCK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<thought>.*?</thought>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<\|thought\|>.*?<\|/thought\|>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<\|think\|>.*?<\|/think\|>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<\|channel\|?>thought.*?<\|?channel\|?>", re.IGNORECASE | re.DOTALL),
)

_UNCLOSED_THOUGHT_RE = re.compile(
    r"(?:<thought>|<think>|<\|thought\|>|<\|think\|>|<\|channel\|?>thought).*$",
    re.IGNORECASE | re.DOTALL,
)

_STRAY_THOUGHT_TAG_RE = re.compile(
    r"</?(?:thought|think)>|<\|/?(?:thought|think)\|>|<\|?channel\|?>",
    re.IGNORECASE,
)

_OUTER_FENCE_RE = re.compile(
    r"^\s*```(?:[A-Za-z0-9_+-]+)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL,
)


def strip_llm_fences_and_thoughts(text: str) -> str:
    """Strip LLM thinking tags (`<thought>`, `<|thought|>`) and outer markdown code fences."""
    if not isinstance(text, str) or not text:
        return ""
    cleaned = text
    for pattern in _THOUGHT_BLOCK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = _UNCLOSED_THOUGHT_RE.sub("", cleaned)
    cleaned = _STRAY_THOUGHT_TAG_RE.sub("", cleaned).strip()

    fence_match = _OUTER_FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return cleaned


def _chmod_0600(path: Path) -> None:
    """Enforce owner-only read/write permissions (0o600) best-effort."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def is_external_models_drive_mounted() -> bool:
    """Return True if the external `/Volumes/Local Models` drive is mounted and not simulated unplugged."""
    if os.environ.get("WBC_SIMULATE_UNPLUGGED") == "1":
        return False
    return Path("/Volumes/Local Models").exists()


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile from a pre-sorted list of floats."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return round(float(sorted_vals[0]), 2)
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    val = sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])
    return round(val, 2)


def _resolve_telemetry_paths(
    shadow_log_path: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve JSONL shadow log and atomic telemetry ledger paths safely."""
    env_shadow = os.environ.get("WBC_ROUTER_SHADOW_LOG", "").strip()
    env_ledger = os.environ.get("WBC_ROUTER_TELEMETRY_LEDGER", "").strip()

    if shadow_log_path is not None:
        resolved_shadow = Path(shadow_log_path).expanduser()
        shadow_overridden = True
    elif env_shadow:
        resolved_shadow = Path(env_shadow).expanduser()
        shadow_overridden = True
    else:
        resolved_shadow = (
            Path.home() / ".whole_body_compute" / "telemetry" / "specialist_router_shadow.jsonl"
        )
        shadow_overridden = False

    if env_ledger:
        resolved_ledger = Path(env_ledger).expanduser()
    elif shadow_overridden:
        resolved_ledger = resolved_shadow.parent / "router_telemetry_ledger.json"
    else:
        resolved_ledger = Path.home() / ".whole_body_compute" / "router_telemetry_ledger.json"

    return resolved_shadow, resolved_ledger


def _is_inside_live_wbc_dir(path: Path) -> bool:
    """Return True if `path` resolves inside the user's live `~/.whole_body_compute` directory."""
    live_root = (Path.home() / ".whole_body_compute").resolve(strict=False)
    try:
        resolved = path.expanduser().resolve(strict=False)
        return resolved.is_relative_to(live_root)
    except (OSError, ValueError):
        return False


def is_shadow_target_safe_for_test(shadow_log_path: Path | None = None) -> bool:
    """
    Return True if shadow logging is safe under pytest (i.e. both resolved shadow log
    and ledger paths live outside `~/.whole_body_compute`).
    """
    try:
        resolved_shadow, resolved_ledger = _resolve_telemetry_paths(shadow_log_path)
    except (OSError, RuntimeError, ValueError):
        return False
    return not (
        _is_inside_live_wbc_dir(resolved_shadow)
        or _is_inside_live_wbc_dir(resolved_ledger)
    )


_INFRA_STATUS_CODES: frozenset[int] = frozenset({401, 403, 502, 503})
_STRICT_AGREEMENT_TASKS: frozenset[str] = frozenset(
    {
        "tool_single_file_diff",
        "single_file_edit_offload",
        "tool_qa_log_triage",
    }
)
_INFRA_EXC_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "CloseError",
        "TimeoutException",
        "TransportError",
        "NetworkError",
    }
)


def is_infra_error(
    status_code: int | None = None,
    exc: BaseException | None = None,
) -> bool:
    """
    Return True if an HTTP status code or exception reflects infrastructure/transport failure
    (HTTP 401/403/502/503, connection refused, socket closed, or timeout before TTFB)
    rather than a local model quality/guardrail failure.
    """
    if status_code is not None:
        try:
            if int(status_code) in _INFRA_STATUS_CODES:
                return True
        except (TypeError, ValueError):
            pass

    if exc is None:
        return False

    if isinstance(exc, urllib.error.HTTPError):
        return getattr(exc, "code", None) in _INFRA_STATUS_CODES

    resp = getattr(exc, "response", None)
    if resp is not None:
        resp_code = getattr(resp, "status_code", None)
        if resp_code is not None:
            try:
                if int(resp_code) in _INFRA_STATUS_CODES:
                    return True
            except (TypeError, ValueError):
                pass

    if isinstance(
        exc,
        (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            OSError,
            http.client.HTTPException,
        ),
    ):
        return True

    return type(exc).__name__ in _INFRA_EXC_CLASS_NAMES


def record_shadow_eval(
    task_type: str,
    engine: str,
    model: str,
    latency_ms: float,
    valid_guardrail: bool,
    *,
    baseline_used: str = "shadow",
    tokens_saved_est: int = 0,
    unplugged_ready: bool = True,
    sla_ms: float = 2000.0,
    extra: dict[str, Any] | None = None,
    shadow_log_path: Path | None = None,
    agreed: bool | None = None,
    bounced: bool = False,
    raw_tool_tokens: int | None = None,
    local_digest_tokens: int | None = None,
    cloud_counterfactual_s: float | None = None,
    retry_penalty_s: float | None = None,
    shadow_only: bool = False,
    infra_skip: bool = False,
    outcome: str | None = None,
) -> dict[str, Any]:
    """
    Best-effort append to `specialist_router_shadow.jsonl` and atomic rollup update to
    `router_telemetry_ledger.json` under an `fcntl.flock` lock. Never raises on I/O errors.
    """
    if os.environ.get("PYTEST_CURRENT_TEST") and not is_shadow_target_safe_for_test(
        shadow_log_path
    ):
        return {}

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    lat_rounded = round(float(latency_ms), 2)
    sla_val = round(float(sla_ms), 2)
    tokens_est = int(tokens_saved_est or 0)
    extra_dict = extra if isinstance(extra, dict) else {}

    resolved_infra_skip = bool(
        infra_skip or outcome in ("infra_error", "infra_skip")
    )
    resolved_bounced = False if resolved_infra_skip else bool(bounced)
    if agreed is not None:
        resolved_agreed = bool(agreed)
    elif "agreed" in extra_dict and extra_dict["agreed"] is not None:
        resolved_agreed = bool(extra_dict["agreed"])
    else:
        resolved_agreed = bool(valid_guardrail) and not resolved_bounced

    if outcome:
        resolved_outcome = str(outcome)
    elif resolved_infra_skip:
        resolved_outcome = "infra_error"
    elif resolved_bounced:
        resolved_outcome = "bounced"
    elif valid_guardrail and resolved_agreed:
        resolved_outcome = "pass"
    else:
        resolved_outcome = "fail"

    if resolved_infra_skip or shadow_only:
        net_tokens_saved = 0
    elif resolved_bounced:
        digest_tok = int(local_digest_tokens if local_digest_tokens is not None else tokens_est)
        net_tokens_saved = -max(0, digest_tok)
    elif resolved_agreed:
        if raw_tool_tokens is not None or local_digest_tokens is not None:
            net_tokens_saved = int(raw_tool_tokens or 0) - int(local_digest_tokens or 0)
        else:
            net_tokens_saved = tokens_est
    else:
        net_tokens_saved = 0

    delta_t_net: float | None = None
    if not resolved_infra_skip and cloud_counterfactual_s is not None:
        retry_pen = float(retry_penalty_s or 0.0) if resolved_bounced else 0.0
        delta_t_net = round(
            float(cloud_counterfactual_s) - (float(latency_ms) / 1000.0) - retry_pen,
            6,
        )

    record: dict[str, Any] = {
        "timestamp": now_iso,
        "task_type": str(task_type),
        "engine": str(engine),
        "model": str(model),
        "latency_ms": lat_rounded,
        "valid_guardrail": bool(valid_guardrail),
        "baseline_used": str(baseline_used),
        "tokens_saved_est": tokens_est,
        "outcome": resolved_outcome,
        "infra_skip": resolved_infra_skip,
        "agreed": resolved_agreed,
        "bounced": resolved_bounced,
        "shadow_only": bool(shadow_only),
        "net_tokens_saved": net_tokens_saved,
    }
    if raw_tool_tokens is not None:
        record["raw_tool_tokens"] = int(raw_tool_tokens)
    if local_digest_tokens is not None:
        record["local_digest_tokens"] = int(local_digest_tokens)
    if cloud_counterfactual_s is not None:
        record["cloud_counterfactual_s"] = round(float(cloud_counterfactual_s), 6)
    if retry_penalty_s is not None:
        record["retry_penalty_s"] = round(float(retry_penalty_s), 6)
    if delta_t_net is not None:
        record["delta_t_net"] = delta_t_net

    for k, v in extra_dict.items():
        if k not in record and k != "_run_emitted":
            record[k] = v

    try:
        resolved_shadow, resolved_ledger = _resolve_telemetry_paths(shadow_log_path)
        resolved_shadow.parent.mkdir(parents=True, exist_ok=True)
        resolved_ledger.parent.mkdir(parents=True, exist_ok=True)

        lock_path = resolved_ledger.with_name(f"{resolved_ledger.name}.lock")
        with open(lock_path, "a+", encoding="utf-8") as lock_fh:
            _chmod_0600(lock_path)
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                with open(resolved_shadow, "a", encoding="utf-8") as log_fh:
                    _chmod_0600(resolved_shadow)
                    log_fh.write(json.dumps(record) + "\n")

                ledger: dict[str, Any] = {
                    "version": 2,
                    "updated_at": now_iso,
                    "total_shadow_evals": 0,
                    "tasks": {},
                }
                if resolved_ledger.is_file():
                    try:
                        loaded = json.loads(resolved_ledger.read_text(encoding="utf-8"))
                        if isinstance(loaded, dict):
                            ledger = loaded
                    except (OSError, json.JSONDecodeError, ValueError) as exc:
                        logger.debug("Starting fresh shadow ledger after read error: %s", exc)

                tasks = ledger.get("tasks")
                if not isinstance(tasks, dict):
                    tasks = {}
                    ledger["tasks"] = tasks

                entry = tasks.get(str(task_type))
                if not isinstance(entry, dict):
                    entry = {}

                count = int(entry.get("count", 0) or 0)
                valid_count = int(entry.get("valid_count", 0) or 0)
                agreed_count = int(
                    entry.get("agreed_count", valid_count)
                    if "agreed_count" not in entry
                    else (entry.get("agreed_count", 0) or 0)
                )
                bounce_count = int(entry.get("bounce_count", 0) or 0)
                infra_error_count = int(entry.get("infra_error_count", 0) or 0)

                recent_raw = entry.get("latencies_ms_recent")
                recent: list[float] = (
                    [float(x) for x in recent_raw if isinstance(x, (int, float))]
                    if isinstance(recent_raw, list)
                    else []
                )

                engines = entry.get("engines")
                engines_map: dict[str, int] = (
                    {str(k): int(v or 0) for k, v in engines.items()}
                    if isinstance(engines, dict)
                    else {}
                )

                models = entry.get("models")
                models_map: dict[str, int] = (
                    {str(k): int(v or 0) for k, v in models.items()}
                    if isinstance(models, dict)
                    else {}
                )

                tokens_total = int(entry.get("tokens_saved_est_total", 0) or 0)
                net_tokens_total = int(entry.get("net_tokens_saved_total", 0) or 0)
                net_time_sum_s = float(entry.get("net_time_saved_sum_s", 0.0) or 0.0)
                net_time_count = int(entry.get("net_time_saved_count", 0) or 0)

                if resolved_infra_skip:
                    infra_error_count += 1
                else:
                    count += 1
                    if valid_guardrail:
                        valid_count += 1
                    if resolved_agreed:
                        agreed_count += 1
                    if resolved_bounced:
                        bounce_count += 1
                    recent.append(lat_rounded)
                    recent = recent[-100:]
                    engines_map[str(engine)] = int(engines_map.get(str(engine), 0) or 0) + 1
                    models_map[str(model)] = int(models_map.get(str(model), 0) or 0) + 1
                    tokens_total += tokens_est
                    net_tokens_total += net_tokens_saved
                    if delta_t_net is not None:
                        net_time_sum_s += delta_t_net
                        net_time_count += 1

                pass_rate = round(valid_count / count, 4) if count > 0 else 0.0
                agreement_rate = round(agreed_count / count, 4) if count > 0 else 0.0
                bounce_rate = round(bounce_count / count, 4) if count > 0 else 0.0

                sorted_lat = sorted(recent)
                p50_ms = _percentile(sorted_lat, 50.0)
                p95_ms = _percentile(sorted_lat, 95.0)
                within_sla_rate = (
                    round(sum(1 for x in recent if x <= sla_val) / len(recent), 4)
                    if recent
                    else 0.0
                )
                mean_net_time_saved_s = (
                    round(net_time_sum_s / net_time_count, 4)
                    if net_time_count > 0
                    else 0.0
                )
                winning_engine = (
                    max(engines_map.items(), key=lambda kv: (kv[1], kv[0]))[0]
                    if engines_map
                    else ""
                )

                min_agreement = (
                    0.98 if str(task_type) in _STRICT_AGREEMENT_TASKS else 0.95
                )
                sla_ok = (p95_ms <= sla_val) if sla_val > 0 else True
                promotion_eligible = bool(
                    count >= 20
                    and agreement_rate >= min_agreement
                    and bounce_rate <= 0.05
                    and sla_ok
                    and mean_net_time_saved_s > 0.0
                )

                tasks[str(task_type)] = {
                    "count": count,
                    "valid_count": valid_count,
                    "agreed_count": agreed_count,
                    "bounce_count": bounce_count,
                    "infra_error_count": infra_error_count,
                    "guardrail_pass_rate": pass_rate,
                    "agreement_rate": agreement_rate,
                    "bounce_rate": bounce_rate,
                    "latencies_ms_recent": recent,
                    "p50_ms": p50_ms,
                    "p95_ms": p95_ms,
                    "sla_ms": sla_val,
                    "within_sla_rate": within_sla_rate,
                    "tokens_saved_est_total": tokens_total,
                    "net_tokens_saved_total": net_tokens_total,
                    "net_time_saved_sum_s": round(net_time_sum_s, 6),
                    "net_time_saved_count": net_time_count,
                    "mean_net_time_saved_s": mean_net_time_saved_s,
                    "engines": engines_map,
                    "models": models_map,
                    "winning_engine": winning_engine,
                    "unplugged_ready": bool(unplugged_ready),
                    "promotion_eligible": promotion_eligible,
                    "last_eval_at": now_iso,
                }

                ledger["version"] = 2
                ledger["updated_at"] = now_iso
                ledger["total_shadow_evals"] = sum(
                    int(t.get("count", 0) or 0)
                    for t in tasks.values()
                    if isinstance(t, dict)
                )

                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(resolved_ledger.parent),
                    prefix=f".{resolved_ledger.name}.",
                    suffix=".tmp",
                )
                try:
                    _chmod_0600(Path(tmp_path))
                    with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_fh:
                        json.dump(ledger, tmp_fh, indent=2, sort_keys=False)
                        tmp_fh.write("\n")
                    os.replace(tmp_path, resolved_ledger)
                    _chmod_0600(resolved_ledger)
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("Shadow ledger update skipped: %s", exc)

    if not extra_dict.get("_run_emitted"):
        try:
            from whole_body_compute.local.run_events import (
                record_local_run_end,
                record_local_run_start,
            )

            passed_run_id = str(extra_dict["run_id"]) if extra_dict.get("run_id") else None
            emitted_id = record_local_run_start(
                role=f"Shadow: {task_type}",
                task_type=str(task_type),
                model=str(model),
                engine=str(engine),
                mode=("live" if baseline_used == "live_metal" else "shadow"),
                run_id=passed_run_id,
            )
            if emitted_id:
                run_status = (
                    "infra_error"
                    if resolved_infra_skip
                    else ("pass" if valid_guardrail else "guardrail_blocked")
                )
                record_local_run_end(
                    emitted_id,
                    status=run_status,
                    latency_ms=latency_ms,
                    valid_guardrail=bool(valid_guardrail),
                    tokens_saved_est=tokens_est,
                    sla_ms=sla_val,
                    detail_ref=extra if isinstance(extra, dict) else None,
                )
        except Exception:  # noqa: BLE001, S110
            pass

    return record
