"""
Insight Post Seed Generator (`insight_post_seed`).

Extracts structured, secret-scrubbed social post seeds (title, hook, problem, fix, metric)
from a session's `StateCapsule` / `CognitiveMetrics` using local Metal endpoints
(`:9380` -> `:9379`), and stamps deterministic provenance links (`conversation://` URI,
`transcript.jsonl` file link, `git` branch@commit, and optional Slack thread URL)
into `post_seeds.jsonl` and `.agents/POST_SEEDS.md`.
"""

from __future__ import annotations

import datetime
import fcntl
import json
import logging
import os
import re
import socket
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from whole_body_compute.local.shadow_ledger import (
    _chmod_0600,
    _is_inside_live_wbc_dir,
    is_external_models_drive_mounted,
    record_shadow_eval,
    strip_llm_fences_and_thoughts,
)
from whole_body_compute.procedural_telemetry import get_wbc_data_dir
from whole_body_compute.redaction import redact_secrets

logger = logging.getLogger("whole_body_compute.local.post_seeds")

DEFAULT_SLA_MS = 2500.0
_SECRET_LEAK_RE = re.compile(
    r"(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{12,}|AIza[0-9A-Za-z_-]{25,})"
)
_REFUSAL_PHRASES = (
    "cannot answer this question",
    "do not have access",
    "i'm sorry",
    "as an ai",
    "no information provided",
)


def _is_port_open(host: str, port: int, timeout: float = 0.025) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _resolve_transcript_path(conv_id: str) -> Path | None:
    cid = str(conv_id or "").strip()
    if not cid:
        return None
    candidate = (
        Path.home()
        / ".gemini"
        / "antigravity"
        / "brain"
        / cid
        / ".system_generated"
        / "logs"
        / "transcript.jsonl"
    )
    return candidate


def _resolve_seeds_jsonl_path(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(os.path.expanduser(str(explicit_path))).resolve()
    env_path = os.environ.get("WBC_POST_SEEDS_LOG", "").strip()
    if env_path:
        return Path(os.path.expanduser(env_path)).resolve()
    return get_wbc_data_dir() / "post_seeds.jsonl"


def _resolve_seeds_md_path(
    workspace_root: str | Path | None = None,
    explicit_md_path: Path | None = None,
) -> Path | None:
    if explicit_md_path is not None:
        return Path(os.path.expanduser(str(explicit_md_path))).resolve()
    env_md = os.environ.get("WBC_POST_SEEDS_MD", "").strip()
    if env_md:
        return Path(os.path.expanduser(env_md)).resolve()
    if workspace_root:
        return Path(os.path.expanduser(str(workspace_root))).resolve() / ".agents" / "POST_SEEDS.md"
    return None


def validate_post_seed(raw_text: str) -> tuple[dict[str, str] | None, bool]:
    """
    Parse and validate a local-model JSON post seed.
    Returns `(parsed_dict_or_None, valid_guardrail)`.
    """
    cleaned = redact_secrets(strip_llm_fences_and_thoughts(str(raw_text or ""))).strip()
    if not cleaned:
        return None, False

    # Extract outer JSON object if wrapped in prose
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None, False

    try:
        obj = json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, False

    if not isinstance(obj, dict):
        return None, False

    title = redact_secrets(str(obj.get("title") or "")).strip()
    hook = redact_secrets(str(obj.get("hook") or "")).strip()
    problem = redact_secrets(str(obj.get("problem") or "")).strip()
    fix = redact_secrets(str(obj.get("fix") or "")).strip()
    metric = redact_secrets(str(obj.get("metric") or "")).strip()

    if len(title) < 8 or len(title) > 120:
        return None, False
    if len(hook) < 20 or len(hook) > 280:
        return None, False
    if len(problem) < 15 or len(problem) > 320:
        return None, False
    if len(fix) < 15 or len(fix) > 320:
        return None, False
    if len(metric) < 3 or len(metric) > 160:
        return None, False

    combined_low = f"{title} {hook} {problem} {fix} {metric}".lower()
    if any(phrase in combined_low for phrase in _REFUSAL_PHRASES):
        return None, False
    if any(
        val.lower().strip(" .!?:;") in ("none", "n/a", "null", "unknown")
        for val in (title, hook, problem, fix, metric)
    ):
        return None, False
    if _SECRET_LEAK_RE.search(f"{title} {hook} {problem} {fix} {metric}"):
        return None, False

    return {
        "title": title,
        "hook": hook,
        "problem": problem,
        "fix": fix,
        "metric": metric,
    }, True


def build_provenance(
    *,
    conv_id: str = "",
    short_id: str = "",
    capsule_id: str = "",
    workspace_root: str | Path | None = None,
    git_state: dict[str, Any] | None = None,
    slack_thread_url: str | None = None,
    parent_conv_id: str | None = None,
) -> dict[str, Any]:
    """Deterministically construct clickable provenance links for a post seed."""
    cid = str(conv_id or "").strip()
    sid = str(short_id or (cid[:8] if cid else "")).strip()
    transcript_p = _resolve_transcript_path(cid) if cid else None
    gs = git_state if isinstance(git_state, dict) else {}
    branch = str(gs.get("branch") or "").strip()
    head = str(gs.get("head") or gs.get("commit") or "").strip()
    git_ref = f"{branch}@{head[:10]}" if branch and head else (branch or head)

    prov: dict[str, Any] = {
        "conv_id": cid,
        "short_id": sid,
        "conv_uri": f"conversation://{cid}" if cid else "",
        "transcript_path": str(transcript_p) if transcript_p else "",
        "capsule_id": str(capsule_id or "").strip(),
        "workspace_root": str(workspace_root or "").strip(),
        "git_ref": git_ref,
        "slack_thread_url": str(slack_thread_url or "").strip(),
        "parent_conv_uri": f"conversation://{parent_conv_id}" if parent_conv_id else "",
    }
    return prov


def format_post_seed_markdown(seed: dict[str, Any]) -> str:
    """Render a post seed + clickable provenance links as a Markdown card."""
    ts = str(seed.get("created_at") or "")[:10] or datetime.date.today().isoformat()
    title = str(seed.get("title") or "Untitled Engineering Insight")
    hook = str(seed.get("hook") or "")
    problem = str(seed.get("problem") or "")
    fix = str(seed.get("fix") or "")
    metric = str(seed.get("metric") or "")
    prov = seed.get("provenance") if isinstance(seed.get("provenance"), dict) else {}

    ctx_parts: list[str] = []
    conv_uri = str(prov.get("conv_uri") or "")
    short_id = str(prov.get("short_id") or "thread")
    if conv_uri:
        ctx_parts.append(f"[Open Convo ({short_id})]({conv_uri})")
    transcript_path = str(prov.get("transcript_path") or "")
    if transcript_path:
        ctx_parts.append(f"[Transcript](file://{transcript_path})")
    parent_uri = str(prov.get("parent_conv_uri") or "")
    if parent_uri:
        p_short = parent_uri.replace("conversation://", "")[:8]
        ctx_parts.append(f"[Parent ({p_short})]({parent_uri})")
    git_ref = str(prov.get("git_ref") or "")
    if git_ref:
        ctx_parts.append(f"`{git_ref}`")
    slack_url = str(prov.get("slack_thread_url") or "")
    if slack_url:
        ctx_parts.append(f"[Slack]({slack_url})")

    ctx_line = " · ".join(ctx_parts) if ctx_parts else "local-session"
    return (
        f"### [{ts}] {title}\n"
        f"- **Hook:** {hook}\n"
        f"- **Problem:** {problem}\n"
        f"- **Fix:** {fix}\n"
        f"- **Metric:** `{metric}`\n"
        f"- **Context:** {ctx_line}\n"
    )


def _persist_post_seed(
    record: dict[str, Any],
    *,
    seeds_jsonl_path: Path | None = None,
    seeds_md_path: Path | None = None,
) -> None:
    resolved_jsonl = _resolve_seeds_jsonl_path(seeds_jsonl_path)
    in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    if in_pytest and _is_inside_live_wbc_dir(resolved_jsonl):
        return

    resolved_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lock_path = resolved_jsonl.with_name(f"{resolved_jsonl.name}.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_fh:
        _chmod_0600(lock_path)
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            with open(resolved_jsonl, "a", encoding="utf-8") as out_fh:
                _chmod_0600(resolved_jsonl)
                out_fh.write(json.dumps(record) + "\n")
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    if seeds_md_path is not None and not (in_pytest and _is_inside_live_wbc_dir(seeds_md_path)):
        seeds_md_path.parent.mkdir(parents=True, exist_ok=True)
        card = format_post_seed_markdown(record)
        header = "# Post Seeds (Auto-Captured from `body flush`)\n\n"
        existing = seeds_md_path.read_text(encoding="utf-8") if seeds_md_path.is_file() else header
        if not existing.startswith("# "):
            existing = header + existing
        body_part = existing[len(header) :] if existing.startswith(header) else existing
        seeds_md_path.write_text(f"{header}{card}\n{body_part.lstrip()}", encoding="utf-8")


def generate_insight_post_seed(
    *,
    goal: str,
    milestone: str,
    next_step: str = "",
    action_summaries: list[str] | None = None,
    touched_files: list[str] | None = None,
    metrics_summary: str = "",
    provenance: dict[str, Any] | None = None,
    seeds_jsonl_path: Path | None = None,
    seeds_md_path: Path | None = None,
    shadow_log_path: Path | None = None,
    record_shadow: bool | None = None,
) -> dict[str, Any] | None:
    """
    Query local Metal (`:9380` -> `:9379`) to synthesize a 5-field `insight_post_seed`
    from session state, validate guardrails, record shadow telemetry, and persist
    with deterministic provenance links.
    """
    actions_txt = "; ".join(str(x) for x in (action_summaries or [])[:6])
    files_txt = ", ".join(str(x) for x in (touched_files or [])[:8])
    raw_context = (
        f"Goal: {goal}\n"
        f"Milestone: {milestone}\n"
        f"Next: {next_step}\n"
        f"Actions: {actions_txt}\n"
        f"Files: {files_txt}\n"
        f"Metrics: {metrics_summary}"
    )
    scrubbed_context = redact_secrets(raw_context)[:1800]

    prompt = (
        "Extract one concise practitioner engineering lesson from this session for a future X/LinkedIn post.\n"
        "Return ONLY valid JSON with exact keys: title, hook, problem, fix, metric.\n"
        "Rules:\n"
        "- title: <= 90 chars, specific technical takeaway\n"
        "- hook: <= 240 chars, punchy practitioner opener (no hashtags, no emojis)\n"
        "- problem: <= 220 chars, concrete failure or bottleneck\n"
        "- fix: <= 220 chars, concrete change made\n"
        "- metric: <= 100 chars, real number or state change from the input\n\n"
        f"Session:\n{scrubbed_context}"
    )

    model_9380 = (
        "gemma4-26b"
        if is_external_models_drive_mounted()
        else "localbench-specialist-router"
    )
    candidates: list[tuple[int, str, str]] = [
        (9380, model_9380, "specialist_router_9380"),
        (9379, "gemma4-2b", "litert_metal_9379"),
        (9379, "body-coder-mlx", "mlx_metal_9379"),
    ]

    parsed: dict[str, str] | None = None
    valid_guardrail = False
    used_engine = "specialist_router_9380"
    used_model = model_9380
    elapsed_ms = 0.0
    attempted = False

    for port, cand_model, engine_name in candidates:
        if not _is_port_open("127.0.0.1", port, timeout=0.025):
            continue
        attempted = True
        used_engine = engine_name
        used_model = cand_model
        t0 = time.perf_counter()
        try:
            payload = json.dumps(
                {
                    "model": cand_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 220,
                    "temperature": 0.2,
                    "stream": False,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.6) as resp:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status != 200:
                    continue
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
            choices = body.get("choices") or []
            raw_ans = str((choices[0].get("message") or {}).get("content") or "") if choices else ""
            used_model = str(body.get("model") or cand_model)
            parsed, valid_guardrail = validate_post_seed(raw_ans)
            if valid_guardrail:
                break
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug("insight_post_seed call on :%s (%s) failed: %s", port, cand_model, exc)
            continue

    if not attempted:
        return None

    shadow_on = (
        bool(record_shadow)
        if record_shadow is not None
        else (os.environ.get("WBC_ROUTER_SHADOW") == "1")
    )
    prov = provenance if isinstance(provenance, dict) else {}
    if shadow_on:
        record_shadow_eval(
            task_type="insight_post_seed",
            engine=used_engine,
            model=used_model,
            latency_ms=elapsed_ms,
            valid_guardrail=valid_guardrail,
            baseline_used="shadow_only",
            sla_ms=DEFAULT_SLA_MS,
            tokens_saved_est=max(0, len(scrubbed_context) // 4),
            extra={
                "conv_id": prov.get("conv_id", ""),
                "hook_preview": (parsed or {}).get("hook", "")[:120],
            },
            shadow_log_path=shadow_log_path,
        )

    if not valid_guardrail or parsed is None:
        return None

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record: dict[str, Any] = {
        "created_at": now_iso,
        "engine": used_engine,
        "model": used_model,
        "latency_ms": round(elapsed_ms, 2),
        **parsed,
        "provenance": prov,
    }
    try:
        _persist_post_seed(
            record,
            seeds_jsonl_path=seeds_jsonl_path,
            seeds_md_path=seeds_md_path,
        )
    except OSError as exc:
        logger.debug("Post seed persistence skipped: %s", exc)
    return record


def dispatch_post_seed_from_flush(
    *,
    capsule: Any,
    metrics: Any = None,
    workspace_root: Path | None = None,
    sync: bool | None = None,
) -> bool:
    """
    Schedule (or run synchronously if `WBC_SHADOW_SYNC=1`) `insight_post_seed`
    extraction from a `body flush` capsule. Never blocks or raises.
    """
    if os.environ.get("WBC_POST_SEEDS") == "0":
        return False
    in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    sync_mode = (os.environ.get("WBC_SHADOW_SYNC") == "1") if sync is None else bool(sync)
    if in_pytest and not sync_mode and os.environ.get("WBC_POST_SEEDS") != "1":
        return False

    ws_root = workspace_root or getattr(capsule, "workspace_root", None)
    git_state = getattr(capsule, "git_state", None)
    cap_metrics = getattr(capsule, "metrics", None) or {}
    parent_cid = None
    resump = str(getattr(capsule, "resumption_prompt", "") or "")
    m_parent = re.search(r"Continuing from (?:conversation://)?([0-9a-fA-F-]{36})", resump)
    if m_parent:
        parent_cid = m_parent.group(1)

    prov = build_provenance(
        conv_id=str(getattr(capsule, "conv_id", "") or ""),
        short_id=str(getattr(capsule, "short_id", "") or ""),
        capsule_id=str(getattr(capsule, "capsule_id", "") or ""),
        workspace_root=ws_root,
        git_state=git_state if isinstance(git_state, dict) else None,
        slack_thread_url=getattr(metrics, "slack_thread_url", None) if metrics else None,
        parent_conv_id=parent_cid,
    )
    csi_val = getattr(metrics, "csi", cap_metrics.get("csi", 0.0)) if metrics else cap_metrics.get("csi", 0.0)
    steps_val = getattr(metrics, "steps", cap_metrics.get("steps", 0)) if metrics else cap_metrics.get("steps", 0)
    kb_val = getattr(metrics, "full_size_kb", cap_metrics.get("full_size_kb", 0.0)) if metrics else cap_metrics.get("full_size_kb", 0.0)
    metrics_summary = f"CSI={csi_val}, steps={steps_val}, transcript_kb={kb_val}"

    seeds_md_path = _resolve_seeds_md_path(workspace_root=ws_root)

    def _run() -> None:
        try:
            generate_insight_post_seed(
                goal=str(getattr(capsule, "goal", "") or ""),
                milestone=str(getattr(capsule, "milestone", "") or ""),
                next_step=str(getattr(capsule, "next_step", "") or ""),
                action_summaries=list(getattr(capsule, "action_summaries", []) or []),
                touched_files=list(getattr(capsule, "touched_files", []) or []),
                metrics_summary=metrics_summary,
                provenance=prov,
                seeds_md_path=seeds_md_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Background insight_post_seed failed: %s", exc)

    if sync_mode:
        _run()
    else:
        threading.Thread(target=_run, name="wbc-post-seed", daemon=True).start()
    return True
