"""
Whole Body Compute FinOps API Proxy (`body proxy`) HTTP/SSE Gateway.

Binds `127.0.0.1:9381` using `aiohttp.web` + `httpx.AsyncClient`:
- Deterministic Anthropic `cache_control` injection (up to 4 breakpoints)
- OpenAI `stream_options.include_usage` injection
- Pre-egress `redact_secrets` scrubbing over message/system text
- Opt-in `<0.5ms` prose-only em-dash & invisible watermark hygiene
- Local model slug fast-path (`gemma*`, `qwen*`, `llama*`, `voicefi-*`, `mlx/*`, `ollama/*`)
- Community 250/day fail-open metering (never returns 402/429)
- Per-client & per-workflow FinOps telemetry in `proxy_finops_ledger`
"""

from __future__ import annotations

import copy
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import httpx
from aiohttp import web

from whole_body_compute.license import LicenseManager
from whole_body_compute.organs.immune_hygiene import (
    auto_heal_em_dash,
    auto_heal_invisible_marks,
)
from whole_body_compute.proxy import ProxyConfig
from whole_body_compute.proxy import chore_router as _chore_router_mod
from whole_body_compute.proxy.cache_injector import (
    inject_anthropic_ephemeral_cache,
    prepare_openai_request,
)
from whole_body_compute.proxy.pricing import (
    compute_proxy_call_cost,
    is_local_model_slug,
)
from whole_body_compute.proxy.sse_stream import SSEFrameScanner
from whole_body_compute.redaction import redact_secrets
from whole_body_compute.shield import get_proxy_client_summary, record_proxy_call

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
    }
)

_FALLBACK_LOCAL_ENDPOINTS: Tuple[str, ...] = (
    "http://127.0.0.1:9379",
    "http://127.0.0.1:11434",
)


def _is_chore_router_available() -> bool:
    return not bool(getattr(_chore_router_mod, "__wbc_pro_stub__", False))


def _resolve_default_local_endpoints() -> Tuple[str, ...]:
    if _is_chore_router_available():
        eps = getattr(_chore_router_mod, "DEFAULT_LOCAL_ENDPOINTS", None)
        if isinstance(eps, (tuple, list)) and eps:
            return tuple(str(e) for e in eps)
    return _FALLBACK_LOCAL_ENDPOINTS


def _safe_load_clients_config() -> Dict[str, Dict[str, Any]]:
    if not _is_chore_router_available():
        return {}
    try:
        loaded = _chore_router_mod.load_clients_config()
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


_DEFAULT_LOCAL_ENDPOINTS: Tuple[str, ...] = _resolve_default_local_endpoints()


def _check_local_specialist_warm() -> bool:
    """Check whether local MLX (:9379) or Ollama (:11434) is warm without probing :9380/:9381."""
    if os.environ.get("WBC_TESTING") == "1" or os.environ.get("PYTEST_CURRENT_TEST"):
        return os.environ.get("WBC_MOCK_SCOUT_WARM") == "1"
    for url in ("http://127.0.0.1:9379/health", "http://127.0.0.1:11434/api/tags"):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=0.05) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


def _filter_upstream_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    """Preserve auth & API version headers verbatim while stripping hop-by-hop headers."""
    out: Dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in _HOP_BY_HOP_HEADERS:
            continue
        out[k] = v
    return out


def _redact_text_value(text: str) -> Tuple[str, int]:
    if not text:
        return text, 0
    redacted = redact_secrets(text)
    if redacted != text:
        return redacted, 1
    return text, 0


def _redact_request_payload(body: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Run pre-egress `redact_secrets` over `system` and `messages[*].content` text
    without mutating the caller's dict or touching auth headers.
    """
    working = copy.deepcopy(body)
    secrets_blocked = 0

    system = working.get("system")
    if isinstance(system, str):
        new_sys, cnt = _redact_text_value(system)
        working["system"] = new_sys
        secrets_blocked += cnt
    elif isinstance(system, list):
        for blk in system:
            if isinstance(blk, dict) and blk.get("type") == "text" and isinstance(blk.get("text"), str):
                new_t, cnt = _redact_text_value(blk["text"])
                blk["text"] = new_t
                secrets_blocked += cnt

    messages = working.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                new_c, cnt = _redact_text_value(content)
                msg["content"] = new_c
                secrets_blocked += cnt
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                        if blk.get("type") in ("thinking", "redacted_thinking"):
                            continue
                        new_t, cnt = _redact_text_value(blk["text"])
                        blk["text"] = new_t
                        secrets_blocked += cnt

    return working, secrets_blocked


def _heal_prose_string(text: str) -> Tuple[str, int]:
    if not text:
        return text, 0
    healed, c1 = auto_heal_em_dash(text)
    healed, c2 = auto_heal_invisible_marks(healed)
    return healed, c1 + c2


def _heal_non_streaming_response(data: Dict[str, Any], provider: str) -> int:
    """Apply opt-in em-dash & invisible mark hygiene to non-streaming prose responses."""
    healed_count = 0
    if provider == "openai":
        choices = data.get("choices")
        if isinstance(choices, list):
            for ch in choices:
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str) and not msg.get("tool_calls"):
                    new_t, cnt = _heal_prose_string(msg["content"])
                    if cnt > 0:
                        msg["content"] = new_t
                        healed_count += cnt
        return healed_count

    content = data.get("content")
    if isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text" and isinstance(blk.get("text"), str):
                new_t, cnt = _heal_prose_string(blk["text"])
                if cnt > 0:
                    blk["text"] = new_t
                    healed_count += cnt
    return healed_count


def _extract_non_streaming_usage(data: Dict[str, Any], provider: str) -> Tuple[int, int, int, int]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0, 0
    if provider == "openai":
        prompt_tok = int(usage.get("prompt_tokens") or 0)
        details = usage.get("prompt_tokens_details")
        cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
        return max(0, prompt_tok - cached), 0, cached, int(usage.get("completion_tokens") or 0)

    return (
        int(usage.get("input_tokens") or 0),
        int(usage.get("cache_creation_input_tokens") or 0),
        int(usage.get("cache_read_input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
    )


def _estimate_prefix_tokens_for_missed_cap(body: Dict[str, Any]) -> int:
    """Conservative token estimate of cacheable prefix when Community 250/day cap is exceeded."""
    total_bytes = 0
    for key in ("tools", "system"):
        val = body.get(key)
        if val:
            total_bytes += len(json.dumps(val, ensure_ascii=False).encode("utf-8"))
    msgs = body.get("messages")
    if isinstance(msgs, list) and len(msgs) >= 2:
        total_bytes += len(json.dumps(msgs[:-1], ensure_ascii=False).encode("utf-8"))
    return total_bytes // 4


def _resolve_client_and_workflow(
    request: web.Request,
    clients_cfg: Dict[str, Dict[str, Any]],
) -> Tuple[str, str, Dict[str, Any]]:
    client_slug = (
        request.match_info.get("client_slug")
        or request.headers.get("x-wbc-client")
        or "untagged"
    ).strip() or "untagged"
    workflow_slug = (
        request.match_info.get("workflow_slug")
        or request.headers.get("x-wbc-workflow")
        or ""
    ).strip()
    return client_slug, workflow_slug, clients_cfg.get(client_slug, {})


def _is_anti_slop_enabled(
    config: ProxyConfig,
    request: web.Request,
    client_entry: Dict[str, Any],
) -> bool:
    hdr = request.headers.get("x-wbc-hygiene", "").strip().lower()
    if hdr in ("em-dash", "all", "1", "true", "on"):
        return True
    if hdr in ("off", "0", "false", "none"):
        return False
    if client_entry.get("anti_slop") is True:
        return True
    return bool(config.anti_slop)


def _check_community_cache_allowance(
    config: ProxyConfig,
    *,
    db_path: Optional[Path | str],
    license_mgr: LicenseManager,
) -> Tuple[bool, int]:
    """
    Fail-open Community 250/day metering.
    Returns `(cache_injection_allowed, cached_calls_today)`. Never raises or blocks.
    """
    try:
        summary = get_proxy_client_summary(window="today", db_path=db_path)
        cached_today = int(summary.get("cached_calls_today") or 0)
    except Exception:
        cached_today = 0

    try:
        tier = license_mgr.get_active_tier()
    except Exception:
        tier = "community"

    if tier in ("pro", "team", "enterprise"):
        return True, cached_today

    return cached_today < config.community_daily_cache_cap, cached_today


def _fallback_anthropic_to_openai_payload(body: Dict[str, Any], model: str) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        sys_parts = [
            str(b.get("text", ""))
            for b in system
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
        ]
        if sys_parts:
            messages.append({"role": "system", "content": "\n".join(sys_parts)})
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts = [
                str(b.get("text", ""))
                for b in content
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
            ]
            messages.append({"role": role, "content": "\n".join(parts)})
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if body.get("max_tokens") is not None:
        payload["max_tokens"] = int(body["max_tokens"])
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    return payload


def _fallback_openai_to_anthropic_response(oai_resp: Dict[str, Any], *, model: str) -> Dict[str, Any]:
    choices = oai_resp.get("choices") or []
    text = ""
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            text = str(msg.get("content") or "")
    usage = oai_resp.get("usage") or {}
    return {
        "id": f"msg_wbc_local_{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


async def _forward_local_fast_path(
    *,
    http_client: httpx.AsyncClient,
    local_endpoints: Iterable[str],
    provider: str,
    body: Dict[str, Any],
    model: str,
) -> Tuple[Dict[str, Any], int]:
    """Forward local model slug (`gemma*`, `qwen*`, etc.) directly to :9379 / :11434."""
    if provider == "anthropic":
        if _is_chore_router_available():
            oai_payload = _chore_router_mod._anthropic_to_openai_payload(body, model)
        else:
            oai_payload = _fallback_anthropic_to_openai_payload(body, model)
    else:
        oai_payload = copy.deepcopy(body)
        oai_payload["stream"] = False

    for base_url in local_endpoints:
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        try:
            resp = await http_client.post(url, json=oai_payload, timeout=30.0)
            if resp.status_code != 200:
                continue
            oai_data = resp.json()
            if provider == "anthropic":
                if _is_chore_router_available():
                    return _chore_router_mod._openai_to_anthropic_response(oai_data, model=model), 200
                return _fallback_openai_to_anthropic_response(oai_data, model=model), 200
            return oai_data, 200
        except Exception:
            continue

    return {"error": {"type": "local_unavailable", "message": "No local Metal/Ollama endpoint reachable"}}, 503


_CFG_KEY = web.AppKey("config", ProxyConfig)
_DB_PATH_KEY = web.AppKey("db_path", Optional[Any])
_LOCAL_ENDPOINTS_KEY = web.AppKey("local_endpoints", Tuple[str, ...])
_LICENSE_MGR_KEY = web.AppKey("license_manager", LicenseManager)
_CLIENTS_CFG_KEY = web.AppKey("clients_config", Dict[str, Dict[str, Any]])
_OWNS_CLIENT_KEY = web.AppKey("owns_http_client", bool)
_HTTP_CLIENT_KEY = web.AppKey("http_client", Optional[httpx.AsyncClient])


def create_proxy_app(
    config: Optional[ProxyConfig] = None,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
    db_path: Optional[Path | str] = None,
    clients_config: Optional[Dict[str, Dict[str, Any]]] = None,
    local_endpoints: Tuple[str, ...] = _DEFAULT_LOCAL_ENDPOINTS,
    license_manager: Optional[LicenseManager] = None,
) -> web.Application:
    """Create and configure the `aiohttp.web.Application` for `body proxy`."""
    cfg = config or ProxyConfig()
    app = web.Application()
    app[_CFG_KEY] = cfg
    app[_DB_PATH_KEY] = db_path
    app[_LOCAL_ENDPOINTS_KEY] = local_endpoints
    app[_LICENSE_MGR_KEY] = license_manager or LicenseManager()
    app[_CLIENTS_CFG_KEY] = clients_config if clients_config is not None else _safe_load_clients_config()
    app[_OWNS_CLIENT_KEY] = http_client is None
    app[_HTTP_CLIENT_KEY] = http_client

    async def _on_startup(a: web.Application) -> None:
        if a[_HTTP_CLIENT_KEY] is None:
            a[_HTTP_CLIENT_KEY] = httpx.AsyncClient(timeout=120.0)

    async def _on_cleanup(a: web.Application) -> None:
        if a[_OWNS_CLIENT_KEY] and a[_HTTP_CLIENT_KEY] is not None:
            await a[_HTTP_CLIENT_KEY].aclose()
            a[_HTTP_CLIENT_KEY] = None

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    async def handle_health(request: web.Request) -> web.Response:
        try:
            summary = get_proxy_client_summary(window="today", db_path=app[_DB_PATH_KEY])
            cached_today = int(summary.get("cached_calls_today") or 0)
            saved_today = float(summary.get("saved_today_usd") or 0.0)
        except Exception:
            cached_today = 0
            saved_today = 0.0
        return web.json_response(
            {
                "status": "ok",
                "service": "whole-body-compute-proxy",
                "cached_calls_today": cached_today,
                "saved_today_usd": round(saved_today, 4),
                "local_specialist_warm": _check_local_specialist_warm(),
            }
        )

    async def handle_passthrough(request: web.Request) -> web.Response:
        client = app[_HTTP_CLIENT_KEY]
        assert client is not None
        upstream_base = cfg.anthropic_upstream.rstrip("/")
        target_url = f"{upstream_base}{request.rel_url}"
        fwd_headers = _filter_upstream_headers(request.headers)
        raw_body = await request.read()
        resp = await client.request(
            request.method,
            target_url,
            content=raw_body if raw_body else None,
            headers=fwd_headers,
            timeout=60.0,
        )
        return web.Response(
            body=resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("content-type", "application/json").split(";")[0],
        )

    async def _handle_completion(request: web.Request, *, provider: str) -> web.StreamResponse:
        start_perf = time.perf_counter()
        client = app[_HTTP_CLIENT_KEY]
        assert client is not None
        client_slug, workflow_slug, client_entry = _resolve_client_and_workflow(
            request, app[_CLIENTS_CFG_KEY]
        )
        anti_slop = _is_anti_slop_enabled(cfg, request, client_entry)

        try:
            raw_payload = await request.json()
        except Exception:
            return web.json_response({"error": {"type": "invalid_request_error", "message": "Invalid JSON"}}, status=400)

        if not isinstance(raw_payload, dict):
            return web.json_response({"error": {"type": "invalid_request_error", "message": "JSON object required"}}, status=400)

        requested_model = str(raw_payload.get("model") or "")
        routed_model = requested_model
        route_mode = "passthrough"
        bp_injected = 0
        missed_usd = 0.0
        secrets_blocked = 0
        slop_healed = 0
        u_in = cw_tok = cr_tok = out_tok = 0
        ttfb_ms = 0.0
        status_code = 200

        # 1. Pre-egress secret scrubbing
        payload, secrets_blocked = _redact_request_payload(raw_payload)

        try:
            # 2. Local model slug fast-path
            if is_local_model_slug(requested_model):
                route_mode = "local_fast_path"
                resp_data, status_code = await _forward_local_fast_path(
                    http_client=client,
                    local_endpoints=app[_LOCAL_ENDPOINTS_KEY],
                    provider=provider,
                    body=payload,
                    model=requested_model,
                )
                ttfb_ms = round((time.perf_counter() - start_perf) * 1000.0, 3)
                if anti_slop and status_code == 200:
                    slop_healed = _heal_non_streaming_response(resp_data, provider)
                u_in, cw_tok, cr_tok, out_tok = _extract_non_streaming_usage(resp_data, provider)
                return web.json_response(resp_data, status=status_code)

            # 3. Bicameral Chore Router (Pro-only; skipped cleanly in Open-Core stub mode)
            chore_enabled = cfg.chore_router or (
                request.headers.get("x-wbc-route", "").strip().lower() in ("auto", "haiku", "local")
            )
            if provider == "anthropic" and chore_enabled and not payload.get("stream") and _is_chore_router_available():
                default_chore = str(client_entry.get("default_route") or "haiku")
                decision = _chore_router_mod.classify_request_complexity(
                    payload,
                    request.headers,
                    default_chore_tier=default_chore if default_chore in ("haiku", "local") else "haiku",
                )
                if decision.should_downshift:
                    route_mode = f"chore_{decision.target_tier}"
                    resp_data, status_code, routed_model = await _chore_router_mod.route_and_execute_chore(
                        payload,
                        request.headers,
                        mode=decision.target_tier,
                        client=client,
                        anthropic_upstream=cfg.anthropic_upstream,
                        local_endpoints=app[_LOCAL_ENDPOINTS_KEY],
                    )
                    ttfb_ms = round((time.perf_counter() - start_perf) * 1000.0, 3)
                    if anti_slop and status_code == 200:
                        slop_healed = _heal_non_streaming_response(resp_data, provider)
                    u_in, cw_tok, cr_tok, out_tok = _extract_non_streaming_usage(resp_data, provider)
                    return web.json_response(resp_data, status=status_code)

            # 4. Cache injection & Community 250/day fail-open metering
            if provider == "anthropic":
                cache_allowed, _ = _check_community_cache_allowance(
                    cfg,
                    db_path=app[_DB_PATH_KEY],
                    license_mgr=app[_LICENSE_MGR_KEY],
                )
                if cache_allowed:
                    payload, bp_injected = inject_anthropic_ephemeral_cache(payload, enabled=True)
                    if bp_injected > 0:
                        route_mode = "ephemeral_cache"
                else:
                    # Fail-open passthrough: never return 402/429, record missed savings
                    route_mode = "community_cap_passthrough"
                    est_prefix_tok = _estimate_prefix_tokens_for_missed_cap(payload)
                    if est_prefix_tok > 0:
                        missed_receipt = compute_proxy_call_cost(
                            requested_model,
                            cache_read_tokens=est_prefix_tok,
                        )
                        missed_usd = max(0.0, missed_receipt.net_usd_saved)
                upstream_url = f"{cfg.anthropic_upstream.rstrip('/')}/v1/messages"
            else:
                payload, injected_usage = prepare_openai_request(payload)
                if injected_usage:
                    route_mode = "openai_usage_capture"
                upstream_url = f"{cfg.openai_upstream.rstrip('/')}/v1/chat/completions"

            fwd_headers = _filter_upstream_headers(request.headers)
            is_stream = bool(payload.get("stream"))

            if not is_stream:
                resp = await client.post(upstream_url, json=payload, headers=fwd_headers, timeout=120.0)
                status_code = resp.status_code
                ttfb_ms = round((time.perf_counter() - start_perf) * 1000.0, 3)
                try:
                    resp_data = resp.json()
                except Exception:
                    return web.Response(body=resp.content, status=status_code)

                if anti_slop and status_code == 200 and isinstance(resp_data, dict):
                    slop_healed = _heal_non_streaming_response(resp_data, provider)
                if isinstance(resp_data, dict):
                    u_in, cw_tok, cr_tok, out_tok = _extract_non_streaming_usage(resp_data, provider)
                return web.json_response(resp_data, status=status_code)

            # Streaming SSE passthrough
            scanner = SSEFrameScanner(
                provider=provider,
                anti_slop=anti_slop,
                start_time=start_perf,
            )
            async with client.stream(
                "POST",
                upstream_url,
                json=payload,
                headers=fwd_headers,
                timeout=120.0,
            ) as upstream_resp:
                status_code = upstream_resp.status_code
                if status_code >= 400:
                    err_bytes = await upstream_resp.aread()
                    ttfb_ms = round((time.perf_counter() - start_perf) * 1000.0, 3)
                    return web.Response(
                        body=err_bytes,
                        status=status_code,
                        content_type=upstream_resp.headers.get("content-type", "application/json").split(";")[0],
                    )

                stream_resp = web.StreamResponse(
                    status=status_code,
                    headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
                )
                await stream_resp.prepare(request)
                async for chunk in upstream_resp.aiter_bytes():
                    processed = scanner.process_chunk(chunk)
                    if processed:
                        await stream_resp.write(processed)
                tail = scanner.finalize()
                if tail:
                    await stream_resp.write(tail)
                await stream_resp.write_eof()

            u_in = scanner.usage.uncached_input_tokens
            cw_tok = scanner.usage.cache_write_tokens
            cr_tok = scanner.usage.cache_read_tokens
            out_tok = scanner.usage.output_tokens
            slop_healed = scanner.usage.slop_healed
            ttfb_ms = scanner.usage.ttfb_ms
            return stream_resp
        finally:
            latency_ms = round((time.perf_counter() - start_perf) * 1000.0, 3)
            baseline_model = requested_model if (routed_model and routed_model != requested_model) else None
            receipt = compute_proxy_call_cost(
                routed_model or requested_model,
                uncached_input_tokens=u_in,
                cache_write_tokens=cw_tok,
                cache_read_tokens=cr_tok,
                output_tokens=out_tok,
                baseline_model=baseline_model,
            )
            record_proxy_call(
                db_path=app[_DB_PATH_KEY],
                client_slug=client_slug,
                workflow_slug=workflow_slug,
                provider=provider,
                requested_model=requested_model,
                routed_model=routed_model or requested_model,
                route_mode=route_mode,
                cache_breakpoints_injected=bp_injected,
                uncached_input_tokens=u_in,
                cache_write_tokens=cw_tok,
                cache_read_tokens=cr_tok,
                output_tokens=out_tok,
                baseline_usd=receipt.baseline_uncached_usd,
                actual_usd=receipt.actual_billed_usd,
                net_usd_saved=receipt.net_usd_saved,
                missed_usd_community_cap=missed_usd,
                secrets_blocked=secrets_blocked,
                slop_healed=slop_healed,
                ttfb_ms=ttfb_ms,
                latency_ms=latency_ms,
                status_code=status_code,
                equivalent_input_tokens_saved=receipt.equivalent_input_tokens_saved,
            )

    async def handle_anthropic_messages(request: web.Request) -> web.StreamResponse:
        return await _handle_completion(request, provider="anthropic")

    async def handle_openai_chat(request: web.Request) -> web.StreamResponse:
        return await _handle_completion(request, provider="openai")

    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_passthrough)
    app.router.add_post("/v1/messages/count_tokens", handle_passthrough)
    app.router.add_post("/v1/messages", handle_anthropic_messages)
    app.router.add_post("/c/{client_slug}/v1/messages", handle_anthropic_messages)
    app.router.add_post("/c/{client_slug}/w/{workflow_slug}/v1/messages", handle_anthropic_messages)
    app.router.add_post("/v1/chat/completions", handle_openai_chat)
    app.router.add_post("/c/{client_slug}/v1/chat/completions", handle_openai_chat)

    return app


def run_proxy_server(config: Optional[ProxyConfig] = None) -> None:
    """Start the `body proxy` aiohttp server on `config.host:config.port`."""
    cfg = config or ProxyConfig()
    app = create_proxy_app(cfg)
    web.run_app(app, host=cfg.host, port=cfg.port, print=None)
