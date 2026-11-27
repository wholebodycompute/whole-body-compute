"""
Pure-stdlib JSON-RPC 2.0 stdio MCP App server for Local Agents (`ui://wbc/local-agents.html`).

Implements SEP-1865 `ui://` resource serving and read-only visibility-split tools:
- `open_local_agents` (model + app): lightweight summary + Antigravity aux_pane/global entrypoints
- `app_list_local_runs` (`["app"]` only): merged, sanitized runs & promotion stats
- `app_get_daemon_vitals` (`["app"]` only): non-blocking local Metal/LiteRT/Thermal vitals
- `app_get_run_detail` (`["app"]` only): bounded, sanitized single-run inspection
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from whole_body_compute import __version__
from whole_body_compute.local.engine import resolve_litert_cache_dir
from whole_body_compute.local.run_events import (
    _sanitize_text,
    _sanitize_value,
    get_local_run_detail,
    list_merged_local_runs,
)
from whole_body_compute.local.shadow_ledger import is_external_models_drive_mounted
from whole_body_compute.local.supervisor import ThermalSupervisor

SERVER_NAME = "whole-body-compute-local-agents"
RESOURCE_URI = "ui://wbc/local-agents.html"
RESOURCE_MIME_TYPE = "text/html;profile=mcp-app"

RESOURCE_META: dict[str, Any] = {
    "ui": {
        "prefersBorder": False,
        "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": [],
        },
    }
}

MAX_OPEN_PAYLOAD_BYTES = 60 * 1024  # strictly < 64 KB
MAX_APP_PAYLOAD_BYTES = 120 * 1024  # strictly < 128 KB
HTTP_PROBE_TIMEOUT_S = 0.35

DEFAULT_UI_HTML_PATH = Path(__file__).parent / "ui" / "local_agents.html"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "open_local_agents",
        "description": "Open the Local Agents pane and return a compact status summary.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum recent runs to summarize (1-100).",
                    "minimum": 1,
                    "maximum": 100,
                },
                "task_type": {
                    "type": "string",
                    "description": "Optional task_type filter.",
                },
            },
            "required": [],
        },
        "_meta": {
            "ui": {"resourceUri": RESOURCE_URI},
            "com.google.antigravity/ui": {
                "entrypoints": [
                    {"type": "aux_pane", "title": "Local Agents"},
                    {"type": "global", "title": "Local Agents"},
                ]
            },
        },
    },
    {
        "name": "app_list_local_runs",
        "description": "List merged local agent runs and shadow promotion stats (app-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "status": {"type": "string"},
                "task_type": {"type": "string"},
                "mode": {"type": "string"},
            },
            "required": [],
        },
        "_meta": {"ui": {"visibility": ["app"]}},
    },
    {
        "name": "app_get_daemon_vitals",
        "description": "Probe local MLX/router daemons, LiteRT cache, and ThermalSupervisor vitals (app-only).",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "_meta": {"ui": {"visibility": ["app"]}},
    },
    {
        "name": "app_get_run_detail",
        "description": "Inspect sanitized detail for a single local run_id (app-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Target local run ID."},
            },
            "required": ["run_id"],
        },
        "_meta": {"ui": {"visibility": ["app"]}},
    },
]


def _default_http_probe(url: str, timeout_s: float = HTTP_PROBE_TIMEOUT_S) -> dict[str, Any]:
    """Perform a fast, non-blocking HTTP GET health probe with strict short timeout."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=min(float(timeout_s), 0.4)) as resp:
            status_code = int(getattr(resp, "status", 200) or 200)
            raw_body = resp.read(2048).decode("utf-8", errors="replace")
            parsed: dict[str, Any] | None = None
            try:
                candidate = json.loads(raw_body)
                if isinstance(candidate, dict):
                    parsed = candidate
            except json.JSONDecodeError:
                parsed = None
            return {
                "reachable": 200 <= status_code < 400,
                "status_code": status_code,
                "payload": parsed,
            }
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {
            "reachable": False,
            "status_code": None,
            "error": str(exc)[:160],
        }


def _json_byte_size(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


class LocalAgentsMCPServer:
    """Stdio JSON-RPC 2.0 MCP App server for Whole Body Compute Local Agents."""

    def __init__(
        self,
        *,
        runs_log_path: str | Path | None = None,
        shadow_log_path: str | Path | None = None,
        ledger_path: str | Path | None = None,
        html_path: str | Path | None = None,
        http_probe_fn: Callable[[str, float], dict[str, Any]] | None = None,
        vitals_probe_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.runs_log_path = Path(runs_log_path).expanduser() if runs_log_path else None
        self.shadow_log_path = Path(shadow_log_path).expanduser() if shadow_log_path else None
        self.ledger_path = Path(ledger_path).expanduser() if ledger_path else None
        self.html_path = Path(html_path).expanduser() if html_path else None
        self._http_probe_fn = http_probe_fn or _default_http_probe
        self._vitals_probe_fn = vitals_probe_fn

    def get_tools(self) -> list[dict[str, Any]]:
        return TOOLS

    def get_resources(self) -> list[dict[str, Any]]:
        return [
            {
                "uri": RESOURCE_URI,
                "name": "Local Agents",
                "description": "Whole Body Compute Local Agents side-pane UI",
                "mimeType": RESOURCE_MIME_TYPE,
                "_meta": RESOURCE_META,
            }
        ]

    def _resolve_ui_html_path(self) -> Path:
        env_html = os.environ.get("WBC_LOCAL_AGENTS_APP_HTML", "").strip()
        if env_html:
            candidate = Path(env_html).expanduser()
            if candidate.is_file():
                return candidate
        if self.html_path is not None and self.html_path.is_file():
            return self.html_path
        return DEFAULT_UI_HTML_PATH

    def read_resource(self, uri: str) -> dict[str, Any] | None:
        if uri != RESOURCE_URI:
            return None
        path = self._resolve_ui_html_path()
        try:
            html_text = path.read_text(encoding="utf-8")
        except OSError:
            html_text = "<!doctype html><html><body><p>Local Agents UI missing.</p></body></html>"
        return {
            "contents": [
                {
                    "uri": RESOURCE_URI,
                    "mimeType": RESOURCE_MIME_TYPE,
                    "text": html_text,
                    "_meta": RESOURCE_META,
                }
            ]
        }

    def _collect_hardware_vitals(self) -> dict[str, Any]:
        if self._vitals_probe_fn is not None:
            return self._vitals_probe_fn()

        try:
            cache_dir = resolve_litert_cache_dir()
        except (OSError, RuntimeError, ValueError) as exc:
            cache_dir = f"unavailable ({exc})"

        try:
            drive_mounted = bool(is_external_models_drive_mounted())
        except (OSError, RuntimeError, ValueError):
            drive_mounted = False

        free_ram_gb = 0.0
        total_ram_gb = 0.0
        thermal_state = "UNKNOWN"
        swap_used_gb = 0.0
        try:
            tel = ThermalSupervisor().get_telemetry()
            free_ram_gb = round(float(getattr(tel, "free_ram_gb", 0.0) or 0.0), 2)
            total_ram_gb = round(float(getattr(tel, "total_ram_gb", 0.0) or 0.0), 2)
            thermal_state = str(getattr(tel, "thermal_state", "UNKNOWN") or "UNKNOWN")
            swap_used_gb = round(float(getattr(tel, "swap_used_gb", 0.0) or 0.0), 2)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            logging.getLogger("whole_body_compute.mcp.local_agents_server").debug(
                "ThermalSupervisor telemetry unavailable: %s", exc
            )

        return {
            "litert_cache_dir": cache_dir,
            "unplugged_ready": True,
            "external_models_drive_mounted": drive_mounted,
            "free_ram_gb": free_ram_gb,
            "total_ram_gb": total_ram_gb,
            "thermal_state": thermal_state,
            "swap_used_gb": swap_used_gb,
        }

    def _fit_runs_payload(
        self,
        structured: dict[str, Any],
        *,
        max_bytes: int,
        text_builder: Callable[[dict[str, Any]], str],
    ) -> dict[str, Any]:
        runs = list(structured.get("runs") or [])
        while True:
            structured["runs"] = runs
            if "summary" in structured and isinstance(structured["summary"], dict):
                structured["summary"]["returned_runs"] = len(runs)
            result = {
                "isError": False,
                "content": [{"type": "text", "text": _sanitize_text(text_builder(structured))}],
                "structuredContent": structured,
            }
            if _json_byte_size(result) < max_bytes:
                return result
            if not runs:
                # Trim promotion_by_task if still too large
                summary = structured.get("summary")
                if isinstance(summary, dict) and summary.get("promotion_by_task"):
                    summary["promotion_by_task"] = {}
                    continue
                result["content"] = [{"type": "text", "text": "Local Agents summary ready."}]
                return result
            runs = runs[: max(0, len(runs) // 2)]

    def _tool_open_local_agents(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_limit = args.get("limit", 25)
        try:
            limit = max(1, min(100, int(raw_limit if raw_limit is not None else 25)))
        except (TypeError, ValueError):
            limit = 25
        task_type = str(args.get("task_type") or "").strip() or None

        merged = list_merged_local_runs(
            limit=limit,
            task_type=task_type,
            runs_log_path=self.runs_log_path,
            shadow_log_path=self.shadow_log_path,
            ledger_path=self.ledger_path,
        )
        summary = merged["summary"]
        compact_runs = [
            {
                "run_id": r["run_id"],
                "role": r["role"],
                "task_type": r["task_type"],
                "model": r["model"],
                "mode": r["mode"],
                "status": r["status"],
                "latency_ms": r["latency_ms"],
                "sla_ms": r["sla_ms"],
                "tokens_saved_est": r["tokens_saved_est"],
                "cost_usd": 0.0,
            }
            for r in merged["runs"]
        ]
        structured = _sanitize_value(
            {
                "summary": summary,
                "runs": compact_runs,
            }
        )

        def _make_text(payload: dict[str, Any]) -> str:
            s = payload["summary"]
            return (
                f"Local Agents: {s['total_runs']} runs "
                f"({s['running_count']} running, {s['pass_count']} pass, "
                f"{s['blocked_count']} blocked, {s['stale_count']} stale), "
                f"{s['total_tokens_saved_est']} est tokens saved ($0.00 cost)."
            )

        return self._fit_runs_payload(
            structured,
            max_bytes=MAX_OPEN_PAYLOAD_BYTES,
            text_builder=_make_text,
        )

    def _tool_app_list_local_runs(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_limit = args.get("limit", 40)
        try:
            limit = max(1, min(100, int(raw_limit if raw_limit is not None else 40)))
        except (TypeError, ValueError):
            limit = 40
        status = str(args.get("status") or "").strip() or None
        task_type = str(args.get("task_type") or "").strip() or None
        mode = str(args.get("mode") or "").strip() or None

        merged = list_merged_local_runs(
            limit=limit,
            status=status,
            task_type=task_type,
            mode=mode,
            runs_log_path=self.runs_log_path,
            shadow_log_path=self.shadow_log_path,
            ledger_path=self.ledger_path,
        )
        structured = _sanitize_value(merged)

        def _make_text(payload: dict[str, Any]) -> str:
            s = payload["summary"]
            return (
                f"Listed {s['returned_runs']} of {s['filtered_runs']} local runs "
                f"({s['running_count']} running)."
            )

        return self._fit_runs_payload(
            structured,
            max_bytes=MAX_APP_PAYLOAD_BYTES,
            text_builder=_make_text,
        )

    def _tool_app_get_daemon_vitals(self, _args: dict[str, Any]) -> dict[str, Any]:
        coder_probe = self._safe_http_probe("http://127.0.0.1:9379/health")
        router_probe = self._safe_http_probe("http://127.0.0.1:9380/health")
        hw = self._collect_hardware_vitals()

        structured = _sanitize_value(
            {
                "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "daemons": {
                    "body_coder_mlx": {
                        "port": 9379,
                        "url": "http://127.0.0.1:9379/health",
                        **coder_probe,
                    },
                    "specialist_router": {
                        "port": 9380,
                        "url": "http://127.0.0.1:9380/health",
                        **router_probe,
                    },
                },
                "storage": {
                    "litert_cache_dir": hw.get("litert_cache_dir", ""),
                    "unplugged_ready": bool(hw.get("unplugged_ready", True)),
                    "external_models_drive_mounted": bool(
                        hw.get("external_models_drive_mounted", False)
                    ),
                },
                "hardware": {
                    "free_ram_gb": hw.get("free_ram_gb", 0.0),
                    "total_ram_gb": hw.get("total_ram_gb", 0.0),
                    "thermal_state": hw.get("thermal_state", "UNKNOWN"),
                    "swap_used_gb": hw.get("swap_used_gb", 0.0),
                },
            }
        )
        text = (
            f"Daemons: :9379={'up' if coder_probe.get('reachable') else 'down'}, "
            f":9380={'up' if router_probe.get('reachable') else 'down'} | "
            f"Thermal={structured['hardware']['thermal_state']} | "
            f"Free RAM={structured['hardware']['free_ram_gb']} GB"
        )
        return {
            "isError": False,
            "content": [{"type": "text", "text": _sanitize_text(text)}],
            "structuredContent": structured,
        }

    def _safe_http_probe(self, url: str) -> dict[str, Any]:
        try:
            res = self._http_probe_fn(url, HTTP_PROBE_TIMEOUT_S)
            if isinstance(res, dict):
                return res
            return {"reachable": False, "status_code": None}
        except (OSError, TimeoutError, RuntimeError, ValueError) as exc:
            return {"reachable": False, "status_code": None, "error": str(exc)[:160]}

    def _tool_app_get_run_detail(self, args: dict[str, Any]) -> dict[str, Any]:
        run_id = str(args.get("run_id") or "").strip()
        if not run_id:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "Missing required argument: run_id"}],
                "structuredContent": {"error": "missing_run_id"},
            }

        detail = get_local_run_detail(
            run_id,
            runs_log_path=self.runs_log_path,
            shadow_log_path=self.shadow_log_path,
            ledger_path=self.ledger_path,
        )
        if detail is None:
            safe_id = _sanitize_text(run_id)
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Run not found: {safe_id}"}],
                "structuredContent": {"error": "not_found", "run_id": safe_id},
            }

        result = {
            "isError": False,
            "content": [
                {
                    "type": "text",
                    "text": _sanitize_text(
                        f"Run {detail['run_id']}: {detail['role']} [{detail['status']}] "
                        f"({detail['model']}, {detail['latency_ms']} ms)"
                    ),
                }
            ],
            "structuredContent": detail,
        }

        # Enforce hard < 128 KB bound by progressively trimming large preview fields if needed.
        if _json_byte_size(result) >= MAX_APP_PAYLOAD_BYTES:
            for cap in (4000, 1000, 256):
                trimmed = self._trim_detail_strings(detail, cap)
                result["structuredContent"] = trimmed
                if _json_byte_size(result) < MAX_APP_PAYLOAD_BYTES:
                    break
            if _json_byte_size(result) >= MAX_APP_PAYLOAD_BYTES:
                result["structuredContent"] = {
                    "run_id": detail["run_id"],
                    "role": detail["role"],
                    "task_type": detail["task_type"],
                    "model": detail["model"],
                    "engine": detail["engine"],
                    "mode": detail["mode"],
                    "status": detail["status"],
                    "latency_ms": detail["latency_ms"],
                    "truncated": True,
                }
        return result

    @staticmethod
    def _trim_detail_strings(value: Any, max_chars: int) -> Any:
        if isinstance(value, str):
            return value if len(value) <= max_chars else value[: max_chars - 3] + "..."
        if isinstance(value, dict):
            return {k: LocalAgentsMCPServer._trim_detail_strings(v, max_chars) for k, v in value.items()}
        if isinstance(value, list):
            return [LocalAgentsMCPServer._trim_detail_strings(v, max_chars) for v in value[:50]]
        return value

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments if isinstance(arguments, dict) else {}
        if name == "open_local_agents":
            return self._tool_open_local_agents(args)
        if name == "app_list_local_runs":
            return self._tool_app_list_local_runs(args)
        if name == "app_get_daemon_vitals":
            return self._tool_app_get_daemon_vitals(args)
        if name == "app_get_run_detail":
            return self._tool_app_get_run_detail(args)
        safe_name = _sanitize_text(str(name))
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Unknown tool: {safe_name}"}],
            "structuredContent": {"error": "unknown_tool", "tool": safe_name},
        }

    def handle_rpc_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(msg, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
        method = str(msg.get("method") or "")
        msg_id = msg.get("id")

        if method == "initialize":
            params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
            proto = str(params.get("protocolVersion") or "2024-11-05")
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": proto,
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": __version__},
                },
            }
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": self.get_tools()}}
        if method == "tools/call":
            params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
            tool_name = str(params.get("name") or "")
            tool_args = params.get("arguments")
            res = self.call_tool(tool_name, tool_args if isinstance(tool_args, dict) else {})
            return {"jsonrpc": "2.0", "id": msg_id, "result": res}
        if method == "resources/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": self.get_resources()}}
        if method == "resources/read":
            params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
            uri = str(params.get("uri") or "")
            res = self.read_resource(uri)
            if res is None:
                if msg_id is not None:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32002, "message": f"Resource not found: {uri}"},
                    }
                return None
            return {"jsonrpc": "2.0", "id": msg_id, "result": res}

        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return None

    def run_stdio(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        in_stream = stdin if stdin is not None else sys.stdin
        out_stream = stdout if stdout is not None else sys.stdout
        for line in in_stream:
            raw = line.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                out_stream.write(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": "Parse error"},
                        }
                    )
                    + "\n"
                )
                out_stream.flush()
                continue
            resp = self.handle_rpc_message(msg)
            if resp is not None:
                out_stream.write(json.dumps(resp) + "\n")
                out_stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Whole Body Compute Local Agents MCP App server")
    parser.add_argument("--runs-log", default=None, help="Optional local_agent_runs.jsonl path")
    parser.add_argument("--shadow-log", default=None, help="Optional specialist_router_shadow.jsonl path")
    parser.add_argument("--ledger", default=None, help="Optional router_telemetry_ledger.json path")
    parser.add_argument("--html", default=None, help="Optional UI HTML bundle override")
    args = parser.parse_args(list(argv) if argv is not None else None)
    server = LocalAgentsMCPServer(
        runs_log_path=args.runs_log,
        shadow_log_path=args.shadow_log,
        ledger_path=args.ledger,
        html_path=args.html,
    )
    server.run_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
