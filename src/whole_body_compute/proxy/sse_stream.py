"""
Zero-buffering incremental SSE frame scanner & prose-only hygiene filter.

Captures exact token usage from Anthropic (`message_start` + `message_delta`)
and OpenAI (`usage` chunks) while forwarding SSE frames immediately.
When `anti_slop=True`, applies <0.5ms `auto_heal_em_dash` and
`auto_heal_invisible_marks` strictly to `text_delta` / `delta.content` prose
outside fenced code blocks, leaving tool calls and thinking blocks untouched.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from whole_body_compute.organs.immune_hygiene import (
    auto_heal_em_dash,
    auto_heal_invisible_marks,
)

_PROTECTED_BLOCK_TYPES = frozenset(
    {"tool_use", "server_tool_use", "thinking", "redacted_thinking"}
)
_PROTECTED_DELTA_TYPES = frozenset(
    {"input_json_delta", "thinking_delta", "signature_delta"}
)


@dataclass
class SSEUsageSummary:
    """Accumulated token usage and hygiene stats for a single SSE stream."""

    uncached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    slop_healed: int = 0
    ttfb_ms: float = 0.0


def _heal_prose_outside_fences(text: str, in_fence: bool) -> Tuple[str, int, bool]:
    """
    Apply `auto_heal_em_dash` + `auto_heal_invisible_marks` only to prose segments
    outside fenced ``` code blocks, preserving `in_fence` across chunks.
    """
    if not text:
        return text, 0, in_fence

    if "```" not in text:
        if in_fence:
            return text, 0, True
        healed, c1 = auto_heal_em_dash(text)
        healed, c2 = auto_heal_invisible_marks(healed)
        return healed, c1 + c2, False

    parts = text.split("```")
    out_parts = []
    total_healed = 0
    curr_fence = in_fence
    for idx, part in enumerate(parts):
        if idx > 0:
            out_parts.append("```")
            curr_fence = not curr_fence
        if curr_fence:
            out_parts.append(part)
        else:
            healed, c1 = auto_heal_em_dash(part)
            healed, c2 = auto_heal_invisible_marks(healed)
            total_healed += c1 + c2
            out_parts.append(healed)
    return "".join(out_parts), total_healed, curr_fence


class SSEFrameScanner:
    """
    Incremental line-oriented SSE scanner for Anthropic and OpenAI streams.
    Forwards complete lines immediately without buffering whole responses.
    """

    def __init__(
        self,
        *,
        provider: str = "anthropic",
        anti_slop: bool = False,
        start_time: Optional[float] = None,
    ) -> None:
        self.provider = (provider or "anthropic").strip().lower()
        self.anti_slop = bool(anti_slop)
        self.start_time = start_time if start_time is not None else time.perf_counter()
        self.usage = SSEUsageSummary()
        self._first_chunk_seen = False
        self._buf = ""
        self._current_event = ""
        self._block_types: Dict[int, str] = {}
        self._block_in_fence: Dict[int, bool] = {}
        self._openai_in_fence: Dict[int, bool] = {}

    def process_chunk(self, chunk: bytes) -> bytes:
        """Process an incoming SSE byte chunk and return the forwarded bytes."""
        if not chunk:
            return b""
        if not self._first_chunk_seen:
            self._first_chunk_seen = True
            self.usage.ttfb_ms = round((time.perf_counter() - self.start_time) * 1000.0, 3)

        self._buf += chunk.decode("utf-8", errors="replace")
        if "\n" not in self._buf:
            return b""

        lines = self._buf.split("\n")
        self._buf = lines.pop()
        out_lines = [self._process_line(line) for line in lines]
        return ("\n".join(out_lines) + "\n").encode("utf-8")

    def finalize(self) -> bytes:
        """Flush any trailing buffered line at EOF."""
        if not self._buf:
            return b""
        tail = self._process_line(self._buf)
        self._buf = ""
        return tail.encode("utf-8")

    def _process_line(self, line: str) -> str:
        stripped = line.rstrip("\r")
        if not stripped:
            self._current_event = ""
            return line
        if stripped.startswith("event:"):
            self._current_event = stripped[6:].strip()
            return line
        if not stripped.startswith("data:"):
            return line

        raw_data = stripped[5:].strip()
        if not raw_data or raw_data == "[DONE]":
            return line

        try:
            payload = json.loads(raw_data)
        except Exception:
            return line

        if not isinstance(payload, dict):
            return line

        if self.provider == "openai":
            modified = self._handle_openai_payload(payload)
        else:
            modified = self._handle_anthropic_payload(payload)

        if modified:
            suffix = "\r" if line.endswith("\r") else ""
            return f"data: {json.dumps(payload, ensure_ascii=False)}{suffix}"
        return line

    def _handle_anthropic_payload(self, payload: Dict[str, Any]) -> bool:
        ev = self._current_event or str(payload.get("type") or "")

        if ev == "message_start":
            msg = payload.get("message")
            if isinstance(msg, dict):
                usage = msg.get("usage")
                if isinstance(usage, dict):
                    self.usage.uncached_input_tokens = int(usage.get("input_tokens") or 0)
                    self.usage.cache_write_tokens = int(usage.get("cache_creation_input_tokens") or 0)
                    self.usage.cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)
            return False

        if ev == "message_delta":
            usage = payload.get("usage")
            if isinstance(usage, dict):
                if "output_tokens" in usage:
                    self.usage.output_tokens = int(usage.get("output_tokens") or 0)
                if "input_tokens" in usage and usage.get("input_tokens") is not None:
                    self.usage.uncached_input_tokens = int(usage.get("input_tokens") or 0)
                if "cache_creation_input_tokens" in usage and usage.get("cache_creation_input_tokens") is not None:
                    self.usage.cache_write_tokens = int(usage.get("cache_creation_input_tokens") or 0)
                if "cache_read_input_tokens" in usage and usage.get("cache_read_input_tokens") is not None:
                    self.usage.cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)
            return False

        if ev == "content_block_start":
            idx = int(payload.get("index") or 0)
            cb = payload.get("content_block")
            if isinstance(cb, dict):
                self._block_types[idx] = str(cb.get("type") or "")
                self._block_in_fence[idx] = False
            return False

        if ev == "content_block_delta" and self.anti_slop:
            idx = int(payload.get("index") or 0)
            block_type = self._block_types.get(idx, "")
            if block_type in _PROTECTED_BLOCK_TYPES:
                return False
            delta = payload.get("delta")
            if not isinstance(delta, dict):
                return False
            delta_type = str(delta.get("type") or "")
            if delta_type in _PROTECTED_DELTA_TYPES or delta_type != "text_delta":
                return False
            if block_type and block_type != "text":
                return False
            text = delta.get("text")
            if not isinstance(text, str) or not text:
                return False
            in_fence = self._block_in_fence.get(idx, False)
            healed, count, next_fence = _heal_prose_outside_fences(text, in_fence)
            self._block_in_fence[idx] = next_fence
            if count > 0 and healed != text:
                delta["text"] = healed
                self.usage.slop_healed += count
                return True

        return False

    def _handle_openai_payload(self, payload: Dict[str, Any]) -> bool:
        usage = payload.get("usage")
        if isinstance(usage, dict):
            prompt_tok = int(usage.get("prompt_tokens") or 0)
            details = usage.get("prompt_tokens_details")
            cached_tok = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
            self.usage.uncached_input_tokens = max(0, prompt_tok - cached_tok)
            self.usage.cache_read_tokens = cached_tok
            self.usage.output_tokens = int(usage.get("completion_tokens") or 0)

        if not self.anti_slop:
            return False

        choices = payload.get("choices")
        if not isinstance(choices, list):
            return False

        modified = False
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if delta.get("tool_calls"):
                continue
            content = delta.get("content")
            if not isinstance(content, str) or not content:
                continue
            c_idx = int(choice.get("index") or 0)
            in_fence = self._openai_in_fence.get(c_idx, False)
            healed, count, next_fence = _heal_prose_outside_fences(content, in_fence)
            self._openai_in_fence[c_idx] = next_fence
            if count > 0 and healed != content:
                delta["content"] = healed
                self.usage.slop_healed += count
                modified = True
        return modified


async def iter_sse_chunks(
    byte_iter: AsyncIterator[bytes],
    *,
    provider: str = "anthropic",
    anti_slop: bool = False,
    scanner: Optional[SSEFrameScanner] = None,
) -> AsyncIterator[bytes]:
    """Yield processed SSE chunks immediately from an async byte iterator."""
    sc = scanner or SSEFrameScanner(provider=provider, anti_slop=anti_slop)
    async for chunk in byte_iter:
        out = sc.process_chunk(chunk)
        if out:
            yield out
    tail = sc.finalize()
    if tail:
        yield tail
