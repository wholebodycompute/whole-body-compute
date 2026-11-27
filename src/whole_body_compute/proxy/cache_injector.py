"""
Deterministic Anthropic `cache_control` 4-slot injector and OpenAI usage enabler.

Anthropic evaluates prompt caches in prefix order: `tools` -> `system` -> `messages`,
with a hard ceiling of 4 `cache_control` breakpoints per request.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple

_EPHEMERAL_MARKER: Dict[str, str] = {"type": "ephemeral"}
_MAX_BREAKPOINTS = 4
_FORBIDDEN_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def get_min_cache_tokens(model: str) -> int:
    """
    Return Anthropic's minimum cacheable prefix token threshold for `model`:
    - 4096: `claude-haiku-4-5*`, `claude-opus-4-5*`, `claude-opus-4-6*`
    - 2048: `claude-3-5-haiku*`, `claude-3-haiku*`
    - 1024: `claude-sonnet-4*`, `claude-3-7-sonnet*`, `claude-3-5-sonnet*`,
      `claude-opus-4*`, `claude-3-opus*` (and default fallback)
    """
    key = (model or "").strip().lower()
    if (
        "haiku-4-5" in key
        or "haiku-4.5" in key
        or "opus-4-5" in key
        or "opus-4.5" in key
        or "opus-4-6" in key
        or "opus-4.6" in key
    ):
        return 4096
    if "3-5-haiku" in key or "3.5-haiku" in key or "3-haiku" in key:
        return 2048
    return 1024


def _est_str_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.encode("utf-8")) // 4


def _est_json_tokens(obj: Any) -> int:
    if not obj:
        return 0
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) // 4


def _est_block_tokens(block: Any) -> int:
    if isinstance(block, str):
        return _est_str_tokens(block)
    if isinstance(block, dict):
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            return _est_str_tokens(block["text"])
        return _est_json_tokens(block)
    return 0


def _est_message_tokens(msg: Any) -> int:
    if not isinstance(msg, dict):
        return 0
    content = msg.get("content")
    if isinstance(content, str):
        return _est_str_tokens(content)
    if isinstance(content, list):
        return sum(_est_block_tokens(b) for b in content)
    return _est_json_tokens(msg)


def _is_eligible_text_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    btype = block.get("type")
    if btype in _FORBIDDEN_BLOCK_TYPES:
        return False
    if "cache_control" in block:
        return False
    if btype == "text":
        return bool(str(block.get("text", "")).strip())
    return False


def _is_eligible_message_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    btype = block.get("type")
    if btype in _FORBIDDEN_BLOCK_TYPES:
        return False
    if "cache_control" in block:
        return False
    if btype == "text":
        return bool(str(block.get("text", "")).strip())
    return True


def count_existing_cache_breakpoints(body: Dict[str, Any]) -> int:
    """Count existing `cache_control` breakpoints across `tools`, `system`, and `messages`."""
    if not isinstance(body, dict):
        return 0
    count = 0

    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and "cache_control" in tool:
                count += 1

    system = body.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                count += 1

    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        count += 1

    return count


def inject_anthropic_ephemeral_cache(
    body: Dict[str, Any],
    *,
    enabled: bool = True,
) -> Tuple[Dict[str, Any], int]:
    """
    Deterministically inject up to `(4 - existing_breakpoints)` `cache_control: {"type": "ephemeral"}`
    markers across `tools`, `system`, and `messages` without mutating the caller's input dict.

    Returns `(body_copy, injected_count)`.
    """
    if not enabled or not isinstance(body, dict):
        return body, 0

    existing = count_existing_cache_breakpoints(body)
    slots_left = _MAX_BREAKPOINTS - existing
    if slots_left <= 0:
        return copy.deepcopy(body), 0

    working = copy.deepcopy(body)
    min_tok = get_min_cache_tokens(str(working.get("model", "")))
    cum_tok = 0
    injected = 0

    # Priority 1: Tool Catalog (`tools[-1]`)
    tools = working.get("tools")
    if isinstance(tools, list) and tools:
        cum_tok += _est_json_tokens(tools)
        if (
            slots_left > 0
            and cum_tok >= min_tok
            and isinstance(tools[-1], dict)
            and "cache_control" not in tools[-1]
        ):
            tools[-1]["cache_control"] = dict(_EPHEMERAL_MARKER)
            slots_left -= 1
            injected += 1

    # Priority 2: System Prompt / Client Brand Bible (`system`)
    system = working.get("system")
    sys_tok = 0
    if isinstance(system, str):
        sys_tok = _est_str_tokens(system)
    elif isinstance(system, list):
        sys_tok = sum(_est_block_tokens(b) for b in system)
    cum_tok += sys_tok

    if slots_left > 0 and sys_tok >= 256 and cum_tok >= min_tok:
        if isinstance(system, str) and system.strip():
            working["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": dict(_EPHEMERAL_MARKER),
                }
            ]
            slots_left -= 1
            injected += 1
        elif isinstance(system, list) and system:
            eligible_indices: List[int] = [
                idx for idx, blk in enumerate(system) if _is_eligible_text_block(blk)
            ]
            if eligible_indices:
                # Multi-block Agency SOP + Brand Bible bonus
                if (
                    len(system) >= 2
                    and slots_left >= 2
                    and eligible_indices[0] == 0
                    and _est_block_tokens(system[0]) >= min_tok
                    and len(eligible_indices) >= 2
                ):
                    system[0]["cache_control"] = dict(_EPHEMERAL_MARKER)
                    slots_left -= 1
                    injected += 1
                last_idx = eligible_indices[-1]
                if slots_left > 0 and "cache_control" not in system[last_idx]:
                    system[last_idx]["cache_control"] = dict(_EPHEMERAL_MARKER)
                    slots_left -= 1
                    injected += 1

    # Priority 3: Multi-Turn History Prefix (`messages[-2]`) or Single-Turn Bundle (`messages[0].content[-2]`)
    messages = working.get("messages")
    if isinstance(messages, list) and slots_left > 0:
        if len(messages) >= 2:
            cum_tok += sum(_est_message_tokens(m) for m in messages[:-1])
            if cum_tok >= min_tok:
                target_msg = messages[-2]
                if isinstance(target_msg, dict):
                    content = target_msg.get("content")
                    if isinstance(content, str) and content.strip():
                        target_msg["content"] = [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": dict(_EPHEMERAL_MARKER),
                            }
                        ]
                        slots_left -= 1
                        injected += 1
                    elif isinstance(content, list) and content:
                        for blk in reversed(content):
                            if _is_eligible_message_block(blk):
                                blk["cache_control"] = dict(_EPHEMERAL_MARKER)
                                slots_left -= 1
                                injected += 1
                                break
        elif len(messages) == 1:
            msg0 = messages[0]
            if isinstance(msg0, dict) and isinstance(msg0.get("content"), list):
                blocks = msg0["content"]
                if len(blocks) >= 2:
                    cum_tok += sum(_est_block_tokens(b) for b in blocks[:-1])
                    if cum_tok >= min_tok:
                        target_blk = blocks[-2]
                        if _is_eligible_message_block(target_blk):
                            target_blk["cache_control"] = dict(_EPHEMERAL_MARKER)
                            slots_left -= 1
                            injected += 1

    return working, injected


def prepare_openai_request(body: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Inject `stream_options: {"include_usage": True}` on streaming `/v1/chat/completions`
    requests when absent so exact token usage is captured from the final SSE chunk.
    """
    if not isinstance(body, dict) or body.get("stream") is not True:
        return body, False

    existing_opts = body.get("stream_options")
    if isinstance(existing_opts, dict) and existing_opts.get("include_usage") is True:
        return body, False

    working = copy.deepcopy(body)
    opts = dict(existing_opts) if isinstance(existing_opts, dict) else {}
    opts["include_usage"] = True
    working["stream_options"] = opts
    return working, True
