"""
Pal Live Edit Mode - Deterministic Spoken Edit Rules & Data Models.

Provides:
1. SelectionSnapshot & PalEditRecord data models.
2. Sub-2ms deterministic spoken punctuation, relational line actions, color change,
   and smart HTML-entity-aware replacement rules.
"""

from __future__ import annotations

import datetime
import html
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("whole_body_compute.pal_editor")
PAL_LOG_DIR = Path.home() / ".voicefi" / "logs"
PAL_LOG_FILE = PAL_LOG_DIR / "pal_editor.log"


def pal_log(message: str, level: str = "info") -> None:
    """Write structured, high-visibility log messages to stderr (/tmp/voicefi.log) and ~/.voicefi/logs/pal_editor.log."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    formatted = f"[{timestamp}] [PalEditor] {message}"

    # 1. Output to sys.stderr so LaunchAgent streams it to /tmp/voicefi.log
    try:
        sys.stderr.write(formatted + "\n")
        sys.stderr.flush()
    except Exception:
        pass

    # 2. Append to dedicated log file ~/.voicefi/logs/pal_editor.log
    try:
        PAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(PAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass

    # 3. Standard Python logging
    log_fn = getattr(logger, level.lower(), logger.info)
    try:
        log_fn(message)
    except Exception:
        pass


ACTIVE_EDITING_JOURNAL = Path.home() / ".voicefi" / "active_editing.json"

# Default local project roots searched when mapping a browser tab to a local source file
DEFAULT_WORKSPACE_ROOTS: Tuple[Path, ...] = (
    Path.home() / "Projects" / "Whole Body Compute",
    Path.home() / "Projects" / "whole_body_compute.org",
    Path.home() / "Projects" / "vifi.co",
    Path.home() / "Projects" / "VoiceFi",
)

# Spoken punctuation & typographic symbol normalizer
SPOKEN_PUNCTUATION_MAP: Dict[str, str] = {
    "colon": ":",
    "a colon": ":",
    "the colon": ":",
    "semicolon": ";",
    "semi colon": ";",
    "semi-colon": ";",
    "a semicolon": ";",
    "a semi colon": ";",
    "a semi-colon": ";",
    "comma": ",",
    "a comma": ",",
    "the comma": ",",
    "period": ".",
    "a period": ".",
    "full stop": ".",
    "a full stop": ".",
    "dot": ".",
    "a dot": ".",
    "em dash": "—",
    "em-dash": "—",
    "emdash": "—",
    "an em dash": "—",
    "an em-dash": "—",
    "long dash": "—",
    "a long dash": "—",
    "en dash": "–",
    "en-dash": "–",
    "an en dash": "–",
    "an en-dash": "–",
    "dash": "-",
    "a dash": "-",
    "hyphen": "-",
    "a hyphen": "-",
    "minus": "-",
    "minus sign": "-",
    "exclamation point": "!",
    "an exclamation point": "!",
    "exclamation mark": "!",
    "an exclamation mark": "!",
    "exclamation": "!",
    "question mark": "?",
    "a question mark": "?",
    "ellipsis": "...",
    "an ellipsis": "...",
    "dot dot dot": "...",
    "three dots": "...",
    "slash": "/",
    "a slash": "/",
    "forward slash": "/",
    "a forward slash": "/",
    "pipe": "|",
    "a pipe": "|",
    "vertical bar": "|",
    "bullet": "•",
    "a bullet": "•",
    "bullet point": "•",
    "a bullet point": "•",
    "ampersand": "&",
    "an ampersand": "&",
    "and sign": "&",
    "plus": "+",
    "a plus": "+",
    "plus sign": "+",
    "a plus sign": "+",
    "equals": "=",
    "equals sign": "=",
    "an equals sign": "=",
    "space": " ",
    "a space": " ",
    "nothing": "",
    "blank": "",
    "empty": "",
}

# Characters that attach directly to the end of the preceding word (no leading space)
ATTACHED_TRAILING_PUNCT = {":", ";", ",", ".", "!", "?"}
# Dash variants that often have spaces around them in web copy
DASH_VARIANTS = {
    "—",
    "–",
    "-",
    "&mdash;",
    "&ndash;",
    "&#8212;",
    "&#8211;",
    "&#x2014;",
    "&#x2013;",
}


@dataclass
class SelectionSnapshot:
    """Represents a captured user selection on screen along with its surrounding context."""

    selected_text: str
    surrounding_text: str = ""
    app_name: str = ""
    window_title: str = ""
    page_url: str = ""
    x: float = 0.0
    y: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PalEditRecord:
    """Represents a single live edit step in Pal Edit Mode (for UI feed and Undo stack)."""

    step_number: int
    action: str  # "select", "edit_applied", "edit_failed"
    selected_text: str
    surrounding_text: str
    instruction: str
    file_path: str
    old_snippet: str
    new_snippet: str
    spoken_summary: str
    app_name: str = ""
    window_title: str = ""
    page_url: str = ""
    engine_used: str = "fast_path"  # "fast_path" | "local_llm"
    elapsed_ms: float = 0.0
    tokens_saved: int = 1200
    applied: bool = False
    undone: bool = False
    timestamp: float = field(default_factory=time.time)
    # Full file contents kept in memory for instant exact undo
    _old_file_content: Optional[str] = field(default=None, repr=False)
    _new_file_content: Optional[str] = field(default=None, repr=False)

    def to_public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_old_file_content", None)
        d.pop("_new_file_content", None)
        return d


def resolve_spoken_replacement(raw_target: str) -> str:
    """
    Convert spoken replacement phrases like 'a colon', 'an em-dash', 'period',
    or quoted literals into their exact character/text representation.
    """
    cleaned = (raw_target or "").strip()
    # Strip outer quotes if present, but preserve inner punctuation
    if (
        len(cleaned) >= 2
        and cleaned[0] == cleaned[-1]
        and cleaned[0] in ("'", '"', "\u201c", "\u201d")
    ):
        cleaned = cleaned[1:-1]

    # Strip trailing polite words ("please", "now", "instead", "for me")
    cleaned = re.sub(
        r"\s+(?:please|now|right\s+now|instead|for\s+me|thanks|thank\s+you)[.!?,]*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # Check exact punctuation map (ignoring trailing period added by STT)
    lookup = cleaned.rstrip(".!?,").strip().lower()
    if lookup in SPOKEN_PUNCTUATION_MAP:
        return SPOKEN_PUNCTUATION_MAP[lookup]

    # If STT added a trailing period to a normal word/phrase replacement (and user didn't ask for a period), strip it
    if cleaned.endswith(".") and not lookup.endswith("period") and not lookup.endswith("dot"):
        cleaned = cleaned[:-1].strip()

    return cleaned


WORD_TO_NUM: Dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def parse_spoken_number(val: Optional[str], default: int = 1) -> int:
    """Parse spoken word numbers ('three', 'two') or digits ('3', '2') into an int."""
    if not val:
        return default
    v = val.strip().lower()
    if v.isdigit():
        return int(v)
    return WORD_TO_NUM.get(v, default)


def extract_relational_line_actions(instruction: str) -> Tuple[str, Set[int], bool]:
    """
    Parse out relational multi-line edit commands from spoken instructions:
      - 'remove the line above and below it' -> offsets {-1, 1}, remove_current=False
      - 'remove this checkbox and the three below it' -> offsets {1, 2, 3}, remove_current=True
      - 'delete the line above' -> offsets {-1}, remove_current=False
      - 'change this to a colon and remove the line below' -> offsets {1}, remaining='change this to a colon'
    Returns:
      (remaining_instruction, relative_offsets_to_remove, remove_current_line)
    """
    offsets: Set[int] = set()
    remove_current = False
    cleaned_instruction = (instruction or "").strip()

    # Pattern A: Above and below both
    p_both = re.compile(
        r"[,;]?\s*(?:and\s+)?(?:remove|delete|drop|cut|clear)\s+(?:both\s+)?the\s+lines?\s+(?:above\s+and\s+below|below\s+and\s+above)(?:\s+it)?",
        re.IGNORECASE,
    )
    if p_both.search(cleaned_instruction):
        offsets.add(-1)
        offsets.add(1)
        cleaned_instruction = p_both.sub("", cleaned_instruction).strip()

    # Pattern B: "remove this [line/item/checkbox] and the [N] below/above it"
    p_this_and = re.compile(
        r"[,;]?\s*(?:and\s+)?(?:remove|delete|drop|cut|clear)\s+this(?:\s+(?:line|item|checkbox|row|element|div|tag))?\s+and\s+(?:the\s+)?(?:(\d+|one|two|three|four|five)\s+)?(?:lines?\s+)?(below|above)(?:\s+it)?",
        re.IGNORECASE,
    )
    m_this = p_this_and.search(cleaned_instruction)
    if m_this:
        remove_current = True
        count = parse_spoken_number(m_this.group(1), 1)
        direction = m_this.group(2).lower()
        if direction == "below":
            for i in range(1, count + 1):
                offsets.add(i)
        else:
            for i in range(1, count + 1):
                offsets.add(-i)
        cleaned_instruction = p_this_and.sub("", cleaned_instruction).strip()

    # Pattern C: Line(s) above
    p_above = re.compile(
        r"[,;]?\s*(?:and\s+)?(?:remove|delete|drop|cut|clear)\s+(?:the\s+)?(?:(\d+|one|two|three|four|five)\s+)?lines?\s+above(?:\s+it)?",
        re.IGNORECASE,
    )
    m_above = p_above.search(cleaned_instruction)
    if m_above:
        count = parse_spoken_number(m_above.group(1), 1)
        for i in range(1, count + 1):
            offsets.add(-i)
        cleaned_instruction = p_above.sub("", cleaned_instruction).strip()

    # Pattern D: Line(s) below
    p_below = re.compile(
        r"[,;]?\s*(?:and\s+)?(?:remove|delete|drop|cut|clear)\s+(?:the\s+)?(?:(\d+|one|two|three|four|five)\s+)?(?:lines?\s+)?below(?:\s+it)?",
        re.IGNORECASE,
    )
    m_below = p_below.search(cleaned_instruction)
    if m_below:
        count = parse_spoken_number(m_below.group(1), 1)
        for i in range(1, count + 1):
            offsets.add(i)
        cleaned_instruction = p_below.sub("", cleaned_instruction).strip()

    # Clean leftover punctuation or trailing conjunctions
    cleaned_instruction = re.sub(r"[,;]\s*$", "", cleaned_instruction).strip()
    cleaned_instruction = re.sub(
        r"\s+and\s*$", "", cleaned_instruction, flags=re.IGNORECASE
    ).strip()

    return cleaned_instruction, offsets, remove_current


def apply_color_change_to_line(
    line_text: str, selected_text: str, new_color: str, at_end: bool = False
) -> str:
    """Apply a spoken color change (e.g. 'red', 'text-cyan-400') to Tailwind classes, inline styles, or text spans."""
    color_val = new_color.strip().lower()
    tailwind_colors = {
        "red",
        "blue",
        "cyan",
        "emerald",
        "green",
        "amber",
        "yellow",
        "purple",
        "pink",
        "rose",
        "indigo",
        "teal",
        "slate",
        "gray",
        "white",
        "black",
    }

    # 1. Existing Tailwind text color class in line (e.g. text-cyan, text-cyan-400)
    tw_match = re.search(r"\btext-([a-z]+)(-[0-9]+)?\b", line_text)
    if tw_match:
        old_shade = tw_match.group(2) or ""
        if color_val.startswith("text-"):
            new_tw_class = color_val
        elif color_val in tailwind_colors:
            shade = (
                old_shade if old_shade else "-500" if color_val not in ("white", "black") else ""
            )
            new_tw_class = f"text-{color_val}{shade}"
        else:
            new_tw_class = f"text-{color_val}"
        return line_text[: tw_match.start()] + new_tw_class + line_text[tw_match.end() :]

    # 2. Existing style="color: ...;"
    style_match = re.search(r'style="([^"]*color:\s*)[^;"]+([^"]*)"', line_text)
    if style_match:
        return (
            line_text[: style_match.start(1)]
            + f"color: {color_val}"
            + line_text[style_match.end(1) :]
        )

    # 3. If selected_text is given and present in line
    if selected_text and selected_text in line_text:
        tw_class = (
            color_val
            if color_val.startswith("text-")
            else f"text-{color_val}-500"
            if color_val in tailwind_colors
            else f"color-{color_val}"
        )
        wrapped = f'<span class="{tw_class}">{selected_text}</span>'
        return line_text.replace(selected_text, wrapped, 1)

    # 4. If at_end is True or instructed 'at the end here'
    m_last = re.search(r"([^<>\n]+)(</[a-zA-Z0-9]+>\s*)$", line_text)
    if m_last:
        end_text = m_last.group(1).rstrip()
        tw_class = (
            color_val
            if color_val.startswith("text-")
            else f"text-{color_val}-500"
            if color_val in tailwind_colors
            else f"color-{color_val}"
        )
        wrapped = f'<span class="{tw_class}">{end_text.strip()}</span>'
        start_pos = m_last.start(1)
        return line_text[:start_pos] + wrapped + line_text[m_last.end(1) :]

    return line_text


def parse_fast_path_edit_instruction(
    instruction: str, selected_text: str = ""
) -> Optional[Tuple[str, Optional[str], str]]:
    """
    Parse a spoken edit instruction into a deterministic operation if possible.

    Returns:
        Optional tuple of (op_type, explicit_source_target, replacement_text)
        where op_type is one of:
          - "replace_selection": replace the selected text (or explicit_source_target within it) with replacement_text
          - "delete_selection": delete the selected text
          - "transform_case": replacement_text is "upper" | "lower" | "title"
    """
    if not instruction or not instruction.strip():
        return None

    # Strip wake prefixes and conversational fillers
    s = instruction.strip()
    s = re.sub(
        r"^(?:(?:hey|hi|ok|okay|yo)\s+)?(?:pal|paul|hal|cal|brok|viv|voicefi)\b[,:\s.!?\-]*",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    s = re.sub(
        r"^(?:(?:uh+|um+|er+|ah+|like|well|so|and|then|also|please|can\s+you|could\s+you|would\s+you|let\'?s|just|go\s+ahead\s+and)\b[,:\s.!?\-]*)+",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()

    s_clean = s.rstrip(".!?").strip()
    s_lower = s_clean.lower()

    # 1. Delete / Remove selection (or explicit trailing punctuation like "remove the period at the end")
    if re.match(
        r"^(?:delete|remove|erase|drop|cut|take\s+out|get\s+rid\s+of)(?:\s+(?:this|that|it|the\s+selection|selected\s+text|this\s+part|this\s+word|this\s+sentence|this\s+line))?$",
        s_lower,
        re.IGNORECASE,
    ):
        return ("delete_selection", None, "")

    m_del_punct = re.match(
        r"^(?:delete|remove|erase|drop|cut|take\s+out|get\s+rid\s+of)\s+(?:the\s+|a\s+|an\s+|this\s+|that\s+)?(period|dot|full\s+stop|comma|colon|semicolon|semi-colon|em\s*dash|em-dash|en\s*dash|en-dash|dash|hyphen|exclamation(?:\s+point|\s+mark)?|question\s+mark)(?:\s+(?:at\s+the\s+end|from\s+the\s+end|here|on\s+this|in\s+this))?$",
        s_lower,
        re.IGNORECASE,
    )
    if m_del_punct:
        punct_char = resolve_spoken_replacement(m_del_punct.group(1))
        return ("delete_selection", punct_char, "")

    # 1b. Line Break before / after selection or explicit sub-word (including STT phonetic "pad a line break")
    m_lb_after = re.search(
        r"\b(?:(?:add|put|insert|pad|make)\s+(?:a\s+)?line\s*break\s+(?:after|following)|line\s*break\s+(?:after|following))(?:\s+(.+))?$",
        s_clean,
        re.IGNORECASE,
    )
    if m_lb_after:
        sub_w = (m_lb_after.group(1) or "").strip(" \t\n\r.,!?:;\"'")
        if sub_w.lower() in ("", "this", "that", "it", "here", "the selection", "selected text"):
            sub_w = None
        return ("add_line_break_after", sub_w, "<br>")

    m_lb_before = re.search(
        r"\b(?:(?:add|put|insert|pad|make)\s+(?:a\s+)?line\s*break(?:\s+(?:before|for|on|in\s+front\s+of|prior\s+to))?|line\s*break\s+(?:before|for|on|here)|(?:put|move)\s+(?:this|that|it)\s+on\s+(?:a\s+)?new\s+line|break\s+(?:the\s+)?line(?:\s+here|\s+before)?)(?:\s+(.+))?$",
        s_clean,
        re.IGNORECASE,
    )
    if m_lb_before:
        sub_w = (m_lb_before.group(1) or "").strip(" \t\n\r.,!?:;\"'")
        if sub_w.lower() in (
            "",
            "this",
            "that",
            "it",
            "here",
            "now",
            "please",
            "for this",
            "on this",
            "before this",
            "the selection",
            "selected text",
        ):
            sub_w = None
        return ("add_line_break_before", sub_w, "<br>")

    # 2. Case transformations
    if re.match(
        r"^(?:make\s+(?:this|that|it)\s+|convert\s+(?:this|that|it)\s+to\s+|change\s+(?:this|that|it)\s+to\s+)?(?:all\s+caps|uppercase|upper\s+case|capital\s+letters)$",
        s_lower,
    ):
        return ("transform_case", None, "upper")
    if re.match(
        r"^(?:make\s+(?:this|that|it)\s+|convert\s+(?:this|that|it)\s+to\s+|change\s+(?:this|that|it)\s+to\s+)?(?:lowercase|lower\s+case|all\s+lowercase)$",
        s_lower,
    ):
        return ("transform_case", None, "lower")
    if re.match(
        r"^(?:make\s+(?:this|that|it)\s+|convert\s+(?:this|that|it)\s+to\s+|change\s+(?:this|that|it)\s+to\s+|capitalize\s+(?:this|that|it)|title\s*case(?:\s+this|\s+that|\s+it)?)$",
        s_lower,
    ):
        return ("transform_case", None, "title")

    # 3. Color transformations
    m_color = re.match(
        r"^(?:change|switch|swap|set|make|update)\s+(?:the\s+)?(?:text\s+)?color(?:\s+at\s+the\s+end(?:\s+here)?)?\s+(?:to|with|into|as)\s+([a-z0-9_\-]+)$",
        s_clean,
        re.IGNORECASE,
    )
    if m_color:
        new_color = m_color.group(1).strip()
        at_end = "at the end" in s_lower
        return ("change_color", "at_end" if at_end else None, new_color)

    # 4. Direct "change/replace/turn/make/swap this [em-dash/word] to/with/into <replacement>"
    m_this = re.match(
        r"^(?:change|replace|turn|swap|switch|set|update|convert)\s+"
        r"(?:this|that|it|the\s+selected|the\s+selection)(?:\s+([a-z0-9\-\s]{1,24}?))?\s+"
        r"(?:to\s+be|to|with|into|as|for)\s+(.+)$",
        s_clean,
        re.IGNORECASE,
    )
    if m_this:
        sub_target_raw = (m_this.group(1) or "").strip()
        rep_raw = (m_this.group(2) or "").strip()
        explicit_src = None
        if sub_target_raw:
            sub_resolved = resolve_spoken_replacement(sub_target_raw)
            if sub_resolved in SPOKEN_PUNCTUATION_MAP.values():
                explicit_src = sub_resolved
        return ("replace_selection", explicit_src, resolve_spoken_replacement(rep_raw))

    # 4b. "make this <replacement>"
    m_make = re.match(
        r"^make\s+(?:this|that|it)\s+(?:say\s+|read\s+)?(.+)$",
        s_clean,
        re.IGNORECASE,
    )
    if m_make:
        rep_raw = (m_make.group(1) or "").strip()
        resolved = resolve_spoken_replacement(rep_raw)
        adjective_rewrites = re.compile(
            r"^(?:shorter|longer|punchier|better|clearer|simpler|bolder|nicer|more\s+\w+|less\s+\w+|active\s+voice|passive\s+voice|professional|casual|friendly|concise)\b",
            re.IGNORECASE,
        )
        if not adjective_rewrites.search(rep_raw):
            return ("replace_selection", None, resolved)

    # 5. Explicit "change/replace <source> to/with <replacement>"
    m_explicit = re.match(
        r"^(?:change|replace|swap|switch|turn)\s+(?:the\s+)?(.+?)\s+(?:to|with|into|for)\s+(.+)$",
        s_clean,
        re.IGNORECASE,
    )
    if m_explicit:
        src_raw = (m_explicit.group(1) or "").strip()
        rep_raw = (m_explicit.group(2) or "").strip()
        src_resolved = resolve_spoken_replacement(src_raw)
        rep_resolved = resolve_spoken_replacement(rep_raw)
        return ("replace_selection", src_resolved, rep_resolved)

    return None


def _normalize_whitespace_and_entities(s: str) -> str:
    """Decode HTML entities, strip HTML tags, and collapse whitespace for fuzzy line matching."""
    no_tags = re.sub(r"<[^>]+>", "", s)
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def apply_smart_replacement_in_line(
    line_text: str,
    selected_text: str,
    replacement_text: str,
    explicit_subtarget: Optional[str] = None,
) -> Optional[str]:
    """
    Replace `selected_text` (or `explicit_subtarget` inside it) within `line_text`,
    handling HTML entity equivalents (e.g., `—` vs `&mdash;` / `&#8212;`) and
    smart typographic spacing around colons/commas/semicolons replacing dashes.
    """
    target = (selected_text or "").strip()
    if explicit_subtarget and explicit_subtarget in target:
        target_to_swap = explicit_subtarget
    else:
        target_to_swap = target or (explicit_subtarget or "")

    if not target_to_swap:
        return None

    # Build candidate representations in source HTML/code for the target string
    candidates: List[str] = [target_to_swap]
    if "—" in target_to_swap:
        candidates.append(target_to_swap.replace("—", "&mdash;"))
        candidates.append(target_to_swap.replace("—", "&#8212;"))
        candidates.append(target_to_swap.replace("—", "&#x2014;"))
        if target_to_swap == "—":
            candidates.extend(["–", "&ndash;", "&#8211;"])
    elif "–" in target_to_swap:
        candidates.append(target_to_swap.replace("–", "&ndash;"))
        candidates.append(target_to_swap.replace("–", "&#8211;"))
        if target_to_swap == "–":
            candidates.extend(["—", "&mdash;", "&#8212;"])
    if "&" in target_to_swap:
        candidates.append(target_to_swap.replace("&", "&amp;"))

    if (
        replacement_text in ATTACHED_TRAILING_PUNCT
        and target_to_swap not in DASH_VARIANTS
        and any(d in target_to_swap for d in ("—", "–", " - "))
    ):
        for d_char in ("—", "–", " - "):
            if d_char in target_to_swap:
                parts = [p.strip() for p in target_to_swap.split(d_char, 1)]
                if len(parts) == 2 and parts[0] and parts[1]:
                    reconstructed = f"{parts[0]}{replacement_text} {parts[1]}"
                    for cand in candidates:
                        if cand in line_text:
                            return line_text.replace(cand, reconstructed, 1)

    for cand in candidates:
        if cand in line_text:
            if cand in ATTACHED_TRAILING_PUNCT and replacement_text == "":
                m_end_punct = re.search(
                    rf"{re.escape(cand)}(?=(?:\s*</[a-zA-Z0-9]+>)+\s*$|\s*$)", line_text
                )
                if m_end_punct:
                    s_i, e_i = m_end_punct.span()
                    return line_text[:s_i] + line_text[e_i:]

            if cand in DASH_VARIANTS and replacement_text in ATTACHED_TRAILING_PUNCT:
                escaped_cand = re.escape(cand)
                pattern = re.compile(rf"[ \t]*{escaped_cand}[ \t]*")
                m = pattern.search(line_text)
                if m:
                    start_idx, end_idx = m.span()
                    after_char = line_text[end_idx : end_idx + 1]
                    needs_trailing_space = bool(
                        after_char and after_char not in (" ", "\t", "\n", "\r", "<")
                    )
                    smart_rep = (
                        f"{replacement_text} " if needs_trailing_space else replacement_text
                    )
                    return line_text[:start_idx] + smart_rep + line_text[end_idx:]

            return line_text.replace(cand, replacement_text, 1)

    # Case-insensitive replacement fallback
    for cand in candidates:
        cand_esc = re.escape(cand.strip())
        if cand_esc:
            m_ci = re.search(rf"[ \t]*{cand_esc}[ \t]*", line_text, re.IGNORECASE)
            if m_ci:
                start_i, end_i = m_ci.span()
                after_char = line_text[end_i : end_i + 1]
                needs_trailing_space = bool(
                    replacement_text
                    and after_char
                    and after_char not in (" ", "\t", "\n", "\r", "<")
                )
                rep_out = f"{replacement_text} " if needs_trailing_space else replacement_text
                return line_text[:start_i] + rep_out + line_text[end_i:]

    # Fallback: match target_to_swap across inline HTML tags
    words = [re.escape(w) for w in target_to_swap.split() if w]
    if len(words) >= 2:
        tag_gap = r"(?:\s*<[^>]+>\s*|\s+)+"
        last_w = words[-1]
        if last_w.endswith(r"\.") or last_w.endswith(r"\,") or last_w.endswith(r"\:"):
            words[-1] = last_w[:-2] + r"(?:</(?:span|strong|em|b|i|a)>)*" + last_w[-2:]
        pattern_str = tag_gap.join(words)
        try:
            m = re.search(pattern_str, line_text)
            if m:
                return line_text[: m.start()] + replacement_text + line_text[m.end() :]
        except Exception:
            pass

    return None
