"""
Blood-Brain Barrier (Organ 05) — Autonomic Typography, Machine-Mark & Lexical Slop Auto-Heal Engine,
and Walken Acoustic Callouts.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from whole_body_compute.organs.immune_vascular import AntibodyRegistry


def _get_wbc_data_dir() -> Path:
    env_dir = os.environ.get("WBC_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    home = Path.home()
    wbc_dir = home / ".whole_body_compute"
    if wbc_dir.exists():
        return wbc_dir
    vifi_dir = home / ".voicefi"
    if vifi_dir.exists():
        return vifi_dir
    return wbc_dir


DEFAULT_IMMUNE_AUDIT_LOG = _get_wbc_data_dir() / "immune_audit.jsonl"
WALKEN_SFX_DIR = _get_wbc_data_dir() / "sfx"
WALKEN_EM_DASH_SFX = WALKEN_SFX_DIR / "walken_em_dash.wav"
WALKEN_MACHINE_MARK_SFX = WALKEN_SFX_DIR / "walken_machine_mark.wav"
WALKEN_SLOP_SFX = WALKEN_SFX_DIR / "walken_slop.wav"
WALKEN_DEBOUNCE_FILE = Path("/tmp/wbc_walken_sfx_ts")
WALKEN_DEBOUNCE_SEC = 2.0

EM_DASH_CHAR = "\u2014"
_EXTENDED_DASH_CHARS = ("\u2014", "\u2015", "\u2e3a", "\u2e3b")
_EXTENDED_DASH_REPEAT_RE = re.compile(r"[\u2014\u2015\u2e3a\u2e3b]{2,}")
_EXTENDED_DASH_LINE_START_RE = re.compile(r"(?:^|(?<=[\n\r]))[\u2014\u2015\u2e3a\u2e3b][ \t]*")
_EXTENDED_DASH_INLINE_RE = re.compile(r"[ \t]*[\u2014\u2015\u2e3a\u2e3b][ \t]*")
_HTML_EM_DASH_RE = re.compile(r"[ \t]*&(?:mdash|#8212|#x2014);?[ \t]*", re.IGNORECASE)
_SPACED_EN_DASH_RE = re.compile(r"[ \t]+\u2013[ \t]+")
_LINE_START_EN_DASH_RE = re.compile(r"(?:^|(?<=[\n\r]))\u2013[ \t]+")
_REPEAT_EN_DASH_RE = re.compile(r"\u2013{2,}")

_INVISIBLE_MARKS_RE = re.compile(r"[\u200b\u200c\u200d\u2060\u00ad\ufeff]")
_NARROW_NBSP_RE = re.compile(r"\u202f")

CODE_CONFIG_EXTENSIONS = {
    ".py", ".sh", ".bash", ".zsh", ".json", ".yaml", ".yml",
    ".toml", ".js", ".ts", ".tsx", ".jsx", ".css", ".sql",
}
_SMART_DOUBLE_QUOTES_RE = re.compile(r"[\u201c\u201d]")
_SMART_SINGLE_QUOTES_RE = re.compile(r"[\u2018\u2019]")

_ANTITHESIS_RE = re.compile(
    r"\b(?:it['\u2019]?s|this\s+is|that['\u2019]?s)\s+not\s+just\s+(?:a\s+|an\s+|about\s+)?[^.!?\n,;]{2,50}?(?:,|\s+-\s+|\s*[\u2014\u2013]\s*|\s*;\s*)\s*it['\u2019]?s\s+",
    re.IGNORECASE,
)
_RLHF_SLOP_RE = re.compile(
    r"\b(?:delve|delves|delving|tapestry|testament\s+to|multifaceted|symphony\s+of|beacon\s+of|in\s+today['\u2019]?s\s+fast-paced\s+world|let['\u2019]?s\s+dive\s+in)\b",
    re.IGNORECASE,
)
_CJK_TOKEN_LEAK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+")

LEXICAL_SLOP_REPLACEMENTS = [
    (re.compile(r"\bdelve\s+into\b", re.IGNORECASE), "examine"),
    (re.compile(r"\bdelves\s+into\b", re.IGNORECASE), "examines"),
    (re.compile(r"\bdelving\s+into\b", re.IGNORECASE), "examining"),
    (re.compile(r"\bdelve\b", re.IGNORECASE), "explore"),
    (re.compile(r"\bdelves\b", re.IGNORECASE), "explores"),
    (re.compile(r"\bdelving\b", re.IGNORECASE), "exploring"),
    (re.compile(r"\btapestry\s+of\b", re.IGNORECASE), "mix of"),
    (re.compile(r"\btestament\s+to\b", re.IGNORECASE), "proof of"),
    (re.compile(r"\bmultifaceted\b", re.IGNORECASE), "complex"),
    (re.compile(r"\bsymphony\s+of\b", re.IGNORECASE), "harmony of"),
    (re.compile(r"\bbeacon\s+of\b", re.IGNORECASE), "example of"),
    (re.compile(r"\bin\s+today['\u2019]?s\s+fast-paced\s+world\b", re.IGNORECASE), "today"),
    (re.compile(r"\blet['\u2019]?s\s+dive\s+in\b", re.IGNORECASE), "here is how it works"),
]


def auto_heal_em_dash(text: str) -> Tuple[str, int]:
    """
    Autonomic Typography Reflex: detect and scrub Unicode em-dashes (U+2014),
    horizontal bars (U+2015), two-em/three-em dashes (U+2E3A, U+2E3B), HTML entities,
    and spaced clause-break en-dashes (U+2013) into clean ' - ' in <0.5ms.
    Preserves unspaced numeric/date/line ranges (e.g. '2020\u20132026', '1065\u20131071').
    """
    if not text:
        return text, 0

    has_ext_dash = any(c in text for c in _EXTENDED_DASH_CHARS)
    has_html_dash = bool(_HTML_EM_DASH_RE.search(text))
    has_spaced_en = bool(
        _SPACED_EN_DASH_RE.search(text)
        or _LINE_START_EN_DASH_RE.search(text)
        or _REPEAT_EN_DASH_RE.search(text)
    )

    if not (has_ext_dash or has_html_dash or has_spaced_en):
        return text, 0

    count = (
        sum(text.count(c) for c in _EXTENDED_DASH_CHARS)
        + len(_HTML_EM_DASH_RE.findall(text))
        + len(_SPACED_EN_DASH_RE.findall(text))
        + len(_LINE_START_EN_DASH_RE.findall(text))
        + len(_REPEAT_EN_DASH_RE.findall(text))
    )

    healed = text
    if has_ext_dash:
        healed = _EXTENDED_DASH_REPEAT_RE.sub(lambda m: "-" * len(m.group(0)), healed)
        healed = _EXTENDED_DASH_LINE_START_RE.sub("- ", healed)
        healed = re.sub(r"[ \t]*[\u2014\u2015\u2e3a\u2e3b][ \t]*(?=[\n\r]|$)", " -", healed)
        healed = _EXTENDED_DASH_INLINE_RE.sub(" - ", healed)

    if has_html_dash:
        healed = _HTML_EM_DASH_RE.sub(" - ", healed)

    if has_spaced_en:
        healed = _REPEAT_EN_DASH_RE.sub(lambda m: "-" * len(m.group(0)), healed)
        healed = _LINE_START_EN_DASH_RE.sub("- ", healed)
        healed = re.sub(r"[ \t]+\u2013[ \t]*(?=[\n\r]|$)", " -", healed)
        healed = _SPACED_EN_DASH_RE.sub(" - ", healed)

    if healed in (" -", "- "):
        healed = "-"

    return healed, max(1, count) if text != healed else 0


def auto_heal_invisible_marks(text: str) -> Tuple[str, int]:
    """
    Autonomic Watermark Reflex: strip zero-width spaces (U+200B), ZWNJ (U+200C),
    ZWJ (U+200D), word joiners (U+2060), soft hyphens (U+00AD), and inline BOMs (U+FEFF).
    Normalizes narrow no-break space (U+202F) to standard ASCII space.
    """
    if not text:
        return text, 0
    inv_count = len(_INVISIBLE_MARKS_RE.findall(text))
    nbsp_count = len(_NARROW_NBSP_RE.findall(text))
    total = inv_count + nbsp_count
    if total == 0:
        return text, 0
    healed = _INVISIBLE_MARKS_RE.sub("", text)
    healed = _NARROW_NBSP_RE.sub(" ", healed)
    return healed, total


def auto_heal_smart_quotes_in_code(text: str, file_path: str = "") -> Tuple[str, int]:
    """
    Code/Config Smart-Quote Auto-Heal: convert curly double quotes (U+201C, U+201D)
    to '\"' and curly single quotes (U+2018, U+2019) to '\'' in code and config files.
    Skips if file_path is immune.py or a test file.
    """
    if not text:
        return text, 0
    if file_path:
        p = Path(file_path)
        name = p.name.lower()
        if (
            name.endswith("immune.py")
            or name.endswith("immune_hygiene.py")
            or name.startswith("test_")
            or name.endswith("_test.py")
        ):
            return text, 0
        if p.suffix.lower() not in CODE_CONFIG_EXTENSIONS:
            return text, 0

    count = len(_SMART_DOUBLE_QUOTES_RE.findall(text)) + len(_SMART_SINGLE_QUOTES_RE.findall(text))
    if count == 0:
        return text, 0
    healed = _SMART_DOUBLE_QUOTES_RE.sub('"', text)
    healed = _SMART_SINGLE_QUOTES_RE.sub("'", healed)
    return healed, count


def detect_and_heal_lexical_machine_marks(
    text: str, auto_heal: bool = True
) -> Tuple[str, int, List[str]]:
    """
    Lexical & Structural AI Slop Detector & Healer:
    Detects high-signal LLM tells (antithesis cliches, RLHF slop words like 'delve',
    'tapestry', 'testament to', 'multifaceted', 'in today\'s fast-paced world', etc.).
    When auto_heal=True, cleans and replaces slop phrases with crisp natural prose.
    Returns (healed_text, total_modifications, triggers_found).
    """
    if not text:
        return text, 0, []

    triggers: List[str] = []
    anti_matches = _ANTITHESIS_RE.findall(text)
    if anti_matches:
        triggers.append("antithesis_cliche")

    slop_matches = _RLHF_SLOP_RE.findall(text)
    for sm in slop_matches:
        sm_norm = sm.lower()
        if sm_norm not in triggers:
            triggers.append(sm_norm)

    if not triggers:
        return text, 0, []

    if not auto_heal:
        return text, 0, triggers

    healed = text
    count = 0

    if anti_matches:
        def _heal_anti(m: re.Match) -> str:
            full = m.group(0)
            return re.sub(
                r"^(?:it['\u2019]?s|this\s+is|that['\u2019]?s)\s+not\s+just\s+",
                "Beyond ",
                full,
                flags=re.IGNORECASE,
            )
        healed, n_anti = _ANTITHESIS_RE.subn(_heal_anti, healed)
        count += n_anti

    for pat, repl in LEXICAL_SLOP_REPLACEMENTS:
        healed, n_repl = pat.subn(repl, healed)
        count += n_repl

    return healed, count, triggers


def auto_heal_cjk_token_leak(text: str) -> Tuple[str, int]:
    """
    Multilingual / CJK Token Leakage Scrubber:
    Strips stray Chinese/CJK characters leaked at the end or within English responses.
    """
    if not text:
        return text, 0
    matches = _CJK_TOKEN_LEAK_RE.findall(text)
    if not matches:
        return text, 0
    count = sum(len(m) for m in matches)
    healed = _CJK_TOKEN_LEAK_RE.sub("", text)
    healed = re.sub(r"[ \t]{2,}", " ", healed)
    return healed, count


def auto_heal_machine_marks(
    text: str, file_path: str = "", heal_lexical: bool = False
) -> Dict[str, Any]:
    """
    Unified Multi-Vector Machine-Mark Auto-Heal:
    Coordinates extended dash healing, invisible watermark stripping, smart quote
    straightening in code, CJK token leakage scrubbing, and lexical slop detection/healing.
    """
    if not text:
        return {
            "text": text,
            "total_healed": 0,
            "em_dash_healed": 0,
            "invisible_healed": 0,
            "smart_quote_healed": 0,
            "lexical_healed": 0,
            "cjk_healed": 0,
            "triggers": [],
        }

    triggers: List[str] = []
    healed_text, dash_cnt = auto_heal_em_dash(text)
    if dash_cnt > 0:
        triggers.append("typography_dash")

    healed_text, inv_cnt = auto_heal_invisible_marks(healed_text)
    if inv_cnt > 0:
        triggers.append("invisible_mark")

    healed_text, sq_cnt = auto_heal_smart_quotes_in_code(healed_text, file_path=file_path)
    if sq_cnt > 0:
        triggers.append("smart_quote")

    healed_text, cjk_cnt = auto_heal_cjk_token_leak(healed_text)
    if cjk_cnt > 0:
        triggers.append("cjk_token_leak")

    is_exempt = False
    if file_path:
        f_lower = Path(file_path).name.lower()
        if (
            f_lower.endswith("immune.py")
            or f_lower.endswith("immune_hygiene.py")
            or f_lower in ("body-immune-hook", "body_immune_hook.sh", "body-immune-watchdog")
            or "editorial_carousel_studio.html" in f_lower
            or f_lower.startswith("test_")
            or f_lower.endswith("_test.py")
        ):
            is_exempt = True

    if is_exempt:
        return {
            "text": text,
            "total_healed": 0,
            "em_dash_healed": 0,
            "invisible_healed": 0,
            "smart_quote_healed": 0,
            "lexical_healed": 0,
            "cjk_healed": 0,
            "triggers": [],
        }

    should_heal_lexical = heal_lexical and not is_exempt
    healed_text, lex_cnt, lex_triggers = detect_and_heal_lexical_machine_marks(
        healed_text, auto_heal=should_heal_lexical
    )
    if lex_triggers:
        triggers.extend(lex_triggers)

    total = dash_cnt + inv_cnt + sq_cnt + lex_cnt + cjk_cnt
    return {
        "text": healed_text,
        "total_healed": total,
        "em_dash_healed": dash_cnt,
        "invisible_healed": inv_cnt,
        "smart_quote_healed": sq_cnt,
        "lexical_healed": lex_cnt,
        "cjk_healed": cjk_cnt,
        "triggers": triggers,
    }


def scrub_machine_marks_in_structure(
    obj: Any, file_path: str = "", heal_lexical: bool = False
) -> Tuple[Any, int, Dict[str, Any]]:
    """
    Recursively scans strings inside dicts/lists or single strings,
    auto-healing all machine mark vectors and returning (scrubbed_obj, total_count, stats).
    """
    if isinstance(obj, str):
        res = auto_heal_machine_marks(obj, file_path=file_path, heal_lexical=heal_lexical)
        return res["text"], res["total_healed"], res
    elif isinstance(obj, dict):
        new_dict = {}
        total = 0
        agg_stats = {
            "em_dash_healed": 0,
            "invisible_healed": 0,
            "smart_quote_healed": 0,
            "lexical_healed": 0,
            "cjk_healed": 0,
            "triggers": [],
        }
        for k, v in obj.items():
            new_v, c, st = scrub_machine_marks_in_structure(v, file_path=file_path, heal_lexical=heal_lexical)
            new_dict[k] = new_v
            total += c
            for sk in ("em_dash_healed", "invisible_healed", "smart_quote_healed", "lexical_healed", "cjk_healed"):
                agg_stats[sk] += st.get(sk, 0)
            for trg in st.get("triggers", []):
                if trg not in agg_stats["triggers"]:
                    agg_stats["triggers"].append(trg)
        return new_dict, total, agg_stats
    elif isinstance(obj, list):
        new_list = []
        total = 0
        agg_stats = {
            "em_dash_healed": 0,
            "invisible_healed": 0,
            "smart_quote_healed": 0,
            "lexical_healed": 0,
            "cjk_healed": 0,
            "triggers": [],
        }
        for item in obj:
            new_item, c, st = scrub_machine_marks_in_structure(item, file_path=file_path, heal_lexical=heal_lexical)
            new_list.append(new_item)
            total += c
            for sk in ("em_dash_healed", "invisible_healed", "smart_quote_healed", "lexical_healed", "cjk_healed"):
                agg_stats[sk] += st.get(sk, 0)
            for trg in st.get("triggers", []):
                if trg not in agg_stats["triggers"]:
                    agg_stats["triggers"].append(trg)
        return new_list, total, agg_stats
    return obj, 0, {}


def scrub_em_dashes_in_structure(obj: Any) -> Tuple[Any, int]:
    """
    Recursively scans strings inside dicts/lists or single strings,
    auto-healing em-dashes and returning (scrubbed_obj, total_count).
    Zero prompt token cost.
    """
    scrubbed, total, _ = scrub_machine_marks_in_structure(obj, heal_lexical=False)
    return scrubbed, total


def _check_walken_audio_debounce() -> bool:
    """Non-blocking 2.0s audio debounce check to prevent afplay overlap."""
    now = time.time()
    immune_mod = sys.modules.get("whole_body_compute.organs.immune")
    debounce_file = getattr(immune_mod, "WALKEN_DEBOUNCE_FILE", WALKEN_DEBOUNCE_FILE)
    debounce_sec = getattr(immune_mod, "WALKEN_DEBOUNCE_SEC", WALKEN_DEBOUNCE_SEC)
    try:
        if debounce_file.is_file():
            raw = debounce_file.read_text(encoding="utf-8").strip()
            if raw:
                last_ts = float(raw)
                if now - last_ts < debounce_sec:
                    return False
        debounce_file.write_text(str(now), encoding="utf-8")
        return True
    except Exception:
        return True


def _trigger_walken_machine_mark_callout(
    vector: str = "typography_em_dash",
    snippet: str = "",
    record_antigen: bool = True,
) -> bool:
    """
    Fire non-blocking (<1ms) Walken acoustic reflex with 2.0s debounce:
    - walken_slop.wav: if 'delve' in snippet
    - walken_machine_mark.wav: for lexical, invisible, smart-quote, CJK marks
    - walken_em_dash.wav: for typography em/en dashes
    Optionally logs an antigen in AntibodyRegistry.
    """
    if record_antigen:
        try:
            AntibodyRegistry.get_instance().record_antigen(
                vector=vector,
                payload=snippet[:120] or EM_DASH_CHAR,
                reason=f"Autonomic Auto-Heal scrubbed machine mark ({vector}) in PreToolUse",
                stage=1,
            )
        except Exception:
            pass

    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("VOICEFI_MOCK_AUDIO") == "1":
        return False

    if not _check_walken_audio_debounce():
        return False

    try:
        s_lower = (snippet or "").lower()
        if "delve" in s_lower or "delves" in s_lower or "delving" in s_lower:
            sfx = WALKEN_SLOP_SFX if WALKEN_SLOP_SFX.is_file() else WALKEN_EM_DASH_SFX
        elif vector in ("lexical_slop", "invisible_mark", "smart_quote", "cjk_token_leak"):
            sfx = WALKEN_MACHINE_MARK_SFX if WALKEN_MACHINE_MARK_SFX.is_file() else WALKEN_EM_DASH_SFX
        elif vector in ("typography_em_dash", "typography_dash"):
            sfx = WALKEN_EM_DASH_SFX if WALKEN_EM_DASH_SFX.is_file() else WALKEN_MACHINE_MARK_SFX
        else:
            sfx = WALKEN_EM_DASH_SFX if WALKEN_EM_DASH_SFX.is_file() else WALKEN_MACHINE_MARK_SFX

        if sfx.is_file():
            import subprocess

            subprocess.Popen(
                ["afplay", str(sfx)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
    except Exception:
        pass
    return False


def _trigger_walken_em_dash_callout(
    snippet: str = "",
    record_antigen: bool = True,
) -> bool:
    """Backward-compatible Walken callout delegating to multi-vector reflex."""
    return _trigger_walken_machine_mark_callout(
        vector="typography_em_dash",
        snippet=snippet,
        record_antigen=record_antigen,
    )
