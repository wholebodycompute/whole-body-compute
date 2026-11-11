"""
Intermediate Representation (IR), Lexical Denoiser, PbE String Synthesizer,
and Hoare Logic Contract Exception Types for the VoiceFi Pal Compiler.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse


# ==============================================================================
# PILLAR 1 & 2 ENUMS AND CORE DATA STRUCTURES
# ==============================================================================


class ExecutionTier(IntEnum):
    """
    Strength-Reduction Execution Hierarchy (ordered from highest reliability / lowest latency to lowest).
    """

    TIER_0_LOCAL = 0  # Pure Python / Filesystem / SQLite / Regex (<10ms, 0 network)
    TIER_1_HEADLESS_API = 10  # Direct Authenticated REST API / CLI / MCP (Drive, Sheets, Slack, gh, webmail)
    TIER_1_5_BROWSER_DOM_JS = 15  # Active-Browser Session DOM/JS Bridge (inherits live auth cookies, 0 mouse drift)
    TIER_2_SEMANTIC_UI = 20  # macOS Accessibility Tree (AXRole+AXTitle) / System Events menu_path
    TIER_3_COORDINATE_GUI = 30  # Raw Quartz (x, y) screen/window coordinates (fallback for opaque GPU canvases)


TIER_RELIABILITY_WEIGHTS: Dict[ExecutionTier, float] = {
    ExecutionTier.TIER_0_LOCAL: 1.00,
    ExecutionTier.TIER_1_HEADLESS_API: 0.99,
    ExecutionTier.TIER_1_5_BROWSER_DOM_JS: 0.95,
    ExecutionTier.TIER_2_SEMANTIC_UI: 0.86,
    ExecutionTier.TIER_3_COORDINATE_GUI: 0.45,
}


class IRPrimitiveType(str, Enum):
    """Canonical Semantic Intermediate Representation (IR) Intent Primitives."""

    FILE_TRANSFORM = "file_transform"
    WEB_COLLECTION_EXTRACT = "web_collection_extract"
    ENRICH_LOOKUP = "enrich_lookup"
    CLOUD_STORAGE_UPLOAD = "cloud_storage_upload"
    SPREADSHEET_ROW_MUTATE = "spreadsheet_row_mutate"
    DATABASE_RECORD = "database_record"
    MESSAGE_DISPATCH = "message_dispatch"
    GIT_RELEASE = "git_release"
    EMAIL_TRIAGE = "email_triage"
    NATIVE_APP_CONTROL = "native_app_control"


@dataclass
class SSAValue:
    """Static Single Assignment (SSA) symbolic register value."""

    name: str  # e.g. "$v1_file", "$v2_clean_stem", "$v3_leads"
    register_alias: str  # e.g. "$selected_file", "$modal_buffer", "$clipboard"
    producer_step: int
    symbolic_expr: str  # e.g. "RegexSub($v1_file.stem, '^APPROVED\\s+|\\s*-\\s*\\d+\\s*SFR$', '')"
    concrete_example: Optional[str] = None
    transform_rule: Optional[str] = None


@dataclass
class DataFlowEdge:
    """Directed def-use edge linking a producer step/node to a consumer step/node via an SSA register."""

    ssa_var: str
    register: str  # e.g. "$clipboard", "$selected_file"
    def_step: int
    use_step: int
    def_app: str
    use_app: str
    symbolic_expr: str
    concrete_value: Optional[str] = None


@dataclass
class LoopExecutionPolicy:
    """Politeness pacing, rate-limit backoff, and crash-safe checkpointing for set-based loops."""

    rate_limit_rps: float = 5.0
    jitter_ms: int = 120
    exponential_backoff_on_429: bool = True
    max_retries: int = 3
    checkpoint_key_field: str = "id"
    checkpoint_skip_predicate: str = "status in ('Enriched', 'Contacted', 'Done')"


@dataclass
class BlastRadiusPolicy:
    """Safety guardrails for batch mutations and external message dispatches."""

    max_unattended_batch: int = 25
    require_template_validation: bool = True
    require_dedupe_ledger: bool = True
    supports_dry_run: bool = True


@dataclass
class HoareContract:
    """
    Formal Hoare Triple {P} C {Q} wrapping a compiled workflow or IR primitive.
    """

    name: str
    preconditions: List[str] = field(default_factory=list)
    idempotency_strategy: str = "UPSERT_BY_NATURAL_KEY"
    natural_key_fields: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    blast_radius: Optional[BlastRadiusPolicy] = None


@dataclass
class IRNode:
    """A single canonical instruction node in the Semantic IR."""

    node_id: str
    primitive: IRPrimitiveType
    description: str
    source_step_numbers: List[int] = field(default_factory=list)
    app_context: str = ""
    window_context: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    preferred_tier: ExecutionTier = ExecutionTier.TIER_1_HEADLESS_API
    fallback_tiers: List[ExecutionTier] = field(default_factory=list)
    contract: Optional[HoareContract] = None
    loop_wrapper: Optional[Dict[str, Any]] = None


@dataclass
class IRMapLoop:
    """Set-based loop construct induced from a single-item demonstration + spoken quantifier."""

    loop_id: str
    quantifier_phrase: str
    collection_source: str
    item_var: str = "$item"
    dedupe_key: str = "phone_or_domain"
    policy: LoopExecutionPolicy = field(default_factory=LoopExecutionPolicy)
    body_node_ids: List[str] = field(default_factory=list)


@dataclass
class DenoisingReport:
    """Telemetry from Pass 1 (TraceDenoiser)."""

    raw_step_count: int
    denoised_step_count: int
    boundary_self_steps_pruned: int = 0
    negative_intent_steps_pruned: int = 0
    key_mashes_collapsed: int = 0
    aborted_scrubs_pruned: int = 0
    pruned_step_numbers: List[int] = field(default_factory=list)


@dataclass
class IRProgram:
    """Complete decompiled Intermediate Representation of a demonstrated workflow."""

    workflow_id: str
    workflow_name: str
    archetype: str
    denoising: DenoisingReport
    ssa_values: Dict[str, SSAValue] = field(default_factory=dict)
    data_flow_edges: List[DataFlowEdge] = field(default_factory=list)
    loops: List[IRMapLoop] = field(default_factory=list)
    nodes: List[IRNode] = field(default_factory=list)
    voice_notes: List[str] = field(default_factory=list)


@dataclass
class CompilationReport:
    """Explainability and reliability summary attached to a compiled PalWorkflowRecipe."""

    archetype: str
    raw_step_count: int
    denoised_step_count: int
    ir_node_count: int
    compiled_step_count: int
    dominant_tier: str
    tier_breakdown: Dict[str, int]
    reliability_score: float
    ssa_edges_resolved: int
    loops_induced: int
    contracts_attached: List[str]
    fallback_ladder: List[str]
    suggested_alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# PASS 1: LEXICAL DENOISER & PEEPHOLE FILTER
# ==============================================================================


class TraceDenoiser:
    """
    Pass 1 of the Pal Compiler:
    1. Strips boundary self-reference steps targeting Antigravity/VoiceFi used to start/stop training.
    2. Strips micro-spans annotated with spoken negative intent ("normally not part of workflow", "oops").
    3. Collapses consecutive identical clipboard/hotkey key-mashes (e.g. 7x Cmd+C -> 1x Cmd+C).
    4. Eliminates aborted arrow-key scrubbing sequences that immediately terminate in Escape.
    """

    SELF_APPS: Set[str] = {"antigravity", "voicefi", "voicefi pal"}
    STOP_PHRASES: Tuple[str, ...] = (
        "stop training",
        "finish training",
        "cancel training",
        "end training",
    )
    NEGATIVE_VOICE_PATTERNS: Tuple[re.Pattern, ...] = (
        re.compile(r"\b(?:normally\s+)?not\s+(?:be\s+)?part\s+of\s+(?:the\s+)?workflow\b", re.I),
        re.compile(r"\b(?:ignore|skip|scratch|delete)\s+(?:that|this|those)\b", re.I),
        re.compile(r"\b(?:wait\s+no|oops|my\s+bad|mistake|accidental)\b", re.I),
    )

    @classmethod
    def is_negative_voice_note(cls, note: Optional[str]) -> bool:
        if not note:
            return False
        # Inspect only the latest segment if pipe-delimited
        segments = [seg.strip() for seg in note.split("|") if seg.strip()]
        check_text = segments[-1] if segments else note
        return any(pat.search(check_text) for pat in cls.NEGATIVE_VOICE_PATTERNS)

    @staticmethod
    def _safe_params(step: Any) -> Dict[str, Any]:
        p = getattr(step, "params", None)
        return p if isinstance(p, dict) else {}

    @classmethod
    def _safe_elem(cls, step: Any) -> Dict[str, Any]:
        p = cls._safe_params(step)
        el = p.get("element")
        return el if isinstance(el, dict) else {}

    @classmethod
    def denoise(cls, raw_steps: List[Any]) -> Tuple[List[Any], DenoisingReport]:
        raw_count = len(raw_steps or [])
        if raw_count == 0:
            return [], DenoisingReport(raw_step_count=0, denoised_step_count=0)

        steps = [s for s in raw_steps if s is not None]
        pruned_nums: List[int] = []
        boundary_pruned = 0
        neg_pruned = 0
        mashes_collapsed = 0
        scrubs_pruned = 0

        # 1. Strip leading self-app focus/clicks before the first real target app interaction
        start_idx = 0
        while start_idx < len(steps):
            s = steps[start_idx]
            s_params = cls._safe_params(s)
            app = str(getattr(s, "app_name", "") or s_params.get("app_name", "") or "").strip().lower()
            elem = cls._safe_elem(s)
            elem_title = str(elem.get("title") or "").strip().lower()
            # Keep if clicking Finder or another app icon in Dock
            if elem_title and elem_title not in cls.SELF_APPS:
                break
            if app in cls.SELF_APPS or (
                not app
                and getattr(s, "action", "") == "click"
                and not str(getattr(s, "voice_note", None) or "").strip()
            ):
                pruned_nums.append(int(getattr(s, "step_number", start_idx + 1) or 0))
                boundary_pruned += 1
                start_idx += 1
            else:
                break

        # 2. Strip trailing self-app / "stop training" steps at the end of the recording
        end_idx = len(steps)
        for i in range(len(steps) - 1, start_idx - 1, -1):
            s = steps[i]
            s_params = cls._safe_params(s)
            text_val = str(s_params.get("text") or "").strip().lower()
            elem = cls._safe_elem(s)
            elem_title = str(elem.get("title") or "").strip().lower()
            if any(p in text_val for p in cls.STOP_PHRASES) or elem_title == "voicefi":
                # Prune from the click on VoiceFi / stop training to the end
                cut_from = i
                while cut_from > start_idx:
                    prev_s = steps[cut_from - 1]
                    prev_params = cls._safe_params(prev_s)
                    prev_elem = cls._safe_elem(prev_s)
                    prev_title = str(prev_elem.get("title") or "").strip().lower()
                    if prev_title == "voicefi" or (
                        getattr(prev_s, "action", "") == "click"
                        and not prev_title
                        and (cut_from - i) >= -3
                        and str(prev_params.get("text") or "") == ""
                    ):
                        cut_from -= 1
                    else:
                        break
                for k in range(cut_from, end_idx):
                    pruned_nums.append(int(getattr(steps[k], "step_number", k + 1) or 0))
                    boundary_pruned += 1
                end_idx = cut_from
                break

        working = steps[start_idx:end_idx]

        # 3. Strip steps carrying explicit spoken negative-intent annotations
        filtered_neg: List[Any] = []
        for s in working:
            note = getattr(s, "voice_note", None)
            if cls.is_negative_voice_note(note) and getattr(s, "action", "") in (
                "click",
                "right_click",
                "press_key",
            ):
                pruned_nums.append(int(getattr(s, "step_number", 0) or 0))
                neg_pruned += 1
                continue
            filtered_neg.append(s)

        # 4. Peephole pass: collapse consecutive identical idempotent hotkeys (e.g. 7x Cmd+C -> 1x Cmd+C)
        #    and prune arrow-key scrubbing immediately aborted by Escape.
        peephole_out: List[Any] = []
        i = 0
        while i < len(filtered_neg):
            curr = filtered_neg[i]
            curr_params = cls._safe_params(curr)

            # Check for arrow-key scrubbing aborted by Escape
            if getattr(curr, "action", "") == "press_key" and curr_params.get("key") in (
                "Left",
                "Right",
                "Up",
                "Down",
            ):
                j = i
                while (
                    j < len(filtered_neg)
                    and getattr(filtered_neg[j], "action", "") == "press_key"
                    and cls._safe_params(filtered_neg[j]).get("key") in ("Left", "Right", "Up", "Down")
                ):
                    j += 1
                # If >= 3 arrow keys are followed immediately (or after 1 uncommitted click) by Escape
                if (j - i) >= 3:
                    next_idx = j
                    if (
                        next_idx < len(filtered_neg)
                        and getattr(filtered_neg[next_idx], "action", "") == "click"
                        and next_idx + 1 < len(filtered_neg)
                        and getattr(filtered_neg[next_idx + 1], "action", "") == "press_key"
                        and cls._safe_params(filtered_neg[next_idx + 1]).get("key") == "Escape"
                    ):
                        for k in range(i, next_idx + 2):
                            pruned_nums.append(int(getattr(filtered_neg[k], "step_number", 0) or 0))
                            scrubs_pruned += 1
                        i = next_idx + 2
                        continue
                    elif (
                        next_idx < len(filtered_neg)
                        and getattr(filtered_neg[next_idx], "action", "") == "press_key"
                        and cls._safe_params(filtered_neg[next_idx]).get("key") == "Escape"
                    ):
                        for k in range(i, next_idx + 1):
                            pruned_nums.append(int(getattr(filtered_neg[k], "step_number", 0) or 0))
                            scrubs_pruned += 1
                        i = next_idx + 1
                        continue

            # Check for consecutive identical Cmd+C / Cmd+A hotkeys
            if getattr(curr, "action", "") == "hotkey" and curr_params.get("combo") in ("Cmd+C", "Cmd+A"):
                combo = curr_params.get("combo")
                peephole_out.append(curr)
                j = i + 1
                while (
                    j < len(filtered_neg)
                    and getattr(filtered_neg[j], "action", "") == "hotkey"
                    and cls._safe_params(filtered_neg[j]).get("combo") == combo
                ):
                    j_params = cls._safe_params(filtered_neg[j])
                    if j_params.get("clipboard_text") and not curr_params.get("clipboard_text"):
                        if isinstance(getattr(curr, "params", None), dict):
                            curr.params["clipboard_text"] = j_params["clipboard_text"]
                    pruned_nums.append(int(getattr(filtered_neg[j], "step_number", 0) or 0))
                    mashes_collapsed += 1
                    j += 1
                i = j
                continue

            peephole_out.append(curr)
            i += 1

        report = DenoisingReport(
            raw_step_count=raw_count,
            denoised_step_count=len(peephole_out),
            boundary_self_steps_pruned=boundary_pruned,
            negative_intent_steps_pruned=neg_pruned,
            key_mashes_collapsed=mashes_collapsed,
            aborted_scrubs_pruned=scrubs_pruned,
            pruned_step_numbers=sorted(pruned_nums),
        )
        return peephole_out, report


# ==============================================================================
# FLASHFILL-STYLE PROGRAMMING-BY-EXAMPLE (PbE) STRING SYNTHESIZER
# ==============================================================================


class PbEStringSynthesizer:
    """
    Synthesizes deterministic Tier-0 string transformation rules from concrete
    (input_example, output_example) pairs captured across Finder/Modal/Clipboard diffs
    or spoken voice descriptions.
    """

    @classmethod
    def induce_transform(
        cls,
        before_str: Optional[str],
        after_str: Optional[str],
        voice_hint: str = "",
    ) -> Tuple[str, str]:
        """
        Returns (rule_name, symbolic_expression) induced from before/after strings or voice hints.
        """
        b = str(before_str or "").replace("\x00", "")[:2048].strip()
        a = str(after_str or "").replace("\x00", "")[:2048].strip()
        hint = str(voice_hint or "").replace("\x00", "")[:4096].lower()

        # 1. Exact PbE match: DFPB / enterprise_corpus APPROVED + SFR count suffix stripping
        if b and a:
            b_no_ext = re.sub(r"\.[a-zA-Z0-9]{2,4}$", "", b)
            a_no_ext = re.sub(r"\.[a-zA-Z0-9]{2,4}$", "", a)
            sfr_cleaned = cls.apply_rule("strip_sfr_package_affixes", b)
            if sfr_cleaned == a_no_ext or f"{sfr_cleaned}.csv" == a:
                return (
                    "strip_sfr_package_affixes",
                    r"RegexSub($input.stem, '^APPROVED\s+|\s*-\s*\d+\s*SFR$', '')",
                )

            # 2. PbE match: URL -> Root Domain extraction (common in Lead Gen enrichment)
            if ("http://" in b or "https://" in b or "www." in b) and "." in a and "/" not in a:
                dom = cls.apply_rule("extract_root_domain", b)
                if dom == a.lower():
                    return ("extract_root_domain", "ExtractRootDomain($input)")

            # 3. PbE match: Phone number normalization
            digits_b = re.sub(r"\D+", "", b)
            digits_a = re.sub(r"\D+", "", a)
            if len(digits_b) >= 10 and digits_b[-10:] == digits_a[-10:] and b != a:
                return ("normalize_us_phone", "NormalizePhone($input)")

            # 4. General prefix/suffix slice induction (if affixes are static/non-numeric)
            if a_no_ext and a_no_ext in b_no_ext:
                idx = b_no_ext.find(a_no_ext)
                prefix = b_no_ext[:idx]
                suffix = b_no_ext[idx + len(a_no_ext) :]
                if not re.search(r"\d", prefix + suffix):
                    return (
                        "strip_literal_affixes",
                        f"StripAffixes($input, prefix={prefix!r}, suffix={suffix!r})",
                    )

            # 5. Local Neural PbE Synthesis (voicefi-coder-mlx :9379 -> Qwen2.5-Coder-14B :8080)
            induced_rule = cls._induce_neural_regex_rule(b, a, hint)
            if induced_rule is not None:
                return induced_rule

            # Fallback to generalized pattern or literal affixes if neural endpoint is offline
            if a_no_ext and a_no_ext in b_no_ext:
                idx = b_no_ext.find(a_no_ext)
                prefix = b_no_ext[:idx]
                suffix = b_no_ext[idx + len(a_no_ext) :]
                return (
                    "strip_literal_affixes",
                    f"StripAffixes($input, prefix={prefix!r}, suffix={suffix!r})",
                )

        # Fallback to voice-hint inference when concrete strings weren't both captured
        if any(k in hint for k in ("after the date", "after the deep", "sfr", "approved", "count suffix")):
            return (
                "strip_sfr_package_affixes",
                r"RegexSub($input.stem, '^APPROVED\s+|\s*-\s*\d+\s*SFR$', '')",
            )
        if "domain" in hint or "website" in hint:
            return ("extract_root_domain", "ExtractRootDomain($input)")
        if "phone" in hint:
            return ("normalize_us_phone", "NormalizePhone($input)")

        return ("identity_strip", "Strip($input)")

    _DYNAMIC_REGEX_RULES: Dict[str, Tuple[str, str]] = {}

    @classmethod
    def _induce_neural_regex_rule(
        cls, before_str: str, after_str: str, voice_hint: str = ""
    ) -> Optional[Tuple[str, str]]:
        """
        Synthesize and verify a deterministic regex substitution `(pattern, repl)` using on-device
        local models (:9379 voicefi-coder-mlx or :8080 Qwen2.5-Coder-14B) when a novel string
        transformation is demonstrated. Verifies `re.sub(pattern, repl, before_str) == after_str`.
        """
        import urllib.request

        if not before_str or not after_str or before_str == after_str:
            return None

        cache_key = f"{before_str}==>{after_str}"
        if cache_key in cls._DYNAMIC_REGEX_RULES:
            pat, rep = cls._DYNAMIC_REGEX_RULES[cache_key]
            rule_id = f"neural_regex_{abs(hash(cache_key)) % 100000}"
            cls._DYNAMIC_REGEX_RULES[rule_id] = (pat, rep)
            return (rule_id, f"RegexSub($input, {pat!r}, {rep!r})")

        prompt = (
            "Return ONLY a JSON object with keys \"pattern\" and \"repl\" for Python `re.sub(pattern, repl, text)` "
            "that transforms INPUT into OUTPUT. Use general regex classes like `\\d+` for dates/numbers so the "
            "rule generalizes to future dates and counts.\n"
            f"INPUT: {before_str!r}\n"
            f"OUTPUT: {after_str!r}\n"
            f"VOICE HINT: {voice_hint!r}\n"
            "JSON:"
        )
        endpoints = [
            ("http://127.0.0.1:8080/v1/chat/completions", "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit", 8.0),
            ("http://127.0.0.1:9379/v1/chat/completions", "voicefi-coder-mlx", 8.0),
        ]
        for url, model_id, timeout_s in endpoints:
            try:
                body = json.dumps(
                    {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 180,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    if resp.status != 200:
                        continue
                    data = json.loads(resp.read().decode("utf-8"))
                    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                    if "<|im_end|>" in content:
                        content = content.split("<|im_end|>", 1)[0]
                    if "</think>" in content:
                        content = content.split("</think>", 1)[-1]
                    m_json = re.search(r"\{.*?\}", content, re.DOTALL)
                    if not m_json:
                        continue
                    parsed = json.loads(m_json.group(0))
                    pat = str(parsed.get("pattern") or "")
                    rep = str(parsed.get("repl") or "")
                    if pat and re.sub(pat, rep, before_str).strip() == after_str.strip():
                        rule_id = f"neural_regex_{abs(hash(cache_key)) % 100000}"
                        cls._DYNAMIC_REGEX_RULES[cache_key] = (pat, rep)
                        cls._DYNAMIC_REGEX_RULES[rule_id] = (pat, rep)
                        return (rule_id, f"RegexSub($input, {pat!r}, {rep!r})")
            except Exception:
                continue
        return None

    @classmethod
    def apply_rule(cls, rule_name: str, value: str, **kwargs: Any) -> str:
        """Execute a synthesized Tier-0 string transformation in <0.05ms."""
        raw = str(value or "").replace("\x00", "")[:4096].strip()
        if rule_name in cls._DYNAMIC_REGEX_RULES:
            pat, rep = cls._DYNAMIC_REGEX_RULES[rule_name]
            return re.sub(pat, rep, raw).strip()
        if rule_name == "strip_sfr_package_affixes":
            stem = re.sub(r"\.csv$", "", raw, flags=re.IGNORECASE).strip()
            stem = re.sub(r"^APPROVED\s+", "", stem, flags=re.IGNORECASE).strip()
            stem = re.sub(r"\s*-\s*\d+\s*SFR.*$", "", stem, flags=re.IGNORECASE).strip()
            return stem

        if rule_name == "extract_root_domain":
            candidate = raw if "://" in raw else f"https://{raw}"
            parsed = urlparse(candidate)
            host = (parsed.netloc or parsed.path.split("/")[0]).lower().strip()
            host = re.sub(r"^www\.", "", host)
            return host

        if rule_name == "normalize_us_phone":
            digits = re.sub(r"\D+", "", raw)
            if len(digits) == 11 and digits.startswith("1"):
                digits = digits[1:]
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            return raw

        if rule_name == "strip_literal_affixes":
            stem = re.sub(r"\.[a-zA-Z0-9]{2,4}$", "", raw)
            prefix = kwargs.get("prefix", "")
            suffix = kwargs.get("suffix", "")
            if prefix and stem.startswith(prefix):
                stem = stem[len(prefix) :]
            if suffix and stem.endswith(suffix):
                stem = stem[: -len(suffix)]
            return stem.strip()

        return raw


# ==============================================================================
# PILLAR 3: HOARE LOGIC CONTRACT VERIFIER ({P} C {Q})
# ==============================================================================


class ContractViolationError(RuntimeError):
    """Base exception raised when a Hoare Triple {P} C {Q} contract is violated."""


class PreconditionFailedError(ContractViolationError):
    """Raised when Precondition Gate {P} fails before mutation C executes."""


class PostconditionFailedError(ContractViolationError):
    """Raised when Closed-Loop Read-Back Gate {Q} detects a state mismatch after C executes."""


class BlastRadiusExceededError(ContractViolationError):
    """Raised when a batch or outbound mutation violates its BlastRadiusPolicy."""
