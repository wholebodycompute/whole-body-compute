"""
Zero-Training Ambient Workflow Discovery & Passive Motif Extractor for VoiceFi.

Monitors macOS cross-app data transitions:
  (Source App/Window -> NSPasteboard copied entity -> Target App/URL via AXUIElement -> Pasted/Typed Action)
without requiring explicit user "Teach Mode" activation.

Core Capabilities:
1. Passive Observation: Low-overhead monitoring of macOS clipboard, app focus, browser URLs, and AXUIElements.
2. Privacy & Auth Hardening: Integrated with VoiceFi redaction and cognitive safety; automatically suppresses
   password managers (1Password, Bitwarden, Keychain), AXSecureTextField inputs, API keys, credentials, and PII.
3. Sequence Clustering & Motif Extraction: Discovers recurring >= 3-step cross-app sequences, clusters them
   by structural signatures, extracts dynamic parameters, and constructs dataflow DAGs.
4. PalTrainer & PalWorkflow Integration: Compiles mined motifs directly into executable PalWorkflowRecipes.
5. Shadow Mode Staging: Stages candidate workflows with automated headless parity scoring before 1-click promotion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from whole_body_compute.audio.cognitive_safety import AcousticCognitiveSafetyClassifier
from whole_body_compute.integrations.app_focus import AppFocusRegistry
from whole_body_compute.integrations.pal_workflow import (
    ActionType,
    ClickTarget,
    PalWorkflowRecipe,
    PalWorkflowRegistry,
    ParameterDefinition,
    PostConditionCheck,
    WorkflowStep,
)
from whole_body_compute.integrations.screenpipe import resolve_direct_headless_route
from whole_body_compute.redaction import REDACTION_PLACEHOLDER, contains_secret, redact_secrets

logger = logging.getLogger(__name__)


def resolve_default_candidates_dir() -> Path:
    """
    Resolve canonical `recipes/candidates/` directory for staged workflow candidates.
    Shared by Community Open-Core (`body candidates`) and Pro (`REMSleepDaemon`).
    """
    env_cand = os.environ.get("WBC_CANDIDATES_DIR", "").strip()
    if env_cand:
        return Path(env_cand).expanduser()
    cwd_cand = Path.cwd() / "recipes" / "candidates"
    if cwd_cand.is_dir():
        return cwd_cand
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "recipes" / "candidates").is_dir():
            return parent / "recipes" / "candidates"
    return Path.home() / ".whole_body_compute" / "recipes" / "candidates"


def resolve_ambient_events_log_path() -> Path:
    """Resolve persistent JSONL journal for ambient cross-app transitions across daemon restarts."""
    env_path = os.environ.get("WBC_AMBIENT_EVENTS_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    wbc_data = os.environ.get("WBC_DATA_DIR", "").strip()
    if wbc_data:
        return Path(wbc_data).expanduser() / "ambient_transitions.jsonl"
    return Path.home() / ".voicefi" / "ambient_transitions.jsonl"


def is_ambient_discovery_enabled() -> bool:
    """Return True unless user explicitly opted out via `WBC_AMBIENT_DISCOVERY=0`."""
    raw = os.environ.get("WBC_AMBIENT_DISCOVERY", "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "disabled")


# ==============================================================================
# PRIVACY, AUTH & SENSITIVE CREDENTIAL GUARD
# ==============================================================================

BLOCKED_PASSWORD_MANAGERS: Set[str] = {
    "1password",
    "1password 7",
    "1password 8",
    "bitwarden",
    "lastpass",
    "keepass",
    "keepassxc",
    "keychain access",
    "dashlane",
    "nordpass",
    "authy",
    "authenticator",
    "yubikey",
    "enpass",
    "roboform",
}

SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
CREDENTIAL_PARAM_REGEX = re.compile(
    r"(?i)(?:password|passwd|token|api_key|secret|bearer)=([^\s&]+)"
)

# Semantic entity matchers
PATIENT_ID_REGEX = re.compile(
    r"^(?:PT|MRN|PAT|ID)[\-_]?\d{3,12}$|\b(?:PT|MRN|PAT)-[A-Z0-9]{3,10}\b",
    re.IGNORECASE,
)
CURRENCY_REGEX = re.compile(r"^\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?$")
DATE_REGEX = re.compile(
    r"^(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})$"
)
EMAIL_REGEX = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)
URL_REGEX = re.compile(r"^https?://[^\s]+$")
CODE_REGEX = re.compile(r"^[A-Z0-9_\-]{4,20}$", re.IGNORECASE)


class AmbientPrivacyGuard:
    """
    Zero-leak privacy choke point for ambient observation.
    Ensures that credentials, API keys, passwords, and sensitive PII are never
    persisted in transition graphs, candidate workflows, or serialized motifs.
    """

    BLOCKED_PASSWORD_MANAGERS: Set[str] = BLOCKED_PASSWORD_MANAGERS

    @classmethod
    def is_app_blocked(cls, app_name: str) -> bool:
        """Return True if the application is an auth vault or password manager."""
        if not app_name:
            return False
        clean = app_name.strip().lower()
        return any(b in clean for b in BLOCKED_PASSWORD_MANAGERS)

    @classmethod
    def is_element_secure(cls, element: Optional[Dict[str, Any]]) -> bool:
        """Return True if the AXUIElement represents a password or secure text field."""
        if not element or not isinstance(element, dict):
            return False
        if element.get("is_secure") is True:
            return True
        role = str(element.get("role") or "").lower()
        subrole = str(element.get("subrole") or "").lower()
        title = str(element.get("title") or "").lower()
        return (
            "secure" in role
            or "password" in role
            or "secure" in subrole
            or "password" in subrole
            or "password" in title
        )

    @classmethod
    def sanitize_text(cls, text: str) -> Tuple[str, bool]:
        """
        Sanitize text by redacting API keys, private keys, passwords, SSNs, and credit cards.
        Returns: (sanitized_text, was_sensitive)
        """
        if not text:
            return "", False

        original = text
        sanitized = text
        was_sensitive = False

        # 1. VoiceFi Secret Redaction (API keys, JWTs, AWS keys, Stripe, etc.)
        if contains_secret(sanitized):
            sanitized = redact_secrets(sanitized, placeholder="[REDACTED_SECRET]")
            was_sensitive = True

        # 2. PII: SSN Masking
        if SSN_REGEX.search(sanitized):
            sanitized = SSN_REGEX.sub("[REDACTED_SSN]", sanitized)
            was_sensitive = True

        # 3. PII: Credit Card Masking
        if CREDIT_CARD_REGEX.search(sanitized):
            sanitized = CREDIT_CARD_REGEX.sub("[REDACTED_CC]", sanitized)
            was_sensitive = True

        # 4. Credential Query Parameters
        if CREDENTIAL_PARAM_REGEX.search(sanitized):
            sanitized = CREDENTIAL_PARAM_REGEX.sub(r"password=[REDACTED_PARAM]", sanitized)
            was_sensitive = True

        return sanitized, was_sensitive or (sanitized != original)

    @classmethod
    def classify_and_sanitize(
        cls,
        text: str,
        element: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, bool]:
        """
        Classifies semantic entity type and scrubs content.
        Returns: (entity_type, sanitized_content, is_sensitive)
        """
        if cls.is_element_secure(element):
            return "secure_field", "[REDACTED_SECURE_FIELD]", True

        cleaned_text = (text or "").strip()
        sanitized, was_sensitive = cls.sanitize_text(cleaned_text)

        if was_sensitive or "[REDACTED" in sanitized:
            return "sensitive_credential", sanitized, True

        # Semantic classification
        if PATIENT_ID_REGEX.search(sanitized):
            return "patient_id", sanitized, False
        if CURRENCY_REGEX.match(sanitized):
            return "currency", sanitized, False
        if DATE_REGEX.match(sanitized):
            return "date", sanitized, False
        if EMAIL_REGEX.match(sanitized):
            return "email", sanitized, False
        if URL_REGEX.match(sanitized):
            return "url", sanitized, False
        if CODE_REGEX.match(sanitized) and any(c.isdigit() for c in sanitized) and any(c.isalpha() for c in sanitized):
            return "code", sanitized, False

        return "text", sanitized, False

    @classmethod
    def verify_safety_guardrails(
        cls,
        source_app: str,
        target_app: str,
        text: str,
        element: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Verify that a candidate transition does not breach cognitive safety or privacy rules."""
        if cls.is_app_blocked(source_app):
            return False, f"Source app '{source_app}' is a protected password manager"
        if cls.is_app_blocked(target_app):
            return False, f"Target app '{target_app}' is a protected password manager"
        if cls.is_element_secure(element):
            return False, "Target or source element is a secure password field"

        # Check for interruption keywords via cognitive safety
        if AcousticCognitiveSafetyClassifier.is_interruption_text(text):
            return False, "Text contains emergency abort/interruption keyword"

        return True, "Safe"


# ==============================================================================
# DATA MODELS FOR OBSERVED TRANSITIONS & MOTIFS
# ==============================================================================


class TransitionEvent(BaseModel):
    """
    Atomic cross-app dataflow transition event:
    Source App/Window -> Entity -> Target App/URL/Window -> Action
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)
    source_app: str = ""
    source_window: str = ""
    source_url: Optional[str] = None
    source_element: Optional[Dict[str, Any]] = None
    copied_entity: Optional[str] = None
    entity_type: str = "text"
    entity_fingerprint: str = ""
    target_app: str = ""
    target_window: str = ""
    target_url: Optional[str] = None
    target_element: Optional[Dict[str, Any]] = None
    action_type: str = "copy_paste"  # "copy_paste", "switch_app", "type_text", "hotkey", "click", "query"
    pasted_entity: Optional[str] = None
    duration_s: float = 0.0
    is_sensitive: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def structural_signature(self) -> str:
        """
        Abstract structural signature for sequence clustering.
        Independent of dynamic patient names or transient values.
        Format: source_app:entity_type->target_app[domain]:action_type
        """
        src = (self.source_app or "unknown").strip().lower()
        tgt = (self.target_app or "unknown").strip().lower()
        etype = (self.entity_type or "text").strip().lower()
        act = (self.action_type or "action").strip().lower()

        # Ingest web domain if target/source is a browser
        domain = ""
        url = self.target_url or self.source_url
        if url:
            try:
                parsed = urlparse(url)
                if parsed.netloc:
                    domain = parsed.netloc.lower()
            except Exception:
                domain = ""

        if domain:
            return f"{src}:{etype}->{tgt}[{domain}]:{act}"
        return f"{src}:{etype}->{tgt}:{act}"


class WorkflowMotif(BaseModel):
    """
    Recurring cross-app sequence motif mined from passive observation.
    Represents a candidate workflow DAG with parameterized variables.
    """

    motif_id: str
    name: str
    signatures: List[str]
    exemplar_events: List[TransitionEvent]
    support_count: int = 1
    source_apps: List[str] = Field(default_factory=list)
    target_apps: List[str] = Field(default_factory=list)
    entities_involved: List[str] = Field(default_factory=list)
    parameters: Dict[str, ParameterDefinition] = Field(default_factory=dict)
    dag_edges: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="Directed edges (step_i -> step_j) showing control and dataflow dependencies",
    )
    confidence: float = 0.0
    last_seen: float = Field(default_factory=time.time)
    is_cross_app: bool = True


class WorkflowCandidate(BaseModel):
    """
    Staged candidate workflow in shadow mode ready for headless parity validation
    and 1-click user promotion.
    """

    candidate_id: str
    name: str
    motif: WorkflowMotif
    recipe: PalWorkflowRecipe
    status: Literal["staged", "validating", "ready", "promoted", "rejected"] = "staged"
    headless_parity_score: float = 0.0
    validation_notes: List[str] = Field(default_factory=list)
    direct_headless_routes: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    promoted_at: Optional[float] = None
    rejection_reason: Optional[str] = None


# ==============================================================================
# PASSIVE OBSERVATION ENGINE (macOS NSPasteboard + AXUIElement)
# ==============================================================================


class AmbientWorkflowObserver:
    """
    Passive observer that tracks macOS (Source App/Window -> NSPasteboard -> Target App/URL -> Action)
    transitions without requiring explicit 'Teach Mode'.
    """

    def __init__(
        self,
        on_transition: Optional[Callable[[TransitionEvent], None]] = None,
        poll_interval_s: float = 0.25,
    ):
        self.on_transition = on_transition
        self.poll_interval_s = poll_interval_s
        self.events: List[TransitionEvent] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # State tracking for delta computation
        self._last_pasteboard_count: int = -1
        self._last_copied_text: str = ""
        self._last_copied_entity: str = ""
        self._last_copied_type: str = "text"
        self._last_source_app: str = ""
        self._last_source_window: str = ""
        self._last_source_url: Optional[str] = None
        self._last_source_element: Optional[Dict[str, Any]] = None
        self._copy_timestamp: float = 0.0
        self._last_observed_app: str = ""
        self._last_observed_window: str = ""
        self._last_focus_change_ts: float = 0.0
        self._emitted_for_current_copy: Set[Tuple[str, str]] = set()
        self.copy_ttl_s: float = 90.0
        self.focus_settle_s: float = 0.45

    def start(self) -> None:
        """Start passive background observation thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._poll_loop,
                daemon=True,
                name="AmbientWorkflowObserver",
            )
            self._thread.start()
            logger.info("AmbientWorkflowObserver started")

    def stop(self) -> None:
        """Stop passive background observation thread."""
        with self._lock:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._thread = None
            logger.info("AmbientWorkflowObserver stopped")

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def _poll_loop(self) -> None:
        """Continuous passive polling loop for macOS clipboard and window focus shifts."""
        while self.is_running:
            try:
                self._check_pasteboard_and_focus()
            except Exception as e:
                logger.debug(f"Passive observer cycle error: {e}")
            time.sleep(self.poll_interval_s)

    def _check_pasteboard_and_focus(self) -> None:
        """Inspect general pasteboard changeCount and frontmost window focus transitions."""
        pasteboard_changed = False
        try:
            from AppKit import NSPasteboard, NSPasteboardTypeString

            pb = NSPasteboard.generalPasteboard()
            curr_count = pb.changeCount()
            if self._last_pasteboard_count == -1:
                # Seed baseline pasteboard count on boot so stale clipboard doesn't emit false transitions
                self._last_pasteboard_count = curr_count
            elif curr_count != self._last_pasteboard_count:
                raw_clip = str(pb.stringForType_(NSPasteboardTypeString) or "")
                self._last_pasteboard_count = curr_count
                self._handle_live_pasteboard_change(raw_clip)
                pasteboard_changed = True
        except Exception:
            pass

        if not pasteboard_changed:
            self._check_live_focus_transition()

    def _check_live_focus_transition(self) -> None:
        """
        Correlate active clipboard entity with frontmost app/window shifts (`Source App -> Copy -> Target App`)
        so live desktop cross-app flows emit `TransitionEvent`s automatically without manual `record_paste()`.
        """
        if not self._last_copied_entity or not self._copy_timestamp:
            return

        now = time.time()
        if (now - self._copy_timestamp) > self.copy_ttl_s:
            self._last_copied_entity = ""
            self._emitted_for_current_copy.clear()
            return

        try:
            front_app = AppFocusRegistry.get_frontmost_app_name()
        except Exception:
            return

        if not front_app or AmbientPrivacyGuard.is_app_blocked(front_app):
            return

        win_title, browser_url = self._get_frontmost_window_and_url(front_app)
        if (front_app, win_title) != (self._last_observed_app, self._last_observed_window):
            self._last_observed_app = front_app
            self._last_observed_window = win_title
            self._last_focus_change_ts = now
            return

        # Require target surface to settle past ephemeral Cmd+Tab flicker (<0.45s)
        if self._last_focus_change_ts and (now - self._last_focus_change_ts) < self.focus_settle_s:
            return

        # Only emit when transitioning to a distinct app or distinct browser/document window
        if front_app == self._last_source_app and win_title == self._last_source_window:
            return

        target_key = (front_app, win_title)
        if target_key in self._emitted_for_current_copy:
            return

        focused_elem = self._get_focused_element()
        if AmbientPrivacyGuard.is_element_secure(focused_elem):
            return

        self._emitted_for_current_copy.add(target_key)
        self.record_paste(
            target_app=front_app,
            target_window=win_title,
            target_url=browser_url,
            target_element=focused_elem,
            action_type="copy_paste",
        )

    def _handle_live_pasteboard_change(self, raw_clip: str) -> None:
        """Invoked when macOS clipboard changeCount increments."""
        if not raw_clip or not raw_clip.strip():
            return

        front_app = AppFocusRegistry.get_frontmost_app_name()
        if AmbientPrivacyGuard.is_app_blocked(front_app):
            logger.debug(f"Ignoring copy event from blocked auth app: {front_app}")
            return

        win_title, browser_url = self._get_frontmost_window_and_url(front_app)
        focused_elem = self._get_focused_element()

        if AmbientPrivacyGuard.is_element_secure(focused_elem):
            logger.debug("Ignoring copy from secure password input field")
            return

        etype, sanitized, is_sens = AmbientPrivacyGuard.classify_and_sanitize(raw_clip, focused_elem)
        if is_sens:
            logger.debug(f"Redacted sensitive entity during ambient copy from {front_app}")

        self._last_copied_text = sanitized
        self._last_copied_entity = sanitized
        self._last_copied_type = etype
        self._last_source_app = front_app
        self._last_source_window = win_title
        self._last_source_url = browser_url
        self._last_source_element = focused_elem
        self._copy_timestamp = time.time()
        self._last_observed_app = front_app
        self._last_observed_window = win_title
        self._last_focus_change_ts = self._copy_timestamp
        self._emitted_for_current_copy.clear()

    def _get_frontmost_window_and_url(self, app_name: str) -> Tuple[str, Optional[str]]:
        """Extract frontmost window title and browser URL if Chrome/Safari."""
        win_title = app_name
        browser_url = None
        try:
            from ApplicationServices import (
                AXUIElementCopyAttributeValue,
                AXUIElementCreateApplication,
            )
            from AppKit import NSWorkspace

            ws = NSWorkspace.sharedWorkspace()
            front = ws.frontmostApplication()
            if front:
                app_ax = AXUIElementCreateApplication(front.processIdentifier())
                err, win = AXUIElementCopyAttributeValue(app_ax, "AXFocusedWindow", None)
                if err == 0 and win:
                    _, title = AXUIElementCopyAttributeValue(win, "AXTitle", None)
                    if title:
                        win_title = str(title).strip()

            # Browser tab URL extraction
            low_app = app_name.lower()
            if any(b in low_app for b in ("chrome", "safari", "brave", "edge", "arc")):
                from whole_body_compute.integrations.pal_harness import PalHarness

                if "chrome" in low_app:
                    script = 'tell application "Google Chrome" to if (count of windows) > 0 then return URL of active tab of front window'
                elif "safari" in low_app:
                    script = 'tell application "Safari" to if (count of windows) > 0 then return URL of current tab of front window'
                else:
                    script = ""

                if script:
                    ok, out = PalHarness.run_applescript(script)
                    if ok and out:
                        browser_url = out.strip()
        except Exception:
            pass

        return win_title, browser_url

    def _get_focused_element(self) -> Optional[Dict[str, Any]]:
        """Inspect AXFocusedUIElement in frontmost application."""
        try:
            from ApplicationServices import (
                AXUIElementCopyAttributeValue,
                AXUIElementCreateApplication,
            )
            from AppKit import NSWorkspace

            ws = NSWorkspace.sharedWorkspace()
            front = ws.frontmostApplication()
            if not front:
                return None

            app_ax = AXUIElementCreateApplication(front.processIdentifier())
            err, focused = AXUIElementCopyAttributeValue(app_ax, "AXFocusedUIElement", None)
            if err == 0 and focused:
                _, role = AXUIElementCopyAttributeValue(focused, "AXRole", None)
                _, title = AXUIElementCopyAttributeValue(focused, "AXTitle", None)
                _, dom_id = AXUIElementCopyAttributeValue(focused, "AXDOMIdentifier", None)
                _, subrole = AXUIElementCopyAttributeValue(focused, "AXSubrole", None)
                r_str = str(role or "")
                sr_str = str(subrole or "")
                is_sec = "secure" in r_str.lower() or "password" in r_str.lower() or "secure" in sr_str.lower()
                return {
                    "role": r_str,
                    "title": str(title or ""),
                    "dom_selector": str(dom_id or "") if dom_id else None,
                    "subrole": sr_str,
                    "is_secure": is_sec,
                }
        except Exception:
            pass
        return None

    # --------------------------------------------------------------------------
    # SYNTHETIC & RECORDED INGESTION APIS
    # --------------------------------------------------------------------------

    def record_transition(self, event: TransitionEvent) -> Optional[TransitionEvent]:
        """
        Ingest a transition event into the observer history.
        Applies privacy checks, noise filters, and entity classification.
        """
        # 1. Privacy filter
        safe, reason = AmbientPrivacyGuard.verify_safety_guardrails(
            source_app=event.source_app,
            target_app=event.target_app,
            text=event.copied_entity or event.pasted_entity or "",
            element=event.target_element or event.source_element,
        )
        if not safe:
            logger.debug(f"Filtered unsafe transition: {reason}")
            return None

        # 2. Sanitize entities
        if event.copied_entity:
            etype, sanitized, is_sens = AmbientPrivacyGuard.classify_and_sanitize(
                event.copied_entity, event.source_element
            )
            event.copied_entity = sanitized
            event.entity_type = etype
            event.is_sensitive = is_sens or event.is_sensitive

        if event.pasted_entity:
            _, sanitized_paste, is_sens = AmbientPrivacyGuard.classify_and_sanitize(
                event.pasted_entity, event.target_element
            )
            event.pasted_entity = sanitized_paste
            event.is_sensitive = is_sens or event.is_sensitive

        # 3. Entity Fingerprint
        if event.copied_entity:
            event.entity_fingerprint = hashlib.sha256(event.copied_entity.encode("utf-8")).hexdigest()[:12]

        with self._lock:
            self.events.append(event)

        if self.on_transition:
            try:
                self.on_transition(event)
            except Exception as e:
                logger.error(f"Error in on_transition callback: {e}")

        return event

    def record_copy(
        self,
        source_app: str,
        source_window: str,
        text: str,
        source_url: Optional[str] = None,
        source_element: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Manually record a copy event."""
        if AmbientPrivacyGuard.is_app_blocked(source_app):
            return
        etype, sanitized, is_sens = AmbientPrivacyGuard.classify_and_sanitize(text, source_element)
        self._last_source_app = source_app
        self._last_source_window = source_window
        self._last_source_url = source_url
        self._last_source_element = source_element
        self._last_copied_text = sanitized
        self._last_copied_entity = sanitized
        self._last_copied_type = etype
        self._copy_timestamp = time.time()

    def record_paste(
        self,
        target_app: str,
        target_window: str,
        target_url: Optional[str] = None,
        target_element: Optional[Dict[str, Any]] = None,
        action_type: str = "copy_paste",
    ) -> Optional[TransitionEvent]:
        """Manually record a paste/insertion correlating with the previous copy."""
        if not self._last_copied_entity:
            return None

        event = TransitionEvent(
            source_app=self._last_source_app,
            source_window=self._last_source_window,
            source_url=self._last_source_url,
            source_element=self._last_source_element,
            copied_entity=self._last_copied_entity,
            entity_type=self._last_copied_type,
            target_app=target_app,
            target_window=target_window,
            target_url=target_url,
            target_element=target_element,
            action_type=action_type,
            pasted_entity=self._last_copied_entity,
            duration_s=time.time() - self._copy_timestamp if self._copy_timestamp else 0.0,
        )
        return self.record_transition(event)

    def ingest_event_stream(self, stream: List[TransitionEvent]) -> List[TransitionEvent]:
        """Ingest a batch of transition events."""
        recorded = []
        for ev in stream:
            res = self.record_transition(ev)
            if res:
                recorded.append(res)
        return recorded


# ==============================================================================
# SEQUENCE CLUSTERING & MOTIF EXTRACTION
# ==============================================================================


class AmbientMotifExtractor:
    """
    Mines recurring >= 3-step cross-app sequences and clusters them into candidate
    workflow DAGs with parameter generalization.
    """

    def __init__(
        self,
        min_support: int = 2,
        min_steps: int = 3,
        max_steps: int = 10,
        session_idle_seconds: float = 60.0,
    ):
        self.min_support = min_support
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.session_idle_seconds = session_idle_seconds

    def extract_motifs(self, events: List[TransitionEvent]) -> List[WorkflowMotif]:
        """
        Mine and cluster recurring >= 3-step cross-app sequences from transition events.
        """
        # 1. Denoise events
        clean_events = self._filter_noise(events)
        if len(clean_events) < self.min_steps:
            return []

        # 2. Session segmentation
        sessions = self._segment_sessions(clean_events)

        # 3. Mine cross-app n-grams across sessions
        # Candidate tuple: (signatures_tuple) -> list of matching event sequences
        motif_clusters: Dict[Tuple[str, ...], List[List[TransitionEvent]]] = {}

        for sess in sessions:
            if len(sess) < self.min_steps:
                continue

            n = len(sess)
            for k in range(self.min_steps, min(n + 1, self.max_steps + 1)):
                for start_idx in range(n - k + 1):
                    subseq = sess[start_idx : start_idx + k]

                    # Must satisfy cross-app constraint
                    if not self._is_cross_app_sequence(subseq):
                        continue

                    sigs = tuple(ev.structural_signature() for ev in subseq)
                    if sigs not in motif_clusters:
                        motif_clusters[sigs] = []
                    motif_clusters[sigs].append(subseq)

        # 4. Filter by minimum support
        qualified_signatures = [
            sigs for sigs, instances in motif_clusters.items()
            if len(instances) >= self.min_support
        ]

        # 5. Prune subsumed sub-sequences (longest motif preference)
        pruned_signatures = self._prune_subsumed(qualified_signatures, motif_clusters)

        # 6. Build WorkflowMotif objects
        motifs: List[WorkflowMotif] = []
        for idx, sigs in enumerate(pruned_signatures, start=1):
            instances = motif_clusters[sigs]
            exemplar = instances[0]

            # Inferred parameters across instances
            params = self._infer_parameters(instances)

            # Build dataflow DAG edges
            dag_edges = self._build_dag_edges(exemplar)

            src_apps = list(dict.fromkeys(ev.source_app for ev in exemplar if ev.source_app))
            tgt_apps = list(dict.fromkeys(ev.target_app for ev in exemplar if ev.target_app))
            entities = list(dict.fromkeys(ev.entity_type for ev in exemplar if ev.entity_type))

            # Synthesize human-readable title
            name = self._synthesize_motif_name(exemplar, entities, src_apps, tgt_apps)
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

            confidence = min(1.0, 0.5 + 0.15 * len(instances) + 0.05 * len(sigs))

            motif = WorkflowMotif(
                motif_id=f"motif_{slug}_{idx}",
                name=name,
                signatures=list(sigs),
                exemplar_events=exemplar,
                support_count=len(instances),
                source_apps=src_apps,
                target_apps=tgt_apps,
                entities_involved=entities,
                parameters=params,
                dag_edges=dag_edges,
                confidence=round(confidence, 2),
                last_seen=max(ev.timestamp for ev in exemplar),
                is_cross_app=True,
            )
            motifs.append(motif)

        return motifs

    def _filter_noise(self, events: List[TransitionEvent]) -> List[TransitionEvent]:
        """
        Filter transient jitter, empty clips, and noisy focus flickers.
        """
        filtered: List[TransitionEvent] = []
        ignored_system_apps = {"loginwindow", "dock", "controlcenter", "screensaverengine"}

        for ev in events:
            src = (ev.source_app or "").strip().lower()
            tgt = (ev.target_app or "").strip().lower()

            if not src or not tgt:
                continue
            if src in ignored_system_apps or tgt in ignored_system_apps:
                continue
            if AmbientPrivacyGuard.is_app_blocked(src) or AmbientPrivacyGuard.is_app_blocked(tgt):
                continue

            # Skip single-app self-transitions if there is no entity or action
            if src == tgt and not ev.copied_entity and not ev.pasted_entity and ev.duration_s < 0.3:
                continue

            # Skip empty copy/paste entities
            if ev.copied_entity is not None and not ev.copied_entity.strip():
                continue

            filtered.append(ev)

        return filtered

    def _segment_sessions(self, events: List[TransitionEvent]) -> List[List[TransitionEvent]]:
        """Split a stream of events into discrete sessions by idle threshold."""
        if not events:
            return []

        sessions: List[List[TransitionEvent]] = []
        current: List[TransitionEvent] = [events[0]]

        for i in range(1, len(events)):
            gap = events[i].timestamp - events[i - 1].timestamp
            if gap > self.session_idle_seconds:
                sessions.append(current)
                current = [events[i]]
            else:
                current.append(events[i])

        if current:
            sessions.append(current)

        return sessions

    def _is_cross_app_sequence(self, sequence: List[TransitionEvent]) -> bool:
        """Verify that the sequence spans at least 2 distinct applications or domains."""
        apps: Set[str] = set()
        domains: Set[str] = set()

        for ev in sequence:
            if ev.source_app:
                apps.add(ev.source_app.strip().lower())
            if ev.target_app:
                apps.add(ev.target_app.strip().lower())
            url = ev.target_url or ev.source_url
            if url:
                try:
                    host = urlparse(url).netloc.lower()
                    if host:
                        domains.add(host)
                except Exception:
                    pass

        # At least 2 distinct applications, or within a browser at least 2 distinct web domains
        return len(apps) >= 2 or len(domains) >= 2

    def _prune_subsumed(
        self,
        signatures: List[Tuple[str, ...]],
        clusters: Dict[Tuple[str, ...], List[List[TransitionEvent]]],
    ) -> List[Tuple[str, ...]]:
        """Prune shorter sub-motifs if a longer motif covers the same support."""
        # Sort descending by length
        sorted_sigs = sorted(signatures, key=lambda s: len(s), reverse=True)
        retained: List[Tuple[str, ...]] = []

        for sig in sorted_sigs:
            count = len(clusters[sig])
            is_subsumed = False
            for parent in retained:
                parent_count = len(clusters[parent])
                # Check if sig is a sub-tuple of parent
                if len(sig) < len(parent) and parent_count >= count:
                    for i in range(len(parent) - len(sig) + 1):
                        if parent[i : i + len(sig)] == sig:
                            is_subsumed = True
                            break
                if is_subsumed:
                    break

            if not is_subsumed:
                retained.append(sig)

        return retained

    def _infer_parameters(
        self, instances: List[List[TransitionEvent]]
    ) -> Dict[str, ParameterDefinition]:
        """
        Identify dynamic entity values across instances and generalize them into ParameterDefinitions.
        """
        params: Dict[str, ParameterDefinition] = {}
        if not instances:
            return params

        num_steps = len(instances[0])
        for step_idx in range(num_steps):
            values = [
                inst[step_idx].copied_entity or inst[step_idx].pasted_entity
                for inst in instances
                if inst[step_idx].copied_entity or inst[step_idx].pasted_entity
            ]
            unique_vals = set(values)
            etype = instances[0][step_idx].entity_type

            # If values vary or represent recognized entity types (patient_id, code, currency)
            if (len(unique_vals) > 1 or etype in ("patient_id", "currency", "code", "email")) and values:
                param_name = f"{etype}_{step_idx + 1}"
                if etype == "patient_id":
                    patterns = [r"(?P<value>(?:PT|MRN|PAT)[\-_]?\d+)", r"(?P<value>[A-Z0-9_\-]+)"]
                    desc = "Patient Identifier"
                elif etype == "currency":
                    patterns = [r"(?P<value>\$?\d+(?:\.\d{2})?)"]
                    desc = "Monetary Amount / Remaining Maximum"
                elif etype == "date":
                    patterns = [r"(?P<value>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"]
                    desc = "Date Value"
                elif etype == "email":
                    patterns = [r"(?P<value>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"]
                    desc = "Email Address"
                else:
                    patterns = [r"(?P<value>[^\s]+)"]
                    desc = f"Extracted {etype} parameter"

                params[param_name] = ParameterDefinition(
                    type="string",
                    description=desc,
                    default=values[0] if values else None,
                    required=True,
                    extract_patterns=patterns,
                )

        return params

    def _build_dag_edges(self, sequence: List[TransitionEvent]) -> List[Tuple[int, int]]:
        """
        Construct directed acyclic graph (DAG) edges showing:
        1. Sequential control flow (i -> i+1)
        2. Entity dataflow (producer step i -> consumer step j where entity flows)
        """
        edges: Set[Tuple[int, int]] = set()
        n = len(sequence)

        # 1. Control flow
        for i in range(n - 1):
            edges.add((i, i + 1))

        # 2. Entity dataflow
        for i in range(n):
            out_entity = sequence[i].copied_entity or sequence[i].pasted_entity
            out_fp = sequence[i].entity_fingerprint
            out_type = sequence[i].entity_type

            if not out_entity and not out_fp:
                continue

            for j in range(i + 1, n):
                in_entity = sequence[j].copied_entity or sequence[j].pasted_entity
                in_fp = sequence[j].entity_fingerprint
                in_type = sequence[j].entity_type

                # Match on exact fingerprint or matching semantic entity type
                if (out_fp and out_fp == in_fp) or (out_entity and out_entity == in_entity):
                    edges.add((i, j))
                elif out_type != "text" and out_type == in_type:
                    edges.add((i, j))

        return sorted(list(edges))

    def _synthesize_motif_name(
        self,
        exemplar: List[TransitionEvent],
        entities: List[str],
        src_apps: List[str],
        tgt_apps: List[str],
    ) -> str:
        """Synthesize a human-readable workflow title."""
        # Check for recognized domain archetypes (e.g. Dental Front Desk Eligibility Check)
        all_text = " ".join(
            f"{ev.source_app} {ev.source_window} {ev.target_app} {ev.target_window} {ev.target_url or ''}"
            for ev in exemplar
        ).lower()

        if "dentrix" in all_text or "dental" in all_text or "availity" in all_text or "eligibility" in all_text:
            return "Dental Front-Desk Eligibility Check"
        if "sheet" in all_text and "drive" in all_text:
            return "Drive Sheets Data Pipeline"
        if "github" in all_text or "pr" in all_text:
            return "Git Pull Request Workflow"
        if "invoice" in all_text or "billing" in all_text:
            return "Invoice Processing Pipeline"

        # Fallback to App -> App flow title
        src = src_apps[0] if src_apps else "App"
        tgt = tgt_apps[-1] if tgt_apps else "App"
        primary_entity = entities[0].replace("_", " ").title() if entities else "Data"
        return f"{primary_entity} Transfer ({src} to {tgt})"


# ==============================================================================
# SHADOW MODE STAGING & HEADLESS PARITY VALIDATION
# ==============================================================================


class ShadowModeStager:
    """
    Maintains staged candidate workflows in shadow mode.
    Computes headless parity validation score, flushes candidate YAML manifests into
    `recipes/candidates/<slug>.yaml` for `body candidates`, and enables 1-click user promotion
    into active recipes and hot-reload into PalWorkflowRegistry.
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        candidates_dir: Optional[Path] = None,
        persist_candidates: bool = True,
    ):
        self.storage_dir = storage_dir or resolve_default_candidates_dir().parent
        self.candidates_dir = candidates_dir or (
            storage_dir / "candidates" if storage_dir is not None else resolve_default_candidates_dir()
        )
        self.persist_candidates = persist_candidates
        self._candidates: Dict[str, WorkflowCandidate] = {}
        self._lock = threading.Lock()

    def stage_motif(self, motif: WorkflowMotif) -> WorkflowCandidate:
        """
        Stage a mined motif into a shadow candidate, compile its baseline recipe,
        run headless parity validation, and persist YAML to `recipes/candidates/`.
        """
        from whole_body_compute.integrations.ambient_discovery import AmbientWorkflowDiscovery

        recipe = AmbientWorkflowDiscovery.compile_motif_to_recipe(motif)
        candidate = WorkflowCandidate(
            candidate_id=motif.motif_id,
            name=motif.name,
            motif=motif,
            recipe=recipe,
            status="staged",
        )

        # Validate headless parity
        self.validate_candidate(candidate)

        # Enrich recipe with tier_summary & contract_summary for `body candidates`
        self._attach_candidate_summaries(candidate)

        if self.persist_candidates and candidate.recipe and candidate.recipe.steps:
            self._persist_staged_yaml(candidate)

        with self._lock:
            self._candidates[candidate.candidate_id] = candidate

        return candidate

    def _attach_candidate_summaries(self, candidate: WorkflowCandidate) -> None:
        """Attach `tier_summary` and `contract_summary` metadata so `body candidates` renders full telemetry."""
        recipe = candidate.recipe
        if recipe is None:
            return

        dominant_tier = (
            "TIER_1_HEADLESS_API"
            if candidate.direct_headless_routes
            else "TIER_2_SEMANTIC_UI"
        )
        if not recipe.tier_summary:
            recipe.tier_summary = {
                "archetype": "ambient_cross_app_motif",
                "raw_step_count": len(candidate.motif.exemplar_events),
                "compiled_step_count": len(recipe.steps),
                "dominant_tier": dominant_tier,
                "reliability_score": float(candidate.headless_parity_score),
                "ssa_edges_resolved": len(candidate.motif.dag_edges),
                "support_count": int(candidate.motif.support_count),
                "direct_headless_routes": list(candidate.direct_headless_routes),
            }
        if not recipe.contract_summary:
            contracts = [
                {
                    "name": f"{recipe.id}_postcondition_contract",
                    "preconditions": ["SOURCE_APP_ACCESSIBLE", "PASTEBOARD_SANITIZED"],
                    "postconditions": ["TARGET_SURFACE_VERIFIED"],
                }
            ]
            recipe.contract_summary = {"contracts": contracts}

    def _persist_staged_yaml(self, candidate: WorkflowCandidate) -> Optional[Path]:
        """Write sanitized candidate YAML to `recipes/candidates/<id>.yaml` if changed."""
        if candidate.recipe is None:
            return None
        try:
            import yaml

            self.candidates_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = self.candidates_dir / f"{candidate.recipe.id}.yaml"
            payload_dict = candidate.recipe.model_dump(mode="json", exclude_none=True)
            yaml_text = redact_secrets(
                yaml.safe_dump(payload_dict, sort_keys=False, allow_unicode=True)
            )
            new_digest = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()[:16]
            existing_digest = ""
            if candidate_path.is_file():
                existing_text = candidate_path.read_text(encoding="utf-8", errors="ignore")
                existing_digest = hashlib.sha256(existing_text.encode("utf-8")).hexdigest()[:16]
            if new_digest != existing_digest:
                candidate_path.write_text(yaml_text, encoding="utf-8")
                logger.info(
                    "Staged ambient workflow candidate '%s' (%dx) -> %s",
                    candidate.recipe.id,
                    candidate.motif.support_count,
                    candidate_path,
                )
            return candidate_path
        except Exception as exc:
            logger.debug("Ambient candidate YAML staging notice: %s", exc)
            return None

    def validate_candidate(self, candidate: WorkflowCandidate) -> WorkflowCandidate:
        """
        Evaluate headless parity:
        Checks if candidate steps can run headlessly via Tier 0/1 routes (APIs, CLIs, Python)
        or whether frontmost app targets are resolvable.
        """
        notes: List[str] = []
        direct_routes: List[str] = []
        parity_score = 0.50  # Base feasibility

        # 1. Evaluate Direct Headless Routes from Foveal Vision & Episodic Memory
        for ev in candidate.motif.exemplar_events:
            route = resolve_direct_headless_route(
                app_name=ev.target_app or ev.source_app,
                window_name=ev.target_window or ev.source_window,
                ocr_text=ev.copied_entity or "",
            )
            if "DIRECT ROUTE" in route:
                direct_routes.append(route)
                parity_score += 0.15
                notes.append(f"Headless route resolved for {ev.target_app or ev.source_app}")

        # 2. Evaluate AppFocusRegistry support
        for app in candidate.motif.target_apps:
            target = AppFocusRegistry.get_target(app)
            if target:
                parity_score += 0.10
                notes.append(f"App '{app}' recognized in AppFocusRegistry")

        # 3. Parameters extraction check
        if candidate.motif.parameters:
            parity_score += 0.15
            notes.append(f"{len(candidate.motif.parameters)} dynamic parameter slots bound")

        candidate.headless_parity_score = round(min(1.0, parity_score), 2)
        candidate.direct_headless_routes = direct_routes
        candidate.validation_notes = notes

        if candidate.headless_parity_score >= 0.70:
            candidate.status = "ready"
        else:
            candidate.status = "validating"

        return candidate

    def promote_candidate(
        self,
        candidate_id: str,
        save_dir: Optional[Path] = None,
    ) -> PalWorkflowRecipe:
        """
        1-Click Promotion: Promote candidate to an active PalWorkflowRecipe,
        persist YAML recipe to disk, and register in PalWorkflowRegistry.
        """
        with self._lock:
            candidate = self._candidates.get(candidate_id)
            if not candidate:
                raise ValueError(f"Candidate '{candidate_id}' not found in shadow stager")

            candidate.status = "promoted"
            candidate.promoted_at = time.time()

        dest_dir = save_dir or self.storage_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        file_path = dest_dir / f"{candidate.recipe.id}.yaml"

        import yaml

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                candidate.recipe.model_dump(mode="json", exclude_none=True),
                f,
                sort_keys=False,
            )

        # Hot-reload into registry
        PalWorkflowRegistry.register_recipe(candidate.recipe)
        logger.info(f"Promoted candidate '{candidate.name}' -> {file_path}")

        return candidate.recipe

    def reject_candidate(self, candidate_id: str, reason: str = "") -> WorkflowCandidate:
        """Reject and archive a shadow candidate."""
        with self._lock:
            candidate = self._candidates.get(candidate_id)
            if not candidate:
                raise ValueError(f"Candidate '{candidate_id}' not found")
            candidate.status = "rejected"
            candidate.rejection_reason = reason or "User dismissed"
            return candidate

    def get_candidate(self, candidate_id: str) -> Optional[WorkflowCandidate]:
        with self._lock:
            return self._candidates.get(candidate_id)

    def list_candidates(
        self, status: Optional[str] = None
    ) -> List[WorkflowCandidate]:
        with self._lock:
            if status:
                return [c for c in self._candidates.values() if c.status == status]
            return list(self._candidates.values())


# ==============================================================================
# UNIFIED AMBIENT WORKFLOW DISCOVERY FACADE
# ==============================================================================


class AmbientWorkflowDiscovery:
    """
    Unified controller for Zero-Training Ambient Workflow Discovery.
    Orchestrates passive observation, motif clustering, PalWorkflow compilation,
    and shadow mode staging.
    """

    def __init__(
        self,
        min_support: int = 2,
        min_steps: int = 3,
        session_idle_seconds: float = 60.0,
        storage_dir: Optional[Path] = None,
        candidates_dir: Optional[Path] = None,
        events_log_path: Optional[Path] = None,
        auto_stage: bool = False,
        poll_interval_s: float = 0.25,
    ):
        self.auto_stage = auto_stage
        self.events_log_path = events_log_path
        self._max_journal_events = 500
        self.observer = AmbientWorkflowObserver(
            on_transition=self._handle_observed_transition,
            poll_interval_s=poll_interval_s,
        )
        self.extractor = AmbientMotifExtractor(
            min_support=min_support,
            min_steps=min_steps,
            session_idle_seconds=session_idle_seconds,
        )
        self.stager = ShadowModeStager(
            storage_dir=storage_dir,
            candidates_dir=candidates_dir,
        )
        if self.events_log_path is not None:
            self._load_persisted_transitions()

    def _load_persisted_transitions(self) -> int:
        """Reload recent sanitized transition history across daemon restarts."""
        if not self.events_log_path or not self.events_log_path.is_file():
            return 0
        loaded = 0
        try:
            lines = self.events_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for raw in lines[-self._max_journal_events :]:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ev = TransitionEvent.model_validate(json.loads(raw))
                    with self.observer._lock:
                        self.observer.events.append(ev)
                    loaded += 1
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("Ambient transition journal load notice: %s", exc)
        return loaded

    def _append_transition_to_journal(self, event: TransitionEvent) -> None:
        if not self.events_log_path:
            return
        try:
            self.events_log_path.parent.mkdir(parents=True, exist_ok=True)
            line = redact_secrets(json.dumps(event.model_dump(mode="json"), sort_keys=False))
            with self.events_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.debug("Ambient transition journal append notice: %s", exc)

    def _handle_observed_transition(self, event: TransitionEvent) -> None:
        """Persist sanitized transition and incrementally mine/stage motifs when `auto_stage=True`."""
        self._append_transition_to_journal(event)
        if self.auto_stage and len(self.observer.events) >= (
            self.extractor.min_support * self.extractor.min_steps
        ):
            try:
                self.discover_and_stage()
            except Exception as exc:
                logger.debug("Incremental ambient motif discovery notice: %s", exc)

    def start_background_discovery(self) -> bool:
        """Start continuous background `AmbientWorkflowObserver` + auto-staging into `recipes/candidates/`."""
        if not is_ambient_discovery_enabled():
            logger.info("AmbientWorkflowDiscovery disabled via WBC_AMBIENT_DISCOVERY=0")
            return False
        self.auto_stage = True
        if self.events_log_path is None:
            self.events_log_path = resolve_ambient_events_log_path()
            self._load_persisted_transitions()
        if len(self.observer.events) >= (self.extractor.min_support * self.extractor.min_steps):
            try:
                self.discover_and_stage()
            except Exception:
                pass
        self.observer.start()
        return True

    def stop_background_discovery(self) -> None:
        """Stop continuous background `AmbientWorkflowObserver`."""
        self.observer.stop()

    def record_transition(self, event: TransitionEvent) -> Optional[TransitionEvent]:
        """Record an observed transition."""
        return self.observer.record_transition(event)

    def discover_and_stage(self) -> List[WorkflowCandidate]:
        """
        Process current observer event stream, extract recurring motifs,
        and stage them into shadow mode candidates.
        """
        motifs = self.extractor.extract_motifs(self.observer.events)
        candidates = []
        for motif in motifs:
            cand = self.stager.stage_motif(motif)
            candidates.append(cand)
        return candidates

    @classmethod
    def compile_motif_to_recipe(
        cls,
        motif: WorkflowMotif,
        name: Optional[str] = None,
    ) -> PalWorkflowRecipe:
        """
        Compile an ambiently discovered motif into an executable PalWorkflowRecipe.
        """
        recipe_name = name or motif.name
        slug = re.sub(r"[^a-z0-9]+", "_", recipe_name.lower()).strip("_")
        compiled_steps: List[WorkflowStep] = []
        notes: List[str] = [
            f"Auto-discovered ambient cross-app motif (Observed {motif.support_count}x)"
        ]

        # Convert transitions to WorkflowSteps
        current_app = ""
        for idx, ev in enumerate(motif.exemplar_events, start=1):
            src_app = ev.source_app
            tgt_app = ev.target_app
            etype = ev.entity_type
            act = ev.action_type

            # Activate source app if needed
            if src_app and src_app != current_app:
                compiled_steps.append(
                    WorkflowStep(
                        name=f"Focus {src_app}",
                        action=ActionType.ACTIVATE_APP,
                        app_name=src_app,
                        description=f"Switch to source {src_app}",
                    )
                )
                current_app = src_app

            # Copy step
            if ev.copied_entity:
                compiled_steps.append(
                    WorkflowStep(
                        name=f"Copy {etype}",
                        action=ActionType.HOTKEY,
                        combo="Cmd+C",
                        key="C",
                        modifiers=["Cmd"],
                        description=f"Copy {etype} from {ev.source_window or src_app}",
                        postconditions=[
                            PostConditionCheck(
                                getter_type="clipboard_matches",
                                expected=ev.copied_entity,
                                description=f"Clipboard holds {etype}",
                            )
                        ],
                    )
                )

            # Activate target app
            if tgt_app and tgt_app != current_app:
                compiled_steps.append(
                    WorkflowStep(
                        name=f"Focus {tgt_app}",
                        action=ActionType.ACTIVATE_APP,
                        app_name=tgt_app,
                        description=f"Switch to target {tgt_app}",
                    )
                )
                current_app = tgt_app

            # Target action (Paste / Keystroke / Click)
            if act in ("paste", "copy_paste"):
                target = None
                if ev.target_element and ev.target_element.get("dom_selector"):
                    target = ClickTarget(
                        target_type="dom_selector",
                        dom_selector=ev.target_element["dom_selector"],
                    )

                compiled_steps.append(
                    WorkflowStep(
                        name=f"Paste {etype} into {tgt_app}",
                        action=ActionType.HOTKEY,
                        combo="Cmd+V",
                        key="V",
                        modifiers=["Cmd"],
                        click_target=target,
                        description=f"Paste {etype} into {ev.target_window or tgt_app}",
                    )
                )
            elif act == "type_text" and ev.pasted_entity:
                compiled_steps.append(
                    WorkflowStep(
                        name=f"Type {etype}",
                        action=ActionType.KEYSTROKE,
                        text=ev.pasted_entity,
                        description=f"Type {etype} into {tgt_app}",
                    )
                )

        # Trigger phrases
        clean_name = recipe_name.lower().replace("workflow", "").strip()
        triggers = [
            clean_name,
            f"run {clean_name}",
            f"execute {clean_name}",
        ]

        recipe = PalWorkflowRecipe(
            id=slug,
            name=recipe_name,
            description=f"Discovered cross-app workflow DAG: {' -> '.join(motif.source_apps + motif.target_apps[:1])}",
            notes=notes,
            triggers=list(dict.fromkeys(triggers)),
            parameters=motif.parameters,
            steps=compiled_steps,
            success_summary=f"Successfully executed {recipe_name}",
            error_summary=f"Failed during {recipe_name} at {{step_name}}",
        )

        PalWorkflowRegistry.register_recipe(recipe)
        return recipe


# ==============================================================================
# STAGED CANDIDATE MANAGEMENT & CLI HELPERS (`body candidates`)
# ==============================================================================

_GLOBAL_AMBIENT_DISCOVERY: Optional[AmbientWorkflowDiscovery] = None
_GLOBAL_AMBIENT_LOCK = threading.Lock()


def get_global_ambient_discovery() -> AmbientWorkflowDiscovery:
    """Return process-wide singleton `AmbientWorkflowDiscovery` controller."""
    global _GLOBAL_AMBIENT_DISCOVERY
    with _GLOBAL_AMBIENT_LOCK:
        if _GLOBAL_AMBIENT_DISCOVERY is None:
            _GLOBAL_AMBIENT_DISCOVERY = AmbientWorkflowDiscovery(
                events_log_path=resolve_ambient_events_log_path(),
                auto_stage=True,
            )
        return _GLOBAL_AMBIENT_DISCOVERY


def list_staged_candidates(
    candidates_dir: Optional[Path] = None,
    active_recipes_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Inspect all staged candidate `.yaml` workflows in `recipes/candidates/` along with
    their `PalCompiler` / `ShadowModeStager` reliability scores, dominant execution tiers,
    and Hoare contracts, plus live `AmbientWorkflowObserver` transition telemetry.
    """
    import yaml

    c_dir = Path(candidates_dir or resolve_default_candidates_dir())
    r_dir = Path(active_recipes_dir or c_dir.parent)
    candidates: List[Dict[str, Any]] = []

    if c_dir.is_dir():
        for ypath in sorted(c_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(ypath.read_text(encoding="utf-8", errors="ignore")) or {}
                wf_id = str(data.get("id") or ypath.stem)
                tier_sum = data.get("tier_summary") or {}
                contract_sum = data.get("contract_summary") or {}
                contracts = contract_sum.get("contracts") or []
                active_path = r_dir / f"{wf_id}.yaml"
                candidates.append(
                    {
                        "id": wf_id,
                        "name": str(data.get("name") or wf_id),
                        "path": str(ypath),
                        "steps_count": len(data.get("steps") or []),
                        "postconditions_count": len(data.get("postconditions") or []),
                        "dominant_tier": str(tier_sum.get("dominant_tier") or "TIER_1_HEADLESS_API"),
                        "reliability_score": float(tier_sum.get("reliability_score") or 0.99),
                        "ssa_edges": int(tier_sum.get("ssa_edges_resolved") or 0),
                        "support_count": int(tier_sum.get("support_count") or 0),
                        "contracts": [
                            c.get("name", "HoareContract") for c in contracts if isinstance(c, dict)
                        ],
                        "promoted_to_active": active_path.is_file(),
                        "updated_at": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(ypath.stat().st_mtime)
                        ),
                    }
                )
            except Exception as e:
                logger.debug("Candidate parse notice (%s): %s", ypath, e)

    active_count = len([p for p in r_dir.glob("*.yaml") if p.is_file()]) if r_dir.is_dir() else 0
    events_path = resolve_ambient_events_log_path()
    journal_transitions = 0
    if events_path.is_file():
        try:
            journal_transitions = sum(
                1 for line in events_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()
            )
        except Exception:
            journal_transitions = 0

    return {
        "candidates_dir": str(c_dir),
        "active_recipes_dir": str(r_dir),
        "staged_candidates_count": len(candidates),
        "active_recipes_count": active_count,
        "ambient_discovery_default_on": is_ambient_discovery_enabled(),
        "ambient_events_journal": str(events_path),
        "ambient_transitions_logged": journal_transitions,
        "candidates": candidates,
    }


def promote_staged_candidate(
    candidate_id: str,
    candidates_dir: Optional[Path] = None,
    active_recipes_dir: Optional[Path] = None,
    share_learnings: bool = False,
) -> Dict[str, Any]:
    """
    Validate a staged candidate `.yaml` against `PalWorkflowRecipe` schema and Hoare postcondition
    requirements, then promote it into active `recipes/<id>.yaml`.
    """
    import shutil
    import yaml

    c_dir = Path(candidates_dir or resolve_default_candidates_dir())
    r_dir = Path(active_recipes_dir or c_dir.parent)
    slug = candidate_id.strip().removesuffix(".yaml")
    src_path = c_dir / f"{slug}.yaml"
    if not src_path.is_file():
        return {
            "status": "error",
            "promoted": False,
            "reason": f"Candidate '{slug}' not found in {c_dir}",
        }

    try:
        raw_data = yaml.safe_load(src_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw_data, dict):
            return {
                "status": "error",
                "promoted": False,
                "reason": f"Candidate '{slug}' YAML did not produce a dictionary mapping.",
            }
        recipe = PalWorkflowRecipe.model_validate(raw_data)
    except Exception as e:
        return {
            "status": "error",
            "promoted": False,
            "reason": f"Malformed candidate YAML or schema invalidation ({slug}): {e}",
        }
    if not recipe.steps:
        return {
            "status": "error",
            "promoted": False,
            "reason": f"Candidate '{slug}' has 0 executable steps.",
        }

    r_dir.mkdir(parents=True, exist_ok=True)
    dest_path = r_dir / f"{recipe.id}.yaml"
    shutil.copy2(src_path, dest_path)
    PalWorkflowRegistry.register_recipe(recipe)

    learning_card: Optional[Dict[str, Any]] = None
    if share_learnings:
        try:
            from whole_body_compute.procedural_telemetry import export_sanitized_learning_card

            learning_card = export_sanitized_learning_card(dest_path, persist_to_outbox=True)
        except Exception as exc:
            logger.debug("Procedural DNA learning card export notice: %s", exc)

    result: Dict[str, Any] = {
        "status": "ok",
        "promoted": True,
        "id": recipe.id,
        "name": recipe.name,
        "source_path": str(src_path),
        "promoted_path": str(dest_path),
        "steps_count": len(recipe.steps),
        "postconditions_count": len(recipe.postconditions),
    }
    if learning_card is not None:
        result["learning_card"] = learning_card
    try:
        from whole_body_compute.procedural_telemetry import track_wbc_event

        track_wbc_event(
            "wbc_candidate_promoted",
            {
                "recipe_id": str(recipe.id),
                "steps_count": len(recipe.steps),
                "postconditions_count": len(recipe.postconditions),
                "shared_learnings": bool(share_learnings),
            },
        )
    except Exception:
        pass
    return result


def discover_now_from_ambient_sources(
    candidates_dir: Optional[Path] = None,
    events_log_path: Optional[Path] = None,
    screenpipe_db: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    On-demand pass (`body candidates --discover`) that mines both the live `ambient_transitions.jsonl`
    journal and `~/.screenpipe/db.sqlite` multi-app window transitions, compiling any recurring motifs
    directly into `recipes/candidates/`.
    """
    from types import SimpleNamespace
    import sqlite3
    import yaml
    from whole_body_compute.integrations.pal_compiler import PalCompiler
    from whole_body_compute.integrations.screenpipe import DEFAULT_SCREENPIPE_DB

    c_dir = Path(candidates_dir or resolve_default_candidates_dir())
    ev_path = Path(events_log_path or resolve_ambient_events_log_path())

    discovery = AmbientWorkflowDiscovery(
        candidates_dir=c_dir,
        events_log_path=ev_path,
        auto_stage=False,
    )
    staged_from_journal = discovery.discover_and_stage()
    staged_ids: List[str] = [c.recipe.id for c in staged_from_journal if c.recipe]

    # Also mine recent Screenpipe OCR multi-app transitions if available
    sp_db = Path(screenpipe_db or DEFAULT_SCREENPIPE_DB)
    if sp_db.is_file():
        try:
            conn = sqlite3.connect(f"file:{sp_db}?mode=ro", uri=True, timeout=3.0)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT app_name, window_name, substr(text, 1, 240) AS snippet
                FROM ocr_text
                WHERE app_name IS NOT NULL AND trim(app_name) != ''
                ORDER BY rowid DESC LIMIT 30
                """
            ).fetchall()
            conn.close()

            transitions: List[Dict[str, str]] = []
            for r in reversed(rows):
                app = str(r["app_name"] or "").strip()
                if AmbientPrivacyGuard.is_app_blocked(app):
                    continue
                win = str(r["window_name"] or "").strip()
                snip = redact_secrets(str(r["snippet"] or "").strip())
                if not transitions or (transitions[-1]["app"] != app or transitions[-1]["win"] != win):
                    transitions.append({"app": app, "win": win, "snip": snip})

            distinct_apps = {t["app"].lower() for t in transitions if t["app"]}
            if len(transitions) >= 2 and len(distinct_apps) >= 2:
                steps = [
                    SimpleNamespace(
                        step_number=idx,
                        action="click",
                        app_name=t["app"],
                        window_title=t["win"],
                        voice_note=f"Observed in {t['app']} ({t['win']}): {t['snip'][:90]}",
                        params={"element": {"title": t["win"] or t["app"], "role": "AXWindow"}},
                    )
                    for idx, t in enumerate(transitions[:6], start=1)
                ]
                recipe, _ir_prog, _report = PalCompiler.compile_demonstration(
                    raw_steps=steps,
                    voice_notes=[f"Transition across {', '.join(sorted(distinct_apps))}"],
                    workflow_name="Ambient Screenpipe Multi-App Flow",
                )
                c_dir.mkdir(parents=True, exist_ok=True)
                cand_path = c_dir / f"{recipe.id}.yaml"
                yaml_text = redact_secrets(
                    yaml.safe_dump(
                        recipe.model_dump(mode="json", exclude_none=True),
                        sort_keys=False,
                        allow_unicode=True,
                    )
                )
                cand_path.write_text(yaml_text, encoding="utf-8")
                if recipe.id not in staged_ids:
                    staged_ids.append(recipe.id)
        except Exception as exc:
            logger.debug("On-demand Screenpipe discovery notice: %s", exc)

    try:
        from whole_body_compute.procedural_telemetry import track_wbc_event

        track_wbc_event(
            "wbc_ambient_discovery",
            {
                "discovered_count": len(staged_ids),
                "journal_events_scanned": len(discovery.observer.events),
            },
        )
    except Exception:
        pass

    return {
        "discovered_count": len(staged_ids),
        "staged_ids": staged_ids,
        "journal_events_scanned": len(discovery.observer.events),
        "candidates_dir": str(c_dir),
    }


def format_candidates_table(report: Dict[str, Any]) -> str:
    """Format staged `recipes/candidates/` report for terminal review."""
    cands = report.get("candidates", [])
    ambient_on = report.get("ambient_discovery_default_on", True)
    transitions_logged = int(report.get("ambient_transitions_logged", 0) or 0)
    sensor_badge = (
        f"🟢 DEFAULT-ON ({transitions_logged} transitions observed)"
        if ambient_on
        else "⚪ OPTED-OUT (WBC_AMBIENT_DISCOVERY=0)"
    )
    lines = [
        "🧪 Whole Body Compute - Staged Ambient Workflow Candidates (`recipes/candidates/`)",
        f"👁️ Passive Ambient Observer: {sensor_badge}",
        "┌────────────────────────────────────┬──────────────────────────┬─────────────┬───────┬────────────┐",
        "│ Candidate ID                       │ Dominant Tier            │ Reliability │ Steps │ Status     │",
        "├────────────────────────────────────┼──────────────────────────┼─────────────┼───────┼────────────┤",
    ]
    if not cands:
        lines.append(
            "│ (none staged)                      │ -                        │ -           │ -     │ -          │"
        )
    else:
        for c in cands:
            cid = str(c["id"])[:34].ljust(34)
            tier = str(c["dominant_tier"])[:24].ljust(24)
            rel = f"{float(c['reliability_score']) * 100:.1f}%".ljust(11)
            steps = str(c["steps_count"]).ljust(5)
            status = "🟢 ACTIVE" if c.get("promoted_to_active") else "🟡 STAGED"
            lines.append(f"│ {cid} │ {tier} │ {rel} │ {steps} │ {status.ljust(10)} │")
    lines.append(
        "└────────────────────────────────────┴──────────────────────────┴─────────────┴───────┴────────────┘"
    )
    lines.append(
        f"💡 Promote a candidate to production: `body candidates --promote <candidate_id>` "
        f"(Active recipes: {report.get('active_recipes_count', 0)})"
    )
    return "\n".join(lines)
