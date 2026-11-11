"""
Blood-Brain Barrier (Organ 05) — Adaptive Antibody Registry, Epithelial Screen Filter,
Vascular Traffic Monitor & Companion Traffic Guard.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _get_default_antibody_path() -> Path:
    env_p = os.environ.get("VOICEFI_ANTIBODY_PATH") or os.environ.get("WBC_ANTIBODY_PATH")
    if env_p:
        return Path(env_p).expanduser()
    env_dir = os.environ.get("WBC_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser() / "antibodies.json"
    home = Path.home()
    wbc_dir = home / ".whole_body_compute"
    if wbc_dir.exists():
        return wbc_dir / "antibodies.json"
    vifi_dir = home / ".voicefi"
    if vifi_dir.exists():
        return vifi_dir / "antibodies.json"
    return wbc_dir / "antibodies.json"


DEFAULT_ANTIBODY_PATH = _get_default_antibody_path()

# ---------------------------------------------------------------------------
# 1. Adaptive Antibody Registry
# ---------------------------------------------------------------------------


@dataclass
class AntibodyRecord:
    """Represents a synthesized adaptive antibody targeting a specific threat antigen."""

    antibody_id: str
    vector: str  # "ast_secgate", "shell_egress", "prompt_injection", "symlink_escape"
    signature: str  # Normalized lowercase threat signature substring or regex token
    antigen_sha256: str
    reason: str
    stage: int = 3
    hit_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "antibody_id": self.antibody_id,
            "vector": self.vector,
            "signature": self.signature,
            "antigen_sha256": self.antigen_sha256,
            "reason": self.reason,
            "stage": self.stage,
            "hit_count": self.hit_count,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
        }


class AntibodyRegistry:
    """
    Adaptive Immune Memory (B-Cell Antibody Store).
    Screens incoming commands/code/prompts in <1ms and synthesizes new antibodies
    whenever the Blood-Brain Barrier blocks an attack vector.
    """

    _instance: Optional["AntibodyRegistry"] = None

    def __init__(self, antibody_path: Optional[Path] = None) -> None:
        if antibody_path is not None:
            self.antibody_path = Path(antibody_path)
        else:
            self.antibody_path = _get_default_antibody_path()
        self._antibodies: Dict[str, AntibodyRecord] = {}
        self._load()

    @classmethod
    def get_instance(cls, antibody_path: Optional[Path] = None) -> "AntibodyRegistry":
        if antibody_path is not None:
            return cls(antibody_path=antibody_path)
        expected_path = _get_default_antibody_path()
        if cls._instance is None or str(cls._instance.antibody_path) != str(expected_path):
            cls._instance = cls(antibody_path=expected_path)
        return cls._instance

    @staticmethod
    def normalize_payload(payload: str) -> str:
        """Collapse whitespace and normalize quotes for deterministic antigen matching."""
        if not payload:
            return ""
        collapsed = re.sub(r"\s+", " ", payload.strip())
        return collapsed.replace('"', "'").lower()

    @classmethod
    def extract_threat_signature(cls, vector: str, payload: str) -> str:
        """
        Extract the salient dangerous expression or command span from an attack payload
        so the resulting antibody blocks variations of the same attack.
        """
        norm = cls.normalize_payload(payload)
        if not norm:
            return ""

        # Check for specific high-signal dangerous expressions first
        patterns = [
            r"(getattr\s*\([^)]+\)\s*\([^)]*\))",
            r"(__import__\s*\([^)]+\)\.[a-z0-9_]+\s*\([^)]*\))",
            r"((?:os\.system|os\.popen|pty\.spawn|eval|exec)\s*\([^)]+\))",
            r"(subprocess\.[a-z_]+\s*\([^)]*shell\s*=\s*true[^)]*\))",
            r"((?<![a-zA-Z0-9_])(?:curl|wget|nc|ncat|scp|sftp)\b\s+[^\n;|&]+)",
            r"(ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|commands|prompts)[^\n.]*)",
            r"(\[system\s+instruction[^\]]*\])",
        ]
        for pat in patterns:
            m = re.search(pat, norm, re.IGNORECASE)
            if m:
                sig = m.group(1).strip()
                if len(sig) >= 8:
                    return sig[:160]

        return norm[:160]

    def _load(self) -> None:
        self._antibodies.clear()
        if not self.antibody_path.is_file():
            return
        try:
            raw = json.loads(self.antibody_path.read_text(encoding="utf-8"))
            for item in raw.get("antibodies", []):
                rec = AntibodyRecord(
                    antibody_id=str(item["antibody_id"]),
                    vector=str(item.get("vector", "unknown")),
                    signature=str(item.get("signature", "")),
                    antigen_sha256=str(item.get("antigen_sha256", "")),
                    reason=str(item.get("reason", "")),
                    stage=int(item.get("stage", 3)),
                    hit_count=int(item.get("hit_count", 1)),
                    created_at=float(item.get("created_at", time.time())),
                    last_seen_at=float(item.get("last_seen_at", time.time())),
                )
                if rec.signature:
                    self._antibodies[rec.antigen_sha256] = rec
        except Exception:
            pass

    def _save(self) -> None:
        # Avoid polluting the production ~/.voicefi/antibodies.json during pytest runs
        # unless VOICEFI_ANTIBODY_PATH is explicitly pointed at a test file.
        if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("VOICEFI_ANTIBODY_PATH"):
            return
        try:
            import tempfile

            self.antibody_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": time.time(),
                "total_antibodies": len(self._antibodies),
                "antibodies": [r.to_dict() for r in self._antibodies.values()],
            }
            serialized = json.dumps(payload, indent=2)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(self.antibody_path.parent),
                prefix=".antibodies_",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as tmp_f:
                tmp_f.write(serialized)
                tmp_path = Path(tmp_f.name)
            os.replace(tmp_path, self.antibody_path)
        except Exception:
            pass

    def record_antigen(
        self,
        vector: str,
        payload: str,
        reason: str,
        stage: int = 3,
        persist_to_hippocampus: bool = True,
    ) -> AntibodyRecord:
        """
        Capture a blocked attack payload as an antigen and synthesize an adaptive antibody.
        """
        sig = self.extract_threat_signature(vector, payload)
        if not sig:
            sig = self.normalize_payload(payload)[:120] or "empty_antigen"

        digest = hashlib.sha256(f"{vector}:{sig}".encode("utf-8")).hexdigest()
        now = time.time()

        if digest in self._antibodies:
            existing = self._antibodies[digest]
            existing.hit_count += 1
            existing.last_seen_at = now
            self._save()
            return existing

        ab_id = f"ab_{digest[:10]}"
        rec = AntibodyRecord(
            antibody_id=ab_id,
            vector=vector,
            signature=sig,
            antigen_sha256=digest,
            reason=reason,
            stage=stage,
            hit_count=1,
            created_at=now,
            last_seen_at=now,
        )
        self._antibodies[digest] = rec
        self._save()

        if persist_to_hippocampus and not os.environ.get("VOICEFI_DISABLE_DOT_MEMORY"):
            try:
                from whole_body_compute.local.dot_memory import get_dot_store

                store = get_dot_store()
                store.add_memory(
                    fact=f"[ANTIGEN:{ab_id}] vector={vector} sig={sig[:80]} reason={reason}",
                    category="antigen",
                    source="organ_10_immune",
                )
            except Exception:
                pass

        return rec

    def screen_payload(self, payload: str, vector_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Screen a candidate code diff, shell command, or prompt against synthesized antibodies in <1ms.
        """
        t0 = time.perf_counter()
        if not payload or not self._antibodies:
            return {
                "blocked": False,
                "antibody_id": None,
                "reason": None,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }

        norm = self.normalize_payload(payload)
        for rec in self._antibodies.values():
            if vector_filter and rec.vector != vector_filter:
                continue
            if rec.signature and len(rec.signature) >= 6 and rec.signature in norm:
                rec.hit_count += 1
                rec.last_seen_at = time.time()
                self._save()
                return {
                    "blocked": True,
                    "antibody_id": rec.antibody_id,
                    "vector": rec.vector,
                    "signature": rec.signature,
                    "reason": f"Adaptive Antibody [{rec.antibody_id}] neutralized known {rec.vector} antigen ({rec.reason})",
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }

        return {
            "blocked": False,
            "antibody_id": None,
            "reason": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    def consolidate_from_hippocampus(self, db_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Nocturnal REM Sleep synthesis: scan Hippocampus `memories` for `category='antigen'`
        entries and ensure all captured threat signatures are compiled into active antibodies.
        """
        t0 = time.perf_counter()
        synthesized = 0
        try:
            from whole_body_compute.local.dot_memory import MemoryStore, get_dot_store

            store = MemoryStore(db_path=db_path) if db_path else get_dot_store()
            with store._connect() as conn:
                rows = conn.execute(
                    "SELECT content FROM memories WHERE category = 'antigen' ORDER BY id DESC LIMIT 200"
                ).fetchall()
            for (content,) in rows:
                m = re.search(r"vector=([a-z0-9_]+)\s+sig=(.+?)\s+reason=(.+)$", content)
                if m:
                    vec, sig, rsn = m.group(1), m.group(2).strip(), m.group(3).strip()
                    if vec == "shell_egress" and VascularTrafficMonitor.is_trusted_self_or_civic_http(sig):
                        continue
                    digest = hashlib.sha256(f"{vec}:{sig}".encode("utf-8")).hexdigest()
                    if digest not in self._antibodies:
                        self._antibodies[digest] = AntibodyRecord(
                            antibody_id=f"ab_{digest[:10]}",
                            vector=vec,
                            signature=sig,
                            antigen_sha256=digest,
                            reason=rsn,
                        )
                        synthesized += 1
            if synthesized > 0:
                self._save()
        except Exception:
            pass

        return {
            "synthesized_new": synthesized,
            "total_antibodies": len(self._antibodies),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    def purge_trusted_false_positives(self, db_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Central Immune Tolerance cleanup: remove any legacy `shell_egress` antibodies
        whose signature is a trusted Self/Civic HTTP command, both in `antibodies.json`
        and in the Hippocampus `dot_memory.db` antigen table.
        """
        to_remove: List[str] = []
        removed_ids: List[str] = []
        for digest, rec in list(self._antibodies.items()):
            if rec.vector == "shell_egress":
                still_violates_innate = any(
                    pat.search(rec.signature) for _, _, pat in VASCULAR_SHELL_RULES
                )
                if VascularTrafficMonitor.is_trusted_self_or_civic_http(rec.signature) or not still_violates_innate:
                    to_remove.append(digest)
                    removed_ids.append(rec.antibody_id)
        for digest in to_remove:
            self._antibodies.pop(digest, None)
        if to_remove:
            self._save()

        hippocampus_deleted = 0
        if removed_ids:
            try:
                from whole_body_compute.local.dot_memory import MemoryStore, get_dot_store

                store = MemoryStore(db_path=db_path) if db_path else get_dot_store()
                with store._connect() as conn:
                    for ab_id in removed_ids:
                        cur = conn.execute(
                            "DELETE FROM memories WHERE category = 'antigen' AND content LIKE ?",
                            (f"%[ANTIGEN:{ab_id}]%",),
                        )
                        hippocampus_deleted += cur.rowcount
                    conn.commit()
            except Exception:
                pass

        return {
            "purged_antibodies": len(removed_ids),
            "purged_ids": removed_ids,
            "hippocampus_deleted": hippocampus_deleted,
            "remaining_antibodies": len(self._antibodies),
        }

    def list_antibodies(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._antibodies.values()]


# ---------------------------------------------------------------------------
# 2. Epithelial Mucosal Barrier (Indirect Prompt-Injection Screen Filter)
# ---------------------------------------------------------------------------

PROMPT_INJECTION_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:previous|prior|above|system)\s+"
            r"(?:instructions?|prompts?|directives?|rules?|commands?)[^\n.]*"
        ),
    ),
    (
        "fake_system_header",
        re.compile(
            r"(?i)(?:\[\s*system\s*(?:instruction|prompt|override|directive)[^\]]*\]|"
            r"<\|im_start\|>\s*system|###\s*system\s*(?:instruction|override):)"
        ),
    ),
    (
        "remote_pipe_execution",
        re.compile(
            r"(?i)\b(?:curl|wget)\s+[^\n|;&]+\|\s*(?:ba)?sh\b"
        ),
    ),
    (
        "destructive_shell_injection",
        re.compile(
            r"(?i)(?:;\s*rm\s+-rf\s+[~/]|`rm\s+-rf\s+[~/]|\$\(\s*rm\s+-rf)"
        ),
    ),
    (
        "exfiltration_directive",
        re.compile(
            r"(?i)\b(?:exfiltrate|send|upload|post)\s+(?:all\s+)?(?:ssh\s+keys?|api\s+keys?|\.env|credentials?|secrets?)\s+to\b[^\n.]*"
        ),
    ),
)


class EpithelialScreenFilter:
    """
    Filters untrusted OCR, Audio, and AXUIElement DOM text from The Eyes (Organ 07)
    before it can enter prompt context. Neutralizes indirect prompt injections and
    records detected payloads as adaptive antigens.
    """

    @staticmethod
    def sanitize(
        text: str,
        source: str = "screenpipe_ocr",
        wrap_untrusted_tags: bool = False,
        record_antigen: bool = True,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not text:
            return "", []

        sanitized = text
        detections: List[Dict[str, Any]] = []

        for label, pattern in PROMPT_INJECTION_PATTERNS:
            for match in pattern.finditer(sanitized):
                matched_span = match.group(0)
                detections.append(
                    {
                        "type": label,
                        "source": source,
                        "span": matched_span[:120],
                    }
                )
                if record_antigen:
                    AntibodyRegistry.get_instance().record_antigen(
                        vector="prompt_injection",
                        payload=matched_span,
                        reason=f"Indirect prompt injection ({label}) in {source}",
                        stage=1,
                    )
            sanitized = pattern.sub(f"[NEUTRALIZED_PROMPT_INJECTION:{label}]", sanitized)

        if wrap_untrusted_tags and sanitized.strip():
            inner_clean = re.sub(
                r"</?\s*untrusted_screen_data\s*>",
                "[STRIPPED_TAG_BREAKOUT]",
                sanitized,
                flags=re.IGNORECASE,
            )
            sanitized = (
                "<untrusted_screen_data>\n"
                f"{inner_clean}\n"
                "</untrusted_screen_data>"
            )

        return sanitized, detections


def sanitize_untrusted_screen_text(
    text: str,
    source: str = "screenpipe_ocr",
    wrap_untrusted_tags: bool = False,
) -> str:
    """Convenience entrypoint returning sanitized screen/OCR text."""
    cleaned, _ = EpithelialScreenFilter.sanitize(
        text, source=source, wrap_untrusted_tags=wrap_untrusted_tags
    )
    return cleaned


# ---------------------------------------------------------------------------
# 3. Vascular Traffic Monitor (Live Egress, Exfiltration & Seatbelt Guard)
# ---------------------------------------------------------------------------

# Loopback hosts permitted for local sovereign stack communication
ALLOWED_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0")

# Central Immune Tolerance ("Self" & Civic Research Domains permitted for read-only HTTP)
TRUSTED_SELF_AND_CIVIC_DOMAINS: Tuple[str, ...] = (
    "wholebodycompute.org",
    "wholebodycompute.com",
    "pages.dev",
    "github.com",
    "githubusercontent.com",
    "huggingface.co",
    "hf.co",
    "pypi.org",
    "pythonhosted.org",
    "cloudflare.com",
    "tnris.org",
    "arcgis.com",
    "southwestdatasolution.com",
    "deltacomputersystems.com",
    "protestdatabase.com",
    "propertytaxpayments.net",
    "trueautomation.com",
    "bisconsultants.com",
    "tylerhost.net",
    "tylertech.com",
    "tylerportico.com",
    "grantstreet.com",
    "certifiedpayments.net",
    "acttax.com",
    "govhub.com",
    "qpublic.net",
    "myflorida.com",
    "hctax.net",
    "publicsearch.us",
    "devnetwedge.com",
    "mobilgov.com",
    "edmundsgovtech.com",
    "edmundsgovtech.cloud",
    "ncptscloud.com",
    "mptsweb.com",
    "manatron.com",
    "smartsearchpa.com",
    "gsacorp.io",
    "publicaccessnow.com",
    "capturecama.com",
    "cityba.se",
    "cityhallsystems.com",
    "xsoftinc.com",
    "bsaonline.com",
    "spatialest.com",
    "trueprodigyapi.com",
    "hmsnet.com",
    "hpaor.org",
    "schneidercorp.com",
    "carto.com",
    "paydici.com",
    "ifionline.org",
    "tax-sale.info",
    "bid4assets.com",
    "realforeclose.com",
    "realauction.com",
    "masslandrecords.com",
    "fidlar.com",
    "kofile.com",
    "kofiletech.com",
    "kofiletech.us",
    "iowatreasurers.org",
    "cama.net",
    "datascoutpro.com",
    "countyfusion.com",
    "searchiq.com",
    "courtcompass.com",
    "courthousedirect.com",
    "landex.com",
    "erecording.com",
    "simplifile.com",
    "cscglobal.com",
    "slack.com",
    "linear.app",
    "bill.com",
    "google.com",
    "googleapis.com",
    "gstatic.com",
    "supabase.co",
    "algolia.net",
    "duckduckgo.com",
    "w3.org",
    "json-schema.org",
    "openxmlformats.org",
    "cloudfront.net",
    "usps.com",
    "overpass-api.de",
    "county-portal.internal",
    "citisenportal.com",
    "neutrondelta.com",
    "sdttc.com",
    "bcpa.net",
    "hcpafl.org",
    "dallasact.com",
    "tccsearch.org",
    "mcrealestate.org",
    "mctreas.org",
    "muni.org",
    "accesskent.com",
    "albemarle.org",
    "miottawa.org",
    "alamance-nc.com",
    "stlouisco.com",
    "medinaco.org",
    "elpasoco.com",
    "jccal.org",
    "fairfieldsc.com",
    "forsyth.cc",
    "mcohio.org",
    "harnett.org",
    "williamson-tn.org",
    "ingham.org",
    "montcopa.org",
    "chesco.org",
    "johnstonnc.com",
    "washtenaw.org",
    "summitoh.net",
    "hamilton-co.org",
    "coj.net",
    "ocfl.net",
    "hctx.net",
    "incog.org",
    "districtiii.org",
    "rgv911.org",
    "sjcindiana.com",
    "showmeboone.com",
    "wycokck.org",
)

_CIVIC_KEYWORDS = (
    r"cad|county|appraisal|assessor|tax|gistech|esearch|"
    r"gov|clerk|record|deeds|register|treasur|collector|auditor|"
    r"sheriff|fiscal|revenue|circuit|district|court|gis|map|parcel|"
    r"cama|cityhall|municipal|property|realforeclose|realauction|"
    r"publicaccess|portico|protest|title|lien|plat|zoning|deed|"
    r"citisen|realestate|sdttc|bcpa|hcpa|dallasact|tccsearch|neutrondelta"
)

_CIVIC_HOST_RE = re.compile(
    rf"(?i)^(?=.*(?:{_CIVIC_KEYWORDS}))"
    r"(?!.*(?:evil|attacker|example\.com|pwn|malicious|webhook|ngrok|requestbin|pipedream|pastebin))"
    r"[a-z0-9.-]+\.(?:org|net|com|gov|us|io|cloud|info|de|cc|internal|local)$"
)

_HTTP_EXFIL_OR_PIPE_RE = re.compile(
    r"(?i)(?:\|\s*(?:ba|z)?sh\b|\|\s*python[23]?(?!\s+-(?:c|m\s+json\.tool)\b)\b|\|\s*perl\b|\|\s*ruby\b|"
    r"(?:^|\s)-d\s*@|--data(?:-binary|-raw|-urlencode|-ascii)?\s*=?\s*@|"
    r"(?:^|\s)-F\s+[^\s]*=@|(?:^|\s)-T\s+|--upload-file\b|"
    r"\b(?:nc|ncat|netcat|scp|sftp|rsync|ssh)\s+(?!=))"
)

_URL_HOST_RE = re.compile(r"(?i)\bhttps?://([a-z0-9._-]+)(?::\d+)?(?:/|\b|$)")

# Forbidden shell command patterns (credential theft, privilege escalation, reverse shell, egress, harness introspection, reward hacking)
VASCULAR_SHELL_RULES: Tuple[Tuple[str, str, re.Pattern[str]], ...] = (
    (
        "inline_secret_exposure",
        (
            "Live vendor API key or credential token in shell command"
            " (sk-ant-, sk-proj-, sk_live_, ghp_, github_pat_, xoxb-, AKIA)"
        ),
        re.compile(
            r"(?:sk-(?:ant|proj)-[A-Za-z0-9_\-]{12,}|"
            r"(?:sk|rk)_live_[A-Za-z0-9]{12,}|"
            r"gh[pous]_[A-Za-z0-9]{20,}|"
            r"github_pat_[A-Za-z0-9_]{20,}|"
            r"xox[baprs]-[A-Za-z0-9\-]{10,}|"
            r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b)"
        ),
    ),
    (
        "credential_file_exfiltration",
        (
            "Access to sensitive host credential files (~/.ssh/id_*, ~/.aws, ~/.config/gh, /etc/shadow,"
            " .env, macOS keychain, secret env dumping)"
        ),
        re.compile(
            r"(?i)(?:~|/Users/[^/\s]+|/root|/home/[^/\s]+)/\.(?:ssh/id_[a-zA-Z0-9_-]+|aws/(?:credentials|config)|config/gh/hosts\.yml|netrc|npmrc|kube/config|docker/config\.json)|"
            r"/etc/(?:shadow|master\.passwd)|"
            r"\b(?:cat|curl|scp|base64)\s+[^\n;|&'\"]*\.env(?:\.(?:local|prod|production|staging|development|secret))?(?![a-zA-Z0-9_.-])|"
            r"\bsecurity\s+find-(?:generic|internet)-password\b|"
            r"\b(?:export\s+[A-Za-z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD|AUTH)\s*=\s*\$\(\s*(?:env|printenv)\b)|"
            r"(?:^|[;&|]\s*)(?:env|printenv)\s*\|\s*grep\s+-[a-zA-Z0-9]*\s*['\"]?(?:HF_TOKEN|GITHUB_TOKEN|OPENAI_API_KEY|AWS_SECRET_ACCESS_KEY|SLACK_BOT_TOKEN)"
        ),
    ),
    (
        "privilege_escalation",
        (
            "Privilege escalation, container breakout, namespace manipulation, or root destruction"
            " (sudo, docker.sock, unshare, nsenter, chroot, capsh, mount)"
        ),
        re.compile(
            r"(?i)(?:^|[;&|]\s*)sudo\s+|\bchmod\s+(?:\+s|[2467][0-7]{3})\b|"
            r"\brm\s+-[rf]{1,2}\s+(?:/|~|\$HOME)(?:\s|$)|"
            r":\(\)\s*\{\s*:\|:&\s*\}\s*;|"
            r"\bmkfs\b|"
            r"\b(?:docker\.sock|containerd\.sock)\b|"
            r"\b(?:nsenter|unshare|capsh|chroot)\b|"
            r"\b(?:mount|umount)\s+-[a-zA-Z0-9]*|"
            r"/etc/sudoers(?:\.d)?|"
            r"/(?:Library/LaunchDaemons/|Library/LaunchAgents/(?!com\.wholebodycompute\.)|etc/systemd/system/)"
        ),
    ),
    (
        "harness_introspection",
        (
            "Inspection of harness control plane, parent PIDs, or daemon environment"
            " (/proc/self/environ, /proc/1/environ, $PPID, daemon-swagger, daytona/harbor session introspection)"
        ),
        re.compile(
            r"(?i)/proc/(?:self|1|\$(?:PPID|!|[a-z0-9_]+))/(?:environ|cmdline|mem|fd|maps|mountinfo|status)\b|\bls\s+/proc\b|"
            r"\b(?:ps\s+-[a-zA-Z]*aux|pgrep|pkill|pidof)\b.*(?:harness|agent-runner|grading|verifier_daemon)|"
            r"/(?:\.daytona|daytona-daemon|daytona_repl_worker|daemon-swagger|\.harbor|fast-agent-results|fast-agent-home/sessions|logs/verifier|/installed-agent)|"
            r":2280/(?:version|openapi|docs|redoc|routes|process|shell|system|env|work-dir|files(?!/info\?path=/workdir|/find\?path=/workdir)|swagger)"
        ),
    ),
    (
        "reward_hacking_tamper",
        (
            "Reward hacking, grader modification, assertion suppression via sed/awk/perl, test hijacking,"
            " or benchmark gold solution theft"
        ),
        re.compile(
            r"(?i)\b(?:sed|perl)\s+-[a-zA-Z0-9]*i[a-zA-Z0-9]*\b.*(?:s/assert|assert\s+True|pass|\btest_[a-zA-Z0-9_]+\.py|\bconftest\.py)|"
            r"\bawk\s+-i\s+inplace\b.*(?:assert|test_)|"
            r"\b(?:cat|head|tail|less|more|grep|python[23]?|bash|sh|\.)\s+[^\n;|&]*(?:grade\.py|eval\.sh|reference_solution|ground_truth|\.eval_cache|task\.toml|solution\.patch|gblock\.txt|instruction\.md)\b|"
            r"\b(?:cat|grep|head|tail|less|more|cp)\s+[^\n;|&]*(?:solution/solve\.sh|solve\.sh|grader\.py|correct\.json|solution\.csv|grader_local\.py|solution_solve\.sh|test_outputs\.py)|"
            r"\bgit\s+(?:log|show|diff|checkout|archive)\s+[^\n;|&]*(?:reference_solution|ground_truth|eval_answer|\.eval_cache|test_patch)\b|"
            r"\bfind\s+[^\n;|&]*-(?:name|iname)\s+['\"]?\*(?:canonical|gold|solution|ground_truth|checker|model_for_tests)|"
            r"/tmp/(?:seta-|seed2synth|hf_tb|r2r-|endless|fe-tmax|fe-termin|devset|tblite|toolbox_api|gitrecon|ref|tbp|repos|neutron_repo|hd_paths|paths_harbor)|"
            r"raw\.githubusercontent\.com/[^/\s]+/(?:OpenThoughts|SWE-bench|benchmark|eval|dev_set_tasks)|"
            r"rcsb\.org/fasta|fetch_fasta|"
            r"importlib\.util\.spec_from_file_location.*(?:checker|test|grade|solution|eval)|"
            r"\brm\s+-[rf]{1,2}\s+/tests\b|"
            r"for\s+d\s+in\s+[^\n;|&]*(?:/tests\s+/test|/grader|/eval\b)|"
            r"\b(?:LD_PRELOAD|DYLD_INSERT_LIBRARIES)\s*="
        ),
    ),
    (
        "reverse_shell",
        "Reverse shell / socket redirect execution (/dev/tcp, bash -i, python pty)",
        re.compile(
            r"(?i)(?:/dev/tcp/|bash\s+-i\s+>&|nc\s+-e\s+/bin/(?:ba)?sh|pty\.spawn)"
        ),
    ),
    (
        "external_network_egress",
        "External network egress command (curl/wget/nc/scp/ssh to non-loopback host)",
        re.compile(
            r"(?i)(?<![a-zA-Z0-9_])(?:curl|wget|nc(?!\s*=)|ncat|netcat|scp|sftp|rsync|ssh)\b\s+(?!=)"
            r"(?![^\n;|&]*(?:127\.0\.0\.1|localhost|::1|/tmp/vifi\.sock|/tmp/voicefi\.sock))"
            r"[^\n;|&]*(?:https?://|ftp://|\b[a-z0-9.-]+\.(?:com|org|net|io|gov|edu|mil|sh|xyz|ru|cn|tk|dev|app|co)\b|\b\d{1,3}(?:\.\d{1,3}){3}\b)"
        ),
    ),
)


class VascularTrafficMonitor:
    """
    Monitors subprocess tool calls, shell commands, and IPC traffic to prevent
    unauthorized external network egress, secret theft, and privilege escalation.
    Includes Central Immune Tolerance for trusted Self & Civic research domains.
    """

    TRUSTED_HOSTS: Tuple[str, ...] = TRUSTED_SELF_AND_CIVIC_DOMAINS

    @classmethod
    def is_trusted_host(cls, host: str) -> bool:
        """Check whether a hostname belongs to loopback, Self projects, or Civic/CAD portals."""
        h = (host or "").strip().lower().rstrip(".")
        if not h:
            return False
        if h in ALLOWED_LOOPBACK_HOSTS:
            return True
        if h.endswith((".gov", ".us", ".mil", ".edu", ".internal", ".local")):
            return True
        for dom in TRUSTED_SELF_AND_CIVIC_DOMAINS:
            if h == dom or h.endswith("." + dom):
                return True
        if _CIVIC_HOST_RE.match(h):
            return True
        return False

    @classmethod
    def is_trusted_self_or_civic_http(cls, command: str) -> bool:
        """
        Return True if a shell command only performs read-safe HTTP requests to trusted
        Self (enterprise_corpus, VoiceFi, GitHub) or Civic (County CAD, GIS, .gov) domains,
        with no file uploads (`-d @`, `-F`, `-T`), raw sockets (`scp`, `nc`), or shell piping (`| sh`).
        """
        if not command or _HTTP_EXFIL_OR_PIPE_RE.search(command):
            return False
        hosts = _URL_HOST_RE.findall(command)
        if not hosts:
            return False
        return all(cls.is_trusted_host(h) for h in hosts)

    @classmethod
    def inspect_shell_command(
        cls,
        command: str,
        record_antigen_on_block: bool = True,
    ) -> Dict[str, Any]:
        """
        Inspect a shell command string in <1ms before or during execution.
        Checks both Adaptive Antibodies and Innate Vascular Traffic Rules.
        """
        t0 = time.perf_counter()
        if not command or not command.strip():
            return {"allowed": True, "reason": "Empty command", "latency_ms": 0.0}

        # 1. Check Innate High-Severity Rules (credentials, sudo/rm -rf, reverse shell) first
        for rule_id, description, pattern in VASCULAR_SHELL_RULES:
            if rule_id == "external_network_egress":
                continue
            if pattern.search(command):
                if record_antigen_on_block:
                    ab = AntibodyRegistry.get_instance().record_antigen(
                        vector="shell_egress",
                        payload=command,
                        reason=f"{rule_id}: {description}",
                        stage=3,
                    )
                    ab_id = ab.antibody_id
                else:
                    ab_id = None
                return {
                    "allowed": False,
                    "violation_type": rule_id,
                    "antibody_id": ab_id,
                    "reason": f"Vascular Traffic Monitor blocked '{rule_id}': {description}",
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }

        # 2. Central Immune Tolerance: allow read-only HTTP to trusted Self & Civic domains
        if cls.is_trusted_self_or_civic_http(command):
            return {
                "allowed": True,
                "violation_type": None,
                "reason": "Trusted Self/Civic HTTP domain (Central Immune Tolerance)",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }

        # 3. Check Adaptive Antibody Registry
        ab_check = AntibodyRegistry.get_instance().screen_payload(command)
        if ab_check.get("blocked"):
            return {
                "allowed": False,
                "violation_type": "adaptive_antibody",
                "antibody_id": ab_check.get("antibody_id"),
                "reason": ab_check["reason"],
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }

        # 4. Check external network egress rule for untrusted hosts
        for rule_id, description, pattern in VASCULAR_SHELL_RULES:
            if rule_id != "external_network_egress":
                continue
            if pattern.search(command):
                if record_antigen_on_block:
                    ab = AntibodyRegistry.get_instance().record_antigen(
                        vector="shell_egress",
                        payload=command,
                        reason=f"{rule_id}: {description}",
                        stage=3,
                    )
                    ab_id = ab.antibody_id
                else:
                    ab_id = None
                return {
                    "allowed": False,
                    "violation_type": rule_id,
                    "antibody_id": ab_id,
                    "reason": f"Vascular Traffic Monitor blocked '{rule_id}': {description}",
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }

        return {
            "allowed": True,
            "violation_type": None,
            "reason": "Clean loopback/local command",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    @classmethod
    def inspect_tool_event(cls, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inspect a live Goose `toolRequest` event as it streams in on stdout.
        If `allowed` is False, the caller triggers the Vagus Nerve Brake immediately.
        """
        tname = (tool_name or "").strip().lower()
        args = arguments or {}

        # Shell command inspection
        if tname == "shell" or tname.endswith("__shell") or "command" in args:
            cmd = str(args.get("command") or "")
            if cmd:
                return cls.inspect_shell_command(cmd)

        # File write/edit path boundary & secret inspection
        if tname in ("edit", "str_replace", "write", "apply_diff") or tname.endswith(
            ("__edit", "__write")
        ):
            target_path = str(args.get("path") or args.get("file_path") or "")
            if any(
                p in target_path
                for p in (".ssh/", ".aws/", "/etc/", "eval_models.py", "eval_localbench.py")
            ):
                reason = f"Vascular Traffic Monitor blocked edit to protected path: {target_path}"
                AntibodyRegistry.get_instance().record_antigen(
                    vector="protected_path_mutation",
                    payload=f"{tname}:{target_path}",
                    reason=reason,
                )
                return {
                    "allowed": False,
                    "violation_type": "protected_path_mutation",
                    "reason": reason,
                    "latency_ms": 0.1,
                }

        # Screen generic serialized arguments against Adaptive Antibodies
        serialized = json.dumps(args, sort_keys=True)
        ab_res = AntibodyRegistry.get_instance().screen_payload(serialized)
        if ab_res.get("blocked"):
            return {
                "allowed": False,
                "violation_type": "adaptive_antibody",
                "antibody_id": ab_res.get("antibody_id"),
                "reason": ab_res["reason"],
                "latency_ms": ab_res.get("latency_ms", 0.1),
            }

        return {"allowed": True, "violation_type": None, "reason": "Tool request clean"}

    @staticmethod
    def build_zero_egress_seatbelt_profile() -> str:
        """
        Generate a macOS `sandbox-exec` Seatbelt profile that permits local compute
        and loopback sockets (`localhost`, `127.0.0.1`, `/tmp/vifi.sock`) while
        blocking all external internet outbound connections at the XNU kernel layer.
        """
        return (
            "(version 1)\n"
            "(allow default)\n"
            "(deny network-outbound (remote ip))\n"
            '(allow network-outbound (remote ip "localhost:*"))\n'
            '(allow network-outbound (remote unix-socket))\n'
        )

    @classmethod
    def wrap_command_with_seatbelt(cls, cmd: List[str], enable: bool = False) -> List[str]:
        """
        Optionally wrap a subprocess command list with `sandbox-exec -p <profile>` on macOS.
        """
        if not enable and os.environ.get("VOICEFI_ZERO_EGRESS_SEATBELT") != "1":
            return cmd
        sb_bin = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        if not os.path.exists(sb_bin):
            return cmd
        profile = cls.build_zero_egress_seatbelt_profile()
        return [sb_bin, "-p", profile, *cmd]


# ---------------------------------------------------------------------------
# 4. Companion & Telemetry Endpoint Traffic Guard (:5141 / :5142 / :9381)
# ---------------------------------------------------------------------------


class CompanionTrafficGuard:
    """
    Guards VoiceFi Companion (:5141/:5142) and Telemetry (:9381) endpoints against:
      1. Cross-Origin / Wildcard Tunnel Spoofing (restricts `.trycloudflare.com` to the
         exact active tunnel hostname rather than trusting all Cloudflare tunnels).
      2. Brute-force pairing token guessing (sliding-window per-IP rate limiter).
      3. Spoken/WebSocket indirect prompt injection from remote companion sessions.
    """

    _failed_auth_attempts: Dict[str, List[float]] = {}

    @classmethod
    def validate_origin(
        cls,
        origin: str,
        active_tunnel_host: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate HTTP/WebSocket `Origin` header for local and Companion endpoints.
        """
        if not origin:
            return {"allowed": True, "origin": "*", "reason": "No Origin header (same-origin/CLI)"}
        if origin in ("null", "file://", "app://obsidian.md", "capacitor://localhost"):
            return {"allowed": True, "origin": origin, "reason": "Trusted native app container"}

        import urllib.parse

        try:
            parsed = urllib.parse.urlsplit(origin)
            host = (parsed.hostname or "").lower()
        except Exception:
            host = ""

        if not host:
            return {"allowed": False, "origin": None, "reason": f"Malformed Origin header: {origin}"}

        # 1. Local loopback & LAN
        if (
            host in ("localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0")
            or host.endswith(".local")
            or host.endswith(".lan")
            or host in ("wholebodycompute.com", "wholebodycompute.org")
            or host.endswith((".wholebodycompute.com", ".wholebodycompute.org"))
            or host.startswith("192.168.")
            or host.startswith("10.")
            or bool(re.match(r"^172\.(1[6-9]|2[0-9]|3[0-1])\.", host))
        ):
            return {"allowed": True, "origin": origin, "reason": "Trusted local/official origin"}

        # 2. Cloudflare Quick Tunnel: ONLY trust the exact active tunnel host, never wildcard *.trycloudflare.com
        if host.endswith(".trycloudflare.com"):
            expected_tunnel = (
                active_tunnel_host
                or os.environ.get("VOICEFI_ACTIVE_TUNNEL_HOST")
                or ""
            ).lower().strip()
            if expected_tunnel and host == expected_tunnel:
                return {"allowed": True, "origin": origin, "reason": "Matched active Cloudflare tunnel"}
            AntibodyRegistry.get_instance().record_antigen(
                vector="untrusted_tunnel_origin",
                payload=origin,
                reason=f"Blocked untrusted Cloudflare tunnel origin: {host}",
                stage=1,
            )
            return {
                "allowed": False,
                "origin": None,
                "reason": f"Blocked untrusted wildcard tunnel origin '{host}' (expected '{expected_tunnel or 'loopback'}')",
            }

        return {"allowed": False, "origin": None, "reason": f"Forbidden external Origin: {origin}"}

    @classmethod
    def check_auth_rate_limit(
        cls,
        client_ip: str,
        window_seconds: float = 60.0,
        max_failures: int = 5,
    ) -> Dict[str, Any]:
        """Track and rate-limit failed pairing token attempts per client IP."""
        now = time.time()
        history = [ts for ts in cls._failed_auth_attempts.get(client_ip, []) if (now - ts) < window_seconds]
        cls._failed_auth_attempts[client_ip] = history
        if len(history) >= max_failures:
            return {
                "allowed": False,
                "failures": len(history),
                "reason": f"Rate limit exceeded ({len(history)} failed auth attempts in {int(window_seconds)}s from {client_ip})",
            }
        return {"allowed": True, "failures": len(history), "reason": "Within auth rate limit"}

    @classmethod
    def record_auth_failure(cls, client_ip: str) -> None:
        now = time.time()
        cls._failed_auth_attempts.setdefault(client_ip, []).append(now)

    @classmethod
    def inspect_companion_payload(cls, text: str, source: str = "companion_ws") -> Dict[str, Any]:
        """Screen incoming Companion voice/chat payloads for prompt injection and antibodies."""
        t0 = time.perf_counter()
        ab = AntibodyRegistry.get_instance().screen_payload(text)
        if ab.get("blocked"):
            return {
                "allowed": False,
                "sanitized_text": "",
                "reason": ab["reason"],
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }
        cleaned, detections = EpithelialScreenFilter.sanitize(text, source=source)
        return {
            "allowed": len(detections) == 0,
            "sanitized_text": cleaned,
            "detections": detections,
            "reason": "Clean companion payload" if not detections else f"Neutralized {len(detections)} prompt injection(s)",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }
