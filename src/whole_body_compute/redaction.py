"""
Secret redaction for text leaving VoiceFi.

Notes, transcripts, and vault snippets routinely contain live credentials —
users paste API keys into their own documents and forget. Any feature that
reads that text and then speaks it aloud, returns it to a caller, or ships it
to a cloud LLM is a credential exfiltration path.

:func:`redact_secrets` is the choke point. Apply it where text leaves its
origin, not where it is consumed, so a new consumer cannot forget to call it.

Design bias: **false positives are cheap, false negatives are not.** Masking a
harmless hex string costs the user a slightly worse answer. Missing a live
Stripe key ships it to a third party.
"""

import re
from typing import List, Pattern, Tuple

REDACTION_PLACEHOLDER = "[redacted]"

# Vendor-prefixed credentials. These prefixes are unambiguous — matching one is
# effectively proof the string is a secret, so they are matched greedily.
_PREFIXED_SECRETS: Tuple[str, ...] = (
    r"\bsk-[A-Za-z0-9_\-]{16,}",  # OpenAI / Anthropic style
    r"\bsk_(?:live|test)_[A-Za-z0-9]{8,}",  # Stripe secret
    r"\bpk_(?:live|test)_[A-Za-z0-9]{8,}",  # Stripe publishable
    r"\brk_(?:live|test)_[A-Za-z0-9]{8,}",  # Stripe restricted
    r"\bwhsec_[A-Za-z0-9+/=]{8,}",  # Stripe webhook signing
    r"\bre_(?=[A-Za-z0-9_\-]*[A-Z0-9])[A-Za-z0-9_\-]{16,}",  # Resend
    r"\bpolar_(?=[A-Za-z0-9_\-]*[A-Z0-9])[A-Za-z0-9_\-]{16,}",  # Polar
    r"\bghp_[A-Za-z0-9]{20,}",  # GitHub PAT (classic)
    r"\bgh[osu]_[A-Za-z0-9]{20,}",  # GitHub OAuth / server / user
    r"\bgithub_pat_[A-Za-z0-9_]{20,}",  # GitHub PAT (fine-grained)
    r"\bglpat-[A-Za-z0-9_\-]{16,}",  # GitLab
    r"\bxox[baprs]-[A-Za-z0-9\-]{10,}",  # Slack
    r"\bAKIA[0-9A-Z]{16}",  # AWS access key
    r"\bASIA[0-9A-Z]{16}",  # AWS temporary key
    r"\bAIza[0-9A-Za-z_\-]{35}",  # Google API key
    r"\bya29\.[0-9A-Za-z_\-]{20,}",  # Google OAuth token
    r"\bdop_v1_[a-f0-9]{40,}",  # DigitalOcean
    r"\bshpat_[a-fA-F0-9]{32}",  # Shopify
    r"\bsq0[a-z]{3}-[0-9A-Za-z_\-]{20,}",  # Square
    r"\bnpm_[A-Za-z0-9]{30,}",  # npm
    r"\bpypi-[A-Za-z0-9_\-]{32,}",  # PyPI
    r"\bhf_[A-Za-z0-9]{20,}",  # Hugging Face
    r"\bph[cx]_[A-Za-z0-9]{20,}",  # PostHog
    r"\blin_api_[A-Za-z0-9]{20,}",  # Linear API
    r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}",  # SendGrid
    r"\bxapp-[0-9]-[A-Za-z0-9\-]{16,}",  # Slack App-Level Token
    r"\bsbp_[A-Za-z0-9]{20,}",  # Supabase
    r"\bAGE-SECRET-KEY-1[A-Z0-9]{20,}",  # age private key
    r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",  # JWT
)

# Words that mark the *value* beside them as sensitive, however it is shaped.
# Excludes non-credential words containing 'auth' such as 'authenticated', 'author', 'authority',
# and excludes non-secret telemetry/LLM-token counters (e.g. 'session_id', 'input_tokens', 'secrets_blocked').
_SECRET_KEYWORD = (
    r"(?:api[_\-\s]?key|secret|token|password|passwd|pwd|credential|"
    r"auth(?!(?:enticat|or|orit|oriz)[a-z]*)|bearer|private[_\-\s]?key|access[_\-\s]?key|"
    r"client[_\-\s]?secret|signing[_\-\s]?secret|webhook[_\-\s]?secret|passphrase)"
)

_NON_CREDENTIAL_KEY_RE = re.compile(
    r"(?i)^\"?(?:"
    r"(?:input|output|max|min|total|raw|compressed|direct|quadratic|cum|cumulative|net|gross|"
    r"saved|billed|cached|uncached|prompt|completion|context|distraction|target|ceiling|limit|"
    r"count|est|estimated|price|rate|usd|num|n)[_\-\s]*tokens?[A-Za-z0-9_\-]*"
    r"|tokens?[_\-\s]+(?:saved|est|count|limit|ceiling|blocked|sent|ingested|burn|rate|budget|per|id|name|type)[A-Za-z0-9_\-]*"
    r"|secrets?[_\-\s]+(?:blocked|scrubbed|detected|redaction|count|in_output|payload|gate|pattern|re|regex)[A-Za-z0-9_\-]*"
    r"|contains_secret|redact_secrets|is_secure"
    r")\"?$"
)

_CODE_EXPRESSION_VALUE_RE = re.compile(
    r"^(?:\$[A-Za-z_]|[\(\[\{]|(?:bool|int|float|str|list|dict|set|tuple|max|min|len|sum|any|all|getattr|setattr|re|os|sys|self|cls)\b|(?:self|cls|os|sys|env|config|settings|args|kwargs|state|ctx)\.[A-Za-z_][A-Za-z0-9_]*$|[A-Za-z_][A-Za-z0-9_.]*\(|[/.~][A-Za-z0-9_.\-/]+)"
)

_COMPILED: List[Pattern[str]] = [
    # 1. PEM private key blocks - mask the whole block, headers included.
    re.compile(
        r"-----BEGIN[ A-Z]*PRIVATE KEY-----.*?-----END[ A-Z]*PRIVATE KEY-----",
        re.DOTALL | re.IGNORECASE,
    ),
    # 2. Vendor-prefixed credentials.
    re.compile("|".join(_PREFIXED_SECRETS)),
    # 3. Authorization headers (evaluated before generic KEY: value so 'Bearer <tok>' is masked cleanly).
    re.compile(r"(?i)\b(authorization\s*:\s*)(?:bearer\s+|basic\s+)?[A-Za-z0-9._\-+/=]{8,}"),
    # 4. KEY=value / KEY: value / "KEY": "value" / KEY is value where the key name is sensitive.
    #    The value is masked; the key name survives so the text still reads.
    #    Excludes '&' so query strings do not greedily swallow subsequent parameters, and
    #    excludes boolean/null/numeric/placeholder literals so YAML/JSON config booleans stay intact.
    re.compile(
        rf"(?i)(\"?\b([A-Za-z0-9_\-]*{_SECRET_KEYWORD}[A-Za-z0-9_\-]*)\"?\s*(?:[:=]|\s+is\s+)\s*)"
        r"(\"|')?(?!(?:true|false|null|none|yes|no|\d+(?:\.\d+)?)\b|\[redacted\])([^\s\"'\}\],;\[&]{6,})(\3)?",
    ),
    # 5. Database URI passwords (e.g. postgres://user:secret@host, mysql://user:p@ss!@host)
    re.compile(
        r"(?i)\b([a-z][a-z0-9+.\-]*://[^:\s/@]+:)"
        r"(?:[^\s/]+)"
        r"(@(?:\[[0-9a-f:]+\]|[a-z0-9.\-]+)(?::\d+)?(?:/[^\s]*)?)"
    ),
    # 6. Sensitive query parameter values in URLs (e.g. ?token=secret, &apiKey=key)
    re.compile(
        r"(?i)([?&](?:token|apiKey|api_key|key|password|passwd|pwd|secret|auth)=)"
        r"([^&\s\"'>]{6,})"
    ),
]

# 5. Long high-entropy strings with no other explanation. Deliberately
#    conservative: 32+ chars mixing case and digits, or 40+ hex.
_HIGH_ENTROPY = re.compile(
    r"\b(?=[A-Za-z0-9+/_\-=]{32,}(?:\b|(?<==)(?![A-Za-z0-9+/_\-=])))"
    r"(?=[A-Za-z0-9+/_\-=]*[a-z])(?=[A-Za-z0-9+/_\-=]*[A-Z])(?=[A-Za-z0-9+/_\-=]*[0-9])"
    r"[A-Za-z0-9+/_\-=]{32,}(?:\b|(?<==)(?![A-Za-z0-9+/_\-=]))"
)
_LONG_HEX = re.compile(r"\b[a-fA-F0-9]{40,}\b")


def redact_secrets(text: str, placeholder: str = REDACTION_PLACEHOLDER) -> str:
    """
    Mask credential-shaped substrings in ``text``.

    Returns the text with secrets replaced by ``placeholder``. Surrounding
    prose is preserved so the result still reads naturally aloud.
    """
    if not text:
        return text

    result = text

    # PEM blocks and prefixed vendor tokens: mask the entire match.
    result = _COMPILED[0].sub(placeholder, result)
    result = _COMPILED[1].sub(placeholder, result)

    # Authorization header & keyword-assigned values: keep the key name, mask the value.
    result = _COMPILED[2].sub(lambda m: f"{m.group(1)}{placeholder}", result)

    def _replace_keyword_assignment(m: re.Match[str]) -> str:
        key_name = (m.group(2) or "").strip()
        val_str = (m.group(4) or "").strip()
        if _NON_CREDENTIAL_KEY_RE.match(key_name):
            return m.group(0)
        if _CODE_EXPRESSION_VALUE_RE.match(val_str):
            return m.group(0)
        return f"{m.group(1)}{placeholder}"

    result = _COMPILED[3].sub(_replace_keyword_assignment, result)
    result = _COMPILED[4].sub(lambda m: f"{m.group(1)}{placeholder}{m.group(2)}", result)
    result = _COMPILED[5].sub(lambda m: f"{m.group(1)}{placeholder}", result)

    # Entropy-based sweep last, so already-masked spans are not re-scanned.
    def _replace_high_entropy(m: re.Match[str]) -> str:
        tok = m.group(0)
        start = m.start()
        if "/" in tok and ((start > 0 and result[start - 1] == "/") or tok.count("/") >= 2):
            return tok
        # Skip snake_case / dunder Python & test identifiers (e.g. test_round4_high_entropy_2026_11_10)
        if tok.count("_") >= 2 and not tok.endswith("="):
            return tok
        return placeholder

    def _replace_long_hex(m: re.Match[str]) -> str:
        start = m.start()
        prefix = result[max(0, start - 28) : start].lower()
        if prefix.endswith(("commit ", "tree ", "parent ")) or any(
            k in prefix for k in ("sha256", "sha1", "md5", "digest", "hash", "blob_id", "commit")
        ):
            return m.group(0)
        return placeholder

    result = _HIGH_ENTROPY.sub(_replace_high_entropy, result)
    result = _LONG_HEX.sub(_replace_long_hex, result)

    return result


def count_new_redactions(
    original_text: str,
    scrubbed_text: str,
    placeholder: str = REDACTION_PLACEHOLDER,
) -> int:
    """Return the exact number of newly inserted redaction placeholders."""
    if not original_text or not scrubbed_text or original_text == scrubbed_text:
        return 0
    return max(0, scrubbed_text.count(placeholder) - original_text.count(placeholder))


def contains_secret(text: str) -> bool:
    """True when :func:`redact_secrets` would mask something in ``text``."""
    return bool(text) and redact_secrets(text) != text


def mask_license_key(key: str) -> str:
    """Mask a VoiceFi license key for UI display (e.g. VF1-PRO-••••••••••••••••)."""
    if not key:
        return ""
    key = key.strip()
    if key.startswith("VF1-"):
        parts = key.split("-")
        if len(parts) >= 3:
            prefix = "-".join(parts[:2])
            return f"{prefix}-" + "•" * 24
    return "•" * min(len(key), 28)
