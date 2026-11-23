"""
SpinalReflexGate: Clean-Room Online Teacher-Distillation Gate for Whole Body Compute.

In plain English:
  When an AI agent inspects a large file or long shell output, Whole Body Compute can
  either return a cheap short summary/compaction or hand over the exact raw code/logs.
  `SpinalReflexGate` is a tiny local "muscle-memory" model that watches what the agent
  actually does next - if it immediately re-reads the raw file or re-runs a command with
  `-vv`, the gate learns to skip the summary and deliver exact detail right away next
  time. It checks 8 fast clues (file size, symbol count, exact symbol mention, recent
  edits, test/log type, RAM/thermal pressure, turn depth) in <0.5ms, stays silent until
  it proves >=95% accuracy, and never touches security rules.

Architectural guarantees:
1. Pure NumPy + stdlib SQLite WAL (~3 KB weights per gate, <0.05 ms inference).
2. Hybrid representation: normalized 768-d EmbeddingGemma 2 / Apple ANE / hash vector
   (namespaced by backend) + 8 structural scalars populated in <0.5ms.
3. Prior-anchored regularization (global baseline -> repo-scoped adaptation) so new
   repos start warm without catastrophic forgetting.
4. 3-state safety lifecycle: SHADOW -> GRADUATED (>=95% 5-fold CV) -> TRIPPED
   (3-strike consecutive-miss circuit breaker + 5% canary audit).
5. Constitutional Firewall: strictly forbidden for security gates (SecGate,
   secret redaction, APFS CoW, destructive shell guards).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import re
import shlex
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


def _ensure_numpy_site_packages() -> None:
    try:
        import numpy  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(os.path.abspath(sys.executable)).parent.parent / "lib" / py_ver / "site-packages",
        Path(sys.prefix) / "lib" / py_ver / "site-packages",
        Path(sys.base_prefix) / "lib" / py_ver / "site-packages",
    ]
    for cand in candidates:
        if cand.is_dir() and str(cand) not in sys.path:
            sys.path.append(str(cand))


_ensure_numpy_site_packages()

import numpy as np

from whole_body_compute.procedural_telemetry import get_wbc_data_dir

logger = logging.getLogger("whole_body_compute.local.reflex_gate")

DEFAULT_EMBED_DIM = 768
NUM_STRUCTURAL_FEATURES = 8
TOTAL_FEATURE_DIM = DEFAULT_EMBED_DIM + NUM_STRUCTURAL_FEATURES
SQLITE_BUSY_TIMEOUT_MS = 30_000
_MAX_LOCK_RETRIES = 6

# Constitutional firewall: security-critical gates must remain 100% deterministic.
FORBIDDEN_SECURITY_GATE_TOKENS = (
    "secgate",
    "secret",
    "redact",
    "apfs",
    "cow",
    "sandbox",
    "destructive",
    "security",
    "auth",
    "permission",
)

STATE_SHADOW = "SHADOW"
STATE_GRADUATED = "GRADUATED"
STATE_TRIPPED = "TRIPPED"

REGISTERED_REFLEX_GATES = (
    ("recon_scout_sufficiency", "awaiting live scout turns"),
    ("read_autoslice_window", "awaiting auto-sliced file reads"),
    ("stdout_compaction", "awaiting ambiguous shell stdout turns"),
)

_VERBOSE_FLAG_RE = re.compile(
    r"(?:^|\s)(?:-[A-Za-z]*v[A-Za-z]*|--verbose|--tb=long|--tb=native|--full-trace|--nocapture)(?:\s|$|=)"
)
_AIRLOCK_BLOB_CMD_RE = re.compile(r"--blob(?:=|\s+)([A-Za-z0-9_-]{6,32})")
_TEST_REPO_KEY_RE = re.compile(
    r"^(?:"
    r"test_[A-Za-z0-9_]*\d+|"
    r"pytest-[A-Za-z0-9_.-]+|"
    r"_pytest|"
    r"tmp(?:[A-Za-z0-9_]{4,})?|"
    r"-private-tmp-.*|"
    r"(?:dja|sym|mpl|pyl|req|sea|xar)\d+_t\d+|"
    r"swe-\d+_.*|"
    r"arm_(?:vanilla|local|wbc)|"
    r"live_probe_.*|"
    r"gauntlet\d+|"
    r"wbc_claude_.*|"
    r"lme\d+|"
    r"[a-z]+_(?:swe\d+|lme\d+|arch-\d+|lien-\d+|hard-.*|mpl\d+)"
    r")$"
)
_EPHEMERAL_PATH_PREFIXES = ("/tmp/", "/private/tmp/", "/var/folders/")


def _is_ephemeral_repo_key(repo_key: str) -> bool:
    if not repo_key or repo_key == "__global__":
        return False
    return bool(_TEST_REPO_KEY_RE.match(repo_key))
_STRIP_SHELL_FLAGS_TOKENS = {
    "-q",
    "-qq",
    "--quiet",
    "-v",
    "-vv",
    "-vvv",
    "--verbose",
    "--tb=short",
    "--tb=long",
    "--tb=line",
    "--tb=native",
    "--tb=no",
    "--tb=auto",
    "--full-trace",
    "--nocapture",
    "-s",
}


def has_verbose_shell_flags(command: str) -> bool:
    """Return True if a shell command includes explicit verbose / full-trace flags."""
    if not command:
        return False
    return bool(_VERBOSE_FLAG_RE.search(command))


def extract_airlock_blob_id(command: str) -> str:
    """Extract `--blob <id>` from a shell command (e.g. `body immune --blob <id> --lines 1:50`)."""
    if not command or "--blob" not in command:
        return ""
    m = _AIRLOCK_BLOB_CMD_RE.search(command)
    return m.group(1) if m else ""


def normalize_shell_command_signature(command: str) -> str:
    """
    Canonicalize a test/build shell command by stripping verbosity/traceback flags so a
    subsequent `-v` / `--tb=long` / `--full-trace` retry matches the original command.
    """
    raw = (command or "").strip()
    if not raw:
        return ""
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()

    cleaned: list[str] = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == "--" and idx + 1 < len(tokens):
            idx += 1
            continue
        if tok in _STRIP_SHELL_FLAGS_TOKENS:
            idx += 1
            continue
        if tok == "--tb" and idx + 1 < len(tokens):
            idx += 2
            continue
        if tok.startswith("--tb="):
            idx += 1
            continue
        if re.fullmatch(r"-[A-Za-z]{2,}", tok) and ("v" in tok or "q" in tok):
            rem = "".join(ch for ch in tok[1:] if ch not in ("v", "q"))
            if rem:
                cleaned.append(f"-{rem}")
            idx += 1
            continue
        cleaned.append(tok)
        idx += 1
    return " ".join(cleaned).strip().lower()


def _assert_non_security_gate(gate_id: str) -> str:
    normalized = (gate_id or "").strip().lower()
    if not normalized:
        raise ValueError("gate_id must be a non-empty string")
    for token in FORBIDDEN_SECURITY_GATE_TOKENS:
        if token in normalized:
            raise ValueError(
                f"Constitutional firewall: '{gate_id}' appears security-related ({token!r}); "
                "SpinalReflexGate is restricted to soft compute/token routing."
            )
    return normalized


def _scalar_sigmoid(z: float) -> float:
    if z <= -30.0:
        return 1.0 / (1.0 + math.exp(30.0))
    if z >= 30.0:
        return 1.0 / (1.0 + math.exp(-30.0))
    return 1.0 / (1.0 + math.exp(-z))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class StructuralSignals:
    """8 bounded structural scalars in [0.0, 1.0] paired with the text embedding."""

    file_lines: int = 0
    symbol_count: int = 0
    exact_symbol_mention: bool = False
    recently_edited: bool = False
    is_test_or_log: bool = False
    ram_pressure: float = 0.0
    thermal_pressure: float = 0.0
    turn_depth: int = 0

    def to_vector(self) -> np.ndarray:
        log_lines = min(1.0, math.log1p(max(0, self.file_lines)) / 10.0)
        denom = max(1.0, self.file_lines / 20.0)
        sym_density = min(1.0, max(0.0, float(self.symbol_count)) / denom)
        return np.array(
            [
                log_lines,
                sym_density,
                1.0 if self.exact_symbol_mention else 0.0,
                1.0 if self.recently_edited else 0.0,
                1.0 if self.is_test_or_log else 0.0,
                min(1.0, max(0.0, float(self.ram_pressure))),
                min(1.0, max(0.0, float(self.thermal_pressure))),
                min(1.0, max(0.0, float(self.turn_depth)) / 20.0),
            ],
            dtype=np.float32,
        )


def build_hybrid_features(
    embedding: np.ndarray,
    signals: StructuralSignals | None = None,
    embed_dim: int = DEFAULT_EMBED_DIM,
) -> np.ndarray:
    """Combine L2-normalized embedding with 8 bounded structural scalars."""
    vec = np.asarray(embedding, dtype=np.float32).ravel()
    if vec.shape[0] != embed_dim:
        raise ValueError(f"Expected embedding dim {embed_dim}, got {vec.shape[0]}")
    norm = float(np.linalg.norm(vec))
    emb_norm = (vec / norm).astype(np.float32) if norm > 0.0 else np.zeros(embed_dim, dtype=np.float32)
    struct_vec = (signals or StructuralSignals()).to_vector()
    return np.concatenate([emb_norm, struct_vec]).astype(np.float32)


@dataclass
class ReflexDecision:
    gate_id: str
    repo_key: str
    state: str
    use_local: bool
    should_run_teacher: bool
    probability: float
    is_canary: bool
    reason: str
    latency_ms: float = 0.0


@dataclass
class GateMetrics:
    gate_id: str
    repo_key: str
    state: str
    sample_count: int
    cv_accuracy: float
    false_local_rate: float
    consecutive_misses: int
    rolling_audit_accuracy: float


def _unpack_sample_rows(rows: list[sqlite3.Row], feature_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack SQLite (feature_blob, label) rows in a single contiguous buffer pass."""
    n = len(rows)
    buf = b"".join(r[0] for r in rows)
    X = np.frombuffer(buf, dtype=np.float32).reshape(n, feature_dim).copy()
    y = np.fromiter((int(r[1]) for r in rows), dtype=np.int32, count=n)
    return X, y


def _fit_prior_anchored_logistic(
    X: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 1.0,
    w_prior: np.ndarray | None = None,
    b_prior: float = 0.0,
    max_iters: int = 80,
    balanced: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Fit L2-regularized logistic regression with optional prior anchoring:
      min_{w,b} weighted_mean(BCE(y, sigmoid(Xw + b))) + (l2_reg / (2N)) * ||w - w_prior||^2
    Uses float32 BLAS Nesterov Accelerated Gradient for deterministic <3ms convergence.
    """
    n_samples, n_features = X.shape
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    y32 = y.astype(np.float32, copy=False)
    w_p = (
        np.zeros(n_features, dtype=np.float32)
        if w_prior is None
        else np.ascontiguousarray(w_prior, dtype=np.float32)
    )
    w = w_p.copy()
    b = float(b_prior)

    n_pos = int(np.sum(y32 >= 0.5))
    n_neg = n_samples - n_pos
    if balanced and n_pos > 0 and n_neg > 0:
        sw = np.where(y32 >= 0.5, 0.5 / float(n_pos), 0.5 / float(n_neg)).astype(np.float32)
    else:
        sw = np.full(n_samples, 1.0 / max(1, n_samples), dtype=np.float32)

    # Smoothness upper bound L <= 0.25 * sum(sw * ||x_i||^2) + l2_reg / N
    sq_norm_mean = float(np.dot(sw, np.einsum("ij,ij->i", X32, X32))) + 1.0
    reg_scale = float(l2_reg) / max(1, n_samples)
    lipschitz = 0.25 * sq_norm_mean + reg_scale
    lr = float(1.0 / max(lipschitz, 1e-6))
    reg_f32 = np.float32(reg_scale)
    lr_f32 = np.float32(lr)

    v_w = w.copy()
    v_b = b
    t = 1.0

    for _ in range(max_iters):
        logits = X32 @ v_w + np.float32(v_b)
        np.clip(logits, -30.0, 30.0, out=logits)
        probs = 1.0 / (1.0 + np.exp(-logits))
        err = (probs - y32) * sw

        grad_w = (X32.T @ err) + reg_f32 * (v_w - w_p)
        grad_b = float(np.sum(err)) + reg_scale * 0.1 * (v_b - b_prior)

        w_next = v_w - lr_f32 * grad_w
        b_next = v_b - lr * grad_b

        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        beta = (t - 1.0) / t_next
        beta_f32 = np.float32(beta)

        v_w = w_next + beta_f32 * (w_next - w)
        v_b = b_next + beta * (b_next - b)
        w = w_next
        b = b_next
        t = t_next

        if float(np.dot(grad_w, grad_w)) < 1e-10 and abs(grad_b) < 1e-5:
            break

    return w.astype(np.float32, copy=False), float(b)


def _stratified_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    l2_reg: float = 1.0,
    w_prior: np.ndarray | None = None,
    b_prior: float = 0.0,
    max_iters: int = 80,
    low_margin: float | None = None,
    high_margin: float | None = None,
    min_decisive_coverage: float = 0.15,
    balanced: bool = True,
) -> tuple[float, float]:
    """
    Run deterministic stratified k-fold cross-validation using batched BLAS sgemm
    across all k folds simultaneously.
    When `low_margin` and `high_margin` are provided, evaluates precision and FLR on
    decisive predictions (`prob <= low_margin` or `prob >= high_margin`), gating out
    abstain-heavy models below `min_decisive_coverage`.
    Returns (cv_accuracy, false_local_rate) where false_local_rate is FP / max(1, N_neg).
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < k or len(neg_idx) < k:
        return 0.0, 1.0

    n, d = X.shape
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    y32 = y.astype(np.float32, copy=False)

    fold_of_sample = np.empty(n, dtype=np.int32)
    fold_of_sample[pos_idx] = np.arange(len(pos_idx), dtype=np.int32) % k
    fold_of_sample[neg_idx] = np.arange(len(neg_idx), dtype=np.int32) % k

    train_mask = (fold_of_sample[:, None] != np.arange(k, dtype=np.int32)[None, :]).astype(np.float32)
    n_train = np.maximum(1.0, np.sum(train_mask, axis=0))

    w_p = (
        np.zeros((d, 1), dtype=np.float32)
        if w_prior is None
        else np.ascontiguousarray(w_prior, dtype=np.float32).reshape(d, 1)
    )
    W = np.repeat(w_p, k, axis=1)
    b = np.full(k, float(b_prior), dtype=np.float32)

    row_sq = np.einsum("ij,ij->i", X32, X32)
    if balanced:
        pos_mask = train_mask * (y32[:, None] >= 0.5)
        neg_mask = train_mask * (y32[:, None] < 0.5)
        n_pos_fold = np.maximum(1.0, np.sum(pos_mask, axis=0))
        n_neg_fold = np.maximum(1.0, np.sum(neg_mask, axis=0))
        sw = (np.where(y32[:, None] >= 0.5, 0.5 / n_pos_fold, 0.5 / n_neg_fold) * train_mask).astype(np.float32)
        sq_norm_mean = (row_sq @ sw) + 1.0
    else:
        sw = (train_mask * (1.0 / n_train)).astype(np.float32)
        sq_norm_mean = (row_sq @ train_mask) / n_train + 1.0

    reg_scale = np.float32(l2_reg) / n_train
    lipschitz = 0.25 * sq_norm_mean + reg_scale
    lr = (1.0 / np.maximum(lipschitz, 1e-6)).astype(np.float32)
    b_prior_f32 = np.float32(b_prior)

    V_W = W.copy()
    v_b = b.copy()
    t = 1.0

    for _ in range(max_iters):
        logits = X32 @ V_W + v_b
        np.clip(logits, -30.0, 30.0, out=logits)
        probs = 1.0 / (1.0 + np.exp(-logits))
        err = (probs - y32[:, None]) * sw

        grad_W = (X32.T @ err) + (V_W - w_p) * reg_scale
        grad_b = np.sum(err, axis=0) + np.float32(0.1) * reg_scale * (v_b - b_prior_f32)

        W_next = V_W - grad_W * lr
        b_next = v_b - grad_b * lr

        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        beta = np.float32((t - 1.0) / t_next)

        V_W = W_next + beta * (W_next - W)
        v_b = b_next + beta * (b_next - b)
        W = W_next
        b = b_next
        t = t_next

        if float(np.max(np.sum(grad_W * grad_W, axis=0))) < 1e-10 and float(np.max(np.abs(grad_b))) < 1e-5:
            break

    val_logits = np.einsum("ij,ij->i", X32, W[:, fold_of_sample].T) + b[fold_of_sample]
    if low_margin is not None and high_margin is not None:
        val_probs = _sigmoid(val_logits)
        hi = val_probs >= float(high_margin)
        lo = val_probs <= float(low_margin)
        decisive = hi | lo
        min_decisive = max(k, int(math.ceil(float(min_decisive_coverage) * n)))
        if int(np.sum(decisive)) < min_decisive:
            return 0.0, 1.0
        cv_acc = float(np.mean((val_probs[decisive] >= 0.5) == y[decisive]))
        false_positives = int(np.sum(hi & (y == 0)))
        false_local_rate = false_positives / float(max(1, len(neg_idx)))
        return cv_acc, false_local_rate

    preds = (val_logits >= 0.0).astype(np.int32)
    correct = int(np.sum(preds == y))
    false_positives = int(np.sum((preds == 1) & (y == 0)))
    total_neg = len(neg_idx)

    cv_acc = correct / float(n)
    false_local_rate = false_positives / float(max(1, total_neg))
    return cv_acc, false_local_rate


class SpinalReflexGate:
    """
    On-device online teacher-distillation router backed by SQLite WAL + NumPy.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        embed_dim: int = DEFAULT_EMBED_DIM,
        min_samples: int = 50,
        min_per_class: int = 8,
        cv_threshold: float = 0.95,
        max_false_local_rate: float = 0.05,
        low_margin: float = 0.25,
        high_margin: float = 0.75,
        canary_rate: float = 0.05,
        strike_limit: int = 3,
        rolling_audit_min_acc: float = 0.90,
        l2_reg: float = 1.0,
        read_only: bool = False,
    ):
        if db_path is None:
            resolved = get_wbc_data_dir() / "reflex_gates.sqlite"
            if not read_only:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(resolved)
        else:
            self.db_path = str(db_path)

        self.embed_dim = embed_dim
        self.feature_dim = embed_dim + NUM_STRUCTURAL_FEATURES
        self.min_samples = min_samples
        self.min_per_class = min_per_class
        self.cv_threshold = cv_threshold
        self.max_false_local_rate = max_false_local_rate
        self.low_margin = low_margin
        self.high_margin = high_margin
        self.canary_rate = canary_rate
        self.strike_limit = strike_limit
        self.rolling_audit_min_acc = rolling_audit_min_acc
        self.l2_reg = l2_reg
        self.read_only = read_only

        self._lock = threading.RLock()
        if read_only and self.db_path != ":memory:":
            uri_path = Path(self.db_path).resolve().as_uri()
            self._conn = sqlite3.connect(
                f"{uri_path}?mode=ro",
                uri=True,
                timeout=SQLITE_BUSY_TIMEOUT_MS / 1000.0,
                check_same_thread=False,
                isolation_level=None,
            )
        else:
            self._conn = sqlite3.connect(
                self.db_path,
                timeout=SQLITE_BUSY_TIMEOUT_MS / 1000.0,
                check_same_thread=False,
                isolation_level=None,
            )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS};")
        self._weights_cache: dict[tuple[str, str], tuple[np.ndarray, float, str, float, int]] = {}
        self._retrain_bg_lock = threading.Lock()
        self._retrain_in_flight_keys: set[tuple[str, str]] = set()
        self._retrain_dirty_keys: set[tuple[str, str]] = set()
        self._closed = False
        if not read_only:
            self._init_db()

    def _run_write_tx(self, fn):
        """Execute a write transaction under `BEGIN IMMEDIATE` with jittered retry on transient lock contention."""
        with self._lock:
            for attempt in range(_MAX_LOCK_RETRIES):
                try:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn()
                        self._conn.execute("COMMIT")
                        return result
                    except BaseException:
                        try:
                            self._conn.execute("ROLLBACK")
                        except sqlite3.Error:
                            pass
                        raise
                except sqlite3.OperationalError as exc:
                    if "locked" in str(exc).lower() and attempt + 1 < _MAX_LOCK_RETRIES:
                        time.sleep(0.004 * (2**attempt) + ((os.getpid() % 17) * 0.0003))
                        continue
                    raise

    def _init_db(self) -> None:
        with self._lock:
            for attempt in range(_MAX_LOCK_RETRIES):
                try:
                    cur = self._conn.cursor()
                    cur.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS};")
                    mode_row = cur.execute("PRAGMA journal_mode;").fetchone()
                    current_mode = str(mode_row[0]).lower() if mode_row else ""
                    if current_mode != "wal" and self.db_path != ":memory:":
                        cur.execute("PRAGMA journal_mode = WAL;")
                    cur.execute("PRAGMA synchronous = NORMAL;")
                    tables = {
                        str(r[0])
                        for r in cur.execute(
                            "SELECT name FROM sqlite_master WHERE type = 'table'"
                        ).fetchall()
                    }
                    required_tables = {
                        "reflex_samples",
                        "pending_scouts",
                        "pending_stdout_compactions",
                        "reflex_models",
                    }
                    if required_tables.issubset(tables):
                        stdout_cols = {
                            str(r[1])
                            for r in cur.execute("PRAGMA table_info(pending_stdout_compactions)").fetchall()
                        }
                        scout_cols = {
                            str(r[1])
                            for r in cur.execute("PRAGMA table_info(pending_scouts)").fetchall()
                        }
                        if (
                            "airlock_blob_id" in stdout_cols
                            and "session_id" in stdout_cols
                            and "gate_id" in scout_cols
                            and "session_id" in scout_cols
                        ):
                            return
                        cur.execute("BEGIN IMMEDIATE")
                        try:
                            stdout_cols_now = {
                                str(r[1])
                                for r in cur.execute("PRAGMA table_info(pending_stdout_compactions)").fetchall()
                            }
                            if "airlock_blob_id" not in stdout_cols_now:
                                cur.execute(
                                    "ALTER TABLE pending_stdout_compactions "
                                    "ADD COLUMN airlock_blob_id TEXT NOT NULL DEFAULT ''"
                                )
                            if "session_id" not in stdout_cols_now:
                                cur.execute(
                                    "ALTER TABLE pending_stdout_compactions "
                                    "ADD COLUMN session_id TEXT NOT NULL DEFAULT ''"
                                )
                                cur.execute(
                                    "CREATE INDEX IF NOT EXISTS idx_pending_stdout_session "
                                    "ON pending_stdout_compactions(session_id, command_sig, created_at);"
                                )
                            scout_cols_now = {
                                str(r[1])
                                for r in cur.execute("PRAGMA table_info(pending_scouts)").fetchall()
                            }
                            if "gate_id" not in scout_cols_now:
                                cur.execute(
                                    "ALTER TABLE pending_scouts "
                                    "ADD COLUMN gate_id TEXT NOT NULL DEFAULT 'recon_scout_sufficiency'"
                                )
                                cur.execute(
                                    "CREATE INDEX IF NOT EXISTS idx_pending_scouts_gate_path "
                                    "ON pending_scouts(gate_id, target_path, created_at);"
                                )
                            if "session_id" not in scout_cols_now:
                                cur.execute(
                                    "ALTER TABLE pending_scouts "
                                    "ADD COLUMN session_id TEXT NOT NULL DEFAULT ''"
                                )
                                cur.execute(
                                    "CREATE INDEX IF NOT EXISTS idx_pending_scouts_session "
                                    "ON pending_scouts(session_id, gate_id, target_path, created_at);"
                                )
                            cur.execute("COMMIT")
                            return
                        except BaseException:
                            try:
                                cur.execute("ROLLBACK")
                            except sqlite3.Error:
                                pass
                            raise

                    cur.execute("BEGIN IMMEDIATE")
                    try:
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS reflex_samples (
                                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                gate_id TEXT NOT NULL,
                                repo_key TEXT NOT NULL,
                                feature_blob BLOB NOT NULL,
                                label INTEGER NOT NULL,
                                is_audit INTEGER NOT NULL DEFAULT 0,
                                predicted_local INTEGER,
                                created_at REAL NOT NULL
                            );
                        """)
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_reflex_samples_gate_repo "
                            "ON reflex_samples(gate_id, repo_key, sample_id);"
                        )
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS pending_scouts (
                                pending_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                gate_id TEXT NOT NULL DEFAULT 'recon_scout_sufficiency',
                                session_id TEXT NOT NULL DEFAULT '',
                                target_path TEXT NOT NULL,
                                repo_key TEXT NOT NULL,
                                feature_blob BLOB NOT NULL,
                                predicted_local INTEGER NOT NULL DEFAULT 0,
                                turns_elapsed INTEGER NOT NULL DEFAULT 0,
                                created_at REAL NOT NULL
                            );
                        """)
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_pending_scouts_path "
                            "ON pending_scouts(target_path, created_at);"
                        )
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_pending_scouts_gate_path "
                            "ON pending_scouts(gate_id, target_path, created_at);"
                        )
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_pending_scouts_session "
                            "ON pending_scouts(session_id, gate_id, target_path, created_at);"
                        )
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS pending_stdout_compactions (
                                pending_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                session_id TEXT NOT NULL DEFAULT '',
                                command_sig TEXT NOT NULL,
                                had_verbose INTEGER NOT NULL DEFAULT 0,
                                airlock_blob_id TEXT NOT NULL DEFAULT '',
                                repo_key TEXT NOT NULL,
                                feature_blob BLOB NOT NULL,
                                predicted_local INTEGER NOT NULL DEFAULT 0,
                                turns_elapsed INTEGER NOT NULL DEFAULT 0,
                                created_at REAL NOT NULL
                            );
                        """)
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_pending_stdout_cmd "
                            "ON pending_stdout_compactions(command_sig, created_at);"
                        )
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_pending_stdout_session "
                            "ON pending_stdout_compactions(session_id, command_sig, created_at);"
                        )
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS reflex_models (
                                gate_id TEXT NOT NULL,
                                repo_key TEXT NOT NULL,
                                state TEXT NOT NULL,
                                weights_blob BLOB NOT NULL,
                                bias REAL NOT NULL,
                                sample_count INTEGER NOT NULL DEFAULT 0,
                                cv_accuracy REAL NOT NULL DEFAULT 0.0,
                                false_local_rate REAL NOT NULL DEFAULT 1.0,
                                consecutive_misses INTEGER NOT NULL DEFAULT 0,
                                updated_at REAL NOT NULL,
                                PRIMARY KEY (gate_id, repo_key)
                            );
                        """)
                        cur.execute("COMMIT")
                        return
                    except BaseException:
                        try:
                            cur.execute("ROLLBACK")
                        except sqlite3.Error:
                            pass
                        raise
                except sqlite3.OperationalError as exc:
                    if "locked" in str(exc).lower() and attempt + 1 < _MAX_LOCK_RETRIES:
                        time.sleep(0.005 * (2**attempt) + ((os.getpid() % 17) * 0.0004))
                        continue
                    raise

    def close(self) -> None:
        self.flush_background_retrains(timeout=2.0)
        with self._lock:
            self._closed = True
            self._conn.close()

    def _is_canary(self, gate_id: str, sample_key: str) -> bool:
        if self.canary_rate <= 0.0:
            return False
        digest = hashlib.blake2b(f"{gate_id}:{sample_key}".encode(), digest_size=4).digest()
        val = int.from_bytes(digest, "big") / float(0xFFFFFFFF)
        return val < self.canary_rate

    def predict(
        self,
        gate_id: str,
        features: np.ndarray,
        repo_key: str = "__global__",
        sample_key: str = "",
    ) -> ReflexDecision:
        """
        Evaluate whether to use the local path or run the teacher/cloud path.
        Never executes a learned shortcut unless state == GRADUATED and outside uncertainty band.
        """
        t0 = time.perf_counter()
        g_id = _assert_non_security_gate(gate_id)
        r_key = (repo_key or "__global__").strip() or "__global__"
        feat = np.asarray(features, dtype=np.float32).ravel()
        if feat.shape[0] != self.feature_dim:
            raise ValueError(f"Expected feature dim {self.feature_dim}, got {feat.shape[0]}")

        with self._lock:
            model = self._load_model_state(g_id, r_key)
            if model is None and r_key != "__global__":
                model = self._load_model_state(g_id, "__global__")

        if model is None:
            return ReflexDecision(
                gate_id=g_id,
                repo_key=r_key,
                state=STATE_SHADOW,
                use_local=False,
                should_run_teacher=True,
                probability=0.5,
                is_canary=False,
                reason="cold_start_shadow",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        w, b, state = model
        if state != STATE_GRADUATED:
            return ReflexDecision(
                gate_id=g_id,
                repo_key=r_key,
                state=state,
                use_local=False,
                should_run_teacher=True,
                probability=0.5,
                is_canary=False,
                reason=f"state_{state.lower()}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        prob = _scalar_sigmoid(float(np.dot(w, feat)) + b)
        is_canary = self._is_canary(g_id, sample_key or str(time.time_ns()))

        if self.low_margin < prob < self.high_margin:
            return ReflexDecision(
                gate_id=g_id,
                repo_key=r_key,
                state=state,
                use_local=False,
                should_run_teacher=True,
                probability=prob,
                is_canary=is_canary,
                reason="uncertain_margin",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        use_local = prob >= self.high_margin
        return ReflexDecision(
            gate_id=g_id,
            repo_key=r_key,
            state=state,
            use_local=use_local,
            should_run_teacher=is_canary or (not use_local),
            probability=prob,
            is_canary=is_canary,
            reason="graduated_canary" if is_canary else ("graduated_local" if use_local else "graduated_escalate"),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _load_model_state(self, gate_id: str, repo_key: str) -> tuple[np.ndarray, float, str] | None:
        cached = self._weights_cache.get((gate_id, repo_key))
        if cached is not None:
            w_c, b_c, state_c, cached_upd, cached_cnt = cached
            if self.db_path == ":memory:":
                return (w_c, b_c, state_c)
            cur = self._conn.cursor()
            cur.execute(
                "SELECT state, updated_at, sample_count FROM reflex_models WHERE gate_id = ? AND repo_key = ?",
                (gate_id, repo_key),
            )
            meta = cur.fetchone()
            if meta is None:
                self._weights_cache.pop((gate_id, repo_key), None)
                return None
            if (
                float(meta["updated_at"]) == cached_upd
                and str(meta["state"]) == state_c
                and int(meta["sample_count"]) == cached_cnt
            ):
                return (w_c, b_c, state_c)

        cur = self._conn.cursor()
        cur.execute(
            "SELECT state, weights_blob, bias, updated_at, sample_count FROM reflex_models WHERE gate_id = ? AND repo_key = ?",
            (gate_id, repo_key),
        )
        row = cur.fetchone()
        if row is None:
            return None
        w = np.frombuffer(row["weights_blob"], dtype=np.float32).copy()
        b_val = float(row["bias"])
        st_val = str(row["state"])
        upd_val = float(row["updated_at"])
        cnt_val = int(row["sample_count"])
        self._weights_cache[(gate_id, repo_key)] = (w, b_val, st_val, upd_val, cnt_val)
        return (w, b_val, st_val)

    def record_outcome(
        self,
        gate_id: str,
        features: np.ndarray,
        actual_local_ok: bool,
        repo_key: str = "__global__",
        predicted_local: bool | None = None,
        is_audit: bool = False,
        retrain: bool = True,
    ) -> GateMetrics:
        """
        Record a behavioral teacher label (1 = local sufficient, 0 = escalate raw)
        and update circuit breaker / graduation state.
        """
        g_id = _assert_non_security_gate(gate_id)
        r_key = (repo_key or "__global__").strip() or "__global__"
        feat = np.asarray(features, dtype=np.float32).ravel()
        if feat.shape[0] != self.feature_dim:
            raise ValueError(f"Expected feature dim {self.feature_dim}, got {feat.shape[0]}")

        label_int = 1 if actual_local_ok else 0
        pred_int = None if predicted_local is None else (1 if predicted_local else 0)
        feat_bytes = feat.tobytes()

        return self._record_outcome_internal(
            g_id=g_id,
            r_key=r_key,
            feat_bytes=feat_bytes,
            label_int=label_int,
            pred_int=pred_int,
            predicted_local=predicted_local,
            actual_local_ok=actual_local_ok,
            is_audit=is_audit,
            retrain=retrain,
            async_retrain=False,
        )

    def _insert_sample_and_update_guard_in_tx(
        self,
        cur: sqlite3.Cursor,
        g_id: str,
        r_key: str,
        feat_bytes: bytes,
        label_int: int,
        pred_int: int | None,
        predicted_local: bool | None,
        actual_local_ok: bool,
        is_audit: bool,
        now: float,
    ) -> bool:
        cur.execute(
            """
            INSERT INTO reflex_samples (
                gate_id, repo_key, feature_blob, label, is_audit, predicted_local, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (g_id, r_key, feat_bytes, label_int, 1 if is_audit else 0, pred_int, now),
        )

        tripped_now = False
        if predicted_local is True:
            cur.execute(
                "SELECT state, consecutive_misses FROM reflex_models WHERE gate_id = ? AND repo_key = ?",
                (g_id, r_key),
            )
            row = cur.fetchone()
            if row is not None:
                misses = int(row["consecutive_misses"])
                if not actual_local_ok:
                    misses += 1
                else:
                    misses = 0
                new_state = STATE_TRIPPED if misses >= self.strike_limit else str(row["state"])
                tripped_now = new_state == STATE_TRIPPED and str(row["state"]) != STATE_TRIPPED
                cur.execute(
                    "UPDATE reflex_models SET consecutive_misses = ?, state = ?, updated_at = ? "
                    "WHERE gate_id = ? AND repo_key = ?",
                    (misses, new_state, now, g_id, r_key),
                )
                if tripped_now:
                    self._weights_cache.pop((g_id, r_key), None)
                    logger.warning(
                        "SpinalReflexGate tripped on %s (%s) after %d misses", g_id, r_key, misses
                    )

        if not tripped_now and pred_int is not None:
            rolling_now = self._rolling_audit_accuracy_locked(g_id, r_key)
            if rolling_now < self.rolling_audit_min_acc:
                cur.execute(
                    "UPDATE reflex_models SET state = ?, updated_at = ? "
                    "WHERE gate_id = ? AND repo_key = ? AND state = ?",
                    (STATE_SHADOW, now, g_id, r_key, STATE_GRADUATED),
                )
                if cur.rowcount > 0:
                    self._weights_cache.pop((g_id, r_key), None)
        return tripped_now

    def _record_outcome_internal(
        self,
        *,
        g_id: str,
        r_key: str,
        feat_bytes: bytes,
        label_int: int,
        pred_int: int | None,
        predicted_local: bool | None,
        actual_local_ok: bool,
        is_audit: bool,
        retrain: bool,
        async_retrain: bool,
        return_metrics: bool = True,
    ) -> GateMetrics | None:
        with self._lock:
            need_sync_snapshot = retrain and not async_retrain

            def _insert_and_check_strikes() -> tuple[
                bool, list[sqlite3.Row], list[sqlite3.Row], sqlite3.Row | None, float
            ]:
                now = time.time()
                cur = self._conn.cursor()
                tripped_now = self._insert_sample_and_update_guard_in_tx(
                    cur=cur,
                    g_id=g_id,
                    r_key=r_key,
                    feat_bytes=feat_bytes,
                    label_int=label_int,
                    pred_int=pred_int,
                    predicted_local=predicted_local,
                    actual_local_ok=actual_local_ok,
                    is_audit=is_audit,
                    now=now,
                )

                if not need_sync_snapshot or tripped_now:
                    return tripped_now, [], [], None, 1.0

                all_rows, repo_rows, existing, rolling_acc = self._fetch_retrain_snapshot_locked(cur, g_id, r_key)
                return tripped_now, all_rows, repo_rows, existing, rolling_acc

            tripped_now, all_rows, repo_rows, existing, rolling_acc = self._run_write_tx(_insert_and_check_strikes)

            if retrain and not tripped_now:
                if async_retrain and self.db_path != ":memory:":
                    self._schedule_background_retrain(g_id, r_key)
                else:
                    self._fit_and_persist_model_locked(g_id, r_key, all_rows, repo_rows, existing, rolling_acc)

            return self.get_metrics(g_id, r_key) if return_metrics else None

    def _schedule_background_retrain(self, gate_id: str, repo_key: str) -> None:
        key = (gate_id, repo_key)
        with self._retrain_bg_lock:
            if self._closed:
                return
            if key in self._retrain_in_flight_keys:
                self._retrain_dirty_keys.add(key)
                return
            self._retrain_in_flight_keys.add(key)

        def _bg_worker() -> None:
            while True:
                try:
                    with self._lock:
                        if getattr(self, "_closed", False):
                            break
                        cur = self._conn.cursor()
                        all_rows, repo_rows, existing, rolling_acc = self._fetch_retrain_snapshot_locked(
                            cur, gate_id, repo_key
                        )
                        g_existing = None
                        g_rolling = 1.0
                        if repo_key != "__global__" and len(all_rows) >= self.min_samples:
                            cur.execute(
                                "SELECT state, consecutive_misses FROM reflex_models "
                                "WHERE gate_id = ? AND repo_key = '__global__'",
                                (gate_id,),
                            )
                            g_existing = cur.fetchone()
                            g_rolling = self._rolling_audit_accuracy_locked(gate_id, "__global__")
                    # Fit NumPy logistic + 5-fold CV OUTSIDE `self._lock` so main-thread PreToolUse never blocks!
                    computed = self._compute_model_fit(
                        repo_key, all_rows, repo_rows, existing, rolling_acc
                    )
                    g_computed = (
                        self._compute_model_fit("__global__", all_rows, all_rows, g_existing, g_rolling)
                        if (repo_key != "__global__" and len(all_rows) >= self.min_samples)
                        else None
                    )
                    if computed is not None or g_computed is not None:
                        with self._lock:
                            if not getattr(self, "_closed", False):
                                if computed is not None:
                                    w, b_val, n_samples, cv_acc, false_local_rate, new_state, consecutive_misses = computed
                                    self._persist_computed_model_locked(
                                        gate_id,
                                        repo_key,
                                        w,
                                        b_val,
                                        n_samples,
                                        cv_acc,
                                        false_local_rate,
                                        new_state,
                                        consecutive_misses,
                                    )
                                if g_computed is not None:
                                    gw, gb, gn, gcv, gflr, gstate, gmiss = g_computed
                                    self._persist_computed_model_locked(
                                        gate_id, "__global__", gw, gb, gn, gcv, gflr, gstate, gmiss
                                    )
                except Exception:
                    pass
                with self._retrain_bg_lock:
                    if key in self._retrain_dirty_keys and not getattr(self, "_closed", False):
                        self._retrain_dirty_keys.discard(key)
                        continue
                    self._retrain_in_flight_keys.discard(key)
                    break

        t = threading.Thread(target=_bg_worker, name=f"wbc-reflex-retrain-{gate_id}", daemon=True)
        t.start()

    def flush_background_retrains(self, timeout: float = 2.0) -> None:
        """Wait for any scheduled background retrains to finish (useful before close or in tests)."""
        bg_lock = getattr(self, "_retrain_bg_lock", None)
        if bg_lock is None:
            return
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with bg_lock:
                if not self._retrain_in_flight_keys:
                    return
            time.sleep(0.002)

    def _fetch_retrain_snapshot_locked(
        self,
        cur: sqlite3.Cursor,
        gate_id: str,
        repo_key: str,
    ) -> tuple[list[sqlite3.Row], list[sqlite3.Row], sqlite3.Row | None, float]:
        cur.execute(
            "SELECT feature_blob, label FROM reflex_samples WHERE gate_id = ? ORDER BY sample_id DESC LIMIT 1000",
            (gate_id,),
        )
        all_rows = cur.fetchall()
        if repo_key == "__global__":
            repo_rows = all_rows
        else:
            cur.execute(
                "SELECT feature_blob, label FROM reflex_samples "
                "WHERE gate_id = ? AND repo_key = ? ORDER BY sample_id DESC LIMIT 1000",
                (gate_id, repo_key),
            )
            repo_rows = cur.fetchall()
        cur.execute(
            "SELECT state, consecutive_misses FROM reflex_models WHERE gate_id = ? AND repo_key = ?",
            (gate_id, repo_key),
        )
        existing = cur.fetchone()
        rolling_acc = self._rolling_audit_accuracy_locked(gate_id, repo_key)
        return all_rows, repo_rows, existing, rolling_acc

    def _compute_model_fit(
        self,
        repo_key: str,
        all_rows: list[sqlite3.Row],
        rows: list[sqlite3.Row],
        existing: sqlite3.Row | None,
        rolling_acc: float,
    ) -> tuple[np.ndarray, float, int, float, float, str, int] | None:
        n_samples = len(rows)
        if n_samples == 0:
            return None

        existing_state = str(existing["state"]) if existing else STATE_SHADOW
        consecutive_misses = int(existing["consecutive_misses"]) if existing else 0
        if existing_state == STATE_TRIPPED:
            return None

        w_global: np.ndarray | None = None
        b_global = 0.0
        if repo_key != "__global__" and len(all_rows) >= self.min_per_class * 2:
            X_all, y_all = _unpack_sample_rows(all_rows, self.feature_dim)
            if int(np.sum(y_all == 1)) >= 2 and int(np.sum(y_all == 0)) >= 2:
                w_global, b_global = _fit_prior_anchored_logistic(X_all, y_all, l2_reg=self.l2_reg)

        prior_w, prior_b = (None, 0.0) if repo_key == "__global__" else (w_global, b_global)
        X, y = _unpack_sample_rows(rows, self.feature_dim)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))

        w, b = _fit_prior_anchored_logistic(
            X, y, l2_reg=self.l2_reg, w_prior=prior_w, b_prior=prior_b
        )

        cv_acc = 0.0
        false_local_rate = 1.0
        new_state = STATE_SHADOW

        if n_samples >= self.min_samples and n_pos >= self.min_per_class and n_neg >= self.min_per_class:
            cv_acc, false_local_rate = _stratified_kfold_cv(
                X,
                y,
                k=5,
                l2_reg=self.l2_reg,
                w_prior=prior_w,
                b_prior=prior_b,
                low_margin=self.low_margin,
                high_margin=self.high_margin,
                min_decisive_coverage=0.15,
                balanced=True,
            )
            if (
                cv_acc >= self.cv_threshold
                and false_local_rate <= self.max_false_local_rate
                and rolling_acc >= self.rolling_audit_min_acc
            ):
                new_state = STATE_GRADUATED
        elif n_samples >= self.min_samples and n_pos == n_samples and n_neg == 0:
            # Unanimous positive history (e.g. stdout_compaction with 0 escalations):
            # graduate with 100% empirical accuracy; any future misses still trip/demote.
            cv_acc = 1.0
            false_local_rate = 0.0
            if rolling_acc >= self.rolling_audit_min_acc:
                new_state = STATE_GRADUATED

        return w, float(b), n_samples, float(cv_acc), float(false_local_rate), new_state, consecutive_misses

    def _persist_computed_model_locked(
        self,
        gate_id: str,
        repo_key: str,
        w: np.ndarray,
        b_val: float,
        n_samples: int,
        cv_acc: float,
        false_local_rate: float,
        new_state: str,
        consecutive_misses: int,
    ) -> None:
        w_bytes = w.tobytes()

        def _persist() -> tuple[str, float]:
            now = time.time()
            cur = self._conn.cursor()
            cur.execute(
                "SELECT state, consecutive_misses FROM reflex_models WHERE gate_id = ? AND repo_key = ?",
                (gate_id, repo_key),
            )
            latest = cur.fetchone()
            if latest is not None and str(latest["state"]) == STATE_TRIPPED:
                return STATE_TRIPPED, float(now)
            latest_misses = int(latest["consecutive_misses"]) if latest is not None else consecutive_misses
            final_state = STATE_TRIPPED if latest_misses >= self.strike_limit else new_state
            if final_state == STATE_GRADUATED:
                # Re-verify live rolling audit accuracy inside BEGIN IMMEDIATE so a stale
                # background retrain can never resurrect GRADUATED over a concurrent demotion.
                live_rolling = self._rolling_audit_accuracy_locked(gate_id, repo_key)
                if live_rolling < self.rolling_audit_min_acc:
                    final_state = STATE_SHADOW
            cur.execute(
                """
                INSERT INTO reflex_models (
                    gate_id, repo_key, state, weights_blob, bias,
                    sample_count, cv_accuracy, false_local_rate, consecutive_misses, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(gate_id, repo_key) DO UPDATE SET
                    state = CASE WHEN reflex_models.state = 'TRIPPED' THEN 'TRIPPED' ELSE excluded.state END,
                    weights_blob = excluded.weights_blob,
                    bias = excluded.bias,
                    sample_count = excluded.sample_count,
                    cv_accuracy = excluded.cv_accuracy,
                    false_local_rate = excluded.false_local_rate,
                    updated_at = excluded.updated_at
                """,
                (
                    gate_id,
                    repo_key,
                    final_state,
                    w_bytes,
                    b_val,
                    n_samples,
                    float(cv_acc),
                    float(false_local_rate),
                    latest_misses,
                    now,
                ),
            )
            return final_state, now

        persisted_state, persisted_ts = self._run_write_tx(_persist)
        if persisted_state == STATE_TRIPPED:
            self._weights_cache.pop((gate_id, repo_key), None)
        else:
            self._weights_cache[(gate_id, repo_key)] = (w, b_val, persisted_state, persisted_ts, n_samples)

    def _fit_and_persist_model_locked(
        self,
        gate_id: str,
        repo_key: str,
        all_rows: list[sqlite3.Row],
        rows: list[sqlite3.Row],
        existing: sqlite3.Row | None,
        rolling_acc: float,
    ) -> None:
        computed = self._compute_model_fit(repo_key, all_rows, rows, existing, rolling_acc)
        if computed is not None:
            w, b_val, n_samples, cv_acc, false_local_rate, new_state, consecutive_misses = computed
            self._persist_computed_model_locked(
                gate_id, repo_key, w, b_val, n_samples, cv_acc, false_local_rate, new_state, consecutive_misses
            )

        # Auto-persist `__global__` baseline across all repos so unseen repos warm-start immediately
        if repo_key != "__global__" and len(all_rows) >= self.min_samples:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT state, consecutive_misses FROM reflex_models WHERE gate_id = ? AND repo_key = '__global__'",
                (gate_id,),
            )
            g_existing = cur.fetchone()
            g_rolling = self._rolling_audit_accuracy_locked(gate_id, "__global__")
            g_computed = self._compute_model_fit("__global__", all_rows, all_rows, g_existing, g_rolling)
            if g_computed is not None:
                gw, gb, gn, gcv, gflr, gstate, gmiss = g_computed
                self._persist_computed_model_locked(
                    gate_id, "__global__", gw, gb, gn, gcv, gflr, gstate, gmiss
                )

    def _retrain_locked(self, gate_id: str, repo_key: str) -> None:
        """Retrain global prior (if needed) and repo-scoped gate with 5-fold CV check."""
        cur = self._conn.cursor()
        all_rows, repo_rows, existing, rolling_acc = self._fetch_retrain_snapshot_locked(cur, gate_id, repo_key)
        self._fit_and_persist_model_locked(gate_id, repo_key, all_rows, repo_rows, existing, rolling_acc)

    def _rolling_audit_accuracy_locked(self, gate_id: str, repo_key: str, window: int = 50) -> float:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT label, predicted_local FROM reflex_samples
            WHERE gate_id = ? AND repo_key = ? AND predicted_local IS NOT NULL
            ORDER BY sample_id DESC LIMIT ?
            """,
            (gate_id, repo_key, window),
        )
        rows = cur.fetchall()
        if len(rows) < 10:
            return 1.0
        matches = sum(1 for r in rows if int(r["label"]) == int(r["predicted_local"]))
        return matches / float(len(rows))

    def reset_circuit_breaker(self, gate_id: str, repo_key: str = "__global__") -> None:
        """Reset a TRIPPED gate back to SHADOW and re-evaluate."""
        g_id = _assert_non_security_gate(gate_id)
        r_key = (repo_key or "__global__").strip() or "__global__"
        with self._lock:
            def _reset_tx() -> None:
                cur = self._conn.cursor()
                cur.execute(
                    "UPDATE reflex_models SET state = ?, consecutive_misses = 0, updated_at = ? "
                    "WHERE gate_id = ? AND repo_key = ?",
                    (STATE_SHADOW, time.time(), g_id, r_key),
                )

            self._run_write_tx(_reset_tx)
            self._weights_cache.pop((g_id, r_key), None)
            self._retrain_locked(g_id, r_key)

    def get_metrics(self, gate_id: str, repo_key: str = "__global__") -> GateMetrics:
        g_id = _assert_non_security_gate(gate_id)
        r_key = (repo_key or "__global__").strip() or "__global__"
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM reflex_models WHERE gate_id = ? AND repo_key = ?",
                (g_id, r_key),
            )
            row = cur.fetchone()
            rolling_acc = self._rolling_audit_accuracy_locked(g_id, r_key)
            if row is None:
                cur.execute(
                    "SELECT COUNT(*) AS cnt FROM reflex_samples WHERE gate_id = ? AND repo_key = ?",
                    (g_id, r_key),
                )
                cnt = int(cur.fetchone()["cnt"])
                return GateMetrics(
                    gate_id=g_id,
                    repo_key=r_key,
                    state=STATE_SHADOW,
                    sample_count=cnt,
                    cv_accuracy=0.0,
                    false_local_rate=1.0,
                    consecutive_misses=0,
                    rolling_audit_accuracy=rolling_acc,
                )
            return GateMetrics(
                gate_id=g_id,
                repo_key=r_key,
                state=str(row["state"]),
                sample_count=int(row["sample_count"]),
                cv_accuracy=float(row["cv_accuracy"]),
                false_local_rate=float(row["false_local_rate"]),
                consecutive_misses=int(row["consecutive_misses"]),
                rolling_audit_accuracy=rolling_acc,
            )

    def record_pending_scout(
        self,
        target_path: str,
        features: np.ndarray,
        repo_key: str = "__global__",
        gate_id: str = "recon_scout_sufficiency",
        session_id: str = "",
    ) -> ReflexDecision:
        """
        Shadow-evaluate a ReconScout or auto-slice read call and queue it for next-turn behavioral labeling.
        If the cloud agent does not re-read `target_path` within 2 turns, it resolves as
        `actual_local_ok=True`; if it re-reads `target_path`, `observe_tool_turn` labels it `False`.
        Scoped by `session_id` so concurrent multi-worker sessions never cross-increment each other's
        `turns_elapsed`. Skips queuing when `decision.reason == "graduated_escalate"` to prevent
        counterfactual self-poisoning.
        """
        g_id = _assert_non_security_gate(gate_id)
        r_key = (repo_key or "__global__").strip() or "__global__"
        sess_key = (session_id or "").strip()
        real_path = os.path.realpath(target_path) if target_path else ""
        decision = self.predict(g_id, features, repo_key=r_key, sample_key=real_path)

        if not self.read_only and decision.reason != "graduated_escalate":
            blob = np.asarray(features, dtype=np.float32).tobytes()
            pred_int = 1 if decision.use_local else 0
            now = time.time()

            def _insert_pending_scout() -> None:
                cur = self._conn.cursor()
                cur.execute(
                    "INSERT INTO pending_scouts "
                    "(gate_id, session_id, target_path, repo_key, feature_blob, predicted_local, turns_elapsed, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                    (g_id, sess_key, real_path, r_key, blob, pred_int, now),
                )

            self._run_write_tx(_insert_pending_scout)
        return decision

    def observe_tool_turn(
        self,
        is_raw_read: bool,
        target_path: str = "",
        gate_id: str | None = None,
        max_turns: int = 2,
        max_age_seconds: float = 120.0,
        async_retrain: bool = False,
        session_id: str = "",
    ) -> int:
        """
        Called on PreToolUse to turn the agent's next tool action into a free teacher label.
        When `session_id` is supplied, only advances and resolves rows belonging to that session
        (or legacy unscoped rows `session_id = ''`), preventing concurrent workers from falsely
        expiring each other's pending observations.
        Returns the number of pending scout/autoslice observations resolved on this turn.
        """
        if self.read_only:
            return 0
        g_filter = _assert_non_security_gate(gate_id) if gate_id is not None else None
        sess_key = (session_id or "").strip()

        with self._lock:
            # Fast sub-10us short-circuit when no pending scouts exist (common case on 99% of turns)
            has_row = self._conn.execute("SELECT 1 FROM pending_scouts LIMIT 1").fetchone()
            if has_row is None:
                return 0

            real_path = os.path.realpath(target_path) if target_path else ""
            now = time.time()

            def _resolve_scout_tx() -> tuple[list[tuple[np.ndarray, bool, str, str, bool]], set[tuple[str, str]]]:
                events: list[tuple[np.ndarray, bool, str, str, bool]] = []
                gates_repos_to_bg_retrain: set[tuple[str, str]] = set()
                cur = self._conn.cursor()

                # Scope clauses: when session_id is known, only touch rows in this session or legacy ''
                sess_where = " AND (session_id = ? OR session_id = '')" if sess_key else ""
                sess_params: tuple[object, ...] = (sess_key,) if sess_key else ()

                if is_raw_read and real_path:
                    if g_filter is not None:
                        cur.execute(
                            "SELECT pending_id, gate_id, repo_key, feature_blob, predicted_local "
                            f"FROM pending_scouts WHERE gate_id = ? AND target_path = ?{sess_where}",
                            (g_filter, real_path, *sess_params),
                        )
                    else:
                        cur.execute(
                            "SELECT pending_id, gate_id, repo_key, feature_blob, predicted_local "
                            f"FROM pending_scouts WHERE target_path = ?{sess_where}",
                            (real_path, *sess_params),
                        )
                    matched = cur.fetchall()
                    if matched:
                        ids = [int(r["pending_id"]) for r in matched]
                        cur.execute(
                            f"DELETE FROM pending_scouts WHERE pending_id IN ({','.join('?' * len(ids))})",
                            ids,
                        )
                        for r in matched:
                            feat = np.frombuffer(r["feature_blob"], dtype=np.float32).copy()
                            row_gid = str(r["gate_id"] or "recon_scout_sufficiency")
                            events.append((feat, False, row_gid, str(r["repo_key"]), bool(r["predicted_local"])))

                if g_filter is not None:
                    cur.execute(
                        f"UPDATE pending_scouts SET turns_elapsed = turns_elapsed + 1 WHERE gate_id = ?{sess_where}",
                        (g_filter, *sess_params),
                    )
                    cur.execute(
                        "SELECT pending_id, gate_id, repo_key, feature_blob, predicted_local FROM pending_scouts "
                        f"WHERE gate_id = ? AND (((turns_elapsed >= ?){sess_where}) OR (? - created_at) > ?)",
                        (g_filter, max_turns, *sess_params, now, max_age_seconds),
                    )
                else:
                    if sess_key:
                        cur.execute(
                            "UPDATE pending_scouts SET turns_elapsed = turns_elapsed + 1 "
                            "WHERE session_id = ? OR session_id = ''",
                            (sess_key,),
                        )
                        cur.execute(
                            "SELECT pending_id, gate_id, repo_key, feature_blob, predicted_local FROM pending_scouts "
                            "WHERE (turns_elapsed >= ? AND (session_id = ? OR session_id = '')) OR (? - created_at) > ?",
                            (max_turns, sess_key, now, max_age_seconds),
                        )
                    else:
                        cur.execute("UPDATE pending_scouts SET turns_elapsed = turns_elapsed + 1")
                        cur.execute(
                            "SELECT pending_id, gate_id, repo_key, feature_blob, predicted_local FROM pending_scouts "
                            "WHERE turns_elapsed >= ? OR (? - created_at) > ?",
                            (max_turns, now, max_age_seconds),
                        )
                expired = cur.fetchall()
                if expired:
                    ids = [int(r["pending_id"]) for r in expired]
                    cur.execute(
                        f"DELETE FROM pending_scouts WHERE pending_id IN ({','.join('?' * len(ids))})",
                        ids,
                    )
                    for r in expired:
                        feat = np.frombuffer(r["feature_blob"], dtype=np.float32).copy()
                        row_gid = str(r["gate_id"] or "recon_scout_sufficiency")
                        events.append((feat, True, row_gid, str(r["repo_key"]), bool(r["predicted_local"])))

                if async_retrain and self.db_path != ":memory:":
                    for feat, actual_ok, row_gid, r_key, pred_local in events:
                        tripped = self._insert_sample_and_update_guard_in_tx(
                            cur=cur,
                            g_id=row_gid,
                            r_key=r_key,
                            feat_bytes=feat.tobytes(),
                            label_int=1 if actual_ok else 0,
                            pred_int=1 if pred_local else 0,
                            predicted_local=pred_local,
                            actual_local_ok=actual_ok,
                            is_audit=False,
                            now=now,
                        )
                        if not tripped:
                            gates_repos_to_bg_retrain.add((row_gid, r_key))
                return events, gates_repos_to_bg_retrain

            resolved_events, bg_pairs = self._run_write_tx(_resolve_scout_tx)

        if async_retrain and self.db_path != ":memory:":
            for row_gid, r_key in bg_pairs:
                self._schedule_background_retrain(row_gid, r_key)
            return len(resolved_events)

        for feat, actual_ok, row_gid, r_key, pred_local in resolved_events:
            self._record_outcome_internal(
                g_id=row_gid,
                r_key=r_key,
                feat_bytes=feat.tobytes(),
                label_int=1 if actual_ok else 0,
                pred_int=1 if pred_local else 0,
                predicted_local=pred_local,
                actual_local_ok=actual_ok,
                is_audit=False,
                retrain=True,
                async_retrain=False,
                return_metrics=False,
            )
        return len(resolved_events)

    def record_pending_stdout_compaction(
        self,
        command: str,
        features: np.ndarray,
        repo_key: str = "__global__",
        gate_id: str = "stdout_compaction",
        airlock_blob_id: str = "",
        session_id: str = "",
    ) -> ReflexDecision:
        """
        Evaluate a `stdout_compaction` decision for an exit-0 shell command with large/ambiguous
        warning output and queue it for next-turn zero-cost behavioral labeling.
        Scoped by `session_id` so parallel workers cannot prematurely expire each other's pending rows.
        """
        g_id = _assert_non_security_gate(gate_id)
        r_key = (repo_key or "__global__").strip() or "__global__"
        sess_key = (session_id or "").strip()
        cmd_sig = normalize_shell_command_signature(command)
        had_verbose = 1 if has_verbose_shell_flags(command) else 0
        blob_ref = (airlock_blob_id or "").strip()
        decision = self.predict(g_id, features, repo_key=r_key, sample_key=cmd_sig or command)

        if cmd_sig and not self.read_only and decision.reason != "graduated_escalate":
            blob = np.asarray(features, dtype=np.float32).tobytes()
            pred_int = 1 if decision.use_local else 0
            now = time.time()

            def _insert_pending_stdout() -> None:
                cur = self._conn.cursor()
                cur.execute(
                    "INSERT INTO pending_stdout_compactions "
                    "(session_id, command_sig, had_verbose, airlock_blob_id, repo_key, feature_blob, predicted_local, turns_elapsed, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                    (sess_key, cmd_sig, had_verbose, blob_ref, r_key, blob, pred_int, now),
                )

            self._run_write_tx(_insert_pending_stdout)
        return decision

    def attach_pending_stdout_blob(
        self,
        command: str,
        airlock_blob_id: str,
        session_id: str = "",
    ) -> None:
        """Attach an Airlock blob ID to the most recent pending `stdout_compaction` row for `command`."""
        if self.read_only:
            return
        cmd_sig = normalize_shell_command_signature(command)
        blob_ref = (airlock_blob_id or "").strip()
        sess_key = (session_id or "").strip()
        if not cmd_sig or not blob_ref:
            return

        def _update_blob() -> None:
            cur = self._conn.cursor()
            if sess_key:
                cur.execute(
                    "UPDATE pending_stdout_compactions SET airlock_blob_id = ? "
                    "WHERE pending_id = ("
                    "  SELECT pending_id FROM pending_stdout_compactions "
                    "  WHERE command_sig = ? AND (session_id = ? OR session_id = '') "
                    "  ORDER BY pending_id DESC LIMIT 1"
                    ")",
                    (blob_ref, cmd_sig, sess_key),
                )
            else:
                cur.execute(
                    "UPDATE pending_stdout_compactions SET airlock_blob_id = ? "
                    "WHERE pending_id = ("
                    "  SELECT pending_id FROM pending_stdout_compactions "
                    "  WHERE command_sig = ? ORDER BY pending_id DESC LIMIT 1"
                    ")",
                    (blob_ref, cmd_sig),
                )

        self._run_write_tx(_update_blob)

    def observe_shell_turn(
        self,
        is_shell_tool: bool,
        command: str = "",
        gate_id: str = "stdout_compaction",
        max_turns: int = 2,
        max_age_seconds: float = 120.0,
        async_retrain: bool = False,
        session_id: str = "",
    ) -> int:
        """
        Called on PreToolUse to turn the agent's next tool action into a free teacher label
        for pending `stdout_compaction` observations, scoped by `session_id`.
        """
        if self.read_only:
            return 0
        g_id = _assert_non_security_gate(gate_id)
        sess_key = (session_id or "").strip()

        with self._lock:
            has_row = self._conn.execute("SELECT 1 FROM pending_stdout_compactions LIMIT 1").fetchone()
            if has_row is None:
                return 0

            cmd_sig = normalize_shell_command_signature(command) if is_shell_tool else ""
            is_verbose_retry = bool(is_shell_tool and cmd_sig and has_verbose_shell_flags(command))
            requested_blob_id = extract_airlock_blob_id(command) if is_shell_tool else ""
            now = time.time()

            def _resolve_tx() -> tuple[list[tuple[np.ndarray, bool, str, bool]], set[str]]:
                events: list[tuple[np.ndarray, bool, str, bool]] = []
                cur = self._conn.cursor()
                matched: list[sqlite3.Row] = []
                sess_where = " AND (session_id = ? OR session_id = '')" if sess_key else ""
                sess_params: tuple[object, ...] = (sess_key,) if sess_key else ()

                if is_verbose_retry:
                    cur.execute(
                        "SELECT pending_id, repo_key, feature_blob, predicted_local "
                        f"FROM pending_stdout_compactions WHERE command_sig = ? AND had_verbose = 0{sess_where}",
                        (cmd_sig, *sess_params),
                    )
                    matched.extend(cur.fetchall())
                if requested_blob_id:
                    cur.execute(
                        "SELECT pending_id, repo_key, feature_blob, predicted_local "
                        f"FROM pending_stdout_compactions WHERE airlock_blob_id = ?{sess_where}",
                        (requested_blob_id, *sess_params),
                    )
                    matched.extend(cur.fetchall())
                if matched:
                    seen_ids: set[int] = set()
                    deduped: list[sqlite3.Row] = []
                    for r in matched:
                        pid = int(r["pending_id"])
                        if pid not in seen_ids:
                            seen_ids.add(pid)
                            deduped.append(r)
                    ids = list(seen_ids)
                    cur.execute(
                        f"DELETE FROM pending_stdout_compactions WHERE pending_id IN ({','.join('?' * len(ids))})",
                        ids,
                    )
                    for r in deduped:
                        feat = np.frombuffer(r["feature_blob"], dtype=np.float32).copy()
                        events.append((feat, False, str(r["repo_key"]), bool(r["predicted_local"])))

                if sess_key:
                    cur.execute(
                        "UPDATE pending_stdout_compactions SET turns_elapsed = turns_elapsed + 1 "
                        "WHERE session_id = ? OR session_id = ''",
                        (sess_key,),
                    )
                    cur.execute(
                        "SELECT pending_id, repo_key, feature_blob, predicted_local FROM pending_stdout_compactions "
                        "WHERE (turns_elapsed >= ? AND (session_id = ? OR session_id = '')) OR (? - created_at) > ?",
                        (max_turns, sess_key, now, max_age_seconds),
                    )
                else:
                    cur.execute("UPDATE pending_stdout_compactions SET turns_elapsed = turns_elapsed + 1")
                    cur.execute(
                        "SELECT pending_id, repo_key, feature_blob, predicted_local FROM pending_stdout_compactions "
                        "WHERE turns_elapsed >= ? OR (? - created_at) > ?",
                        (max_turns, now, max_age_seconds),
                    )
                expired = cur.fetchall()
                if expired:
                    ids = [int(r["pending_id"]) for r in expired]
                    cur.execute(
                        f"DELETE FROM pending_stdout_compactions WHERE pending_id IN ({','.join('?' * len(ids))})",
                        ids,
                    )
                    for r in expired:
                        feat = np.frombuffer(r["feature_blob"], dtype=np.float32).copy()
                        events.append((feat, True, str(r["repo_key"]), bool(r["predicted_local"])))

                repos_to_bg: set[str] = set()
                if async_retrain and self.db_path != ":memory:":
                    for feat, actual_ok, r_key, pred_local in events:
                        tripped = self._insert_sample_and_update_guard_in_tx(
                            cur=cur,
                            g_id=g_id,
                            r_key=r_key,
                            feat_bytes=feat.tobytes(),
                            label_int=1 if actual_ok else 0,
                            pred_int=1 if pred_local else 0,
                            predicted_local=pred_local,
                            actual_local_ok=actual_ok,
                            is_audit=False,
                            now=now,
                        )
                        if not tripped:
                            repos_to_bg.add(r_key)
                return events, repos_to_bg

            resolved_events, bg_repos = self._run_write_tx(_resolve_tx)

        if async_retrain and self.db_path != ":memory:":
            for r_key in bg_repos:
                self._schedule_background_retrain(g_id, r_key)
            return len(resolved_events)

        for feat, actual_ok, r_key, pred_local in resolved_events:
            self._record_outcome_internal(
                g_id=g_id,
                r_key=r_key,
                feat_bytes=feat.tobytes(),
                label_int=1 if actual_ok else 0,
                pred_int=1 if pred_local else 0,
                predicted_local=pred_local,
                actual_local_ok=actual_ok,
                is_audit=False,
                retrain=True,
                async_retrain=False,
                return_metrics=False,
            )
        return len(resolved_events)

    def list_all_metrics(self) -> list[GateMetrics]:
        """Return metrics for all tracked (gate_id, repo_key) pairs."""
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT DISTINCT gate_id, repo_key FROM reflex_models
                    UNION
                    SELECT DISTINCT gate_id, repo_key FROM reflex_samples
                    ORDER BY gate_id, repo_key
                    """
                )
                pairs = [(str(r["gate_id"]), str(r["repo_key"])) for r in cur.fetchall()]
            except sqlite3.OperationalError:
                return []
        return [self.get_metrics(g, r) for g, r in pairs]

    def count_pending_scouts(self) -> int:
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute("SELECT COUNT(*) AS cnt FROM pending_scouts")
                return int(cur.fetchone()["cnt"])
            except sqlite3.OperationalError:
                return 0

    def count_pending_stdout_compactions(self) -> int:
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute("SELECT COUNT(*) AS cnt FROM pending_stdout_compactions")
                return int(cur.fetchone()["cnt"])
            except sqlite3.OperationalError:
                return 0

    def prune_test_repo_keys(self) -> int:
        """Remove pytest/temp `repo_key` entries from reflex tables and return deleted row count."""
        if self.read_only:
            return 0

        def _prune_tx() -> int:
            cur = self._conn.cursor()
            tables = (
                "reflex_models",
                "reflex_samples",
                "pending_scouts",
                "pending_stdout_compactions",
            )
            deleted = 0
            for tbl in tables:
                try:
                    rows = cur.execute(f"SELECT DISTINCT repo_key FROM {tbl}").fetchall()
                except sqlite3.OperationalError:
                    continue
                bad_keys = [
                    str(r["repo_key"])
                    for r in rows
                    if _TEST_REPO_KEY_RE.match(str(r["repo_key"]))
                ]
                if not bad_keys:
                    continue
                cur.execute(
                    f"DELETE FROM {tbl} WHERE repo_key IN ({','.join('?' * len(bad_keys))})",
                    bad_keys,
                )
                deleted += max(0, cur.rowcount)
            return deleted

        removed = self._run_write_tx(_prune_tx)
        if removed > 0:
            with self._lock:
                self._weights_cache.clear()
        return removed

    def consolidate_on_sleep(
        self,
        max_samples_per_gate: int = 2000,
        max_pending_age_seconds: float = 120.0,
    ) -> dict[str, int]:
        """
        Nocturnal REM Sleep maintenance for SpinalReflexGate:
        1. Prunes stale pytest/temp repo_key rows.
        2. Expires orphaned `pending_scouts` and `pending_stdout_compactions` older than `max_pending_age_seconds`.
        3. Compacts `reflex_samples` per `(gate_id, repo_key)` to `max_samples_per_gate`.
        4. Refreshes `__global__` baseline models across all active gates.
        """
        if self.read_only:
            return {"pruned_test_rows": 0, "expired_pending": 0, "compacted_samples": 0, "refreshed_gates": 0}

        pruned_test_rows = self.prune_test_repo_keys()
        now = time.time()
        cap = max(self.min_samples, int(max_samples_per_gate))

        def _sleep_tx() -> tuple[int, int, list[str]]:
            cur = self._conn.cursor()
            expired_pending = 0
            for ptbl in ("pending_scouts", "pending_stdout_compactions"):
                try:
                    cur.execute(
                        f"DELETE FROM {ptbl} WHERE (? - created_at) > ?",
                        (now, max_pending_age_seconds),
                    )
                    expired_pending += max(0, cur.rowcount)
                except sqlite3.OperationalError:
                    pass

            compacted = 0
            pairs = cur.execute(
                "SELECT DISTINCT gate_id, repo_key FROM reflex_samples"
            ).fetchall()
            gate_ids: set[str] = set()
            for r in pairs:
                gid = str(r["gate_id"])
                rkey = str(r["repo_key"])
                gate_ids.add(gid)
                cur.execute(
                    """
                    DELETE FROM reflex_samples
                    WHERE gate_id = ? AND repo_key = ? AND sample_id NOT IN (
                        SELECT sample_id FROM reflex_samples
                        WHERE gate_id = ? AND repo_key = ?
                        ORDER BY sample_id DESC LIMIT ?
                    )
                    """,
                    (gid, rkey, gid, rkey, cap),
                )
                compacted += max(0, cur.rowcount)
            return expired_pending, compacted, sorted(gate_ids)

        expired_pending, compacted_samples, gate_ids = self._run_write_tx(_sleep_tx)
        refreshed_gates = 0
        with self._lock:
            for gid in gate_ids:
                self._retrain_locked(gid, "__global__")
                refreshed_gates += 1

        return {
            "pruned_test_rows": pruned_test_rows,
            "expired_pending": expired_pending,
            "compacted_samples": compacted_samples,
            "refreshed_gates": refreshed_gates,
        }


_DEFAULT_GATE_LOCK = threading.Lock()
_DEFAULT_REFLEX_GATE: SpinalReflexGate | None = None


def get_default_reflex_gate() -> SpinalReflexGate:
    """Return the process-wide SQLite-backed SpinalReflexGate singleton."""
    global _DEFAULT_REFLEX_GATE
    with _DEFAULT_GATE_LOCK:
        if _DEFAULT_REFLEX_GATE is None:
            _DEFAULT_REFLEX_GATE = SpinalReflexGate()
        return _DEFAULT_REFLEX_GATE


def get_reflex_gate_status_summary(db_path: Path | str | None = None) -> dict[str, object]:
    """
    Non-invasively inspect reflex_gates.sqlite for `body shield` and `body doctor`.
    Does not create the database file if no observations exist yet.
    """
    resolved = Path(db_path) if db_path is not None else (get_wbc_data_dir() / "reflex_gates.sqlite")
    if not resolved.is_file():
        return {
            "initialized": False,
            "pending_scouts": 0,
            "pending_stdout_compactions": 0,
            "gates": [],
            "short_tag": "reflex:standby (0/50)",
        }
    gate = SpinalReflexGate(db_path=resolved, read_only=True)
    try:
        metrics = gate.list_all_metrics()
        pending = gate.count_pending_scouts()
        pending_stdout = gate.count_pending_stdout_compactions()
        min_samples = gate.min_samples
    finally:
        gate.close()

    graduated = len({m.gate_id for m in metrics if m.state == STATE_GRADUATED})
    total_samples = sum(m.sample_count for m in metrics)
    if graduated > 0:
        best_cv = max((m.cv_accuracy for m in metrics if m.state == STATE_GRADUATED), default=0.0)
        short_tag = f"reflex:{graduated} graduated ({best_cv * 100:.1f}% CV)"
    else:
        top_cnt = max((m.sample_count for m in metrics), default=0)
        short_tag = f"reflex:shadow ({top_cnt}/{min_samples})"

    return {
        "initialized": True,
        "pending_scouts": pending,
        "pending_stdout_compactions": pending_stdout,
        "total_samples": total_samples,
        "min_samples": min_samples,
        "gates": [
            {
                "gate_id": m.gate_id,
                "repo_key": m.repo_key,
                "state": m.state,
                "sample_count": m.sample_count,
                "cv_accuracy": round(m.cv_accuracy, 4),
                "false_local_rate": round(m.false_local_rate, 4),
                "consecutive_misses": m.consecutive_misses,
            }
            for m in metrics
        ],
        "short_tag": short_tag,
    }


def format_reflex_gate_status_lines(
    summary: dict | None = None,
    *,
    min_display_samples: int = 10,
    verbose: bool = False,
) -> list[str]:
    """Render a compact SpinalReflexGate status section for `body savings`, `body shield`, and `body doctor`."""
    data = summary if summary is not None else get_reflex_gate_status_summary()
    gates = data.get("gates") or []
    if not data.get("initialized") or not gates:
        return ["🧠 SpinalReflexGate: standby (no gates recorded yet)"]
    min_samples = int(data.get("min_samples", 50))
    lines = [f"🧠 SpinalReflexGate Status (min_samples={min_samples}):"]
    present_gate_ids = {str(g.get("gate_id", "")) for g in gates}
    hidden_low_sample = 0
    for g in gates:
        repo_key = str(g.get("repo_key", "__global__"))
        state = str(g.get("state", STATE_SHADOW))
        sample_count = int(g.get("sample_count", 0))
        if not verbose and repo_key != "__global__" and state == STATE_SHADOW:
            if _is_ephemeral_repo_key(repo_key) or sample_count < min_display_samples:
                hidden_low_sample += 1
                continue
        cv_pct = float(g.get("cv_accuracy", 0.0)) * 100.0
        lines.append(
            f"  • {g.get('gate_id', 'unknown')} [{repo_key}] | "
            f"{state} | {sample_count}/{min_samples} | "
            f"CV={cv_pct:.1f}% | misses={int(g.get('consecutive_misses', 0))}"
        )
    for reg_gate_id, _ in REGISTERED_REFLEX_GATES:
        if reg_gate_id not in present_gate_ids:
            lines.append(
                f"  • {reg_gate_id} [__global__] | {STATE_SHADOW} | 0/{min_samples} | CV=0.0% | misses=0"
            )
    if hidden_low_sample > 0:
        lines.append(
            f"  • (+{hidden_low_sample} low-sample/ephemeral shadow repos < {min_display_samples} samples)"
        )
    return lines


@functools.lru_cache(maxsize=256)
def _resolve_repo_key_for_dir(dir_str: str) -> str:
    p = Path(dir_str).resolve()
    p_str = str(p)
    if any(p_str.startswith(prefix) for prefix in _EPHEMERAL_PATH_PREFIXES) or "/.gemini/antigravity/brain/" in p_str:
        return "__global__"
    for parent in (p, *p.parents):
        if (parent / ".git").exists():
            name = parent.name or "__global__"
            return "__global__" if _is_ephemeral_repo_key(name) else name
    return "__global__"


def resolve_repo_key(target_path: str) -> str:
    """Derive a stable, lightweight repo identifier from a file path without git subprocesses."""
    if not target_path:
        return "__global__"
    p = Path(target_path)
    dir_key = str(p if p.is_dir() else p.parent)
    return _resolve_repo_key_for_dir(dir_key)


_FAST_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FAST_QUERY_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_GENERIC_SYMBOL_STOPWORDS = frozenset(
    {
        "__init__",
        "__repr__",
        "__str__",
        "__eq__",
        "__enter__",
        "__exit__",
        "main",
        "run",
        "get",
        "set",
        "read",
        "write",
        "close",
        "open",
        "to_dict",
        "from_dict",
        "test",
        "setup",
        "teardown",
    }
)
_TEST_OR_LOG_SUFFIXES = frozenset({".log", ".err", ".out", ".trace", ".jsonl"})
_SESSION_STATE_CACHE_LOCK = threading.Lock()
_SESSION_STATE_CACHE: dict[str, tuple[float, dict]] = {}
_SYMBOL_SCAN_CACHE_LOCK = threading.Lock()
_SYMBOL_SCAN_CACHE: dict[tuple[str, int, str, str, str], frozenset[str]] = {}
_EMBED_ENGINE_LOCK = threading.Lock()
_DEFAULT_EMBED_ENGINE: object | None = None
_EMBED_THREAD_LOCAL = threading.local()


def get_last_embed_backend() -> str:
    """Return the embedding backend tag recorded by the most recent `embed_for_reflex_gate*` call on this thread."""
    return str(getattr(_EMBED_THREAD_LOCAL, "last_backend", "hash") or "hash")


@functools.lru_cache(maxsize=512)
def _cached_realpath(path_str: str) -> str:
    try:
        return os.path.realpath(path_str)
    except OSError:
        return path_str


def _get_shared_embed_engine():
    global _DEFAULT_EMBED_ENGINE
    if _DEFAULT_EMBED_ENGINE is None:
        with _EMBED_ENGINE_LOCK:
            if _DEFAULT_EMBED_ENGINE is None:
                from whole_body_compute.local.embed import EmbeddingEngine

                _DEFAULT_EMBED_ENGINE = EmbeddingEngine()
    return _DEFAULT_EMBED_ENGINE


def fast_extract_symbol_names(raw_content: str, suffix: str = "") -> set[str]:
    """
    Extract top-level & member symbol identifiers in <0.25ms using C-accelerated `str.find`
    without full `ast.parse` or tree-sitter allocations on the PreToolUse hot path.
    """
    if not raw_content:
        return set()

    ext = (suffix or "").lower()
    mid = len(raw_content) // 2
    cache_sig = (ext, len(raw_content), raw_content[:64], raw_content[mid : mid + 64], raw_content[-64:])
    with _SYMBOL_SCAN_CACHE_LOCK:
        cached_syms = _SYMBOL_SCAN_CACHE.get(cache_sig)
    if cached_syms is not None:
        return set(cached_syms)

    syms: set[str] = set()
    nl = "\n"

    if ext in (".md", ".markdown"):
        pos = 0
        while True:
            idx = raw_content.find("\n#", pos)
            if idx == -1:
                break
            end_nl = raw_content.find(nl, idx + 1)
            heading = raw_content[idx + 1 : end_nl if end_nl != -1 else len(raw_content)].lstrip("#").strip()
            if heading:
                for tok in _FAST_QUERY_TOKEN_RE.findall(heading[:80]):
                    if len(tok) >= 4:
                        syms.add(tok)
            pos = idx + 2
        if raw_content.startswith("#"):
            end_nl = raw_content.find(nl)
            heading = raw_content[: end_nl if end_nl != -1 else len(raw_content)].lstrip("#").strip()
            if heading:
                for tok in _FAST_QUERY_TOKEN_RE.findall(heading[:80]):
                    if len(tok) >= 4:
                        syms.add(tok)
        with _SYMBOL_SCAN_CACHE_LOCK:
            if len(_SYMBOL_SCAN_CACHE) >= 256:
                _SYMBOL_SCAN_CACHE.clear()
            _SYMBOL_SCAN_CACHE[cache_sig] = frozenset(syms)
        return syms

    if ext in (".py", ".pyi"):
        kws: tuple[str, ...] = ("def ", "class ")
        valid_prefixes: frozenset[str] | None = frozenset({"", "async"})
    elif ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
        kws = ("function ", "class ", "interface ", "type ", "const ")
        valid_prefixes = frozenset(
            {"", "export", "export default", "async", "export async", "declare", "export declare"}
        )
    elif ext == ".rs":
        kws = ("fn ", "struct ", "enum ", "trait ", "impl ")
        valid_prefixes = frozenset({"", "pub", "pub(crate)", "pub(super)", "async", "pub async"})
    elif ext == ".go":
        kws = ("func ", "type ")
        valid_prefixes = None
    else:
        kws = ("def ", "class ", "function ", "fn ", "func ", "struct ", "interface ")
        valid_prefixes = None

    for kw in kws:
        pos = 0
        kw_len = len(kw)
        while True:
            idx = raw_content.find(kw, pos)
            if idx == -1:
                break
            line_start = raw_content.rfind(nl, 0, idx) + 1
            prefix = raw_content[line_start:idx].strip()
            if valid_prefixes is None or prefix in valid_prefixes:
                m = _FAST_IDENT_RE.match(raw_content, idx + kw_len)
                if m:
                    syms.add(m.group(0))
            pos = idx + kw_len
    with _SYMBOL_SCAN_CACHE_LOCK:
        if len(_SYMBOL_SCAN_CACHE) >= 256:
            _SYMBOL_SCAN_CACHE.clear()
        _SYMBOL_SCAN_CACHE[cache_sig] = frozenset(syms)
    return syms


def _load_cached_session_state(state_path: Path) -> dict:
    key = str(state_path)
    try:
        st = state_path.stat()
    except OSError:
        return {}
    mtime = st.st_mtime
    with _SESSION_STATE_CACHE_LOCK:
        cached = _SESSION_STATE_CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
    try:
        parsed = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            parsed = {}
    except Exception:
        parsed = {}
    with _SESSION_STATE_CACHE_LOCK:
        _SESSION_STATE_CACHE[key] = (mtime, parsed)
    return parsed


def _resolve_session_state_entry(
    conv_id: str | None = None,
    streak_file: Path | str | None = None,
) -> dict:
    if streak_file is not None:
        sf_path = Path(streak_file)
    else:
        env_streak = os.environ.get("VOICEFI_BBB_STREAK_PATH") or os.environ.get("WBC_BBB_STREAK_PATH")
        if env_streak:
            sf_path = Path(env_streak)
        else:
            env_audit = os.environ.get("VOICEFI_IMMUNE_AUDIT_PATH") or os.environ.get("WBC_IMMUNE_AUDIT_PATH")
            if env_audit:
                sf_path = Path(env_audit).parent / "vifi_bbb_streak.json"
            elif os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("VOICEFI_STRICT_TEST_HOOK") == "1":
                sf_path = Path("/tmp/vifi_bbb_streak.json")
            else:
                sf_path = get_wbc_data_dir() / "bbb_streak_state.json"

    all_state = _load_cached_session_state(sf_path)
    if not all_state:
        return {}

    resolved_conv = (
        conv_id
        or os.environ.get("WBC_CONVERSATION_ID")
        or os.environ.get("AGY_CONVERSATION_ID")
        or ""
    ).strip()
    if resolved_conv and isinstance(all_state.get(resolved_conv), dict):
        return all_state[resolved_conv]
    if isinstance(all_state.get("local"), dict):
        return all_state["local"]
    for val in all_state.values():
        if isinstance(val, dict) and ("edits" in val or "py_file_edits" in val or "tactile_churn" in val):
            return val
    return {}


def _check_recently_edited_in_session(
    target_path: str,
    c_state: dict,
    recent_mtime_window_sec: float = 900.0,
) -> bool:
    if not target_path:
        return False

    candidates = {target_path}
    real_p = _cached_realpath(target_path)
    if real_p:
        candidates.add(real_p)

    if c_state:
        for map_key in ("edits", "py_file_edits"):
            emap = c_state.get(map_key)
            if isinstance(emap, dict):
                for cand in candidates:
                    if int(emap.get(cand, 0) or 0) > 0:
                        return True
        churn_map = c_state.get("tactile_churn")
        if isinstance(churn_map, dict):
            for cand in candidates:
                entry = churn_map.get(cand)
                if isinstance(entry, dict) and (
                    int(entry.get("same_topology_edits", 0) or 0) > 0
                    or len(entry.get("seen_ast_digests") or []) > 1
                ):
                    return True

    if recent_mtime_window_sec > 0.0 and not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            st = os.stat(target_path)
            if (time.time() - st.st_mtime) <= recent_mtime_window_sec:
                return True
        except OSError:
            pass
    return False


def build_structural_signals(
    target_path: str | Path = "",
    raw_content: str = "",
    query_or_hint: str = "",
    *,
    file_lines: int | None = None,
    symbol_names: set[str] | list[str] | tuple[str, ...] | None = None,
    symbol_count: int | None = None,
    is_test_or_log: bool | None = None,
    recently_edited: bool | None = None,
    turn_depth: int | None = None,
    session_state: dict | None = None,
    conv_id: str | None = None,
    streak_file: Path | str | None = None,
    thermal_override: str | None = None,
    supervisor: object | None = None,
) -> StructuralSignals:
    """
    Populate all 8 `StructuralSignals` in <0.5ms using zero-subprocess hardware pressure,
    mtime-cached session edit state, and C-accelerated symbol scanning.
    """
    t_str = str(target_path or "")
    p = Path(t_str) if t_str else Path()
    suffix = p.suffix.lower()
    name_low = p.name.lower()

    # 1. file_lines
    resolved_lines = (
        int(file_lines)
        if file_lines is not None
        else (max(1, raw_content.count("\n") + 1) if raw_content else 0)
    )

    # 2 & 3. symbol_count & exact_symbol_mention
    if symbol_names is not None:
        syms_set = set(symbol_names)
    elif raw_content:
        syms_set = fast_extract_symbol_names(raw_content, suffix=suffix)
    else:
        syms_set = set()

    resolved_sym_count = int(symbol_count) if symbol_count is not None else len(syms_set)

    exact_mention = False
    if query_or_hint and syms_set:
        q_tokens = set(_FAST_QUERY_TOKEN_RE.findall(query_or_hint))
        matched = (syms_set & q_tokens) - _GENERIC_SYMBOL_STOPWORDS
        if matched:
            exact_mention = True
        else:
            q_low = {t.lower() for t in q_tokens if len(t) >= 5}
            for s in syms_set:
                if len(s) >= 5 and s.lower() not in _GENERIC_SYMBOL_STOPWORDS and s.lower() in q_low:
                    exact_mention = True
                    break

    # 4 & 8. session_state -> recently_edited & turn_depth
    c_state = (
        session_state
        if session_state is not None
        else _resolve_session_state_entry(conv_id=conv_id, streak_file=streak_file)
    )

    if recently_edited is not None:
        resolved_edited = bool(recently_edited)
    else:
        resolved_edited = _check_recently_edited_in_session(
            t_str,
            c_state,
            recent_mtime_window_sec=0.0 if session_state is not None else 900.0,
        )

    if turn_depth is not None:
        resolved_turn_depth = max(0, int(turn_depth))
    else:
        env_step = os.environ.get("WBC_STEP_IDX")
        if env_step is not None:
            try:
                resolved_turn_depth = max(0, int(env_step))
            except ValueError:
                resolved_turn_depth = 0
        elif c_state:
            reads_map = c_state.get("reads") if isinstance(c_state.get("reads"), dict) else {}
            edits_map = c_state.get("edits") if isinstance(c_state.get("edits"), dict) else {}
            resolved_turn_depth = max(
                int(c_state.get("streak", 0) or 0),
                sum(int(v or 0) for v in reads_map.values()) + sum(int(v or 0) for v in edits_map.values()),
            )
        else:
            resolved_turn_depth = 0

    # 5. is_test_or_log
    if is_test_or_log is not None:
        resolved_test_or_log = bool(is_test_or_log)
    else:
        norm_p = t_str.replace("\\", "/").lower()
        resolved_test_or_log = bool(
            suffix in _TEST_OR_LOG_SUFFIXES
            or name_low.startswith("test_")
            or name_low.endswith(("_test.py", ".test.ts", ".test.tsx", ".spec.ts", ".spec.js"))
            or "/tests/" in norm_p
            or "/logs/" in norm_p
            or norm_p.startswith(("tests/", "logs/"))
            or "/receipts/" in norm_p
            or ("Traceback (most recent call last):" in raw_content[:2048] if raw_content else False)
        )

    # 6 & 7. ram_pressure & thermal_pressure from ThermalSupervisor (<0.05ms)
    try:
        from whole_body_compute.local.supervisor import default_supervisor

        sup = supervisor or default_supervisor
        ram_p, therm_p = sup.get_fast_pressures(thermal_override=thermal_override)
    except Exception:
        ram_p, therm_p = 0.0, 0.0

    return StructuralSignals(
        file_lines=resolved_lines,
        symbol_count=resolved_sym_count,
        exact_symbol_mention=exact_mention,
        recently_edited=resolved_edited,
        is_test_or_log=resolved_test_or_log,
        ram_pressure=ram_p,
        thermal_pressure=therm_p,
        turn_depth=resolved_turn_depth,
    )


def embed_for_reflex_gate_with_backend(
    text: str,
    signals: StructuralSignals | None = None,
    *,
    prefer_semantic: bool | None = None,
    engine: object | None = None,
) -> tuple[np.ndarray, str]:
    """
    Generate `(776-d hybrid feature vector, backend_tag)` where `backend_tag` is one of
    `"litert_gemma2"`, `"mlx_uds"`, `"apple_ane"`, or `"hash"`.
    """
    from whole_body_compute.local.embed import DEFAULT_MLX_UDS_PATH
    from whole_body_compute.redaction import redact_secrets

    eng = engine if engine is not None else _get_shared_embed_engine()
    if not text or not text.strip():
        _EMBED_THREAD_LOCAL.last_backend = "hash"
        return (
            build_hybrid_features(np.zeros(DEFAULT_EMBED_DIM, dtype=np.float32), signals=signals),
            "hash",
        )

    safe_text = redact_secrets(text.strip())
    use_semantic = (
        prefer_semantic
        if prefer_semantic is not None
        else (not bool(os.environ.get("PYTEST_CURRENT_TEST")))
    )

    vec: np.ndarray | None = None
    backend = "hash"
    if use_semantic:
        # 1. Warm in-process LiteRT EmbeddingGemma 2 runner
        litert_runner = getattr(eng, "_litert_runner", None)
        if litert_runner is not None:
            try:
                raw_v = litert_runner.embed(f"task: search target | {safe_text}")
                arr = np.asarray(raw_v, dtype=np.float32).ravel()
                if arr.shape[0] == DEFAULT_EMBED_DIM:
                    vec = arr
                    backend = "litert_gemma2"
            except Exception:
                vec = None

        # 2. Warm MLX Unix Domain Socket (sub-ms IPC, never blocks on cold TCP timeouts)
        uds_path = getattr(eng, "uds_path", DEFAULT_MLX_UDS_PATH)
        if vec is None and uds_path and os.path.exists(uds_path):
            try:
                svc_v = eng._embed_via_service(f"task: search target | {safe_text}")
                if svc_v is not None and svc_v.shape[0] == DEFAULT_EMBED_DIM:
                    vec = svc_v
                    backend = "mlx_uds"
            except Exception:
                vec = None

        # 3. Warm Apple Silicon ANE / CoreML NaturalLanguage zero-GPU-contention embedder
        #    (Only called if NLEmbedding is already warm in-process or WBC_ALLOW_COLD_ANE_EMBED=1,
        #     preventing a ~15ms first-call Objective-C framework load inside PreToolUse hooks.)
        if vec is None:
            try:
                from whole_body_compute.local.metal_accel import get_ane_embedder

                ane = get_ane_embedder()
                if (
                    getattr(ane, "_init_done", False)
                    or os.environ.get("WBC_ALLOW_COLD_ANE_EMBED") == "1"
                    or prefer_semantic is True
                ) and ane.is_available():
                    ane_v = ane.embed(safe_text)
                    if ane_v is not None and ane_v.shape[0] == DEFAULT_EMBED_DIM:
                        vec = ane_v
                        backend = "apple_ane"
            except Exception:
                vec = None

    if vec is None:
        vec = eng._heuristic_embed(safe_text)
        backend = "hash"
    _EMBED_THREAD_LOCAL.last_backend = backend
    return build_hybrid_features(vec, signals=signals), backend


def scope_gate_id_for_backend(gate_id: str, backend: str) -> str:
    """
    Namespace `gate_id` by embedding backend (`hash` keeps the canonical un-suffixed name;
    semantic backends like `litert_gemma2`, `mlx_uds`, `apple_ane` isolate their 768-d basis
    so offline hash vectors and semantic vectors never mix in the same logistic regression).
    """
    g_id = _assert_non_security_gate(gate_id)
    b_norm = (backend or "hash").strip().lower()
    if b_norm in ("", "hash"):
        return g_id
    return f"{g_id}@{b_norm}"


def embed_for_reflex_gate(
    text: str,
    signals: StructuralSignals | None = None,
    *,
    prefer_semantic: bool | None = None,
    engine: object | None = None,
) -> np.ndarray:
    """
    Generate a 776-d hybrid feature vector without blocking on cold HTTP/UDS timeouts:
      1. Warm in-process LiteRT `EmbeddingGemma 2` runner (`engine._litert_runner`), if loaded.
      2. Warm MLX Unix Domain Socket (`DEFAULT_MLX_UDS_PATH`), if online.
      3. Warm Apple Silicon ANE / CoreML `NaturalLanguage` sentence embedder (`get_ane_embedder()`), if available.
      4. Fast deterministic lexical multi-hash fallback (`_heuristic_embed`).
    """
    feat, _ = embed_for_reflex_gate_with_backend(
        text,
        signals=signals,
        prefer_semantic=prefer_semantic,
        engine=engine,
    )
    return feat


def _parse_scout_shell_command(cmd: str) -> tuple[str, str]:
    """Extract `(target_path, query)` from a `body scout` / `local.scout` / `body airlock` shell command."""
    if not cmd or not any(k in cmd for k in ("body scout", "local.scout", "local_ai scout", "body airlock")):
        return "", ""
    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    query = ""
    target = ""
    for idx, tok in enumerate(toks):
        if tok in ("-q", "--query") and idx + 1 < len(toks):
            query = toks[idx + 1]
    for idx, tok in enumerate(toks):
        if tok in ("scout", "skeleton", "slice", "zoom", "bundle") or tok.endswith(".scout"):
            for cand in toks[idx + 1 :]:
                if cand in ("-q", "--query"):
                    break
                if not cand.startswith("-"):
                    target = cand
                    break
            if target:
                break
    return target, query


def _is_raw_shell_file_dump(cmd: str, basename: str) -> bool:
    if not basename or basename not in cmd:
        return False
    return any(w in cmd for w in ("cat ", "sed -n", "head ", "tail ", "awk ", "open("))


def _thermal_str_to_pressure(thermal_state: object) -> float:
    val = str(thermal_state or "").strip().upper()
    if val == "FAIR":
        return 0.33
    if val == "SERIOUS":
        return 0.67
    if val == "CRITICAL":
        return 1.0
    return 0.0


def bootstrap_reflex_gates_from_logs(
    gate: SpinalReflexGate | None = None,
    audit_log_path: Path | None = None,
    brain_dir: Path | None = None,
    max_transcripts: int = 40,
    reset_existing: bool = False,
) -> dict[str, object]:
    """
    Warm-bootstrap `recon_scout_sufficiency`, `read_autoslice_window`, and `stdout_compaction`
    from local `immune_audit.jsonl` and Antigravity `transcript.jsonl` history (zero API cost).
    """
    target_gate = gate if gate is not None else get_default_reflex_gate()
    if reset_existing and not target_gate.read_only:
        def _clear_all() -> None:
            cur = target_gate._conn.cursor()
            for tbl in ("reflex_samples", "reflex_models", "pending_scouts", "pending_stdout_compactions"):
                cur.execute(f"DELETE FROM {tbl}")

        target_gate._run_write_tx(_clear_all)
        with target_gate._lock:
            target_gate._weights_cache.clear()

    audit_path = audit_log_path if audit_log_path is not None else (get_wbc_data_dir() / "immune_audit.jsonl")
    antigravity_brain = (
        brain_dir
        if brain_dir is not None
        else (Path.home() / ".gemini" / "antigravity" / "brain")
    )

    scout_added = 0
    autoslice_added = 0
    stdout_added = 0
    touched_pairs: set[tuple[str, str]] = set()
    read_tools = {"read_file", "view_file", "Read", "NotebookRead"}
    edit_tools = {"edit_file", "write_file", "Edit", "Write", "MultiEdit", "replace_file_content"}
    shell_tools = {"run_command", "Bash", "execute_command", "developer__shell", "bash"}
    file_meta_cache: dict[str, tuple[int, int]] = {}

    def _get_file_meta(fpath_str: str, fallback_lines: int = 200) -> tuple[int, int]:
        if fpath_str in file_meta_cache:
            return file_meta_cache[fpath_str]
        p = Path(fpath_str)
        if p.is_file():
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
                lines = max(1, raw.count("\n") + 1)
                syms = len(fast_extract_symbol_names(raw, suffix=p.suffix))
                res = (lines, syms)
            except Exception:
                res = (max(1, fallback_lines), 10)
        else:
            res = (max(1, fallback_lines), 10)
        file_meta_cache[fpath_str] = res
        return res

    def _process_turn_sequence(events: list[dict[str, object]], default_repo: str) -> None:
        nonlocal scout_added, autoslice_added, stdout_added
        edited_names: set[str] = set()
        for idx, ev in enumerate(events):
            kind = str(ev.get("kind", ""))
            if kind == "edit_file":
                t_edit = str(ev.get("target", ""))
                if t_edit:
                    edited_names.add(Path(t_edit).name)
                continue

            window = events[idx + 1 : idx + 3]
            if kind in ("scout_cmd", "auto_slice"):
                tpath = str(ev.get("target", ""))
                if not tpath:
                    continue
                bname = Path(tpath).name
                if not bname:
                    continue
                reread = any(
                    (
                        str(nxt.get("kind")) in ("read_event", "auto_slice")
                        and Path(str(nxt.get("target", ""))).name == bname
                    )
                    or (
                        str(nxt.get("kind")) in ("shell_cmd", "shell_sliced")
                        and _is_raw_shell_file_dump(str(nxt.get("command", "")), bname)
                    )
                    for nxt in window
                )
                r_key = str(ev.get("repo_key") or default_repo or "__global__")
                fallback_lines = int(ev.get("lines", 200) or 200)
                lines, syms = _get_file_meta(tpath, fallback_lines=fallback_lines)
                q_text = str(ev.get("query", "") or "")
                sig = StructuralSignals(
                    file_lines=lines,
                    symbol_count=syms,
                    exact_symbol_mention=bool(q_text and "_" in q_text),
                    recently_edited=(bname in edited_names),
                    is_test_or_log=(
                        bname.lower().startswith("test_")
                        or tpath.endswith((".log", ".err", ".jsonl"))
                    ),
                    thermal_pressure=float(ev.get("thermal_pressure", 0.0) or 0.0),
                    turn_depth=int(ev.get("step_idx", idx) or idx),
                )
                suffix = Path(tpath).suffix
                if kind == "scout_cmd":
                    txt = f"{suffix} | {bname} | {q_text[:160]}" if q_text else f"{suffix} | {bname}"
                    gate_name = "recon_scout_sufficiency"
                    scout_added += 1
                else:
                    txt = f"{suffix} | {bname}"
                    gate_name = "read_autoslice_window"
                    autoslice_added += 1
                feat = embed_for_reflex_gate(txt, signals=sig)
                target_gate.record_outcome(
                    gate_name,
                    feat,
                    actual_local_ok=not reread,
                    repo_key=r_key,
                    retrain=False,
                )
                touched_pairs.add((gate_name, r_key))
            elif kind == "shell_sliced":
                cmd = str(ev.get("command", ""))
                cmd_sig = normalize_shell_command_signature(cmd)
                if not cmd_sig or has_verbose_shell_flags(cmd):
                    continue
                retried = any(
                    str(nxt.get("kind")) in ("shell_cmd", "shell_sliced")
                    and (
                        (
                            normalize_shell_command_signature(str(nxt.get("command", ""))) == cmd_sig
                            and has_verbose_shell_flags(str(nxt.get("command", "")))
                        )
                        or bool(extract_airlock_blob_id(str(nxt.get("command", ""))))
                    )
                    for nxt in window
                )
                r_key = str(ev.get("repo_key") or default_repo or "__global__")
                sig = StructuralSignals(
                    file_lines=int(ev.get("lines", 300) or 300),
                    symbol_count=int(ev.get("warnings", 10) or 10),
                    is_test_or_log=True,
                    thermal_pressure=float(ev.get("thermal_pressure", 0.0) or 0.0),
                    turn_depth=int(ev.get("step_idx", idx) or idx),
                )
                feat = embed_for_reflex_gate(f"shell_exit0_warn | {cmd_sig}", signals=sig)
                target_gate.record_outcome(
                    "stdout_compaction",
                    feat,
                    actual_local_ok=not retried,
                    repo_key=r_key,
                    retrain=False,
                )
                touched_pairs.add(("stdout_compaction", r_key))
                stdout_added += 1

    # 1. Mine immune_audit.jsonl grouped by conversation_id
    if audit_path.is_file():
        by_conv: dict[str, list[dict[str, object]]] = {}
        pre_shell_idx: dict[tuple[str, int, str], int] = {}
        try:
            with open(audit_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    cid = str(obj.get("conversation_id") or "local")
                    tool = str(obj.get("tool") or "")
                    phase = str(obj.get("phase") or "pre")
                    target = str(obj.get("target") or "")
                    reason = str(obj.get("reason") or "")
                    step_i = int(obj.get("step_idx") or 0)
                    therm_p = _thermal_str_to_pressure(obj.get("thermal_state"))
                    if phase == "pre" and tool in edit_tools and target:
                        by_conv.setdefault(cid, []).append({"kind": "edit_file", "target": target, "step_idx": step_i})
                    elif phase == "pre" and tool in read_tools and target:
                        is_auto = bool(
                            obj.get("read_auto_sliced")
                            or "Auto-Slice" in reason
                            or "Auto-Clamp" in reason
                            or obj.get("context_gate_triggered")
                        )
                        if is_auto:
                            saved_tok = int(obj.get("direct_tokens_saved") or 1200)
                            by_conv.setdefault(cid, []).append(
                                {
                                    "kind": "auto_slice",
                                    "target": target,
                                    "lines": max(100, (saved_tok // 8) + 120),
                                    "repo_key": resolve_repo_key(target) if Path(target).is_absolute() else "__global__",
                                    "thermal_pressure": therm_p,
                                    "step_idx": step_i,
                                }
                            )
                        else:
                            by_conv.setdefault(cid, []).append({"kind": "read_event", "target": target, "step_idx": step_i})
                    elif phase == "pre" and tool in shell_tools and target:
                        st, sq = _parse_scout_shell_command(target)
                        if st:
                            by_conv.setdefault(cid, []).append(
                                {
                                    "kind": "scout_cmd",
                                    "target": st,
                                    "query": sq,
                                    "repo_key": resolve_repo_key(st) if Path(st).is_absolute() else "__global__",
                                    "thermal_pressure": therm_p,
                                    "step_idx": step_i,
                                }
                            )
                        elif any(k in target for k in ("pytest", "cargo test", "cargo build", "go test", "jest", "npm test", "vitest")):
                            cmd_sig = normalize_shell_command_signature(target)
                            ev_list = by_conv.setdefault(cid, [])
                            pre_shell_idx[(cid, step_i, cmd_sig)] = len(ev_list)
                            ev_list.append(
                                {
                                    "kind": "shell_sliced",
                                    "command": target,
                                    "repo_key": "__global__",
                                    "thermal_pressure": therm_p,
                                    "step_idx": step_i,
                                }
                            )
                        else:
                            by_conv.setdefault(cid, []).append({"kind": "shell_cmd", "command": target, "step_idx": step_i})
                    elif phase == "post" and tool in shell_tools and target:
                        cmd_sig = normalize_shell_command_signature(target)
                        key = (cid, step_i, cmd_sig)
                        if key in pre_shell_idx:
                            if obj.get("post_output_sliced"):
                                by_conv[cid][pre_shell_idx[key]]["kind"] = "shell_sliced"
                        else:
                            by_conv.setdefault(cid, []).append(
                                {
                                    "kind": "shell_sliced" if obj.get("post_output_sliced") else "shell_cmd",
                                    "command": target,
                                    "repo_key": "__global__",
                                    "thermal_pressure": therm_p,
                                    "step_idx": step_i,
                                }
                            )
        except Exception:
            pass
        for evs in by_conv.values():
            _process_turn_sequence(evs, "__global__")

    # 2. Mine recent Antigravity transcripts
    if antigravity_brain.is_dir():
        transcript_files = sorted(
            antigravity_brain.glob("*/.system_generated/logs/transcript.jsonl"),
            key=lambda p: p.stat().st_mtime if p.is_file() else 0.0,
            reverse=True,
        )[: max(0, int(max_transcripts))]
        for tf in transcript_files:
            evs: list[dict[str, object]] = []
            try:
                with open(tf, "r", encoding="utf-8", errors="replace") as fh:
                    for step_i, line in enumerate(fh):
                        if '"tool_calls"' not in line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        for tc in obj.get("tool_calls") or []:
                            if not isinstance(tc, dict):
                                continue
                            name = str(tc.get("name") or "")
                            targs = tc.get("args") or tc.get("arguments") or {}
                            if not isinstance(targs, dict):
                                continue
                            if name in edit_tools:
                                p_str = str(targs.get("Path") or targs.get("TargetFile") or targs.get("file_path") or "")
                                if p_str:
                                    evs.append({"kind": "edit_file", "target": p_str, "step_idx": step_i})
                            elif name in read_tools:
                                p_str = str(targs.get("Path") or targs.get("file_path") or targs.get("AbsolutePath") or "")
                                if p_str:
                                    evs.append({"kind": "read_event", "target": p_str, "step_idx": step_i})
                            elif name in shell_tools:
                                cmd_s = str(targs.get("CommandLine") or targs.get("command") or "")
                                if cmd_s:
                                    st, sq = _parse_scout_shell_command(cmd_s)
                                    if st:
                                        evs.append(
                                            {
                                                "kind": "scout_cmd",
                                                "target": st,
                                                "query": sq,
                                                "repo_key": resolve_repo_key(st) if Path(st).is_absolute() else "__global__",
                                                "step_idx": step_i,
                                            }
                                        )
                                    elif any(k in cmd_s for k in ("pytest", "cargo test", "cargo build", "go test", "jest", "npm test", "vitest")):
                                        evs.append({"kind": "shell_sliced", "command": cmd_s, "step_idx": step_i})
                                    else:
                                        evs.append({"kind": "shell_cmd", "command": cmd_s, "step_idx": step_i})
            except Exception:
                continue
            if evs:
                _process_turn_sequence(evs, "__global__")

    with target_gate._lock:
        for gid, rkey in sorted(touched_pairs):
            target_gate._retrain_locked(gid, rkey)

    metrics = target_gate.list_all_metrics()
    return {
        "recon_scout_samples": scout_added,
        "read_autoslice_samples": autoslice_added,
        "stdout_compaction_samples": stdout_added,
        "gates": [
            {
                "gate_id": m.gate_id,
                "repo_key": m.repo_key,
                "state": m.state,
                "sample_count": m.sample_count,
                "cv_accuracy": round(m.cv_accuracy, 4),
            }
            for m in metrics
        ],
    }

