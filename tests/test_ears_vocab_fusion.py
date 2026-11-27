"""Tests for Community Free `body ears` / `body transcribe` AST Vocab-Fusion (Initiative 0.6)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from whole_body_compute.cli import main as cli_main
from whole_body_compute.stt.local_ears import LocalEarsEngine
from whole_body_compute.stt.vocab_fusion import (
    ASTSymbolSnapper,
    ASTVocabHydrator,
    VocabBundle,
    split_identifier_to_words,
)


def _create_mock_symbols_db(db_path: Path, workspace: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            kind TEXT,
            file_path TEXT,
            start_line INTEGER,
            end_line INTEGER,
            parent_class TEXT,
            signature TEXT DEFAULT '',
            docstring TEXT DEFAULT '',
            callees TEXT DEFAULT '[]',
            invariants TEXT DEFAULT '[]'
        )
        """
    )
    rows = [
        ("SpinalReflexGate", "class", str(workspace / "src/whole_body_compute/local/reflex_gate.py")),
        ("ZeroCopySharedSlab", "class", str(workspace / "src/whole_body_compute/local/metal_accel.py")),
        ("LocalSymbolIndex", "class", str(workspace / "src/whole_body_compute/local/symbols.py")),
        ("check_organ_the_eyes", "function", str(workspace / "src/whole_body_compute/doctor.py")),
        ("run", "function", str(workspace / "src/whole_body_compute/organs/immune_vault.py")),
        ("window", "function", str(workspace / "src/whole_body_compute/organs/immune_vault.py")),
    ]
    for name, kind, fp in rows:
        conn.execute(
            "INSERT INTO symbols (name, kind, file_path, start_line, end_line) VALUES (?, ?, ?, 1, 20)",
            (name, kind, fp),
        )
    conn.commit()
    conn.close()


def test_split_identifier_to_words():
    assert split_identifier_to_words("SpinalReflexGate") == ["spinal", "reflex", "gate"]
    assert split_identifier_to_words("check_organ_the_eyes") == ["check", "organ", "the", "eyes"]
    assert split_identifier_to_words("ZeroCopySharedSlab") == ["zero", "copy", "shared", "slab"]


def test_ast_hydrator_reads_symbols_db_and_formats_prompts(tmp_path: Path):
    db_path = tmp_path / "symbols.db"
    _create_mock_symbols_db(db_path, tmp_path)

    hydrator = ASTVocabHydrator(workspace=tmp_path, db_path=db_path)
    bundle = hydrator.hydrate(
        active_file_override="src/whole_body_compute/local/reflex_gate.py",
        skip_eyes=True,
    )

    assert "SpinalReflexGate" in bundle.p0_focus_symbols
    assert "ZeroCopySharedSlab" in bundle.p2_repo_symbols
    assert "run" not in bundle.all_symbols  # common English blocklisted
    assert "window" not in bundle.all_symbols

    ane_strings = bundle.to_ane_contextual_strings(limit=50)
    assert "SpinalReflexGate" in ane_strings
    assert "spinal reflex gate" in ane_strings

    whisper_prompt = bundle.to_whisper_prompt(max_terms=32)
    assert "SpinalReflexGate" in whisper_prompt
    assert "ZeroCopySharedSlab" in whisper_prompt


def test_ast_symbol_snapper_exact_file_and_fuzzy_snaps():
    bundle = VocabBundle(
        workspace="/repo",
        active_file="src/whole_body_compute/organs/immune_vault.py",
        p0_focus_symbols=["SpinalReflexGate", "ZeroCopySharedSlab"],
        p1_git_symbols=["check_organ_the_eyes"],
        p2_repo_symbols=["LocalSymbolIndex"],
        file_paths=[
            "src/whole_body_compute/organs/immune_vault.py",
            "tests/test_immune_bbb.py",
        ],
    )
    snapper = ASTSymbolSnapper(bundle)

    # Exact CamelCase + spoken file extension + PhoneticNormalizer ("pie test" -> pytest)
    out, receipts = snapper.snap(
        "Run pie test on test immune bbb dot pie and inspect spinal reflex gate in immune vault dot py"
    )
    assert "pytest" in out
    assert "tests/test_immune_bbb.py" in out
    assert "SpinalReflexGate" in out
    assert "immune_vault.py" in out
    assert len(receipts) >= 3

    # Constrained edit-distance<=1 fuzzy snap ("zero copy share slab" -> ZeroCopySharedSlab)
    fuzzy_out, fuzzy_receipts = snapper.snap("Benchmark zero copy share slab on Apple Silicon")
    assert "ZeroCopySharedSlab" in fuzzy_out
    assert any(r.match_type == "fuzzy" and r.snapped == "ZeroCopySharedSlab" for r in fuzzy_receipts)


def test_ast_symbol_snapper_blocks_common_english_false_positives():
    bundle = VocabBundle(
        workspace="/repo",
        p0_focus_symbols=["run", "window", "test", "memory", "SpinalReflexGate"],
        file_paths=[],
    )
    snapper = ASTSymbolSnapper(bundle)
    out, receipts = snapper.snap("Please run the test in this window to check memory")
    assert out == "Please run the test in this window to check memory"
    assert receipts == []


def test_local_ears_engine_vocab_fusion_receipt(tmp_path: Path):
    db_path = tmp_path / "symbols.db"
    _create_mock_symbols_db(db_path, tmp_path)
    engine = LocalEarsEngine(workspace=tmp_path, db_path=db_path)

    res = engine.apply_vocab_fusion("inspect spinal reflex gate and local symbol index", backend="ane_sfspeech")
    d = res.to_dict()
    assert "SpinalReflexGate" in d["text"]
    assert "LocalSymbolIndex" in d["text"]
    assert d["backend"] == "ane_sfspeech"
    assert d["vocab_context"]["symbols_biased"] >= 2
    assert len(d["vocab_context"]["ast_snaps"]) == 2


def test_cli_ears_status_json(capsys, tmp_path: Path):
    rc = cli_main(["ears", "--status", "--workspace", str(tmp_path), "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["hotkey"] == "fn"
    assert payload["hands_free_chord"] == "fn+space"
    assert payload["highlight_enabled"] is True
    assert payload["highlight_target"] == "auto"
    assert "backend" in payload
    assert "fn_key_check" in payload


def test_hotkey_state_machine_fn_hold_and_hands_free_lock():
    from whole_body_compute.organs.ears import HotkeyStateMachine, _FLAG_FN, _FLAG_OPTION, _KEYCODE_SPACE

    events = []
    sm = HotkeyStateMachine(
        hotkey="fn",
        on_start=lambda mode: events.append(("start", mode)),
        on_stop=lambda: events.append(("stop", None)),
    )

    # 1. Hold Fn -> start_ptt; release Fn -> stop_ptt
    assert sm.handle_flags_changed(_FLAG_FN) == "start_ptt"
    assert sm.handle_flags_changed(0) == "stop_ptt"
    assert events == [("start", "ptt"), ("stop", None)]

    # 2. Hold Fn -> tap Space -> upgrades to hands-free; releasing Fn does NOT stop; second Fn tap stops
    events.clear()
    assert sm.handle_flags_changed(_FLAG_FN) == "start_ptt"
    assert sm.handle_key_down(_KEYCODE_SPACE, _FLAG_FN) == "lock_hands_free"
    assert sm.handle_flags_changed(0) is None
    assert sm.recording is True and sm.hands_free is True
    assert sm.handle_flags_changed(_FLAG_FN) == "stop_hands_free"
    assert events == [("start", "ptt"), ("stop", None)]

    # 3. opt+space toggle mode
    events.clear()
    sm_opt = HotkeyStateMachine(
        hotkey="opt+space",
        on_start=lambda mode: events.append(("start", mode)),
        on_stop=lambda: events.append(("stop", None)),
    )
    assert sm_opt.handle_key_down(_KEYCODE_SPACE, _FLAG_OPTION) == "start_opt_space"
    assert sm_opt.handle_key_down(_KEYCODE_SPACE, _FLAG_OPTION) == "stop_opt_space"
    assert events == [("start", "opt+space"), ("stop", None)]


def test_live_differential_typer_streams_and_snaps_cleanly():
    from whole_body_compute.organs.ears import LiveDifferentialTyper

    buf = []

    def _type(s: str) -> bool:
        buf.extend(list(s))
        return True

    def _bs(n: int) -> None:
        for _ in range(n):
            if buf:
                buf.pop()

    typer = LiveDifferentialTyper(type_fn=_type, backspace_fn=_bs)
    assert typer.update("check spinal") == (0, "check spinal")
    assert "".join(buf) == "check spinal"

    # Incremental append
    assert typer.update("check spinal reflex gate") == (0, " reflex gate")
    assert "".join(buf) == "check spinal reflex gate"

    # Final AST snap replaces "spinal reflex gate" -> "SpinalReflexGate "
    bs, suf = typer.finalize("check SpinalReflexGate")
    assert bs == len("spinal reflex gate")
    assert suf == "SpinalReflexGate "
    assert "".join(buf) == "check SpinalReflexGate "


def test_live_differential_typer_preserves_spoken_words_across_ane_reset():
    from whole_body_compute.organs.ears import LiveDifferentialTyper

    buf: list[str] = []

    def _type(s: str) -> bool:
        buf.extend(list(s))
        return True

    def _bs(n: int) -> None:
        for _ in range(n):
            if buf:
                buf.pop()

    typer = LiveDifferentialTyper(type_fn=_type, backspace_fn=_bs)

    # 1. User speaks initial words while holding Fn
    typer.update("Right before I said this")
    assert "".join(buf) == "Right before I said this"

    # 2. SFSpeechRecognizer resets its active buffer on a pause and emits a brand-new short partial ("Wiping")
    bs, suf = typer.update("Wiping")
    assert bs == 0  # Must NOT backspace previously spoken words!
    assert suf == " Wiping"
    assert "".join(buf) == "Right before I said this Wiping"

    # 3. Continues streaming the post-pause phrase
    typer.update("Wiping what is in the box")
    assert "".join(buf) == "Right before I said this Wiping what is in the box"

    # 4. Empty partial reset mid-stream also preserves all spoken words
    assert typer.update("") == (0, "")
    assert "".join(buf) == "Right before I said this Wiping what is in the box"

    # 5. Finalization preserves both committed pre-reset speech and post-reset speech
    typer.finalize("Wiping what is in the box")
    assert "".join(buf) == "Right before I said this Wiping what is in the box "


def test_doctor_and_foveal_mcp_transcribe_audio_local(tmp_path: Path):
    from whole_body_compute.doctor import check_organ_mouth_and_ears
    from whole_body_compute.integrations.foveal_memory import FovealVisionMCPServer
    from whole_body_compute.organs import OrganEars as ExportedOrganEars

    assert ExportedOrganEars is not None

    doc = check_organ_mouth_and_ears()
    assert "stt_backend" in doc["vital_signs"]
    assert "ast_symbols_indexed" in doc["vital_signs"]
    assert "STT:" in doc["details"]

    db_path = tmp_path / "foveal_memory.sqlite"
    mcp = FovealVisionMCPServer(db_path=db_path)
    tool_names = [t["name"] for t in mcp.get_tools()]
    assert "transcribe_audio_local" in tool_names

    res = mcp.call_tool(
        "transcribe_audio_local",
        {"raw_text": "run pie test on spinal reflex gate"},
    )
    assert not res.get("isError")
    payload = json.loads(res["content"][0]["text"])
    assert "pytest" in payload["text"]
    assert payload["backend"] == "vocab_fusion"


def test_meeting_memory_ast_utterance_snapping():
    from whole_body_compute.audio.meeting_memory import _snap_meeting_utterance

    bundle = VocabBundle(
        workspace="/repo",
        p0_focus_symbols=["SpinalReflexGate", "ZeroCopySharedSlab"],
        file_paths=["src/whole_body_compute/organs/immune_vault.py"],
    )
    snapped = _snap_meeting_utterance(
        "Let us wire spinal reflex gate into immune vault dot py",
        bundle=bundle,
    )
    assert "SpinalReflexGate" in snapped
    assert "immune_vault.py" in snapped


def test_personal_vocab_hydrates_and_snaps_alongside_symbols_db(tmp_path: Path):
    from whole_body_compute.stt.vocab_fusion import add_personal_vocab_word, load_personal_vocab

    db_path = tmp_path / "symbols.db"
    _create_mock_symbols_db(db_path, tmp_path)
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("# Personal terms\n\nVoiceFi\nAntigravity\nlaunchd\n", encoding="utf-8")

    # add_personal_vocab_word appends new term and deduplicates existing
    _, added_new, terms = add_personal_vocab_word("PyPI", vocab_path=vocab_path)
    assert added_new is True
    assert terms == ["VoiceFi", "Antigravity", "launchd", "PyPI"]
    _, added_dup, _ = add_personal_vocab_word("VoiceFi", vocab_path=vocab_path)
    assert added_dup is False
    assert load_personal_vocab(vocab_path) == ["VoiceFi", "Antigravity", "launchd", "PyPI"]

    hydrator = ASTVocabHydrator(workspace=tmp_path, db_path=db_path, vocab_path=vocab_path)
    bundle = hydrator.hydrate(
        active_file_override="src/whole_body_compute/local/reflex_gate.py",
        skip_eyes=True,
    )
    assert bundle.personal_vocab == ["VoiceFi", "Antigravity", "launchd", "PyPI"]
    assert "VoiceFi" in bundle.all_symbols
    assert "SpinalReflexGate" in bundle.all_symbols
    assert "VoiceFi" in bundle.to_ane_contextual_strings(limit=50)
    assert "PyPI" in bundle.to_whisper_prompt(max_terms=32)

    snapper = ASTSymbolSnapper(bundle)
    out, receipts = snapper.snap("Deploy voice fi and antigravity to py pi via launchd and spinal reflex gate")
    assert "VoiceFi" in out
    assert "Antigravity" in out
    assert "PyPI" in out
    assert "launchd" in out
    assert "SpinalReflexGate" in out
    assert any(r.match_type == "personal" and r.snapped == "VoiceFi" for r in receipts)


def test_cli_ears_add_word_json(capsys, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    rc = cli_main(["ears", "--add-word", "VoiceFi", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["term"] == "VoiceFi"
    assert payload["added"] is True
    assert payload["total_terms"] == 1
    assert (tmp_path / ".whole_body_compute" / "vocab.txt").read_text(encoding="utf-8").strip() == "VoiceFi"

    # Second call is idempotent
    rc2 = cli_main(["ears", "--add-word", "VoiceFi", "--json"])
    assert rc2 == 0
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["added"] is False
    assert payload2["total_terms"] == 1


def test_organ_ears_highlight_lifecycle_and_opt_out(monkeypatch, tmp_path: Path):
    import time

    from whole_body_compute.organs import ears as ears_mod
    from whole_body_compute.organs.ears import OrganEars
    from whole_body_compute.stt.local_ears import TranscriptionResult

    calls = []

    def _fake_broker(action: str = "show", *, mode: str = "ptt", target: str = "auto", max_duration_s: float = 95.0):
        calls.append((action, mode, target))
        return {"status": "ok", "visible": action != "hide", "mode": mode, "target_kind": "text_box"}

    monkeypatch.setattr(ears_mod, "broker_ears_highlight", _fake_broker)
    monkeypatch.setattr(ears_mod, "is_voicefi_dictation_active", lambda: False)
    monkeypatch.setenv("WBC_EARS_DISABLE_MEMORY", "1")

    class _DummySession:
        def __init__(self):
            self.bundle = VocabBundle(workspace=str(tmp_path), p0_focus_symbols=["SpinalReflexGate"])

        def finish(self):
            return TranscriptionResult(
                text="check SpinalReflexGate",
                raw_acoustic="check spinal reflex gate",
                backend="ane_sfspeech",
                latency_ms=12.0,
                vocab_bundle=self.bundle,
            )

    ears = OrganEars(workspace=tmp_path, highlight=True, highlight_target="auto")
    monkeypatch.setattr(ears.engine, "start_ptt_mic_session", lambda **kw: _DummySession())
    monkeypatch.setattr(ears_mod, "type_unicode_direct", lambda s: True)
    monkeypatch.setattr(ears_mod, "_send_backspaces", lambda n: None)

    ears._on_ptt_start("ptt")
    assert ("show", "ptt", "auto") in calls

    ears._show_highlight(mode="hands_free")
    assert ("show", "hands_free", "auto") in calls

    ears._on_ptt_stop()
    deadline = time.time() + 2.0
    while ("hide", "ptt", "auto") not in calls and time.time() < deadline:
        time.sleep(0.02)
    assert ("show", "finalizing", "auto") not in calls
    assert ("hide", "ptt", "auto") in calls

    # Opt-out (`highlight=False`) emits zero highlight broker calls
    calls.clear()
    ears_off = OrganEars(workspace=tmp_path, highlight=False, highlight_target="window")
    assert ears_off.status()["highlight_enabled"] is False
    assert ears_off.status()["highlight_target"] == "window"
    ears_off._show_highlight("ptt")
    ears_off._hide_highlight()
    assert calls == []
