"""
Vault Agent Integration for VoiceFi.
Provides conversational Q&A, active note summaries, and auditory briefings for Obsidian vaults.

Question answering runs in two stages:

1. **Retrieval** — a ripgrep sweep across every markdown note in the vault
   (:meth:`VaultAgent.search_vault`). This needs no API key and no network.
2. **Synthesis** — an optional LLM pass over the retrieved snippets. When no
   provider is configured (``get_active_provider() == "heuristic"``), the agent
   speaks the strongest snippet with its source note instead. Adding a Gemini
   key or starting Ollama upgrades the answers with no code change.
"""

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    from whole_body_compute.config import VoiceFiConfig, load_config
except ImportError:
    from whole_body_compute.config import VoiceFiConfig, load_config
from whole_body_compute.redaction import redact_secrets

logger = logging.getLogger(__name__)

# Directories that are never user knowledge.
_EXCLUDED_GLOBS = (
    "!.obsidian/**",
    "!.trash/**",
    "!.git/**",
    "!node_modules/**",
)

# Dropped from spoken queries before retrieval — they match everything.
_STOPWORDS: Set[str] = {
    # Short function words — these match nearly every note.
    "all",
    "and",
    "any",
    "are",
    "but",
    "can",
    "day",
    "did",
    "not",
    "for",
    "get",
    "got",
    "had",
    "has",
    "her",
    "him",
    "his",
    "how",
    "its",
    "let",
    "may",
    "new",
    "now",
    "off",
    "one",
    "our",
    "out",
    "own",
    "per",
    "put",
    "say",
    "see",
    "the",
    "too",
    "two",
    "use",
    "via",
    "was",
    "way",
    "who",
    "why",
    "yet",
    "you",
    "about",
    "after",
    "again",
    "against",
    "another",
    "anything",
    "around",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "could",
    "did",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "even",
    "ever",
    "every",
    "from",
    "further",
    "have",
    "having",
    "here",
    "hers",
    "herself",
    "himself",
    "into",
    "itself",
    "just",
    "like",
    "look",
    "made",
    "make",
    "many",
    "more",
    "most",
    "much",
    "must",
    "myself",
    "need",
    "only",
    "other",
    "over",
    "own",
    "recall",
    "remember",
    "said",
    "same",
    "should",
    "show",
    "some",
    "such",
    "tell",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "things",
    "this",
    "those",
    "through",
    "under",
    "until",
    "very",
    "want",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
    "would",
    "your",
    "yours",
    "yourself",
}

_SEARCH_TIMEOUT_S = 5.0
_SYNTHESIS_TIMEOUT_S = 6.0
_MAX_FILE_BYTES = 2_000_000


class VaultAgent:
    """Conversational assistant for Obsidian knowledge vaults."""

    def __init__(self, config: Optional[VoiceFiConfig] = None):
        self.config = config or load_config()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def answer_vault_query(
        self,
        query: str,
        note_title: str = "",
        note_content: str = "",
        vault_path: Optional[Path] = None,
        search_vault: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a spoken query against the active note and the wider vault.
        Generates an auditory-friendly response calibrated for speech synthesis.
        """
        clean_query = (query or "").strip()
        lower_q = clean_query.lower()

        # Phonetic normalization for common STT phonetic slips
        if any(
            w in lower_q
            for w in [
                "some racist",
                "some raise",
                "some race",
                "some race is",
                "summer eyes",
                "summerize",
                "summarise",
                "summarize",
            ]
        ):
            lower_q = "summarize this note"

        # 1. Handle direct summarization requests
        if any(
            w in lower_q
            for w in [
                "summarize",
                "summary",
                "give me a summary",
                "overview",
                "what is this note about",
            ]
        ):
            return self._summarize_note(note_title, note_content)

        # 2. Handle task / action item extraction
        if any(
            w in lower_q
            for w in ["tasks", "action items", "todos", "what do i need to do", "blockers"]
        ):
            return self._extract_action_items(note_title, note_content)

        # 3. Handle contextual question answering
        return self._answer_question(
            clean_query,
            note_title,
            note_content,
            vault_path=vault_path,
            search_vault=search_vault,
        )

    def ask_spoken_rag(
        self,
        query: str,
        vault_path: Optional[Path] = None,
        socratic: bool = False,
        persona: str = "Christopher",
        include_git: bool = True,
        speak_aloud: bool = False,
        index_db_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute sub-20ms Spoken RAG against vault notes and git history using
        SQLite FTS5 and LocalVaultSynthesizer.
        """
        from whole_body_compute.integrations.vault_index import LocalVaultSynthesizer, VaultRAGIndex

        vp = self.resolve_vault(vault_path) or Path.cwd()
        idx = VaultRAGIndex(db_path=index_db_path)
        synth = LocalVaultSynthesizer(index=idx)
        res = synth.ask(
            query=query,
            vault_path=vp,
            mode="socratic" if socratic else "qa",
            persona=persona,
            include_git=include_git,
            speak_aloud=speak_aloud,
        )
        return res.to_dict()

    def socratic_challenge(
        self,
        idea: str,
        vault_path: Optional[Path] = None,
        persona: str = "Sonia",
        speak_aloud: bool = False,
        index_db_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Brainstorm and challenge assumptions against vault knowledge and git history
        using Socratic Partner mode.
        """
        return self.ask_spoken_rag(
            query=idea,
            vault_path=vault_path,
            socratic=True,
            persona=persona,
            speak_aloud=speak_aloud,
            index_db_path=index_db_path,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def extract_search_terms(query: str) -> List[str]:
        """Reduce a spoken query to the distinctive words worth searching for."""
        words = re.findall(r"\b\w{3,}\b", (query or "").lower())
        terms: List[str] = []
        for w in words:
            if w in _STOPWORDS or w.isdigit():
                continue
            if w not in terms:
                terms.append(w)
        return terms

    def resolve_vault(self, vault_path: Optional[Path] = None) -> Optional[Path]:
        """Resolve the vault to search, honouring an explicit override."""
        if vault_path:
            vp = Path(vault_path).expanduser()
            return vp if vp.is_dir() else None
        from whole_body_compute.integrations.obsidian import get_primary_vault

        return get_primary_vault(self.config)

    def search_vault(
        self,
        query: str,
        vault_path: Optional[Path] = None,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search every markdown note in the vault for the query terms.

        Returns a ranked list of ``{note, path, line, snippet, matched_terms, score}``.
        Ranking favours notes matching the most distinct terms, then match density.
        Never raises — retrieval failure degrades to an empty list.
        """
        terms = self.extract_search_terms(query)
        if not terms:
            return []

        vault = self.resolve_vault(vault_path)
        if not vault:
            return []

        raw = self._ripgrep(terms, vault)
        if raw is None:
            raw = self._python_scan(terms, vault)

        # Group hits by note and score.
        by_file: Dict[str, Dict[str, Any]] = {}
        for path_str, line_no, text in raw:
            entry = by_file.setdefault(
                path_str,
                {"matched_terms": set(), "lines": [], "best_line": line_no, "best_count": 0},
            )
            lowered = text.lower()
            hit_terms = {t for t in terms if t in lowered}
            entry["matched_terms"].update(hit_terms)
            entry["lines"].append((line_no, text))
            if len(hit_terms) > entry["best_count"]:
                entry["best_count"] = len(hit_terms)
                entry["best_line"] = line_no

        # Adaptive term weighting: a term appearing in most matched notes is
        # uninformative regardless of whether it is in the stopword list, so
        # weight each term by inverse document frequency and drop the dregs.
        doc_freq: Dict[str, int] = {}
        for entry in by_file.values():
            for t in entry["matched_terms"]:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        n_files = len(by_file)
        if n_files >= 4:
            ubiquitous = {t for t, df in doc_freq.items() if df >= 0.5 * n_files}
            if len(ubiquitous) < len(doc_freq):  # never drop every term
                for entry in by_file.values():
                    entry["matched_terms"] -= ubiquitous

        results: List[Dict[str, Any]] = []
        for path_str, entry in by_file.items():
            if not entry["matched_terms"]:
                continue
            p = Path(path_str)
            score = sum(10.0 / doc_freq[t] for t in entry["matched_terms"])
            score += min(len(entry["lines"]), 5) * 0.2
            score = round(score, 2)
            results.append(
                {
                    "note": p.stem,
                    "path": path_str,
                    "line": entry["best_line"],
                    # Choke point: snippets feed TTS, the WebSocket/relay
                    # broadcast, the MCP result, and the LLM context. Redact here
                    # so no downstream consumer can forget to.
                    "snippet": redact_secrets(self._extract_context(p, entry["best_line"])),
                    "matched_terms": sorted(entry["matched_terms"]),
                    "score": score,
                }
            )

        results.sort(key=lambda r: (r["score"], r["note"]), reverse=True)
        return results[:max_results]

    def _ripgrep(self, terms: List[str], vault: Path) -> Optional[List[tuple]]:
        """Run ripgrep across the vault. Returns None if rg is unavailable."""
        rg = shutil.which("rg")
        if not rg:
            return None

        cmd = [
            rg,
            "--no-heading",
            "--with-filename",
            "--line-number",
            "--ignore-case",
            "--no-messages",
            "--max-columns",
            "400",
            "--glob",
            "*.md",
        ]
        for g in _EXCLUDED_GLOBS:
            cmd.extend(["--glob", g])
        for t in terms:
            cmd.extend(["-e", re.escape(t)])
        cmd.append(str(vault))

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_SEARCH_TIMEOUT_S)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("vault ripgrep failed (%s); falling back to python scan", e)
            return None

        # rg exits 1 on "no matches", which is not an error for us.
        if proc.returncode not in (0, 1):
            logger.warning("vault ripgrep exit %s; falling back", proc.returncode)
            return None

        hits: List[tuple] = []
        for line in proc.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            path_str, line_no_str, text = parts
            try:
                hits.append((path_str, int(line_no_str), text))
            except ValueError:
                continue
        return hits

    def _python_scan(self, terms: List[str], vault: Path) -> List[tuple]:
        """Pure-python fallback scan when ripgrep is unavailable."""
        hits: List[tuple] = []
        skip = {".obsidian", ".trash", ".git", "node_modules"}
        for md in vault.rglob("*.md"):
            if any(part in skip for part in md.parts):
                continue
            try:
                if md.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = md.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                lowered = line.lower()
                if any(t in lowered for t in terms):
                    hits.append((str(md), i, line[:400]))
        return hits

    @staticmethod
    def _extract_context(path: Path, line_no: int, window: int = 2) -> str:
        """Pull the lines surrounding a match so the snippet reads as prose."""
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return ""
        start = max(0, line_no - 1 - window)
        end = min(len(lines), line_no + window)
        chunk = " ".join(x.strip() for x in lines[start:end] if x.strip())
        return re.sub(r"\s+", " ", chunk).strip()

    # ------------------------------------------------------------------
    # Note-scoped handlers (unchanged behaviour)
    # ------------------------------------------------------------------

    def _summarize_note(self, title: str, content: str) -> Dict[str, Any]:
        """Create a punchy spoken summary of the note."""
        if not content.strip():
            spoken = f"Your note {title or 'document'} is currently empty."
            return {"spoken_response": spoken, "text_summary": spoken, "action": "summarize"}

        # Extract headings and bullet points
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        headings = [re.sub(r"^#+\s*", "", l) for l in lines if l.startswith("#")]
        bullets = [
            re.sub(r"^[-*]\s*(\[[ xX]\]\s*)?", "", l)
            for l in lines
            if l.startswith("-") or l.startswith("*")
        ]

        total_words = len(content.split())
        title_str = title if title else "active note"

        if headings and bullets:
            top_heading = headings[0]
            bullet_sample = ", ".join(bullets[:3])
            spoken = f"Here is a summary of {title_str}. It focuses on {top_heading}, with key items including {bullet_sample}."
        elif headings:
            sections_str = ", ".join(headings[:3])
            spoken = (
                f"{title_str} contains {total_words} words across sections covering {sections_str}."
            )
        elif bullets:
            spoken = f"{title_str} has {len(bullets)} items. The top points are: {', '.join(bullets[:3])}."
        else:
            first_few = " ".join(lines[:2])
            # Strip markdown formatting
            first_few = re.sub(r"[*_#`\[\]]", "", first_few)
            spoken = f"{title_str} is about {first_few[:180]}."

        return {
            "spoken_response": spoken,
            "text_summary": spoken,
            "action": "summarize",
            "title": title,
            "word_count": total_words,
        }

    def _extract_action_items(self, title: str, content: str) -> Dict[str, Any]:
        """Extract open checkboxes and tasks from the note."""
        open_tasks = re.findall(r"^[-*]\s*\[ \]\s*(.+)$", content, re.MULTILINE)
        completed_tasks = re.findall(r"^[-*]\s*\[[xX]\]\s*(.+)$", content, re.MULTILINE)

        if open_tasks:
            count = len(open_tasks)
            task_str = ", ".join(open_tasks[:3])
            more_str = f", plus {count - 3} more" if count > 3 else ""
            spoken = f"You have {count} open task{'s' if count > 1 else ''} in {title or 'this note'}: {task_str}{more_str}."
        elif completed_tasks:
            spoken = f"All {len(completed_tasks)} tasks in {title or 'this note'} are marked complete! Nice work."
        else:
            spoken = f"I didn't find any checkbox tasks in {title or 'this note'}."

        return {
            "spoken_response": spoken,
            "open_tasks": open_tasks,
            "completed_tasks": completed_tasks,
            "action": "tasks",
        }

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def _answer_question(
        self,
        query: str,
        title: str,
        content: str,
        vault_path: Optional[Path] = None,
        search_vault: bool = True,
    ) -> Dict[str, Any]:
        """Answer a question using vault-wide retrieval, then optional synthesis."""
        hits = self.search_vault(query, vault_path=vault_path) if search_vault else []

        # Fall back to the active note when the vault yields nothing.
        if not hits and content.strip():
            snippet = redact_secrets(self._best_paragraph(query, content))
            if snippet:
                hits = [
                    {
                        "note": title or "active note",
                        "path": "",
                        "line": 0,
                        "snippet": snippet,
                        "matched_terms": [],
                        "score": 1,
                    }
                ]

        if not hits:
            title_str = title or "your vault"
            return {
                "spoken_response": f"I searched {title_str} but found nothing matching '{query}'.",
                "action": "answer",
                "query": query,
                "sources": [],
                "provider": "none",
                "matched_snippet": "",
            }

        spoken = self._synthesize_answer(query, hits)
        provider = "llm"
        if not spoken:
            spoken = self._speak_snippet(hits[0])
            provider = "retrieval"

        return {
            "spoken_response": spoken,
            "action": "answer",
            "query": query,
            "sources": [{"note": h["note"], "path": h["path"], "line": h["line"]} for h in hits],
            "provider": provider,
            # Back-compat with callers that read the old single-snippet key.
            "matched_snippet": hits[0]["snippet"],
        }

    @staticmethod
    def _best_paragraph(query: str, content: str) -> str:
        """Keyword-overlap pick from a single note (legacy in-note fallback)."""
        query_words = set(re.findall(r"\b\w{4,}\b", query.lower()))
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        best_p, best_score = "", 0
        for p in paragraphs:
            p_words = set(re.findall(r"\b\w{4,}\b", p.lower()))
            overlap = len(query_words.intersection(p_words))
            if overlap > best_score:
                best_score, best_p = overlap, p
        return best_p if best_score > 0 else ""

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """Strip markdown syntax that reads badly through a TTS engine."""
        text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links / images
        text = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", text)  # wikilinks
        text = re.sub(r"[*_#`>]", "", text)
        text = re.sub(r"^\s*[-+]\s*(\[[ xX]\]\s*)?", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _speak_snippet(self, hit: Dict[str, Any]) -> str:
        """Retrieval-only answer: read the strongest snippet with its source."""
        # Redact before cleaning — see _synthesize_answer for why order matters.
        snippet = self._clean_for_speech(redact_secrets(hit.get("snippet", "")))[:240]
        note = hit.get("note") or "your vault"
        if not snippet:
            return f"I found a match in {note}, but couldn't read the surrounding text."
        return f"From your note {note}: {snippet}"

    def _synthesize_answer(self, query: str, hits: List[Dict[str, Any]]) -> Optional[str]:
        """
        Ask the configured LLM to turn retrieved snippets into a spoken answer.
        Returns None when no provider is available or the call fails, which is
        the normal path on a machine with no API key.
        """
        try:
            from voicefi.integrations.gemini_ai import GeminiIntelligenceEngine

            engine = GeminiIntelligenceEngine(self.config)
            if not engine.is_available():
                return None

            # Redact BEFORE cleaning: _clean_for_speech strips underscores and
            # markdown, which mangles a secret past its own pattern (re_xxx ->
            # rexxx) while leaving it perfectly usable. Redaction must see raw
            # text. Snippets from search_vault are already clean; this re-run
            # covers callers that build hits by hand.
            context = "\n\n".join(
                f"[{h['note']}] {self._clean_for_speech(redact_secrets(h['snippet']))[:600]}"
                for h in hits
            )
            prompt = (
                f"Question: {query}\n\n"
                f"Excerpts from the user's Obsidian vault:\n{context}\n\n"
                "Answer the question using only these excerpts."
            )
            system = (
                "You are Viv, a knowledge companion reading the user's own notes aloud. "
                "Answer in at most two sentences of plain spoken English. "
                "Name the note you drew from. Use no markdown, lists, or symbols. "
                "If the excerpts do not answer the question, say so plainly."
            )

            result = engine.generate_completion(
                prompt=prompt,
                system_instruction=system,
                max_output_tokens=200,
                timeout=_SYNTHESIS_TIMEOUT_S,
            )
            if not result:
                return None
            return self._clean_for_speech(result)
        except Exception as e:
            logger.warning("vault answer synthesis failed: %s", e)
            return None

    def detect_thesis_drift(self, statement: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Search vault and memories for claims overlapping with `statement`.
        Detects potential contradictions (heuristic check + local LLM fallback).
        Returns {"statement": statement, "has_contradiction": bool, "conflicting_notes": list[dict], "analysis": str}.
        """
        clean_statement = statement.strip()
        if not clean_statement:
            return {
                "statement": statement,
                "has_contradiction": False,
                "conflicting_notes": [],
                "analysis": "Empty statement provided.",
            }

        hits = self.search_vault(clean_statement, max_results=top_k)

        try:
            from whole_body_compute.local.dot_memory import get_dot_store

            dot_hits = get_dot_store().search(
                clean_statement, top_k=top_k, use_embeddings=False
            )
            for dh in dot_hits:
                hits.append(
                    {
                        "note": f"Memory:{dh.get('category')}",
                        "path": f"dot_memory.db#{dh.get('id')}",
                        "line": 1,
                        "snippet": dh.get("fact", ""),
                        "score": dh.get("score", 0.0),
                    }
                )
        except Exception:
            pass

        conflicting_notes: List[Dict[str, Any]] = []

        NEGATION_WORDS = {
            "not",
            "never",
            "no",
            "neither",
            "nor",
            "cannot",
            "don't",
            "dont",
            "won't",
            "wont",
            "deprecated",
            "prohibited",
            "avoid",
            "forbid",
            "forbidden",
            "disabled",
            "disable",
            "unsupported",
            "false",
        }
        ASSERTION_WORDS = {
            "always",
            "must",
            "required",
            "mandatory",
            "essential",
            "enable",
            "enabled",
            "supported",
            "true",
            "recommended",
        }

        st_words = set(re.findall(r"\b[A-Za-z0-9_]{3,}\b", clean_statement.lower()))
        st_has_negation = bool(st_words & NEGATION_WORDS)
        st_has_assertion = bool(st_words & ASSERTION_WORDS)

        for hit in hits:
            snippet = hit.get("snippet", "")
            snip_words = set(re.findall(r"\b[A-Za-z0-9_]{3,}\b", snippet.lower()))
            overlap_words = (st_words & snip_words) - NEGATION_WORDS - ASSERTION_WORDS

            if len(overlap_words) >= 1:
                snip_has_negation = bool(snip_words & NEGATION_WORDS)
                snip_has_assertion = bool(snip_words & ASSERTION_WORDS)

                is_polarity_conflict = (
                    (
                        st_has_negation
                        and not snip_has_negation
                        and (snip_has_assertion or len(overlap_words) >= 2)
                    )
                    or (
                        not st_has_negation
                        and snip_has_negation
                        and (st_has_assertion or len(overlap_words) >= 2)
                    )
                    or (st_has_assertion and snip_has_negation)
                    or (st_has_negation and snip_has_assertion)
                )

                conflict_pairs = [
                    ({"sqlite", "postgres"}, {"sqlite", "postgres"}),
                    ({"async", "sync"}, {"async", "sync"}),
                    ({"local", "cloud"}, {"local", "cloud"}),
                    ({"mlx", "cuda"}, {"mlx", "cuda"}),
                ]
                for set_a, set_b in conflict_pairs:
                    matched_a = set_a & st_words
                    matched_b = set_b & snip_words
                    if matched_a and matched_b and matched_a != matched_b:
                        is_polarity_conflict = True

                if is_polarity_conflict:
                    conflicting_notes.append(
                        {
                            "note": hit.get("note") or hit.get("path"),
                            "path": hit.get("path", ""),
                            "snippet": snippet,
                            "conflict_terms": list(overlap_words),
                            "score": hit.get("score", 1.0),
                        }
                    )

        has_contradiction = len(conflicting_notes) > 0
        if has_contradiction:
            note_names = ", ".join(str(cn["note"]) for cn in conflicting_notes[:2])
            analysis = (
                f"Contradiction / Thesis drift detected: '{clean_statement}' conflicts with prior knowledge in {note_names}."
            )
        else:
            analysis = (
                "No thesis drift or contradictions detected against vault notes and active memories."
            )

        return {
            "statement": clean_statement,
            "has_contradiction": has_contradiction,
            "conflicting_notes": conflicting_notes,
            "analysis": analysis,
        }

    def decompose_voice_memo(
        self,
        transcript: str,
        vault_path: Optional[Union[str, Path]] = None,
        memory_store: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Decompose a raw voice memo transcript into:
        - atomic_notes: list of {title, body, wikilinks} written to vault if path provided
        - engineering_tickets: list of action/todo/fix/build/implement statements
        - open_questions: list of question clauses, saved to memory_store with category 'open_question'
        """
        clean_transcript = (transcript or "").strip()
        if not clean_transcript:
            return {
                "atomic_notes": [],
                "engineering_tickets": [],
                "open_questions": [],
            }

        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+|\n+", clean_transcript)
            if s.strip()
        ]

        TICKET_KEYWORDS = (
            "todo",
            "to do",
            "need to",
            "needs to",
            "we should",
            "i should",
            "fix",
            "build",
            "implement",
            "wire up",
            "refactor",
            "create",
            "optimize",
            "investigate",
            "ship",
            "add support",
            "must add",
            "make sure to",
            "ensure that",
            "deprecate",
        )
        QUESTION_INDICATORS = (
            "?",
            "how do",
            "how can",
            "how should",
            "should we",
            "can we",
            "could we",
            "what is",
            "what are",
            "what should",
            "why does",
            "why do",
            "is there",
            "wondering if",
            "wonder if",
            "do we need",
        )

        engineering_tickets: List[str] = []
        open_questions: List[str] = []
        conceptual_statements: List[str] = []

        for sentence in raw_sentences:
            s_lower = sentence.lower()
            if sentence.endswith("?") or any(qi in s_lower for qi in QUESTION_INDICATORS):
                open_questions.append(sentence)
                if memory_store and hasattr(memory_store, "add_memory"):
                    try:
                        memory_store.add_memory(
                            fact=sentence,
                            category="open_question",
                            source="voice_memo",
                        )
                    except Exception as e:
                        logger.debug("Failed saving open question to memory_store: %s", e)
            elif any(tk in s_lower for tk in TICKET_KEYWORDS):
                engineering_tickets.append(sentence)
            else:
                conceptual_statements.append(sentence)

        atomic_notes: List[Dict[str, Any]] = []

        base_title = "Voice Concept Note"
        for sent in (conceptual_statements or raw_sentences):
            words = [
                w
                for w in re.findall(r"\b[A-Za-z0-9_]{3,}\b", sent)
                if w.lower() not in _QUESTION_OPENERS and w.lower() not in _LOOKUP_PHRASES
            ]
            if words:
                base_title = " ".join(w.capitalize() for w in words[:4])
                break

        explicit_links = re.findall(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]", clean_transcript)
        wikilinks = [m[0].strip() for m in explicit_links if m[0].strip()]
        if not wikilinks:
            nouns = [
                w.capitalize()
                for w in re.findall(r"\b[A-Za-z]{4,}\b", clean_transcript)
                if w.lower()
                not in (
                    "with",
                    "that",
                    "this",
                    "from",
                    "have",
                    "been",
                    "were",
                    "what",
                    "when",
                    "where",
                    "should",
                    "could",
                    "would",
                )
            ]
            seen = set()
            for n in nouns[:3]:
                if n not in seen and n != base_title:
                    seen.add(n)
                    wikilinks.append(n)

        note_body_lines = [
            f"# {base_title}",
            "",
            "## Summary",
            clean_transcript,
            "",
        ]
        if conceptual_statements:
            note_body_lines.append("## Concepts")
            for cs in conceptual_statements:
                note_body_lines.append(f"- {cs}")
            note_body_lines.append("")

        if engineering_tickets:
            note_body_lines.append("## Action Items")
            for et in engineering_tickets:
                note_body_lines.append(f"- [ ] {et}")
            note_body_lines.append("")

        if open_questions:
            note_body_lines.append("## Open Questions")
            for oq in open_questions:
                note_body_lines.append(f"- {oq}")
            note_body_lines.append("")

        if wikilinks:
            note_body_lines.append("## Related Notes")
            for wl in wikilinks:
                note_body_lines.append(f"- [[{wl}]]")
            note_body_lines.append("")

        note_body = "\n".join(note_body_lines)

        note_entry = {
            "title": base_title,
            "body": note_body,
            "wikilinks": wikilinks,
        }

        if vault_path:
            vp = Path(vault_path).expanduser().resolve()
            zettel_dir = vp / "Zettelkasten"
            zettel_dir.mkdir(parents=True, exist_ok=True)
            clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", base_title).strip("_")
            if not clean_name:
                clean_name = "Voice_Concept_Note"
            safe_filename = clean_name + ".md"
            note_path = zettel_dir / safe_filename
            note_path.write_text(note_body, encoding="utf-8")
            note_entry["file_path"] = str(note_path)

        atomic_notes.append(note_entry)

        return {
            "atomic_notes": atomic_notes,
            "engineering_tickets": engineering_tickets,
            "open_questions": open_questions,
        }


# ----------------------------------------------------------------------
# Intent detection
# ----------------------------------------------------------------------

_QUESTION_OPENERS = (
    "what",
    "when",
    "where",
    "who",
    "whose",
    "why",
    "how",
    "which",
    "did",
    "do",
    "does",
    "is",
    "are",
    "was",
    "were",
    "can",
    "could",
    "should",
    "would",
    "will",
    "have",
    "has",
    "had",
)

_LOOKUP_PHRASES = (
    "remind me",
    "look up",
    "look for",
    "search for",
    "search my",
    "find my",
    "find the",
    "find that",
    "what did i",
    "where did i",
    "when did i",
    "do i have",
    "tell me about",
    "pull up",
    "read me",
)


def is_vault_question(text: str) -> bool:
    """
    Decide whether spoken text is a question to answer from the vault or a
    thought to capture into the daily note.

    Capture is the safe default: a misrouted question is merely unhelpful, but
    a misrouted thought is lost from the note the user expected it in.
    """
    clean = text.strip()
    if not clean:
        return False

    lowered = clean.lower()

    if lowered.startswith(_LOOKUP_PHRASES) or any(p in lowered for p in _LOOKUP_PHRASES):
        return True

    # An explicit question mark is a strong signal on its own.
    if clean.endswith("?"):
        return True

    first = re.findall(r"\b\w+\b", lowered)
    if first and first[0] in _QUESTION_OPENERS:
        # Short openers are exclamations, not lookups: "what a mess",
        # "how annoying". A real spoken question carries more words.
        return len(first) >= 4

    return False
