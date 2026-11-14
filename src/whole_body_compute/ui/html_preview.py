#!/usr/bin/env python3
"""Zero-dependency Local HTML Preview & Live Visual Copy Editor Server (`127.0.0.1:8788`).

Automatically awakened by Organ 05 (`PostToolUse` hook on `.html` files) with zero `.md` rule bloat:
- Serves `site/`, `docs/`, and workspace HTML/PDF/image assets at 1-click `http://127.0.0.1:8788/...` URLs.
- Injects a lightweight toggle pill (`✏️ Edit Copy` / `?edit=1`) enabling direct click-to-edit on any text
  element, `<br>`-safe Enter key handling (never breaking `<p>` tags with `<div>`s), and `Cmd+S` surgical
  disk saves across mirrored `docs/` + `site/` files.
"""

from __future__ import annotations

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from urllib.parse import parse_qs, unquote, urlparse

DEFAULT_PORT = int(os.environ.get("WBC_HTML_PREVIEW_PORT", "8788"))

ROUTE_ALIASES = {
    "/": "index.html",
    "/open-core": "open_core.html",
    "/org": "open_core.html",
    "/roi": "finops_roi_calculator.html",
    "/calculator": "finops_roi_calculator.html",
    "/telemetry": "body_telemetry.html",
    "/hud": "body_telemetry.html",
    "/hal": "hal_pareto_leaderboard.html",
    "/patents": "patents.html",
    "/blog": "blog/index.html",
    "/blog/decision-1": "blog/decision-1-spinal-reflex.html",
    "/decision-1": "blog/decision-1-spinal-reflex.html",
    "/begin": "begin.sh",
    "/audit": "audit.py",
    "/studio": "editorial_carousel_studio.html",
}

VIEW_PILL_SNIPPET = """
<!-- WBC_LOCAL_PREVIEW_PILL -->
<div id="wbc-view-pill" style="position:fixed;right:18px;bottom:18px;z-index:2147483647;font:600 12px/1 -apple-system,BlinkMacSystemFont,'Inter',sans-serif;">
  <a href="?edit=1" style="display:inline-flex;align-items:center;gap:6px;padding:9px 14px;background:#131312;color:#FAF8F5;border:1px solid rgba(255,42,42,0.7);border-radius:999px;text-decoration:none;box-shadow:0 10px 28px rgba(0,0,0,0.28);">✏️ Click-to-Edit Copy</a>
</div>
"""

EDIT_OVERLAY_SNIPPET = """
<!-- WBC_LOCAL_LIVE_EDIT_OVERLAY -->
<style id="wbc-live-edit-style">
  [data-wbc-editable]:hover {
    outline: 1.5px dashed rgba(255, 42, 42, 0.65) !important;
    outline-offset: 2px !important;
    cursor: text !important;
  }
  [data-wbc-editable]:focus {
    outline: 2px solid #FF2A2A !important;
    outline-offset: 3px !important;
    background: rgba(255, 42, 42, 0.05) !important;
    border-radius: 3px;
  }
  #wbc-edit-bar {
    position: fixed;
    right: 18px;
    bottom: 18px;
    z-index: 2147483647;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: #131312;
    color: #FAF8F5;
    border: 1px solid #FF2A2A;
    border-radius: 999px;
    font: 600 12.5px/1 -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
    box-shadow: 0 12px 32px rgba(0,0,0,0.35);
  }
  #wbc-edit-bar button, #wbc-edit-bar a {
    border: none;
    border-radius: 999px;
    padding: 7px 12px;
    font: 600 12px/1 -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
    cursor: pointer;
    text-decoration: none;
  }
  #wbc-save-btn { background: #FF2A2A; color: #fff; }
  #wbc-save-btn:disabled { background: #3A3936; color: #9A968E; cursor: default; }
  #wbc-view-btn { background: #292826; color: #FAF8F5; }
</style>
<div id="wbc-edit-bar" contenteditable="false">
  <span id="wbc-edit-status">✏️ Click any text to edit (0 changes)</span>
  <button id="wbc-save-btn" type="button" disabled>💾 Save to Disk</button>
  <a id="wbc-view-btn" href="?">👁️ View Clean</a>
</div>
<script id="wbc-live-edit-script">
(function() {
  const relPath = __WBC_REL_PATH_JSON__;
  const edits = new Map();
  const statusEl = document.getElementById("wbc-edit-status");
  const saveBtn = document.getElementById("wbc-save-btn");
  const viewBtn = document.getElementById("wbc-view-btn");
  viewBtn.href = window.location.pathname;

  function sanitizeEditedHtml(raw) {
    // Convert any browser-inserted <div>...</div> wrappers inside text blocks into clean <br> breaks
    return raw
      .replace(/<div><br\\s*\\/?><\\/div>/gi, "<br>")
      .replace(/<div>/gi, "<br>")
      .replace(/<\\/div>/gi, "");
  }

  const candidates = document.querySelectorAll(
    "h1,h2,h3,h4,h5,h6,p,li,a,button,td,th,label,blockquote,figcaption,span,strong,em,div"
  );
  candidates.forEach((el) => {
    if (el.closest("#wbc-edit-bar") || el.closest("script") || el.closest("style") || el.closest("svg")) return;
    // Skip structural container divs that hold block children; allow leaf text divs & inline-formatted copy
    const hasBlockChild = Array.from(el.children).some((c) =>
      /^(DIV|UL|OL|TABLE|TBODY|THEAD|TR|SECTION|HEADER|FOOTER|MAIN|NAV|ARTICLE|ASIDE|FORM|P|H1|H2|H3|H4|LI|SVG)$/i.test(c.tagName)
    );
    if (hasBlockChild) return;
    // If parent is already marked editable (e.g. <em> inside <h1> or <span> inside <p>), edit at the parent block level
    if (el.parentElement && el.parentElement.closest("[data-wbc-editable]")) return;

    const textSample = (el.textContent || "").trim();
    const rawHtml = el.innerHTML.trim();
    if (!textSample || !rawHtml || rawHtml.length > 2500) return;

    el.setAttribute("contenteditable", "true");
    el.setAttribute("spellcheck", "true");
    el.dataset.wbcEditable = "1";
    el.dataset.wbcOrigHtml = el.innerHTML;

    el.addEventListener("click", (e) => {
      if ((el.tagName === "A" || el.closest("a")) && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
      }
    });

    // Keep Enter key inside <p>/<h1>/etc. as <br> so HTML5 never splits <p> with invalid <div> tags
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        if (document.queryCommandSupported && document.queryCommandSupported("insertLineBreak")) {
          document.execCommand("insertLineBreak");
        } else {
          const sel = window.getSelection();
          if (sel && sel.rangeCount) {
            const range = sel.getRangeAt(0);
            range.deleteContents();
            const br = document.createElement("br");
            range.insertNode(br);
            range.setStartAfter(br);
            range.collapse(true);
            sel.removeAllRanges();
            sel.addRange(range);
          }
        }
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });

    el.addEventListener("input", () => {
      const orig = el.dataset.wbcOrigHtml;
      const cur = sanitizeEditedHtml(el.innerHTML);
      if (cur !== orig) {
        edits.set(el, { old_html: orig, new_html: cur });
      } else {
        edits.delete(el);
      }
      statusEl.textContent = `✏️ Editing (${edits.size} unsaved change${edits.size === 1 ? "" : "s"})`;
      saveBtn.disabled = edits.size === 0;
    });
  });

  async function saveChanges() {
    if (edits.size === 0) return;
    saveBtn.disabled = true;
    statusEl.textContent = "⏳ Saving to site/ & docs/...";
    const changes = Array.from(edits.values());
    try {
      const resp = await fetch("/__wbc_save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rel_path: relPath, changes }),
      });
      const data = await resp.json();
      if (!resp.ok || data.status !== "ok") {
        throw new Error(data.error || "Save failed");
      }
      edits.forEach((val, el) => { el.dataset.wbcOrigHtml = val.new_html; });
      edits.clear();
      statusEl.textContent = `✅ Saved ${data.applied} edit(s) to ${data.updated_files.join(" & ")}`;
    } catch (err) {
      saveBtn.disabled = false;
      statusEl.textContent = `❌ ${err.message}`;
    }
  }

  saveBtn.addEventListener("click", saveChanges);
  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      saveChanges();
    }
  });
})();
</script>
"""


def _clean_div_breaks(fragment: str) -> str:
    cleaned = re.sub(r"<div>\s*<br\s*/?>\s*</div>", "<br>", fragment, flags=re.IGNORECASE)
    cleaned = re.sub(r"<div>", "<br>", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</div>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.replace("&nbsp;", " ")


def resolve_preview_url_for_file(file_path: str | Path, repo_root: Path | None = None, port: int = DEFAULT_PORT) -> tuple[str, str]:
    """Return `(view_url, edit_url)` on `127.0.0.1:<port>` for a given `.html` path."""
    p = Path(file_path).resolve()
    root = (repo_root or p.parent).resolve()
    reverse_alias = {v: k for k, v in ROUTE_ALIASES.items() if k in {"/", "/open-core", "/roi", "/telemetry", "/hal", "/patents", "/studio"}}
    if p.name in reverse_alias and p.parent.name in {"site", "docs"}:
        route = reverse_alias[p.name]
    else:
        try:
            rel = p.relative_to(root / "site").as_posix()
            route = "/" + rel
        except ValueError:
            try:
                rel = p.relative_to(root / "docs").as_posix()
                route = "/" + rel
            except ValueError:
                try:
                    rel = p.relative_to(root).as_posix()
                    route = "/" + rel
                except ValueError:
                    route = "/" + p.name
    base = f"http://127.0.0.1:{port}{route}"
    return base, f"{base}?edit=1"


def ensure_preview_server_running(repo_root: Path, port: int = DEFAULT_PORT) -> bool:
    """Ensure the lightweight local HTML preview daemon is listening on `127.0.0.1:<port>` (<3ms check)."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.04):
            return True
    except OSError:
        pass

    try:
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--root", str(repo_root.resolve()), "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception:
        return False


def make_handler(repo_root: Path):
    root = repo_root.resolve()
    site_dir = root / "site"
    docs_dir = root / "docs"

    def _resolve_target(url_path: str) -> tuple[Path | None, str]:
        clean = unquote(url_path)
        if clean in ROUTE_ALIASES:
            clean = "/" + ROUTE_ALIASES[clean]
        rel = clean.lstrip("/") or "index.html"
        for base in (site_dir, docs_dir, root):
            if not base.exists():
                continue
            cand = (base / rel).resolve()
            if cand.is_file() and str(cand).startswith(str(root)):
                return cand, str(cand.relative_to(root))
            if cand.is_dir() and str(cand).startswith(str(root)):
                cand_idx = (cand / "index.html").resolve()
                if cand_idx.is_file():
                    return cand_idx, str(cand_idx.relative_to(root))
            cand_html = (base / (rel + ".html")).resolve()
            if cand_html.is_file() and str(cand_html).startswith(str(root)):
                return cand_html, str(cand_html.relative_to(root))
        return None, rel

    class PreviewHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/__wbc_health":
                self._send_json(200, {"status": "ok", "root": str(root)})
                return
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            target, rel = _resolve_target(parsed.path)
            if target is None:
                self.send_error(404, f"Not found: {parsed.path}")
                return

            ctype, _ = mimetypes.guess_type(target.name)
            if target.name in {"audit.py", "begin.sh", "begin"}:
                ctype = "text/plain; charset=utf-8"
            elif target.suffix.lower() == ".html":
                ctype = "text/html; charset=utf-8"
            elif not ctype:
                ctype = "application/octet-stream"

            data = target.read_bytes()
            if target.suffix.lower() == ".html":
                text = data.decode("utf-8", errors="replace")
                if qs.get("edit") == ["1"]:
                    injected = EDIT_OVERLAY_SNIPPET.replace("__WBC_REL_PATH_JSON__", json.dumps(rel))
                elif qs.get("clean") == ["1"] or target.name == "editorial_carousel_studio.html":
                    injected = ""
                else:
                    injected = VIEW_PILL_SNIPPET
                if injected:
                    if "</body>" in text:
                        text = text.replace("</body>", injected + "\n</body>", 1)
                    else:
                        text = text + injected
                data = text.encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if self.path != "/__wbc_save":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            rel_arg = str(payload.get("rel_path", ""))
            fname = Path(rel_arg).name
            sub_rel = re.sub(r"^(site|docs)/", "", rel_arg)
            changes = payload.get("changes", [])

            candidates = []
            exact = (root / rel_arg).resolve()
            if exact.is_file() and str(exact).startswith(str(root)):
                candidates.append(exact)
            for mirror in (docs_dir / sub_rel, site_dir / sub_rel, docs_dir / fname, site_dir / fname):
                m_res = mirror.resolve()
                if m_res.is_file() and m_res not in candidates:
                    candidates.append(m_res)

            if not candidates:
                self._send_json(400, {"status": "error", "error": f"Unknown target {rel_arg}"})
                return

            applied = 0
            updated_files: list[str] = []
            for tpath in candidates:
                src = tpath.read_text(encoding="utf-8")
                cur = src
                local_hits = 0
                for ch in changes:
                    old_h = ch.get("old_html", "")
                    new_h = _clean_div_breaks(ch.get("new_html", ""))
                    matched = False
                    for cand_old in (old_h, html.unescape(old_h), old_h.strip(), html.unescape(old_h).strip()):
                        if cand_old and cand_old in cur:
                            cur = cur.replace(cand_old, new_h, 1)
                            local_hits += 1
                            matched = True
                            break
                    if not matched and old_h.strip():
                        # Whitespace-insensitive fallback for multi-line formatted HTML blocks
                        tokens = [re.escape(tok) for tok in re.split(r"\s+", old_h.strip()) if tok]
                        if tokens:
                            flex_pat = re.compile(r"\s+".join(tokens))
                            cur_new, count = flex_pat.subn(lambda _m: new_h, cur, count=1)
                            if count > 0:
                                cur = cur_new
                                local_hits += 1
                if cur != src:
                    tpath.write_text(cur, encoding="utf-8")
                    updated_files.append(str(tpath.relative_to(root)))
                    applied = max(applied, local_hits)

            if not updated_files:
                self._send_json(409, {"status": "error", "error": "Could not match HTML span on disk"})
                return

            self._send_json(200, {"status": "ok", "applied": applied, "updated_files": updated_files})

        def _send_json(self, code: int, obj: dict) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return PreviewHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="Whole Body Compute Local HTML Preview & Live Copy Editor")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Workspace root directory")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Localhost port (default: 8788)")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.root))
    print(f"WBC HTML Preview & Live Editor listening on http://127.0.0.1:{args.port} (root={args.root})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
