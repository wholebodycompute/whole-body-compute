# Whole Body Compute (`body`) - Security & macOS TCC Architecture FAQ

Whole Body Compute sits between AI coding agents and local Apple Silicon / Linux hardware. Because desktop embodiment (`body eyes` and `body hands`) can optionally use macOS Accessibility (`kTCCServiceAccessibility`), security-conscious engineers rightly ask what permissions are required, what the blast radius is, and how pre-egress boundaries are enforced.

This document answers those questions directly - separating **OS-level macOS TCC realities** from **Whole Body Compute's deterministic choke points** - and provides 1-command local verification suites so you never have to take policy prose on faith.

---

## 1. Permission Matrix: Zero-TCC Core vs. Opt-In Desktop Embodiment

**You do not need to grant macOS Accessibility (or any macOS TCC permission) to use Whole Body Compute's core token shield, local Metal compute, memory, or FinOps auditing.**

Every organ is modular and degrades cleanly when macOS permissions are withheld:

| Subsystem / CLI | macOS TCC Permission Required | Default Network Egress | Works in Zero-TCC / Linux Mode? |
| :--- | :--- | :--- | :---: |
| **Blood-Brain Barrier (`bin/body-immune-hook`, `body bbb`)**<br>`<3ms` AST Symbol Slicer, CLI stdout digests, Secret Blindfold, Vascular Shell Guard & 3-Stage AST SecGate | **None** | **None** (`100%` local POSIX/Python) | **Yes** |
| **The Heart (`body heart`, `body doctor`, `body context`)**<br>Thermal pressure, Unified RAM headroom & Context Saturation Index | **None** | **None** | **Yes** |
| **The Brainstem (`body scout`, `body perfusion`)**<br>On-device Apple Silicon Metal / Ollama inference & KV prefix cache | **None** | Loopback only (`127.0.0.1`) | **Yes** |
| **The Hippocampus (`body memory`, `body flush`, `body audit-spend`)**<br>SQLite WAL memory, `StateCapsule` handoffs & 30-day FinOps receipts | **None** | **None** | **Yes** |
| **The Ears (`body ears` / `body transcribe`)** *(Optional)*<br>Hold-`Fn` 100% local voice-to-code dictation via `Whole Body Compute.app` | **Microphone on `Whole Body Compute.app` only** *(0 Terminal/IDE TCC)* | **None** (`$0` local weights) | Opt-in |
| **The Eyes & Hands (`body eyes`, `body hands`)** *(Optional)*<br>Live `AXUIElement` DOM inspection & deterministic UI actuation via `Whole Body Compute.app` | **Accessibility on `Whole Body Compute.app` only** *(0 Terminal/IDE TCC; Screen Recording only if fallback OCR is invoked)* | **None** | Opt-in |

### Running in Strict Zero-TCC / Air-Gapped Mode

For locked-down developer laptops, regulated enterprise environments, or engineers who only want context compression and tool-call security:

```bash
# 1. Disable anonymous operational vitals (100% offline Tier-0 mode)
export DO_NOT_TRACK=1
body telemetry disable

# 2. Disable passive cross-app workflow motif observation
export WBC_AMBIENT_DISCOVERY=0

# 3. Wire Blood-Brain Barrier IDE hooks (requires ZERO macOS TCC permissions)
body begin
```

---

## 2. Frequently Asked Security Questions

### Q1: "Isn't granting macOS Accessibility (`kTCCServiceAccessibility`) effectively granting GUI-session root?"

**Yes - at the macOS OS level, Accessibility is one of the highest-privilege TCC entitlements on macOS, and treating it lightly is a mistake.**

Apple's TCC framework does not scope `kTCCServiceAccessibility` to a single target window. Any process trusted for Accessibility can query `AXUIElement` trees and post `Quartz.CGEvent` inputs across user-space apps.

Whole Body Compute addresses this in three ways:
1. **Strictly Optional:** Accessibility is never required for `body begin`, `body bbb`, `body audit-spend`, `body scout`, `body flush`, or `body memory`. Leave Accessibility **OFF** unless you actively want an agent to inspect or drive live macOS GUI apps via `body eyes` / `body hands`.
2. **Deterministic Pre-LLM Vault, Origin & Field Blindfolds:** Even when Accessibility is enabled, `body eyes` blocks DOM extraction entirely on password managers and auth apps (`1Password`, `Bitwarden`, `Apple Passwords`, `LastPass`, `KeePassXC`, `Keychain Access`, `Dashlane`, `NordPass`, `Authy`, `YubiKey`, `Enpass`, `RoboForm`), returning `[BLOCKED_SECURE_VAULT]` with zero elements. Browser tabs on web password vaults or root IAM consoles (`my.1password.com`, `vault.bitwarden.com`, `console.aws.amazon.com/iam`) return `[BLOCKED_SENSITIVE_ORIGIN]`. Any `AXSecureTextField` or password subrole in any other app has its value masked to `[REDACTED_PASSWORD]` in Python *before* serialization.
3. **Engine-Level Actuation Denylists, GUI-Terminal Guards & Zero-Cursor Background Isolation:** `FovealVisionEngine` (`actuate`, `click_element`, `click_xy`, `type_text`, `paste_element`, `attach_file`, `trigger_menu_item`, `dispatch_hotkey`), `ax_act`, and `PalWorkflow` reject attempts to click or type inside password vaults, macOS `System Settings` / `SecurityAgent` TCC dialogs, OS partition & account utilities (`Disk Utility`, `GParted`, `Users & Groups`), anti-bot CAPTCHA challenges (`reCAPTCHA`, `hCaptcha`, `Turnstile`), sensitive web vault origins, or `AXSecureTextField` password inputs - and block GUI keystroke/paste smuggling of `sudo rm -rf`, `.bashrc` persistence, or live vendor secrets into Terminal windows. When targeting background apps (e.g., Chrome), `body hands` uses `AXPress` and PID-targeted events rather than hijacking the physical mouse cursor (`CGEventPost` global HID escalation is blocked while you are working).

---

### Q2: "How is `body eyes` safer than screenshot-based 'Computer Use' agents?"

Standard computer-use loops take full-resolution desktop screenshots every 1-2 seconds (`7,650-15,000` tokens per frame) and upload raw PNG pixels to frontier cloud APIs. Every Slack banner, `.env` terminal split, browser tab, or password toast visible on your monitor is transmitted off-device as pixels - where regex secret scrubbers cannot redact it before cloud ingestion.

By contrast, `body eyes` (`FovealVisionEngine`):
- **Reads structured `AXUIElement` metadata locally in `<15ms` with zero screenshots** (`96.9%-99.5%` fewer tokens).
- **Scrubs secrets before context assembly:** Every element title, label, description, and value passes through a `<1ms` 23-vendor credential redactor (`sk-`, `sk-ant-`, `ghp_`, `AKIA`, `ya29.`, JWTs, Postgres/MySQL/Mongo URIs, Base64 high-entropy secrets, SSNs, and credit cards) *before* leaving the Foveal parser.
- **Neutralizes DOM prompt injections on-device:** Control characters, markdown fence breakouts (` ``` `), invisible Plane-14 Unicode Tags (`U+E0020`-`U+E007E`), Base64/Hex/Binary/Caesar token smuggling, `#`-wrapped multi-line comments, and instruction-hijack headers (`[SYSTEM INSTRUCTION...`, `<SYSTEM MODE>`, `Ignore all previous instructions...`) are escaped and tagged via `EpithelialScreenFilter` in `<0.2ms`.

---

### Q3: "If I grant Accessibility to my Terminal or IDE, doesn't every child process inherit it?"

**Yes - which is why Whole Body Compute isolates `kTCCServiceAccessibility` and `kTCCServiceMicrophone` inside one unified notarized app: `Whole Body Compute.app` (`com.wholebodycompute.app`, Team ID `Y25SHBA3P5`).**

On macOS, TCC attributes permissions to the **responsible parent application bundle** (e.g., `Terminal.app`, `iTerm.app`, `Cursor.app`). If `Terminal.app` holds Accessibility, any rogue `npm postinstall` script or untrusted binary run inside that terminal inherits that TCC attribution.

**How `Whole Body Compute.app` eliminates Terminal/IDE TCC inheritance:**
- **Zero Terminal / Zero IDE / Zero Python TCC:** Grant Accessibility (and Microphone if using `body ears`) **only** to `Whole Body Compute.app` (`macos/WholeBodyCompute/`), a single zero-Python Swift menu-bar broker (`LSUIElement`) signed with **Hardened Runtime** and **Library Validation ON** (Team ID `Y25SHBA3P5`, `0` third-party/Python dylibs). Your terminal, IDE, and Python CLI never need `kTCCServiceAccessibility`.
- **Authenticated Local IPC (`foveal.sock`):** `~/.whole_body_compute/foveal.sock` locks its parent directory to `0700`, binds socket & per-session `foveal.token` at `0600` (owner-only), caps frames at `64 KiB`, and verifies every caller with kernel **`LOCAL_PEERPID`** + **`LOCAL_PEERCRED` UID** + **HMAC-SHA256** (`op_bind` + 30s replay protection) + **`SecCodeCopyGuestWithAttributes`**.
- **Native Swift Blindfolds & 1-Click Kill Switches:** `PrivacyBlindfold.swift` blocks password vaults (`1Password`, `Bitwarden`, `Passwords`, `Keychain Access`), `System Settings` / `SecurityAgent`, sensitive web origins, and `AXSecureTextField` *inside Swift before IPC serialization*. `Option`-click the menu-bar icon to pause all organs in 1 click (or toggle **Pause Eyes & Hands** / **Pause Ears** independently in the menu).
- **Install & Verify:** Run `body eyes --install-app` (pinned SHA-256 + Team ID `Y25SHBA3P5` verification), `brew install --cask wholebodycompute/tap/whole-body-compute`, or `body daemon install`. Verify with `body doctor`.

---

### Q4: "What stops Indirect Prompt Injection from hijacking my shell, reading `.env` / `~/.ssh`, or tampering with tests?"

LLM system prompts cannot stop prompt injection reliably; only **deterministic out-of-loop OS gates** can. Whole Body Compute enforces five pre/post-execution gates inside `bin/body-immune-hook` and Organ 05 (`BloodBrainBarrier`):

```
Untrusted Input / Tool Call
   │
   ├─► 1. Epithelial Screen & DOM Filter (<0.1ms)
   │      Strips [SYSTEM INSTRUCTION] / newline / fence breakouts in OCR & AX labels
   │
   ├─► 2. Secret Blindfold + Symlink Resolver (<1ms)
   │      Blocks direct & symlinked reads of .env, ~/.ssh/*, ~/.aws/*, ~/.config/gh/* & Keychain
   │
   ├─► 3. Vascular Traffic Monitor (<0.5ms)
   │      Allows 127.0.0.1 loopback; blocks external curl/wget POST exfil & pipe-to-shell (| sh)
   │
   ├─► 4. APFS Copy-on-Write Airlock (<2ms)
   │      Isolates autonomous mutations in clonefile() shadow trees; blocks outside-repo symlinks
   │
   └─► 5. 3-Stage AST SecGate (<1ms)
          Blocks dynamic RCE (eval, exec, getattr(os, 'system')) & reward hacking (assert True)
```

---

### Q5: "Does Whole Body Compute watch my clipboard or send code/screen data to external servers?"

1. **Zero Code, Zero Prompts, Zero Screen Data in Telemetry (`TELEMETRY.md`):**
   - Whole Body Compute **never** transmits source code, AST slice text, file paths, repo names, prompts, clipboard strings, window titles, screen pixels, or IP addresses (`"$ip": null`, `"$geoip_disable": true`).
   - Inspect the exact JSON payload on your machine anytime with `body telemetry show`, or disable network telemetry completely with `body telemetry disable` or `export DO_NOT_TRACK=1`.
2. **Ambient Workflow Discovery (`AmbientWorkflowObserver`) & Concealed Pasteboard Respect:**
   - When the Foveal daemon runs, passive motif discovery looks for recurring cross-app steps locally on-device (suppressing blocked password managers, `AXSecureTextField` elements, credentials, SSNs, and payment cards).
   - Clipboard items carrying password-manager autofill/concealed markers (`org.nspasteboard.ConcealedType`, `org.nspasteboard.TransientType`, `org.nspasteboard.AutoGeneratedType`, `com.agilebits.onepassword`) are dropped before `NSPasteboardTypeString` is ever read - even when copied from a browser extension inside Chrome.
   - If you do not want passive pasteboard/focus observation on your machine, set `export WBC_AMBIENT_DISCOVERY=0` to disable it completely.

---

## 3. Empirical Security Scorecard & 60-Second Local Verification

Whole Body Compute's choke points are continuously tested against live A/B and verbatim upstream adversarial payloads across **AgentDojo** (live `36`-pair / `72`-trial A/B + choke-point), **MCPTox** (`485` poisoned MCP tools across `45` real servers + `1,348` server bundles + `100` schema rug-pulls + `386` benign MCP schemas), **OS-Harm** (`51` GUI/DOM IPI + `83` boundary + `375` benign), Meta **CyberSecEval 3/4** (`500` Interpreter + `1,255` PI + `750` `MITRE-FRR` benign), verbatim upstream **InjecAgent** (`2,108` IPI cases + `1,066` benign controls), and verbatim upstream **RedCode-Exec** (`1,410` Python/Bash payloads) (`benchmarks/AGENTDOJO_LIVE_AB_REPORT_2026_11_26.md`, `benchmarks/UPSTREAM_MCPTOX_SECURITY_REPORT_2026_11_27.md`, `benchmarks/UPSTREAM_OSHARM_CYBERSECEVAL_REPORT_2026_11_26.md`, `benchmarks/UPSTREAM_REDCODE_INJECAGENT_REPORT_2026_11_26.md`) - executing the real `bin/body-immune-hook` binary and in-process organ gates with zero mocked verdicts:

| Benchmark / Metric | Unshielded Baseline | Agent + Whole Body Compute (`BBB`) |
| :--- | :---: | :---: |
| **Live Upstream `AgentDojo` A/B (`72` trials: `Workspace`, `Slack`, `Banking`)** | `0.0% ASR` (`0/36`) · `91.7%` Utility (`33/36`) on `Gemini 3.8 Flash` | **`0.0% ASR` (`0/36`) · `91.7%` Utility parity (`33/36`) · `36/36` neutralized at ingress in `0.148ms`** |
| **`AgentDojo` + `CyberSecEval 3` Choke-Point (`12` attacks / `6` benign)** | `100.0% ASR` (`12/12` compromised) | **`0.0% ASR` (`12/12` blocked · `0.0% FPR`)** |
| **Verbatim Upstream `MCPTox` Tool-Poisoning, Rug-Pulls & Shadowing (`485` tools / `1,348` bundles / `100` rug-pulls)** | `0.82%` pre-hardening (`4/485`) · `50.0% ASR` (`10/20` live local LLM) | **`100.0%` Neutralized (`485/485` tools & `1,348/1,348` bundles · `100/100` rug-pulls · `386/386` `0.0% FPR` in `0.28ms`) · `50% -> 5%` live LLM ASR** |
| **Verbatim Upstream `OS-Harm` GUI/DOM & Boundary (`51` IPI / `83` boundary)** | `0%` deterministic Foveal/OS gate | **`0.0% ASR` (`51/51` GUI IPI & `83/83` boundary blocked · `375/375` `0.0% FPR`)** |
| **Verbatim Upstream `CyberSecEval 3/4` (`500` Interpreter / `55` EN Indirect)** | `25.45% ASR` (`14/55` live local LLM) · `0%` AST/egress gate | **`97.6%` Interpreter blocked (`488/500`) · `0.0% ASR` (`55/55` EN PI neutralized & defended)** |
| **Verbatim Upstream `InjecAgent` (`2,108` IPI cases / `1,066` benign)** | `0%` deterministic pre-tool defense | **`0.0% ASR` (`2,108/2,108` defended · `0.0% FPR` in `0.07ms`)** |
| **Verbatim Upstream `RedCode-Exec` Host/OS Boundary (`750` payloads)** | `0%` deterministic AST/shell gate | **`94.53%` blocked (`709/750` · benign `/proc` & `json.loads` preserved)** |
| **Combined Benign Developer & User Utility (`2,583` controls)** | `100.0%` | **`100.0%` (`2,583/2,583` allowed · `0.0% FPR`)** |

### Verify Every Gate Yourself

Clone the repository and run the offensive red-team, Accessibility vault-blindfold, symlink-bypass, and chaos suites locally in under 60 seconds:

```bash
# 1. Run the live 18-scenario AgentDojo + CyberSecEval 3 choke-point benchmark (<1s)
uv run python3 benchmarks/agentdojo_wbc_security_suite.py

# 1b. Run the 4,584-evaluation verbatim upstream InjecAgent + RedCode-Exec sweep (~11s, $0)
uv run python3 benchmarks/upstream_redcode_injecagent_suite.py

# 1c. Run the 3,031-evaluation verbatim upstream OS-Harm + CyberSecEval 3/4 sweep (~3s, $0)
uv run python3 benchmarks/upstream_osharm_cyberseceval_suite.py

# 1d. Run the 2,361-evaluation verbatim upstream MCPTox + rug-pull + shadowing sweep (~2s, $0)
uv run python3 benchmarks/upstream_mcptox_security_suite.py

# 2. Verify WBC Eyes.app Hardened Runtime, Kill Switch, vault blocking & actuation gates
WBC_HEADLESS=1 WBC_TESTING=1 uv run pytest -q tests/test_wbc_eyes_helper.py tests/test_bbb_computer_use_wiring.py tests/test_hands_zero_cursor_background.py

# 3. Verify AST SecGate, APFS symlink escape blocks, Keychain/SSH guards & hook symlink resolution
WBC_HEADLESS=1 WBC_TESTING=1 uv run pytest -q tests/test_red_team_hardened.py tests/test_safeagents_organ5_upgrades.py tests/test_neon_xhigh_audit_hardening.py tests/test_agentdojo_live_ab_suite.py tests/test_upstream_osharm_cyberseceval_suite.py

# 4. Run the 5-organ adversarial chaos & concurrency fuzzer
uv run body chaos
```

---

## 4. Reporting Security Issues

If you discover a bypass in `bin/body-immune-hook`, `VascularTrafficMonitor`, `FovealVisionEngine`, `APFSCoWSandbox`, or `redact_secrets`, please report it directly to **`hello@wholebodycompute.com`** with a minimal reproduction so we can ship a deterministic regression test and credit your finding.
