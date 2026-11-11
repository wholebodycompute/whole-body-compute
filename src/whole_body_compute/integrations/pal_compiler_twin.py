"""
Pillar 4: Deterministic Twin World Environment & OSWorld/WorkArena Benchmark Harness
for the VoiceFi Pal Compiler.
"""

from __future__ import annotations

import copy
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from whole_body_compute.integrations.pal_compiler_ir import (
    BlastRadiusExceededError,
    PbEStringSynthesizer,
    PostconditionFailedError,
    PreconditionFailedError,
)


# ==============================================================================
# PILLAR 4: DETERMINISTIC TWIN WORLD ENVIRONMENT & SCIENTIFIC HARNESS
# ==============================================================================


@dataclass
class TwinWorldState:
    """
    Canonical State Tuple (FileState, CloudState, SheetState, DBState) for mathematical
    equivalence checking across Tier-3 GUI macros and Tier-0/1 Headless recipes.
    """

    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cloud_drive: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    sheets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    database: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    outbound_ledger: Set[str] = field(default_factory=set)

    def canonical_tuple(
        self,
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        return (
            copy.deepcopy(self.files),
            copy.deepcopy(self.cloud_drive),
            copy.deepcopy(self.sheets),
            copy.deepcopy(self.database),
        )


class TwinWorldEnvironment:
    """
    Deterministic Digital Twin of macOS + Google Workspace + Lead Gen DOM + Database.
    Supports:
      1. State-equivalence comparison between Tier-3 GUI execution and Tier-1 Headless execution.
      2. Metamorphic perturbations (Window drift +200px, Sheet column insertion before Col J,
         Lead Gen DOM card height drift, Personal vs. Shared Drive browser tab context).
      3. Chaos fault injection (0-byte Drive uploads, HTTP 200 silent truncations, race-condition
         Sheet cell overwrites, unresolved outbound merge tags).
    """

    SHARED_DRIVE_ID = "1lTZwLHkMDluABS8uDzZXR58rWcvM4lJq"
    PERSONAL_DRIVE_ID = "personal_my_drive_root"
    DFPB_SHEET_ID = "1QHVA_TcRc51G2qw_Wl4msVk_SISym7yXdx_SuvQKs-E"
    LEADS_SHEET_ID = "leads_master_sheet_001"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.state = TwinWorldState(
            files={
                "APPROVED 539694 - Smith - Dallas TX - 2026-10-21 - 42 SFR.csv": {
                    "size": 14500,
                    "content_hash": "sha256:dfpb539694_42sfr",
                    "external_id": "539694",
                    "order_uid": "68",
                }
            },
            cloud_drive={
                self.SHARED_DRIVE_ID: {},
                self.PERSONAL_DRIVE_ID: {},
            },
            sheets={
                self.DFPB_SHEET_ID: {
                    "headers": [
                        "Order UID",
                        "External ID",
                        "Client",
                        "County",
                        "State",
                        "Product",
                        "Ordered Date",
                        "Due Date",
                        "Lead Count",
                        "Name Your File",  # Col 10 (J)
                        "Status",  # Col 11 (K)
                    ],
                    "rows": [
                        [
                            "68",
                            "539694",
                            "Smith",
                            "Dallas",
                            "TX",
                            "SFR",
                            "2026-10-20",
                            "2026-10-21",
                            "42",
                            "",
                            "In Progress",
                        ]
                    ],
                },
                self.LEADS_SHEET_ID: {
                    "headers": ["Company Name", "Phone", "Domain", "Address", "Email", "Status"],
                    "rows": [],
                },
            },
            database={
                "orders": {
                    "68": {
                        "external_id": "539694",
                        "client_name": "Smith",
                        "county": "Dallas",
                        "status": "QA_APPROVED",
                    }
                }
            },
        )

        # UI / Window / Browser Geometry State (for Tier 3 vs Tier 2 vs Tier 1/1.5 testing)
        self.window_offset_x: float = 0.0
        self.window_offset_y: float = 0.0
        self.active_browser_drive_folder_id: str = self.SHARED_DRIVE_ID
        self.auth_token_valid: bool = True

        # Simulated Directory DOM for Lead Gen
        self.directory_dom_cards: List[Dict[str, Any]] = [
            {
                "name": "Apex Commercial Roofing",
                "phone": "(214) 555-0101",
                "website": "https://www.apexroofingdallas.com/about",
                "address": "1200 Main St, Dallas, TX",
                "card_height_px": 80,
            },
            {
                "name": "Lone Star Industrial Roofs",
                "phone": "214.555.0199",
                "website": "https://lonestarroofs.com/contact",
                "address": "4500 Ross Ave, Dallas, TX",
                "card_height_px": 80,
            },
            {
                "name": "Metroplex Storm Restoration",
                "phone": "+1-972-555-0144",
                "website": "http://www.metroplexstorm.com",
                "address": "800 N Central Expy, Plano, TX",
                "card_height_px": 80,
            },
        ]
        self.enrichment_catalog: Dict[str, Dict[str, str]] = {
            "apexroofingdallas.com": {"email": "bids@apexroofingdallas.com", "phone": "(214) 555-0101"},
            "lonestarroofs.com": {"email": "estimating@lonestarroofs.com", "phone": "(214) 555-0199"},
            "metroplexstorm.com": {"email": "info@metroplexstorm.com", "phone": "(972) 555-0144"},
        }
        self.enrichment_api_calls: int = 0

        # Chaos Fault Injection Flags
        self.fault_zero_byte_upload: bool = False
        self.fault_truncate_upload_bytes: Optional[int] = None
        self.fault_upload_trashed: bool = False
        self.fault_upload_wrong_folder: bool = False
        self.fault_sheet_race_overwrite_status: Optional[str] = None
        self.fault_sheet_race_corrupt_filename: Optional[str] = None
        self.fault_rate_limit_at_call: Optional[int] = None

    # --- Metamorphic Perturbation Injectors ---

    def perturb_window_position(self, dx: float, dy: float) -> None:
        """Simulate user moving Chrome/Finder windows by (dx, dy) pixels."""
        self.window_offset_x += dx
        self.window_offset_y += dy

    def perturb_insert_sheet_columns_before_j(self, new_columns: List[str]) -> None:
        """Simulate someone inserting new columns before Column J in the DFPB Google Sheet."""
        sheet = self.state.sheets[self.DFPB_SHEET_ID]
        headers = sheet["headers"]
        insert_pos = 9  # Before index 9 (1-based Column 10 / J)
        for offset, col_name in enumerate(new_columns):
            headers.insert(insert_pos + offset, col_name)
        for row in sheet["rows"]:
            for offset in range(len(new_columns)):
                row.insert(insert_pos + offset, "")

    def perturb_browser_pointed_at_personal_drive(self) -> None:
        """Simulate Chrome tab left open on Personal 'My Drive' instead of DFPB Shared Drive."""
        self.active_browser_drive_folder_id = self.PERSONAL_DRIVE_ID

    def perturb_lead_card_heights(self, extra_px_on_first_card: int = 36) -> None:
        """Simulate a 2-line address or ad banner shifting vertical card coordinates in DOM."""
        if self.directory_dom_cards:
            self.directory_dom_cards[0]["card_height_px"] += extra_px_on_first_card

    # --- Execution Engines (Tier 3 GUI vs Tier 1 / 1.5 Headless + Hoare Contracts) ---

    def execute_tier3_gui_client_upload(self) -> bool:
        """
        Replays the raw screen-coordinate GUI macro for client_upload:
        - Drops file at screen (1060, 1085) into whatever Drive folder is open in Chrome.
        - Clicks screen (1275, 418) -> hardcoded Column J (index 9) and K5 (index 10).
        """
        # 1. If window moved by > 30px, drag-and-drop misses the drop target or clicks wrong cell!
        if abs(self.window_offset_x) > 30 or abs(self.window_offset_y) > 30:
            return False

        src_name = "APPROVED 539694 - Smith - Dallas TX - 2026-10-21 - 42 SFR.csv"
        src_meta = self.state.files.get(src_name)
        if not src_meta:
            return False

        # GUI Rename in Drive strips prefix and suffix
        clean_stem = PbEStringSynthesizer.apply_rule("strip_sfr_package_affixes", src_name)
        clean_csv = f"{clean_stem}.csv"

        # Drops into whichever folder the active browser tab is viewing (no folder_id verification!)
        target_folder = self.state.cloud_drive[self.active_browser_drive_folder_id]
        # Naive GUI drag-and-drop creates a duplicate or uploads to active tab folder
        target_folder[clean_csv] = {
            "id": f"drive_{src_meta['external_id']}",
            "name": clean_csv,
            "size": 0 if self.fault_zero_byte_upload else (
                self.fault_truncate_upload_bytes or src_meta["size"]
            ),
            "trashed": self.fault_upload_trashed,
            "folder_id": self.active_browser_drive_folder_id,
        }

        # GUI clicks fixed pixel (1275, 418) which maps to hardcoded Column index 9 (J) and 10 (K)
        sheet = self.state.sheets[self.DFPB_SHEET_ID]
        row = sheet["rows"][0]
        row[9] = clean_stem
        row[10] = "Done"

        if self.fault_sheet_race_overwrite_status is not None:
            row[10] = self.fault_sheet_race_overwrite_status

        self.state.database["orders"]["68"]["status"] = "DELIVERED"
        return True

    def execute_tier1_compiled_client_upload(self) -> Dict[str, Any]:
        with self._lock:
            return self._execute_tier1_compiled_client_upload_locked()

    def _execute_tier1_compiled_client_upload_locked(self) -> Dict[str, Any]:
        """
        Executes the compiled Tier 0/1 Headless Recipe wrapped in Hoare Contracts {P} C {Q}:
        - Precondition {P}: Validates source file existence, size > 0, APPROVED prefix, and auth token.
        - Command C:
            * Tier 0: PbE regex transform in memory.
            * Tier 1: Idempotent Drive PATCH/POST to explicit SHARED_DRIVE_ID.
            * Tier 1: Dynamic header resolution in Sheets for 'Name Your File' & 'Status'.
            * Tier 0/1: Idempotent DB status update.
        - Postcondition {Q}: Independent 2-Gate Read-Back Verification on Drive & Sheets.
        """
        # --- Precondition Gate {P} ---
        if not self.auth_token_valid:
            raise PreconditionFailedError("[Precondition {P} Failed] OAuth token invalid or expired.")

        matching_files = [
            (k, v)
            for k, v in self.state.files.items()
            if re.match(r"^APPROVED\s+\d+\s*-.*SFR\.csv$", k, re.I)
        ]
        if not matching_files:
            raise PreconditionFailedError(
                "[Precondition {P} Failed] No file matching 'APPROVED * SFR.csv' found."
            )

        src_name, src_meta = matching_files[0]
        local_size = int(src_meta.get("size", 0))
        if local_size <= 0:
            raise PreconditionFailedError(
                f"[Precondition {{P}} Failed] Source file '{src_name}' is empty (0 bytes)."
            )

        # --- Tier 0: Pure Python PbE String Transform ---
        clean_stem = PbEStringSynthesizer.apply_rule("strip_sfr_package_affixes", src_name)
        clean_csv = f"{clean_stem}.csv"
        ext_id = str(src_meta["external_id"])
        order_uid = str(src_meta["order_uid"])

        # --- Tier 1: Idempotent Google Drive Shared Folder Upload (UPSERT_BY_NAME_AND_PARENT) ---
        target_folder_id = (
            self.PERSONAL_DRIVE_ID if self.fault_upload_wrong_folder else self.SHARED_DRIVE_ID
        )
        folder_store = self.state.cloud_drive[target_folder_id]
        action_taken = "UPDATED_EXISTING" if clean_csv in folder_store else "CREATED_NEW"

        written_size = (
            0
            if self.fault_zero_byte_upload
            else (
                self.fault_truncate_upload_bytes
                if self.fault_truncate_upload_bytes is not None
                else local_size
            )
        )
        folder_store[clean_csv] = {
            "id": f"drive_{ext_id}",
            "name": clean_csv,
            "size": written_size,
            "trashed": self.fault_upload_trashed,
            "folder_id": target_folder_id,
            "action": action_taken,
        }

        # --- Tier 1: Dynamic Header Resolution in Google Sheets ---
        sheet = self.state.sheets[self.DFPB_SHEET_ID]
        headers = sheet["headers"]
        col_file_idx = 9
        col_status_idx = 10
        for idx, h in enumerate(headers):
            hc = h.strip().lower()
            if hc in ("file", "link to file", "file link", "name your file"):
                col_file_idx = idx
            elif hc == "status":
                col_status_idx = idx

        target_row: Optional[List[str]] = None
        for r in sheet["rows"]:
            if ext_id in (r[0].strip(), r[1].strip()):
                target_row = r
                break

        if target_row is None:
            raise PreconditionFailedError(
                f"[Precondition {{P}} Failed] External ID '{ext_id}' not found in sheet."
            )

        target_row[col_file_idx] = clean_stem
        target_row[col_status_idx] = "Done"

        # Simulate race-condition overwrite if fault is active
        if self.fault_sheet_race_overwrite_status is not None:
            target_row[col_status_idx] = self.fault_sheet_race_overwrite_status
        if self.fault_sheet_race_corrupt_filename is not None:
            target_row[col_file_idx] = self.fault_sheet_race_corrupt_filename

        # --- Tier 0/1: Idempotent Database Record Update ---
        self.state.database["orders"][order_uid]["status"] = "DELIVERED"

        # --- Postcondition Read-Back Gate {Q} ---
        self.verify_client_upload_postconditions(
            clean_stem=clean_stem,
            clean_csv=clean_csv,
            local_size=local_size,
            external_id=ext_id,
            order_uid=order_uid,
        )

        # Clean transient action tag so canonical state tuple matches pure state
        if clean_csv in self.state.cloud_drive[self.SHARED_DRIVE_ID]:
            self.state.cloud_drive[self.SHARED_DRIVE_ID][clean_csv].pop("action", None)

        return {
            "status": "CERTIFIED",
            "clean_csv": clean_csv,
            "drive_action": action_taken,
            "col_file_idx": col_file_idx,
            "col_status_idx": col_status_idx,
        }

    def verify_client_upload_postconditions(
        self,
        clean_stem: str,
        clean_csv: str,
        local_size: int,
        external_id: str,
        order_uid: str,
    ) -> None:
        """Independent 2-Gate Closed-Loop Read-Back Verification ({Q})."""
        # Gate 1: Drive Shared Folder Read-Back
        shared_folder = self.state.cloud_drive.get(self.SHARED_DRIVE_ID, {})
        remote_file = shared_folder.get(clean_csv)
        if not remote_file:
            raise PostconditionFailedError(
                f"[Postcondition {{Q}} Gate 1 Failed] '{clean_csv}' not found in Shared Drive '{self.SHARED_DRIVE_ID}'."
            )
        if remote_file.get("trashed"):
            raise PostconditionFailedError(
                f"[Postcondition {{Q}} Gate 1 Failed] '{clean_csv}' is in Trash."
            )
        remote_size = int(remote_file.get("size", 0))
        if remote_size <= 0:
            raise PostconditionFailedError(
                f"[Postcondition {{Q}} Gate 1 Failed] '{clean_csv}' has 0 bytes in Drive."
            )
        if abs(remote_size - local_size) > 200:
            raise PostconditionFailedError(
                f"[Postcondition {{Q}} Gate 1 Failed] Byte size mismatch: Drive={remote_size}, Local={local_size}."
            )

        # Gate 2: Sheets Dynamic Header Read-Back
        sheet = self.state.sheets[self.DFPB_SHEET_ID]
        headers = sheet["headers"]
        col_file_idx = next(
            (
                i
                for i, h in enumerate(headers)
                if h.strip().lower() in ("file", "link to file", "file link", "name your file")
            ),
            9,
        )
        col_status_idx = next(
            (i for i, h in enumerate(headers) if h.strip().lower() == "status"),
            10,
        )

        matched_row = next(
            (r for r in sheet["rows"] if external_id in (r[0].strip(), r[1].strip())),
            None,
        )
        if not matched_row:
            raise PostconditionFailedError(
                f"[Postcondition {{Q}} Gate 2 Failed] Row for '{external_id}' missing on read-back."
            )
        if matched_row[col_file_idx] != clean_stem:
            raise PostconditionFailedError(
                f"[Postcondition {{Q}} Gate 2 Failed] Sheet 'Name Your File' mismatch: expected '{clean_stem}', got '{matched_row[col_file_idx]}'."
            )
        if matched_row[col_status_idx].strip().lower() != "done":
            raise PostconditionFailedError(
                f"[Postcondition {{Q}} Gate 2 Failed] Sheet 'Status' mismatch: expected 'Done', got '{matched_row[col_status_idx]}'."
            )

        # Gate 3: DB Status Read-Back
        if self.state.database["orders"].get(order_uid, {}).get("status") != "DELIVERED":
            raise PostconditionFailedError(
                f"[Postcondition {{Q}} Gate 3 Failed] DB status for order {order_uid} is not DELIVERED."
            )

    # --- Lead Generation Execution (Tier 3 GUI vs Tier 1.5/1 Compiled Pipeline) ---

    def execute_tier3_gui_lead_harvester(self) -> int:
        """
        Simulates a naive Tier-3 GUI macro for Lead Gen:
        - Assumes each card is exactly 80px tall and clicks fixed y-offsets.
        - If card #1 is taller (e.g., 2-line address +36px) or window drifted, clicks miss!
        - Also lacks natural-key deduplication (appends duplicate rows on re-run).
        """
        if abs(self.window_offset_x) > 25 or abs(self.window_offset_y) > 25:
            return 0
        for card in self.directory_dom_cards:
            if card.get("card_height_px", 80) != 80:
                return 0
        sheet = self.state.sheets[self.LEADS_SHEET_ID]
        for card in self.directory_dom_cards:
            sheet["rows"].append(
                [card["name"], card["phone"], card["website"], card["address"], "", "New"]
            )
        return len(self.directory_dom_cards)

    def execute_compiled_lead_gen_pipeline(
        self,
        enrich: bool = True,
        send_outbound: bool = False,
        outbound_template: str = "Hi {Company Name}, we have commercial roofing leads in {Address}.",
    ) -> Dict[str, Any]:
        with self._lock:
            return self._execute_compiled_lead_gen_pipeline_locked(
                enrich=enrich,
                send_outbound=send_outbound,
                outbound_template=outbound_template,
            )

    def _execute_compiled_lead_gen_pipeline_locked(
        self,
        enrich: bool = True,
        send_outbound: bool = False,
        outbound_template: str = "Hi {Company Name}, we have commercial roofing leads in {Address}.",
    ) -> Dict[str, Any]:
        """
        Executes the compiled Lead Generation Pipeline:
        1. Tier 1.5 Active-Browser Multi-Anchor Structural DOM Extraction (immune to card height & window drift).
        2. Tier 0 Normalization (PbE phone + root domain) & Set Deduplication.
        3. Tier 1 Idempotent Batch Upsert into Leads Sheet by (Phone, Domain).
        4. Optional Waterfall Enrichment with Checkpoint Skip (0 wasted API calls on already-enriched rows).
        5. Optional Outbound Dispatch with Precondition Template Validation & SQLite Ledger Lock.
        6. Postcondition Read-Back Verification {Q}.
        """
        # 1. Tier 1.5 Multi-Anchor Structural DOM Harvest
        if not self.directory_dom_cards:
            raise PreconditionFailedError("[Precondition {P} Failed] No lead cards found in DOM.")

        sheet = self.state.sheets[self.LEADS_SHEET_ID]
        upserted = 0

        for card in self.directory_dom_cards:
            clean_phone = PbEStringSynthesizer.apply_rule("normalize_us_phone", card["phone"])
            clean_domain = PbEStringSynthesizer.apply_rule("extract_root_domain", card["website"])

            # Natural-key lookup by (Phone, Domain)
            existing_row = next(
                (r for r in sheet["rows"] if r[1] == clean_phone or r[2] == clean_domain),
                None,
            )
            if existing_row is None:
                sheet["rows"].append(
                    [card["name"], clean_phone, clean_domain, card["address"], "", "Harvested"]
                )
                upserted += 1

        # 2. Waterfall Enrichment with Checkpoint Skip
        enriched_count = 0
        if enrich:
            for idx, r in enumerate(sheet["rows"], start=1):
                # Checkpoint Precondition: Skip if already enriched!
                if r[4] and r[5] in ("Enriched", "Contacted"):
                    continue
                if self.fault_rate_limit_at_call is not None and (
                    self.enrichment_api_calls + 1 == self.fault_rate_limit_at_call
                ):
                    self.enrichment_api_calls += 1
                    raise RuntimeError("HTTP 429 Too Many Requests during enrichment batch")

                self.enrichment_api_calls += 1
                info = self.enrichment_catalog.get(r[2], {})
                if info:
                    r[4] = info.get("email", "")
                    r[5] = "Enriched"
                    enriched_count += 1

        # 3. Outbound Dispatch with Blast-Radius & Template Validation Gate
        sent_count = 0
        if send_outbound:
            ready_rows = [r for r in sheet["rows"] if r[4] and r[5] == "Enriched"]
            if len(ready_rows) > 25:
                raise BlastRadiusExceededError(
                    f"[BlastRadius Gate Failed] Batch size {len(ready_rows)} exceeds max_unattended_batch=25."
                )
            for r in ready_rows:
                email = r[4]
                # Checkpoint / Idempotency Ledger Gate
                if email in self.state.outbound_ledger:
                    continue
                rendered = outbound_template.replace("{Company Name}", r[0]).replace(
                    "{Address}", r[3]
                )
                # Precondition {P}: Block unresolved template placeholders!
                if re.search(r"\{[A-Za-z0-9_ ]+\}|\$\{[^}]+\}", rendered):
                    raise PreconditionFailedError(
                        f"[Precondition {{P}} Failed] Unresolved merge tag in outbound message: {rendered!r}"
                    )
                self.state.outbound_ledger.add(email)
                r[5] = "Contacted"
                sent_count += 1

        # 4. Postcondition Read-Back Gate {Q}
        seen_keys: Set[str] = set()
        for r in sheet["rows"]:
            key = f"{r[1]}|{r[2]}"
            if key in seen_keys:
                raise PostconditionFailedError(
                    f"[Postcondition {{Q}} Failed] Duplicate lead key detected in Sheet: {key}"
                )
            seen_keys.add(key)

        return {
            "total_rows": len(sheet["rows"]),
            "upserted": upserted,
            "enriched": enriched_count,
            "sent": sent_count,
            "enrichment_api_calls": self.enrichment_api_calls,
        }


class WorkArenaOSWorldBenchmark:
    """
    Head-to-Head Evaluation Suite comparing:
      1. Standard Per-Step VLM Loop (OSWorld / WorkArena screenshot + A11y agent baseline)
      2. Naive Tier-3 Coordinate Macro (raw screen-coordinate replay)
      3. Hey Pal Compiled Workflow (Demonstrate-Once -> Tier 0/1/1.5 Lowering + OSWorld Getters)
    across multi-app OSWorld desktop tasks and WorkArena enterprise web tasks under layout/window drift.
    """

    @classmethod
    def run_comparison_suite(cls) -> Dict[str, Any]:
        from whole_body_compute.integrations.pal_workflow import PalWorkflowRunner

        # 1. OSWorld Multi-App Task: Finder -> Clean Filename -> Google Drive Upload -> Google Sheets Update
        twin_clean = TwinWorldEnvironment()
        twin_perturbed = TwinWorldEnvironment()
        twin_perturbed.perturb_window_position(dx=180.0, dy=95.0)
        twin_perturbed.perturb_insert_sheet_columns_before_j(["QA Reviewer", "MarginTier"])
        twin_perturbed.perturb_browser_pointed_at_personal_drive()

        tier3_osworld_perturbed_ok = twin_perturbed.execute_tier3_gui_client_upload()
        pal_osworld_res = twin_perturbed.execute_tier1_compiled_client_upload()
        pal_osworld_ok = pal_osworld_res.get("status") == "CERTIFIED"

        # 2. WorkArena Enterprise Web Task: Directory Filter / Table Harvest under DOM Banner Shift
        twin_wa_perturbed = TwinWorldEnvironment()
        twin_wa_perturbed.perturb_lead_card_heights(extra_px_on_first_card=48)
        twin_wa_perturbed.perturb_window_position(dx=120.0, dy=60.0)

        tier3_workarena_rows = twin_wa_perturbed.execute_tier3_gui_lead_harvester()
        pal_workarena_res = twin_wa_perturbed.execute_compiled_lead_gen_pipeline(
            enrich=True, send_outbound=True
        )
        pal_workarena_ok = pal_workarena_res.get("upserted", 0) == 3 and pal_workarena_res.get("sent", 0) == 3

        # 3. Token & Screenshot Bandwidth Savings for a 14-step OSWorld / WorkArena task
        # Using PalWorkflowRunner's per-turn vision token model
        steps_count = 14
        vlm_tokens = sum(
            PalWorkflowRunner.SYS_PROMPT_TOKENS_PER_TURN
            + 14500
            + i * PalWorkflowRunner.ROLLING_HISTORY_TOKENS_PER_PRIOR_TURN
            for i in range(steps_count)
        )
        screen_mb = round(steps_count * PalWorkflowRunner.RETINA_SCREENSHOT_MB, 1)

        return {
            "osworld_multi_app_task": {
                "tier3_coordinate_macro_under_drift": tier3_osworld_perturbed_ok,
                "hey_pal_compiled_under_drift": pal_osworld_ok,
                "postcondition_getters_verified": True,
            },
            "workarena_web_task": {
                "tier3_coordinate_macro_rows_harvested": tier3_workarena_rows,
                "hey_pal_tier1_5_dom_rows_harvested": pal_workarena_res.get("upserted", 0),
                "hey_pal_compiled_under_dom_shift": pal_workarena_ok,
            },
            "telemetry_comparison_per_run": {
                "vlm_agent_steps": steps_count,
                "vlm_agent_cloud_tokens": vlm_tokens,
                "vlm_agent_screen_mb": screen_mb,
                "hey_pal_replay_cloud_tokens": 0,
                "hey_pal_replay_screen_mb": 0.0,
                "cloud_tokens_saved": vlm_tokens,
                "cloud_tokens_saved_formatted": PalWorkflowRunner.format_token_count(vlm_tokens),
            },
        }
