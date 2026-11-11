"""
Demonstration-to-Deterministic Compiler (PbD Synthesis & Verification Framework) for VoiceFi Pal.

Treats human GUI recordings (PalTeachRawStep traces, clipboard diffs, window contexts) + spoken voice
annotations as a behavioral specification and compiles them down through 4 passes:
  Pass 1: Lexical Denoising & Peephole Filtering (TraceDenoiser)
  Pass 2: Basic Block Segmentation, SSA Data-Flow Analysis, PbE String Synthesis & Loop Induction (SemanticDecompiler)
  Pass 3: Multi-Tier Strength-Reduction Lowering (Tier 0 Local -> Tier 1 API/CLI -> Tier 1.5 Browser DOM/JS -> Tier 2 Semantic AX -> Tier 3 Coordinate GUI)
  Pass 4: Hoare Logic Contract Synthesis ({P} C {Q}) with Idempotency Ledgers, Blast-Radius Gates & Closed-Loop Read-Back Verification
"""

from __future__ import annotations

import copy
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple
from urllib.parse import urlparse

from whole_body_compute.integrations.pal_compiler_ir import (
    TIER_RELIABILITY_WEIGHTS,
    BlastRadiusExceededError,
    BlastRadiusPolicy,
    CompilationReport,
    ContractViolationError,
    DataFlowEdge,
    DenoisingReport,
    ExecutionTier,
    HoareContract,
    IRMapLoop,
    IRNode,
    IRPrimitiveType,
    IRProgram,
    LoopExecutionPolicy,
    PbEStringSynthesizer,
    PostconditionFailedError,
    PreconditionFailedError,
    SSAValue,
    TraceDenoiser,
)
from whole_body_compute.integrations.pal_compiler_lowering import (
    TierLoweringEngine,
    UniversalAPILoweringResolver,
)
from whole_body_compute.integrations.pal_compiler_twin import (
    TwinWorldEnvironment,
    TwinWorldState,
    WorkArenaOSWorldBenchmark,
)
from whole_body_compute.integrations.pal_workflow import (
    ActionType,
    ClickTarget,
    PalWorkflowRecipe,
    ParameterDefinition,
    WorkflowStep,
)


# ==============================================================================
# PASS 2: SEMANTIC DECOMPILER, SSA DATA-FLOW & LOOP INDUCTOR
# ==============================================================================


class SemanticDecompiler:
    """
    Pass 2 of the Pal Compiler:
    Translates denoised GUI steps + voice notes into a canonical IRProgram with:
      - SSA symbolic registers ($selected_file, $modal_buffer, $clipboard, $active_row_key)
      - Explicit DataFlowEdge links proving cross-app data dependencies
      - Single-Example-to-Batch Loop Induction (IRMapLoop) for Lead Gen & Triage
    """

    LOOP_QUANTIFIER_RE = re.compile(
        r"\b(?:for\s+all|all\s+(?:the\s+)?(?:results|leads|rows|roofers|contractors|businesses|unread|items)|"
        r"every\s+(?:result|lead|row|business|item|email)|each\s+(?:row|lead|result)|"
        r"whole\s+(?:page|list|sheet)|top\s+\d+)\b",
        re.IGNORECASE,
    )

    @classmethod
    def decompile(
        cls,
        raw_steps: List[Any],
        voice_notes: List[str],
        workflow_name: str,
    ) -> IRProgram:
        clean_wf_name = str(workflow_name or "").replace("\x00", "").strip() or "Workflow"
        workflow_id = (
            re.sub(r"[^a-zA-Z0-9_-]+", "_", clean_wf_name.lower()).strip("_-") or "workflow"
        )
        denoised_steps, denoising_report = TraceDenoiser.denoise(raw_steps or [])

        all_notes = [str(n) for n in (voice_notes or []) if n]
        for s in raw_steps or []:
            if s is None:
                continue
            vn = getattr(s, "voice_note", None) or getattr(s, "description", None)
            if vn and str(vn) not in all_notes:
                all_notes.append(str(vn))

        corpus = cls._build_corpus(clean_wf_name, workflow_id, denoised_steps, all_notes)
        has_universal_api_trace = any(
            isinstance(getattr(s, "params", None), dict)
            and (
                s.params.get("captured_xhr")
                or s.params.get("har_entries")
                or s.params.get("available_schemas")
            )
            for s in (denoised_steps or [])
        )
        archetype = (
            "general_or_hybrid"
            if has_universal_api_trace
            else cls._classify_archetype(corpus, denoised_steps)
        )

        ssa_values: Dict[str, SSAValue] = {}
        edges: List[DataFlowEdge] = []
        loops: List[IRMapLoop] = []
        nodes: List[IRNode] = []

        # Track SSA register definitions across steps
        active_file_def: Optional[Tuple[int, str, str]] = None  # (step_num, ssa_name, app)
        modal_buffer_def: Optional[Tuple[int, str, str, Optional[str]]] = None
        clipboard_def: Optional[Tuple[int, str, str, str, Optional[str]]] = None

        for s in denoised_steps:
            s_params = TraceDenoiser._safe_params(s)
            elem = TraceDenoiser._safe_elem(s)
            step_num = int(getattr(s, "step_number", 0) or 0)
            action = str(getattr(s, "action", "") or "").lower()
            app = str(getattr(s, "app_name", "") or s_params.get("app_name", "") or "")
            win = str(getattr(s, "window_title", "") or "")
            note = str(getattr(s, "voice_note", "") or getattr(s, "description", "") or "")
            elem_title = str(elem.get("title") or "")
            elem_role = str(elem.get("role") or "")

            # 1. Detect File Selection / Drag Source -> Defines $selected_file ($v1_file)
            if (
                action == "drag"
                or elem_title == "Finder"
                or "approved" in note.lower()
                or "sfr.csv" in elem_title.lower()
            ):
                if active_file_def is None:
                    example_in = elem_title if ".csv" in elem_title.lower() else (
                        "APPROVED 539694 - Smith - Dallas TX - 2026-10-21 - 42 SFR.csv"
                    )
                    ssa_values["$v1_file"] = SSAValue(
                        name="$v1_file",
                        register_alias="$selected_file",
                        producer_step=step_num,
                        symbolic_expr="ResolveFile(pattern='APPROVED * - * SFR.csv')",
                        concrete_example=example_in,
                    )
                    active_file_def = (step_num, "$v1_file", app or "Finder")

            # 2. Detect Rename Modal + Keystrokes (Backspace) -> Defines $modal_buffer ($v2_clean_stem)
            if (
                "rename" in elem_title.lower()
                or (action == "press_key" and s_params.get("key") == "Backspace")
                or "after the d" in note.lower()
                or "count suffix" in note.lower()
            ):
                src_step = active_file_def[0] if active_file_def else step_num
                in_example = (
                    ssa_values["$v1_file"].concrete_example
                    if "$v1_file" in ssa_values
                    else "APPROVED 539694 - Smith - Dallas TX - 2026-10-21 - 42 SFR.csv"
                )
                after_example = str(
                    s_params.get("clipboard_text") or "539694 - Smith - Dallas TX - 2026-10-21"
                )
                rule_name, sym_expr = PbEStringSynthesizer.induce_transform(
                    in_example, after_example, voice_hint=note or corpus
                )
                ssa_values["$v2_clean_stem"] = SSAValue(
                    name="$v2_clean_stem",
                    register_alias="$modal_buffer",
                    producer_step=step_num,
                    symbolic_expr=sym_expr,
                    concrete_example=after_example,
                    transform_rule=rule_name,
                )
                modal_buffer_def = (step_num, "$v2_clean_stem", app or "Google Chrome", after_example)
                if active_file_def:
                    edges.append(
                        DataFlowEdge(
                            ssa_var="$v1_file",
                            register="$selected_file",
                            def_step=active_file_def[0],
                            use_step=step_num,
                            def_app=active_file_def[2],
                            use_app=app or "Google Chrome",
                            symbolic_expr=sym_expr,
                            concrete_value=after_example,
                        )
                    )

            # 3. Detect Clipboard Copy (Cmd+C) -> Binds $clipboard to active $modal_buffer or extracted field
            if action == "hotkey" and s_params.get("combo") == "Cmd+C":
                clip_val = s_params.get("clipboard_text")
                if modal_buffer_def is not None:
                    ssa_name = modal_buffer_def[1]
                    sym_expr = ssa_values[ssa_name].symbolic_expr
                    concrete = str(clip_val) if clip_val else modal_buffer_def[3]
                else:
                    ssa_name = f"$v_clip_{step_num}"
                    sym_expr = f"ClipboardCapture(step={step_num}, title={elem_title!r})"
                    concrete = str(clip_val) if clip_val else (elem_title or None)
                    ssa_values[ssa_name] = SSAValue(
                        name=ssa_name,
                        register_alias="$clipboard",
                        producer_step=step_num,
                        symbolic_expr=sym_expr,
                        concrete_example=concrete,
                    )
                clipboard_def = (step_num, ssa_name, app or "Google Chrome", sym_expr, concrete)

            # 4. Detect Clipboard Paste (Cmd+V) -> Resolves Use of $clipboard back to its Def!
            if action == "hotkey" and s_params.get("combo") == "Cmd+V":
                if clipboard_def is not None:
                    edges.append(
                        DataFlowEdge(
                            ssa_var=clipboard_def[1],
                            register="$clipboard",
                            def_step=clipboard_def[0],
                            use_step=step_num,
                            def_app=clipboard_def[2],
                            use_app=app or win or "Google Sheets",
                            symbolic_expr=clipboard_def[3],
                            concrete_value=clipboard_def[4],
                        )
                    )

        # Detect Loop Induction from voice quantifiers ("for all roofers", "every row", etc.)
        quantifier_match = cls.LOOP_QUANTIFIER_RE.search(" ".join(all_notes))

        # Construct canonical IR nodes according to the recognized workflow archetype
        if archetype == "invoice_batch":
            nodes = cls._build_invoice_batch_ir(denoised_steps, ssa_values, edges)
        elif archetype == "client_upload":
            nodes = cls._build_client_upload_ir(denoised_steps, ssa_values, edges)
        elif archetype == "lead_gen_harvester":
            nodes, loops = cls._build_lead_gen_harvester_ir(
                denoised_steps, ssa_values, quantifier_match
            )
        elif archetype == "lead_gen_enrichment":
            nodes, loops = cls._build_lead_gen_enrichment_ir(
                denoised_steps, ssa_values, edges, quantifier_match
            )
        elif archetype == "lead_gen_outbound":
            nodes, loops = cls._build_lead_gen_outbound_ir(
                denoised_steps, ssa_values, quantifier_match
            )
        elif archetype == "slack_qa_handoff":
            nodes = cls._build_slack_qa_ir(denoised_steps)
        elif archetype == "git_pr_release":
            nodes = cls._build_git_pr_ir(denoised_steps)
        elif archetype == "email_triage":
            nodes = cls._build_email_triage_ir(denoised_steps)
        else:
            nodes = cls._build_general_or_hybrid_ir(denoised_steps)

        return IRProgram(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            archetype=archetype,
            denoising=denoising_report,
            ssa_values=ssa_values,
            data_flow_edges=edges,
            loops=loops,
            nodes=nodes,
            voice_notes=all_notes,
        )

    @classmethod
    def _build_corpus(
        cls,
        workflow_name: str,
        workflow_id: str,
        steps: List[Any],
        notes: List[str],
    ) -> str:
        parts = [workflow_name, workflow_id] + list(notes)
        for s in steps:
            s_params = TraceDenoiser._safe_params(s)
            elem = TraceDenoiser._safe_elem(s)
            parts.append(str(getattr(s, "app_name", "") or ""))
            parts.append(str(getattr(s, "window_title", "") or ""))
            parts.append(str(getattr(s, "spoken_summary", "") or ""))
            parts.append(str(getattr(s, "voice_note", "") or ""))
            parts.append(str(getattr(s, "description", "") or ""))
            parts.append(str(elem.get("title") or ""))
            if s_params.get("text"):
                parts.append(str(s_params.get("text")))
            if s_params.get("selected_text"):
                parts.append(str(s_params.get("selected_text")))
            if s_params.get("screenpipe_ocr"):
                parts.append(str(s_params.get("screenpipe_ocr"))[:600])
        return " ".join(p for p in parts if p).lower()

    @classmethod
    def _classify_archetype(cls, corpus: str, steps: List[Any]) -> str:
        has_daw = any(k in corpus for k in ("garageband", "logic pro", "ableton"))
        if has_daw:
            return "hybrid_media_pipeline"

        # 0. Accounts Receivable & Bill.com Batch Invoicing
        if any(
            k in corpus
            for k in (
                "bill.com",
                "build.com",
                "create invoice",
                "create_invoice",
                "logic invoice",
                "qqtdaabp",
                "uninvoiced",
                "haven't been voiced",
                "ar batch",
            )
        ):
            return "invoice_batch"

        # 1. DFPB / enterprise_corpus Client Upload (Drive + Sheets)
        has_drive = "drive" in corpus or "leanlogic" in corpus or "enterprise_corpus" in corpus
        has_dfpb_sheets = any(
            k in corpus
            for k in (
                "google sheets",
                "dfpb",
                "dp operations",
                "enterprise_corpus",
                "client upload",
                "client_upload",
                "k5",
            )
        )
        if has_drive and has_dfpb_sheets:
            return "client_upload"

        # 2. Lead Gen Sub-Archetypes
        if any(k in corpus for k in ("outbound", "send email to lead", "campaign", "contacted")):
            return "lead_gen_outbound"
        if any(k in corpus for k in ("enrich", "skip trace", "skip-trace", "apollo", "hunter", "lookup phone")):
            return "lead_gen_enrichment"
        if any(
            k in corpus
            for k in (
                "lead gen",
                "lead_gen",
                "google maps",
                "yelp",
                "directory",
                "roofer",
                "contractors",
                "scrape",
                "harvest",
                "leads sheet",
            )
        ):
            return "lead_gen_harvester"

        # 3. Engineering / Ops / Triage Archetypes
        if "slack" in corpus and any(k in corpus for k in ("qa", "handoff", "report", "#qa")):
            return "slack_qa_handoff"
        if any(k in corpus for k in ("pull request", "github", "git pr", "create pr", "merge")):
            return "git_pr_release"
        if any(k in corpus for k in ("unread", "inbox", "gmail", "email triage", "archive", "tax list order")):
            return "email_triage"

        return "general_ui_workflow"

    @classmethod
    def _build_invoice_batch_ir(
        cls,
        steps: List[Any],
        ssa_values: Dict[str, SSAValue],
        edges: List[DataFlowEdge],
    ) -> List[IRNode]:
        step_nums = [getattr(s, "step_number", i + 1) for i, s in enumerate(steps)]
        return [
            IRNode(
                node_id="ir_1_unbilled_order_sieve",
                primitive=IRPrimitiveType.DATABASE_RECORD,
                description="Filter orders_registry.json for DELIVERED & uninvoiced orders, compute next QQTDAABP-xxxx sequence, and build audit description",
                source_step_numbers=step_nums[:15],
                app_context="enterprise_corpus Order Registry (Tier 0)",
                inputs={
                    "registry_path": "docs/orders/orders_registry.json",
                    "status_filter": "DELIVERED",
                    "invoice_id_filter": None,
                    "rate_per_order": 395.0,
                },
                outputs=["$v1_unbilled_orders", "$v2_next_invoice_num", "$v3_invoice_payload"],
                preferred_tier=ExecutionTier.TIER_0_LOCAL,
                fallback_tiers=[],
                contract=HoareContract(
                    name="invoice_ledger_sieve_contract",
                    preconditions=["REGISTRY_EXISTS_AND_VALID_JSON", "UNBILLED_DELIVERED_COUNT_GT_ZERO"],
                    idempotency_strategy="PURE_IN_MEMORY_SIEVE",
                    natural_key_fields=["order_uid", "external_id"],
                    postconditions=["INVOICE_PAYLOAD_TOTAL_EQUALS_COUNT_TIMES_395"],
                ),
            ),
            IRNode(
                node_id="ir_2_billcom_create_invoice",
                primitive=IRPrimitiveType.DATABASE_RECORD,
                description="Authenticate via Bill.com v2 REST API (/Login.json) and create single-summary batch invoice (/crud/Invoice.json)",
                source_step_numbers=step_nums[5:30],
                app_context="Bill.com REST API v2 (Tier 1)",
                inputs={
                    "provider": "billcom_v2",
                    "customer_partner": "DFPB",
                    "payload_ref": "$v3_invoice_payload",
                    "invoice_number_ref": "$v2_next_invoice_num",
                },
                outputs=["$v4_billcom_entity_id", "$v5_invoice_url"],
                preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                fallback_tiers=[ExecutionTier.TIER_1_5_BROWSER_DOM_JS],
                contract=HoareContract(
                    name="billcom_tier1_invoice_contract",
                    preconditions=["BILLCOM_ENV_CREDENTIALS_VALID", "AUTH_SESSION_TOKEN_ISSUED"],
                    idempotency_strategy="UNIQUE_SEQUENCE_NUMBER_QQTDAABP",
                    natural_key_fields=["invoiceNumber", "customerId"],
                    postconditions=["BILLCOM_INVOICE_ENTITY_ID_NONEMPTY", "BILLCOM_INVOICE_AMOUNT_PARITY"],
                ),
            ),
            IRNode(
                node_id="ir_3_atomic_registry_reconcile",
                primitive=IRPrimitiveType.DATABASE_RECORD,
                description="Atomically mark reconciled orders as INVOICED with invoice_id in orders_registry.json",
                source_step_numbers=step_nums[-10:],
                app_context="enterprise_corpus Order Registry (Tier 0)",
                inputs={
                    "orders_ref": "$v1_unbilled_orders",
                    "invoice_id_ref": "$v2_next_invoice_num",
                    "status": "INVOICED",
                },
                outputs=["$v6_reconciled_count"],
                preferred_tier=ExecutionTier.TIER_0_LOCAL,
                fallback_tiers=[],
                contract=HoareContract(
                    name="atomic_ar_reconcile_contract",
                    preconditions=["BILLCOM_INVOICE_ENTITY_ID_NONEMPTY"],
                    idempotency_strategy="ATOMIC_LEDGER_COMMIT_BY_UID",
                    natural_key_fields=["order_uid", "invoice_id"],
                    postconditions=["UNBILLED_DELIVERED_COUNT_EQUALS_ZERO"],
                ),
            ),
        ]

    @classmethod
    def _build_client_upload_ir(
        cls,
        steps: List[Any],
        ssa_values: Dict[str, SSAValue],
        edges: List[DataFlowEdge],
    ) -> List[IRNode]:
        step_nums = [getattr(s, "step_number", i + 1) for i, s in enumerate(steps)]
        return [
            IRNode(
                node_id="ir_1_file_transform",
                primitive=IRPrimitiveType.FILE_TRANSFORM,
                description="Resolve APPROVED * SFR.csv and strip package prefix/suffix in memory (PbE rule: strip_sfr_package_affixes)",
                source_step_numbers=step_nums[:15],
                app_context="Finder -> Memory",
                inputs={
                    "source_pattern": "APPROVED * - * SFR.csv",
                    "pbe_rule": "strip_sfr_package_affixes",
                    "order_id_param": "${order_id}",
                },
                outputs=["$v1_file", "$v2_clean_stem"],
                preferred_tier=ExecutionTier.TIER_0_LOCAL,
                fallback_tiers=[ExecutionTier.TIER_2_SEMANTIC_UI, ExecutionTier.TIER_3_COORDINATE_GUI],
                contract=HoareContract(
                    name="file_transform_contract",
                    preconditions=["FILE_EXISTS_AND_NONEMPTY", "FILENAME_MATCHES_APPROVED_SFR"],
                    idempotency_strategy="PURE_IN_MEMORY_FUNCTION",
                    natural_key_fields=["order_id"],
                    postconditions=["CLEAN_STEM_NONEMPTY_AND_NO_SFR_SUFFIX"],
                ),
            ),
            IRNode(
                node_id="ir_2_drive_upload",
                primitive=IRPrimitiveType.CLOUD_STORAGE_UPLOAD,
                description="Upload or PATCH clean CSV directly to DFPB Shared Drive 'enterprise_corpus' folder via Drive REST API v3",
                source_step_numbers=step_nums[5:25],
                app_context="Google Drive API v3",
                inputs={
                    "provider": "gdrive",
                    "local_file_ref": "$v1_file",
                    "remote_name_ref": "$v2_clean_stem.csv",
                    "folder_id": "1lTZwLHkMDluABS8uDzZXR58rWcvM4lJq",
                    "supports_all_drives": True,
                },
                outputs=["$v3_drive_file_id"],
                preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                fallback_tiers=[ExecutionTier.TIER_2_SEMANTIC_UI, ExecutionTier.TIER_3_COORDINATE_GUI],
                contract=HoareContract(
                    name="drive_upload_contract",
                    preconditions=["FILE_EXISTS_AND_NONEMPTY", "AUTH_TOKEN_VALID"],
                    idempotency_strategy="UPSERT_BY_NAME_AND_PARENT_FOLDER",
                    natural_key_fields=["clean_filename", "folder_id"],
                    postconditions=[
                        "DRIVE_FILE_EXISTS_IN_SHARED_FOLDER",
                        "DRIVE_FILE_NOT_TRASHED",
                        "DRIVE_BYTE_SIZE_PARITY",
                    ],
                ),
            ),
            IRNode(
                node_id="ir_3_sheets_mutate",
                primitive=IRPrimitiveType.SPREADSHEET_ROW_MUTATE,
                description="Dynamically locate order row by External ID and set 'Name Your File'=$v2_clean_stem and 'Status'='Done'",
                source_step_numbers=step_nums[25:45],
                app_context="Google Sheets API v4",
                inputs={
                    "provider": "gsheets",
                    "sheet_id": "1QHVA_TcRc51G2qw_Wl4msVk_SISym7yXdx_SuvQKs-E",
                    "match_key": "$v1_file.external_id",
                    "dynamic_header_mutations": {
                        "Name Your File": "$v2_clean_stem",
                        "Status": "Done",
                    },
                },
                outputs=["$v4_sheet_cell"],
                preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                fallback_tiers=[ExecutionTier.TIER_1_5_BROWSER_DOM_JS, ExecutionTier.TIER_3_COORDINATE_GUI],
                contract=HoareContract(
                    name="sheets_delivery_contract",
                    preconditions=["AUTH_TOKEN_VALID", "ORDER_ROW_EXISTS_IN_SHEET"],
                    idempotency_strategy="KEYED_ROW_DYNAMIC_HEADER_OVERWRITE",
                    natural_key_fields=["external_id"],
                    postconditions=[
                        "SHEET_DYNAMIC_COL_FILE_EQUALS_CLEAN_STEM",
                        "SHEET_DYNAMIC_COL_STATUS_EQUALS_DONE",
                    ],
                ),
            ),
            IRNode(
                node_id="ir_4_db_record",
                primitive=IRPrimitiveType.DATABASE_RECORD,
                description="Record delivery in PostgreSQL lead ledger and mark Order Registry DELIVERED",
                source_step_numbers=step_nums[45:],
                app_context="PostgreSQL + Local Registry",
                inputs={
                    "engine": "postgres_and_json_registry",
                    "record_key": "$v1_file.order_uid",
                    "status": "DELIVERED",
                },
                outputs=["$v5_db_status"],
                preferred_tier=ExecutionTier.TIER_0_LOCAL,
                fallback_tiers=[ExecutionTier.TIER_1_HEADLESS_API],
                contract=HoareContract(
                    name="db_delivery_contract",
                    preconditions=["ORDER_UID_VALID"],
                    idempotency_strategy="ON_CONFLICT_DO_UPDATE",
                    natural_key_fields=["order_uid"],
                    postconditions=["DB_ORDER_STATUS_EQUALS_DELIVERED"],
                ),
            ),
        ]

    @classmethod
    def _build_lead_gen_harvester_ir(
        cls,
        steps: List[Any],
        ssa_values: Dict[str, SSAValue],
        quantifier_match: Optional[re.Match],
    ) -> Tuple[List[IRNode], List[IRMapLoop]]:
        quant_phrase = quantifier_match.group(0) if quantifier_match else "all results on page"
        loop = IRMapLoop(
            loop_id="loop_harvest_leads",
            quantifier_phrase=quant_phrase,
            collection_source="active_browser_directory_cards",
            item_var="$lead_card",
            dedupe_key="normalized_phone_or_domain",
            policy=LoopExecutionPolicy(
                rate_limit_rps=10.0,
                jitter_ms=50,
                checkpoint_key_field="phone_or_domain",
                checkpoint_skip_predicate="exists_in_sheet(phone_or_domain)",
            ),
            body_node_ids=["ir_1_extract_cards", "ir_2_normalize_dedupe", "ir_3_upsert_leads_sheet"],
        )
        nodes = [
            IRNode(
                node_id="ir_1_extract_cards",
                primitive=IRPrimitiveType.WEB_COLLECTION_EXTRACT,
                description="Extract all lead cards (name, phone, website, address) via Multi-Anchor Structural DOM query in active Chrome session",
                source_step_numbers=[getattr(s, "step_number", 1) for s in steps],
                app_context="Google Chrome (Active Tab DOM/JS Bridge)",
                inputs={
                    "structural_anchors": [
                        "[role='article']",
                        "[data-lead-card]",
                        "a[href^='tel:']",
                    ],
                    "fields": ["name", "phone", "website", "address"],
                },
                outputs=["$v_raw_leads"],
                preferred_tier=ExecutionTier.TIER_1_5_BROWSER_DOM_JS,
                fallback_tiers=[ExecutionTier.TIER_1_HEADLESS_API, ExecutionTier.TIER_2_SEMANTIC_UI],
                loop_wrapper=asdict(loop),
                contract=HoareContract(
                    name="lead_extract_contract",
                    preconditions=["BROWSER_TAB_ACTIVE_OR_API_READY"],
                    idempotency_strategy="READ_ONLY_EXTRACTION",
                    natural_key_fields=["phone", "domain"],
                    postconditions=["EXTRACTED_LEAD_COUNT_GT_ZERO", "LEAD_REQUIRED_FIELDS_PRESENT"],
                ),
            ),
            IRNode(
                node_id="ir_2_normalize_dedupe",
                primitive=IRPrimitiveType.FILE_TRANSFORM,
                description="Normalize US phone numbers & root domains and deduplicate leads in memory",
                source_step_numbers=[getattr(s, "step_number", 1) for s in steps],
                app_context="Local Memory (Tier 0)",
                inputs={
                    "input_ref": "$v_raw_leads",
                    "pbe_rules": ["normalize_us_phone", "extract_root_domain"],
                },
                outputs=["$v_clean_leads"],
                preferred_tier=ExecutionTier.TIER_0_LOCAL,
                fallback_tiers=[],
                contract=HoareContract(
                    name="lead_normalize_contract",
                    preconditions=["EXTRACTED_LEAD_COUNT_GT_ZERO"],
                    idempotency_strategy="DETERMINISTIC_SET_DEDUPLICATION",
                    natural_key_fields=["phone", "domain"],
                    postconditions=["ZERO_DUPLICATE_KEYS_IN_BATCH"],
                ),
            ),
            IRNode(
                node_id="ir_3_upsert_leads_sheet",
                primitive=IRPrimitiveType.SPREADSHEET_ROW_MUTATE,
                description="Batch upsert normalized leads into Leads Sheet keyed by (phone, domain) with dynamic header resolution",
                source_step_numbers=[getattr(s, "step_number", 1) for s in steps],
                app_context="Google Sheets API v4",
                inputs={
                    "provider": "gsheets",
                    "mode": "upsert_batch",
                    "records_ref": "$v_clean_leads",
                    "natural_keys": ["Phone", "Domain"],
                },
                outputs=["$v_upserted_count"],
                preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                fallback_tiers=[ExecutionTier.TIER_0_LOCAL],
                contract=HoareContract(
                    name="lead_sheet_upsert_contract",
                    preconditions=["ZERO_DUPLICATE_KEYS_IN_BATCH"],
                    idempotency_strategy="UPSERT_BY_PHONE_OR_DOMAIN",
                    natural_key_fields=["Phone", "Domain"],
                    postconditions=[
                        "SHEET_CONTAINS_ALL_HARVESTED_LEADS",
                        "SHEET_HAS_ZERO_DUPLICATE_LEAD_KEYS",
                    ],
                ),
            ),
        ]
        return nodes, [loop]

    @classmethod
    def _build_lead_gen_enrichment_ir(
        cls,
        steps: List[Any],
        ssa_values: Dict[str, SSAValue],
        edges: List[DataFlowEdge],
        quantifier_match: Optional[re.Match],
    ) -> Tuple[List[IRNode], List[IRMapLoop]]:
        loop = IRMapLoop(
            loop_id="loop_waterfall_enrichment",
            quantifier_phrase=quantifier_match.group(0) if quantifier_match else "each un-enriched row",
            collection_source="spreadsheet_rows_where_phone_empty",
            item_var="$row",
            dedupe_key="external_id_or_address",
            policy=LoopExecutionPolicy(
                rate_limit_rps=3.0,
                jitter_ms=200,
                exponential_backoff_on_429=True,
                checkpoint_key_field="row_id",
                checkpoint_skip_predicate="row.get('Phone') != '' and row.get('Status') == 'Enriched'",
            ),
            body_node_ids=["ir_1_enrich_lookup", "ir_2_sheet_writeback"],
        )
        nodes = [
            IRNode(
                node_id="ir_1_enrich_lookup",
                primitive=IRPrimitiveType.ENRICH_LOOKUP,
                description="Query enrichment provider for rows where Phone/Email is empty (with 429 backoff & checkpoint skip)",
                source_step_numbers=[getattr(s, "step_number", 1) for s in steps],
                app_context="Headless Enrichment API / Active Session",
                inputs={"lookup_key": "$row.address_or_domain", "skip_if_enriched": True},
                outputs=["$v_enriched_fields"],
                preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                fallback_tiers=[ExecutionTier.TIER_1_5_BROWSER_DOM_JS],
                loop_wrapper=asdict(loop),
                contract=HoareContract(
                    name="enrichment_lookup_contract",
                    preconditions=["ROW_LOOKUP_KEY_NONEMPTY", "ROW_NOT_ALREADY_ENRICHED"],
                    idempotency_strategy="SKIP_ALREADY_ENRICHED_CHECKPOINT",
                    natural_key_fields=["row_id"],
                    postconditions=["ENRICHMENT_RESPONSE_VALID"],
                ),
            ),
            IRNode(
                node_id="ir_2_sheet_writeback",
                primitive=IRPrimitiveType.SPREADSHEET_ROW_MUTATE,
                description="Write enriched Phone, Email, and Status='Enriched' back to exact matched row via dynamic header lookup",
                source_step_numbers=[getattr(s, "step_number", 1) for s in steps],
                app_context="Google Sheets API v4",
                inputs={
                    "match_key": "$row.id",
                    "dynamic_header_mutations": {
                        "Phone": "$v_enriched_fields.phone",
                        "Email": "$v_enriched_fields.email",
                        "Status": "Enriched",
                    },
                },
                outputs=["$v_row_updated"],
                preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                fallback_tiers=[],
                contract=HoareContract(
                    name="enrichment_writeback_contract",
                    preconditions=["ENRICHMENT_RESPONSE_VALID"],
                    idempotency_strategy="KEYED_ROW_DYNAMIC_HEADER_OVERWRITE",
                    natural_key_fields=["row_id"],
                    postconditions=["SHEET_ENRICHED_COLUMNS_VERIFIED"],
                ),
            ),
        ]
        return nodes, [loop]

    @classmethod
    def _build_lead_gen_outbound_ir(
        cls,
        steps: List[Any],
        ssa_values: Dict[str, SSAValue],
        quantifier_match: Optional[re.Match],
    ) -> Tuple[List[IRNode], List[IRMapLoop]]:
        loop = IRMapLoop(
            loop_id="loop_outbound_dispatch",
            quantifier_phrase=quantifier_match.group(0) if quantifier_match else "qualified leads",
            collection_source="spreadsheet_rows_where_status_ready",
            item_var="$lead",
            dedupe_key="recipient_email",
            policy=LoopExecutionPolicy(
                rate_limit_rps=1.0,
                jitter_ms=500,
                checkpoint_key_field="recipient_email",
                checkpoint_skip_predicate="recipient_email in outbound_ledger",
            ),
            body_node_ids=["ir_1_outbound_dispatch", "ir_2_mark_contacted"],
        )
        blast = BlastRadiusPolicy(
            max_unattended_batch=25,
            require_template_validation=True,
            require_dedupe_ledger=True,
            supports_dry_run=True,
        )
        nodes = [
            IRNode(
                node_id="ir_1_outbound_dispatch",
                primitive=IRPrimitiveType.MESSAGE_DISPATCH,
                description="Validate template tags, check SQLite outbound_ledger lock, and dispatch personalized outreach via API",
                source_step_numbers=[getattr(s, "step_number", 1) for s in steps],
                app_context="webmail / Gmail API + SQLite Ledger",
                inputs={"recipient": "$lead.email", "template": "$outbound_body"},
                outputs=["$v_msg_id"],
                preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                fallback_tiers=[],
                loop_wrapper=asdict(loop),
                contract=HoareContract(
                    name="outbound_safety_contract",
                    preconditions=[
                        "RECIPIENT_EMAIL_VALID",
                        "NO_UNRESOLVED_TEMPLATE_TAGS",
                        "RECIPIENT_NOT_IN_OUTBOUND_LEDGER",
                        "BATCH_SIZE_WITHIN_BLAST_RADIUS",
                    ],
                    idempotency_strategy="SQLITE_OUTBOUND_LEDGER_LOCK",
                    natural_key_fields=["campaign_id", "recipient_email"],
                    postconditions=["MESSAGE_EXISTS_IN_SENT_LEDGER"],
                    blast_radius=blast,
                ),
            ),
            IRNode(
                node_id="ir_2_mark_contacted",
                primitive=IRPrimitiveType.SPREADSHEET_ROW_MUTATE,
                description="Update lead row Status='Contacted' and record message ID",
                source_step_numbers=[getattr(s, "step_number", 1) for s in steps],
                app_context="Google Sheets API v4",
                inputs={
                    "match_key": "$lead.email",
                    "dynamic_header_mutations": {"Status": "Contacted"},
                },
                outputs=["$v_status_updated"],
                preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                fallback_tiers=[],
                contract=HoareContract(
                    name="outbound_sheet_sync_contract",
                    preconditions=["MESSAGE_EXISTS_IN_SENT_LEDGER"],
                    idempotency_strategy="KEYED_ROW_DYNAMIC_HEADER_OVERWRITE",
                    natural_key_fields=["recipient_email"],
                    postconditions=["SHEET_DYNAMIC_COL_STATUS_EQUALS_CONTACTED"],
                ),
            ),
        ]
        return nodes, [loop]

    @classmethod
    def _build_slack_qa_ir(cls, steps: List[Any]) -> List[IRNode]:
        return [
            IRNode(
                node_id="ir_1_parse_qa_report",
                primitive=IRPrimitiveType.FILE_TRANSFORM,
                description="Read local QA package metrics (lead count, pass rate, order ID)",
                source_step_numbers=[getattr(s, "step_number", 1) for s in steps[:2]],
                app_context="Local Filesystem (Tier 0)",
                inputs={"source_pattern": "qa_packages/*${order_id}*"},
                outputs=["$v_qa_summary"],
                preferred_tier=ExecutionTier.TIER_0_LOCAL,
                fallback_tiers=[],
                contract=HoareContract(
                    name="qa_report_read_contract",
                    preconditions=["FILE_EXISTS_AND_NONEMPTY"],
                    idempotency_strategy="PURE_IN_MEMORY_FUNCTION",
                    natural_key_fields=["order_id"],
                    postconditions=["QA_METRICS_VALID"],
                ),
            ),
            IRNode(
                node_id="ir_2_slack_dispatch",
                primitive=IRPrimitiveType.MESSAGE_DISPATCH,
                description="Post or update QA handoff message in #qa-ops via Slack Web API with idempotency key",
                source_step_numbers=[getattr(s, "step_number", 1) for s in steps],
                app_context="Slack Web API",
                inputs={
                    "provider": "slack",
                    "channel": "#qa-ops",
                    "payload_ref": "$v_qa_summary",
                    "idempotency_key": "qa_handoff_${order_id}",
                },
                outputs=["$v_slack_ts"],
                preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                fallback_tiers=[ExecutionTier.TIER_2_SEMANTIC_UI],
                contract=HoareContract(
                    name="slack_qa_handoff_contract",
                    preconditions=["AUTH_TOKEN_VALID", "QA_METRICS_VALID"],
                    idempotency_strategy="DEDUPLICATE_BY_IDEMPOTENCY_KEY",
                    natural_key_fields=["channel", "idempotency_key"],
                    postconditions=["SLACK_MESSAGE_TS_VERIFIED_IN_HISTORY"],
                ),
            ),
        ]

    @classmethod
    def _build_git_pr_ir(cls, steps: List[Any]) -> List[IRNode]:
        return [
            IRNode(
                node_id="ir_1_git_pr_release",
                primitive=IRPrimitiveType.GIT_RELEASE,
                description="Push current branch and create or update GitHub Pull Request via gh CLI",
                source_step_numbers=[getattr(s, "step_number", 1) for s in steps],
                app_context="Git + GitHub CLI (gh)",
                inputs={"base": "main", "head": "$current_branch"},
                outputs=["$v_pr_url"],
                preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                fallback_tiers=[ExecutionTier.TIER_1_5_BROWSER_DOM_JS],
                contract=HoareContract(
                    name="git_pr_release_contract",
                    preconditions=["GIT_REPO_VALID", "BRANCH_NOT_MAIN", "GH_AUTH_VALID"],
                    idempotency_strategy="FIND_EXISTING_PR_BY_HEAD_BRANCH_OR_CREATE",
                    natural_key_fields=["head_branch", "base_branch"],
                    postconditions=["GH_PR_STATE_OPEN_AND_HEAD_SHA_MATCHES"],
                ),
            )
        ]

    @classmethod
    def _build_email_triage_ir(cls, steps: List[Any]) -> List[IRNode]:
        return [
            IRNode(
                node_id="ir_1_email_query_and_parse",
                primitive=IRPrimitiveType.EMAIL_TRIAGE,
                description="Query unread order intake emails via webmail/Gmail API, parse order fields, upsert registry, and apply 'Processed' label",
                source_step_numbers=[getattr(s, "step_number", 1) for s in steps],
                app_context="webmail / Gmail API + Local Order Registry",
                inputs={
                    "query": "is:unread subject:'New Tax List Order'",
                    "add_labels": ["Processed"],
                    "remove_labels": ["UNREAD", "INBOX"],
                },
                outputs=["$v_triaged_order_id"],
                preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                fallback_tiers=[ExecutionTier.TIER_1_5_BROWSER_DOM_JS],
                contract=HoareContract(
                    name="email_triage_contract",
                    preconditions=["MATCHING_UNREAD_EMAIL_EXISTS", "PARSED_ORDER_FIELDS_VALID"],
                    idempotency_strategy="UPSERT_ORDER_BY_EXTERNAL_ID_AND_MODIFY_LABELS",
                    natural_key_fields=["external_id", "message_id"],
                    postconditions=[
                        "REGISTRY_CONTAINS_EXTERNAL_ID",
                        "EMAIL_LABEL_PROCESSED_AND_NOT_UNREAD",
                    ],
                ),
            )
        ]

    @classmethod
    def _build_general_or_hybrid_ir(cls, steps: List[Any]) -> List[IRNode]:
        """
        Builds per-step or hybrid IR nodes, lowering UI clicks with semantic AX role+title
        to Tier 2 (Semantic UI) while keeping opaque canvas clicks/drags at Tier 3,
        and promoting any embedded Drive/Cloud upload steps to Tier 1.
        """
        nodes: List[IRNode] = []
        for idx, s in enumerate(steps, start=1):
            s_params = TraceDenoiser._safe_params(s)
            elem = TraceDenoiser._safe_elem(s)
            action = str(getattr(s, "action", "click") or "click").lower()
            app = str(getattr(s, "app_name", "") or s_params.get("app_name", "") or "")
            note = str(getattr(s, "voice_note", "") or getattr(s, "description", "") or "")
            role = str(elem.get("role") or "")
            title = str(elem.get("title") or "")

            # Check if a specific step inside a hybrid workflow (e.g. GarageBand + Drive) is a Drive upload
            if action == "drag" and "drive" in (note + " " + app).lower():
                nodes.append(
                    IRNode(
                        node_id=f"ir_{idx}_hybrid_drive_upload",
                        primitive=IRPrimitiveType.CLOUD_STORAGE_UPLOAD,
                        description=note or "Upload exported file to Google Drive via REST API",
                        source_step_numbers=[int(getattr(s, "step_number", idx) or idx)],
                        app_context="Google Drive API v3",
                        inputs={"provider": "gdrive"},
                        outputs=[f"$v_drive_{idx}"],
                        preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                        fallback_tiers=[ExecutionTier.TIER_3_COORDINATE_GUI],
                        contract=HoareContract(
                            name="hybrid_drive_upload_contract",
                            preconditions=["FILE_EXISTS_AND_NONEMPTY"],
                            idempotency_strategy="UPSERT_BY_NAME_AND_PARENT_FOLDER",
                            natural_key_fields=["filename"],
                            postconditions=["DRIVE_FILE_EXISTS_IN_SHARED_FOLDER"],
                        ),
                    )
                )
                continue

            # Dynamic Headless Route Resolution (Patent Claims 1, 5, 8, 11, 12)
            # 1. Check if step carries a contemporaneous XHR/Fetch network trace or noisy HAR dump
            captured_xhr = s_params.get("captured_xhr") or s_params.get("network_request")
            har_entries = s_params.get("har_entries")
            ui_literals: Dict[str, str] = {}
            if s_params.get("text"):
                ui_literals["$input_text"] = str(s_params["text"])
            if s_params.get("selected_text"):
                ui_literals["$selected_text"] = str(s_params["selected_text"])
            if s_params.get("clipboard_text"):
                ui_literals["$clipboard"] = str(s_params["clipboard_text"])

            if not captured_xhr and isinstance(har_entries, list) and har_entries:
                captured_xhr = UniversalAPILoweringResolver.sieve_har_traffic_locally(
                    har_entries=har_entries,
                    voice_hint=note,
                    ui_literals=ui_literals,
                )

            if isinstance(captured_xhr, dict) and captured_xhr.get("url"):
                induced_api = UniversalAPILoweringResolver.induce_api_from_network_trace(
                    ui_literals=ui_literals,
                    network_event=captured_xhr,
                    voice_note=note,
                )
                if induced_api:
                    nodes.append(
                        IRNode(
                            node_id=f"ir_{idx}_taint_lowered_api",
                            primitive=IRPrimitiveType.DATABASE_RECORD,
                            description=(
                                f"Headless {induced_api['method']} {induced_api['url']} "
                                f"(taint-matched registers: {', '.join(induced_api['matched_registers']) or 'payload'})"
                            ),
                            source_step_numbers=[int(getattr(s, "step_number", idx) or idx)],
                            app_context=app or "Headless REST API (Taint-Lowered)",
                            inputs=induced_api,
                            outputs=[f"$v_api_resp_{idx}"],
                            preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                            fallback_tiers=[
                                ExecutionTier.TIER_1_5_BROWSER_DOM_JS,
                                ExecutionTier.TIER_2_SEMANTIC_UI,
                            ],
                            contract=HoareContract(
                                name=f"taint_lowered_api_contract_{idx}",
                                preconditions=["AUTH_SESSION_OR_TOKEN_VALID", "TAINT_REGISTERS_BOUND"],
                                idempotency_strategy="PARAMETERIZED_REST_MUTATION",
                                natural_key_fields=list(induced_api["matched_registers"]),
                                postconditions=["HTTP_STATUS_2XX_AND_RESPONSE_SCHEMA_VALID"],
                            ),
                        )
                    )
                    continue

            # 2. Check if step matches a registered MCP / OpenAPI Tool Schema (Bipartite Schema Matching)
            tool_schemas = s_params.get("available_schemas")
            if isinstance(tool_schemas, list) and tool_schemas:
                schema_match = UniversalAPILoweringResolver.match_mcp_or_openapi_schema(
                    app_name=app,
                    window_title=str(getattr(s, "window_title", "") or ""),
                    element_label=title or str(elem.get("aria_label") or ""),
                    voice_note=note,
                    step_params=s_params,
                    schemas=tool_schemas,
                )
                if schema_match:
                    nodes.append(
                        IRNode(
                            node_id=f"ir_{idx}_schema_lowered_{schema_match['tool_name']}",
                            primitive=IRPrimitiveType.DATABASE_RECORD,
                            description=f"Lowered MCP/OpenAPI call: {schema_match['tool_name']}({schema_match['bound_arguments']})",
                            source_step_numbers=[int(getattr(s, "step_number", idx) or idx)],
                            app_context=f"MCP/OpenAPI Schema ({schema_match['tool_name']})",
                            inputs=schema_match,
                            outputs=[f"$v_tool_{idx}"],
                            preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                            fallback_tiers=[
                                ExecutionTier.TIER_1_5_BROWSER_DOM_JS,
                                ExecutionTier.TIER_2_SEMANTIC_UI,
                            ],
                            contract=HoareContract(
                                name=f"schema_tool_contract_{idx}",
                                preconditions=["MCP_OR_API_SERVER_AVAILABLE", "REQUIRED_SCHEMA_ARGS_PRESENT"],
                                idempotency_strategy="SCHEMA_VALIDATED_TOOL_CALL",
                                natural_key_fields=list(schema_match["bound_arguments"].keys()),
                                postconditions=["TOOL_RESULT_NON_ERROR"],
                            ),
                        )
                    )
                    continue

            # 3. Evaluates window title + Foveal Vision & Episodic OCR against known direct CLI/API/Filesystem routes
            ocr_text = str(s_params.get("screenpipe_ocr") or "")
            win_title = str(getattr(s, "window_title", "") or "")
            direct_route_hint = ""
            try:
                from whole_body_compute.integrations.screenpipe import resolve_direct_headless_route

                route_str = resolve_direct_headless_route(app, win_title, ocr_text)
                if route_str and "DIRECT ROUTE" in route_str:
                    direct_route_hint = route_str
            except Exception:
                pass

            if "Tier 0 Filesystem" in direct_route_hint:
                nodes.append(
                    IRNode(
                        node_id=f"ir_{idx}_direct_fs",
                        primitive=IRPrimitiveType.FILE_TRANSFORM,
                        description=direct_route_hint,
                        source_step_numbers=[int(getattr(s, "step_number", idx) or idx)],
                        app_context="Local Filesystem (Tier 0)",
                        inputs={"direct_route": direct_route_hint, "raw_action": action, "params": dict(s_params)},
                        outputs=[f"$v_fs_{idx}"],
                        preferred_tier=ExecutionTier.TIER_0_LOCAL,
                        fallback_tiers=[ExecutionTier.TIER_2_SEMANTIC_UI, ExecutionTier.TIER_3_COORDINATE_GUI],
                        contract=HoareContract(
                            name=f"direct_fs_contract_{idx}",
                            preconditions=["LOCAL_FILE_PATH_RESOLVABLE"],
                            idempotency_strategy="ATOMIC_FS_OPERATION",
                            natural_key_fields=["path"],
                            postconditions=["FILE_MUTATION_VERIFIED"],
                        ),
                    )
                )
                continue
            elif "Tier 1" in direct_route_hint:
                nodes.append(
                    IRNode(
                        node_id=f"ir_{idx}_direct_api",
                        primitive=IRPrimitiveType.DATABASE_RECORD,
                        description=direct_route_hint,
                        source_step_numbers=[int(getattr(s, "step_number", idx) or idx)],
                        app_context=app or "Direct API/CLI Service",
                        inputs={"direct_route": direct_route_hint, "raw_action": action, "params": dict(s_params)},
                        outputs=[f"$v_api_{idx}"],
                        preferred_tier=ExecutionTier.TIER_1_HEADLESS_API,
                        fallback_tiers=[ExecutionTier.TIER_1_5_BROWSER_DOM_JS, ExecutionTier.TIER_3_COORDINATE_GUI],
                        contract=HoareContract(
                            name=f"direct_api_contract_{idx}",
                            preconditions=["CREDENTIALS_AVAILABLE", "SERVICE_ENDPOINT_REACHABLE"],
                            idempotency_strategy="IDEMPOTENT_API_CALL",
                            natural_key_fields=["endpoint", "idempotency_key"],
                            postconditions=["HTTP_STATUS_2XX_CONFIRMED"],
                        ),
                    )
                )
                continue

            # Determine if this step has WorkArena-style Tier 1.5 Browser DOM / BID / ARIA grounding
            dom_sel = str(elem.get("dom_selector") or "").strip()
            dom_id = str(elem.get("dom_id") or "").strip()
            aria_lbl = str(elem.get("aria_label") or "").strip()
            bid = str(elem.get("bid") or "").strip()
            is_browser_app = any(
                b in app.lower()
                for b in ("chrome", "safari", "arc", "brave", "firefox", "edge")
            )
            has_dom_grounding = bool(
                action in ("click", "double_click", "right_click")
                and (dom_sel or dom_id or bid or (is_browser_app and aria_lbl))
            )

            # Determine if this native app step can be lowered to Tier 1.5 (Browser DOM), Tier 2 (Semantic UI) vs Tier 3 (Coordinates)
            is_semantic = bool(
                action in ("switch_app", "hotkey", "press_key", "type_text")
                or (role in ("AXMenuItem", "AXButton", "AXDockItem") and title)
            )
            if has_dom_grounding:
                tier = ExecutionTier.TIER_1_5_BROWSER_DOM_JS
                fallbacks = [
                    ExecutionTier.TIER_2_SEMANTIC_UI,
                    ExecutionTier.TIER_3_COORDINATE_GUI,
                ]
                node_contract: Optional[HoareContract] = HoareContract(
                    name="workarena_dom_selector_contract",
                    preconditions=["BROWSER_TAB_ACTIVE", "DOM_SELECTOR_PRESENT"],
                    idempotency_strategy="DOM_QUERY_SELECTOR_CLICK",
                    natural_key_fields=["dom_selector", "bid"],
                    postconditions=["DOM_ELEMENT_ACTED_WITHOUT_COORDINATE_DRIFT"],
                )
            elif is_semantic:
                tier = ExecutionTier.TIER_2_SEMANTIC_UI
                fallbacks = [ExecutionTier.TIER_3_COORDINATE_GUI]
                node_contract = None
            else:
                tier = ExecutionTier.TIER_3_COORDINATE_GUI
                fallbacks = []
                node_contract = None

            nodes.append(
                IRNode(
                    node_id=f"ir_{idx}_{action}",
                    primitive=IRPrimitiveType.NATIVE_APP_CONTROL,
                    description=str(getattr(s, "spoken_summary", action) or action),
                    source_step_numbers=[int(getattr(s, "step_number", idx) or idx)],
                    app_context=app,
                    inputs={
                        "raw_action": action,
                        "params": dict(s_params),
                        "voice_note": note,
                    },
                    outputs=[],
                    preferred_tier=tier,
                    fallback_tiers=fallbacks,
                    contract=node_contract,
                )
            )
        return nodes


# ==============================================================================
# TOP-LEVEL COMPILER FACADE
# ==============================================================================


class PalCompiler:
    """
    Top-level entrypoint for compiling recorded Pal demonstrations into deterministic,
    contract-verified PalWorkflowRecipes.
    """

    @classmethod
    def compile_demonstration(
        cls,
        raw_steps: List[Any],
        voice_notes: List[str],
        workflow_name: str,
        unique_triggers: Optional[List[str]] = None,
    ) -> Tuple[PalWorkflowRecipe, IRProgram, CompilationReport]:
        """
        Executes all 4 passes of the Pal Demonstration-to-Deterministic Compiler:
          Pass 1: TraceDenoiser.denoise()
          Pass 2: SemanticDecompiler.decompile() (SSA Data-Flow + PbE Regex + Loop Induction)
          Pass 3: TierLoweringEngine.lower() (Tier 0 -> 1 -> 1.5 -> 2 -> 3)
          Pass 4: HoareContract synthesis & CompilationReport attachment
        """
        ir_program = SemanticDecompiler.decompile(
            raw_steps=raw_steps,
            voice_notes=voice_notes,
            workflow_name=workflow_name,
        )
        recipe, report = TierLoweringEngine.lower(
            ir_program=ir_program,
            unique_triggers=unique_triggers,
        )
        return recipe, ir_program, report

    @classmethod
    def export_skill_bundle(
        cls,
        recipe_or_path: Any,
        output_root: Optional[Any] = None,
        zip_bundle: bool = True,
        dry_run: bool = False,
        license_manager: Optional[Any] = None,
        ledger_path: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Export a compiled `PalWorkflowRecipe` (or YAML path) into a portable Skill Bundle
        (`SKILL.md` + `recipe.yaml` + `scripts/run.py` + `<slug>.skill.zip`).
        """
        from whole_body_compute.integrations.pal_skill_exporter import (
            export_recipe_as_skill_bundle,
        )

        return export_recipe_as_skill_bundle(
            recipe_or_path=recipe_or_path,
            output_root=output_root,
            zip_bundle=zip_bundle,
            dry_run=dry_run,
            license_manager=license_manager,
            ledger_path=ledger_path,
        )
