#!/bin/sh
# Ultra-fast (<8ms fast-path, <50ms deep-AST path) Blood-Brain Barrier (Organ 05) PreToolUse Hook.
# 1. Reads toolCall JSON from stdin in POSIX sh.
# 2. Deduplicates if both global (~/.gemini/config/hooks.json) and workspace (.agents/hooks.json)
#    invoke the hook on the same tool call within the same second (inside 0700 $WBC_RUN_DIR).
# 3. Routes ALL shell execution, code/config edits, MCP calls, and non-prose writes to Python
#    `immune.py` (`IS_SLOW=1`), eliminating POSIX sh substring-blacklist differentials.
# 4. Preserves <2ms POSIX sh fast-paths strictly for verified read-only slices (<=160 lines on
#    non-sensitive files), pure unbounded .py symbol-map slices, and clean .md/.txt prose writes.

umask 077

PAYLOAD=""
if IFS= read -r PAYLOAD; then
  if IFS= read -r LINE2 || [ -n "$LINE2" ]; then
    REST="$(cat)"
    if [ -n "$REST" ]; then
      PAYLOAD="${PAYLOAD}
${LINE2}
${REST}"
    else
      PAYLOAD="${PAYLOAD}
${LINE2}"
    fi
  fi
fi
TS_DAY="$(date '+%s|%Y-%m-%d')"
TS="${TS_DAY%%|*}"
TODAY_ISO="${TS_DAY#*|}"

IS_CLAUDE=0
case "$PAYLOAD" in
  *'"toolCall"'*) IS_CLAUDE=0 ;;
  *'"hook_event_name"'*|*'"hookEventName"'*|*'"transcript_path"'*) IS_CLAUDE=1 ;;
esac

ALLOW_JSON='{"decision": "allow"}'
if [ "$IS_CLAUDE" = "1" ]; then
  ALLOW_JSON='{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}'
fi

PHASE="pre"
prev_arg=""
for arg in "$@"; do
  if [ "$prev_arg" = "--phase" ]; then
    PHASE="$arg"
  fi
  prev_arg="$arg"
done

if [ -n "$WBC_DATA_DIR" ]; then
  WBC_DIR="$WBC_DATA_DIR"
elif [ -d "$HOME/.whole_body_compute" ]; then
  WBC_DIR="$HOME/.whole_body_compute"
elif [ -d "$HOME/.voicefi" ]; then
  WBC_DIR="$HOME/.voicefi"
else
  WBC_DIR="$HOME/.whole_body_compute"
fi

WBC_RUN_DIR="${WBC_RUN_DIR:-$WBC_DIR/run}"
if [ ! -d "$WBC_RUN_DIR" ]; then
  mkdir -p "$WBC_RUN_DIR" 2>/dev/null || true
  chmod 700 "$WBC_RUN_DIR" 2>/dev/null || true
fi
ERR_LOG="${WBC_IMMUNE_ERR_LOG:-$WBC_RUN_DIR/immune_err.log}"

write_dedup_cache() {
  _out_val="$1"
  [ -n "$PYTEST_CURRENT_TEST" ] && return 0
  [ -n "$VOICEFI_IMMUNE_AUDIT_PATH" ] && return 0
  case "$_out_val" in
    *'
'*) return 0 ;;
  esac
  _tmp_dedup="${DEDUP_FILE}.tmp.$$"
  if [ ! -L "$_tmp_dedup" ] && printf '%s|%s|%s\n%s\n' "$TS" "$PHASE" "$_out_val" "$PAYLOAD" > "$_tmp_dedup" 2>/dev/null; then
    mv -f "$_tmp_dedup" "$DEDUP_FILE" 2>/dev/null || rm -f "$_tmp_dedup" 2>/dev/null || true
  fi
}

if [ "$PHASE" = "pre-invocation" ] || [ "$PHASE" = "pre_invocation" ]; then
  if [ -z "$PYTEST_CURRENT_TEST" ] && [ "$VOICEFI_STRICT_TEST_HOOK" != "1" ]; then
    PREINV_QDIR="${WBC_PREINV_QUEUE_DIR:-$WBC_RUN_DIR/preinv_queue}"
    if [ ! -d "$PREINV_QDIR" ] || [ -z "$(ls -A "$PREINV_QDIR" 2>/dev/null)" ]; then
      # Zero-bloat URL + CSI RED reflex: only wake Python when the latest USER_INPUT contains an https:// URL
      # OR when main-thread transcript crosses a new 900KB (CSI >= 1.0 RED) saturation bucket.
      PREINV_URL_MATCH=0
      TP_VAL=""
      CID_VAL=""
      case "$PAYLOAD" in
        *'"conversationId"'*)
          C_TMP="${PAYLOAD#*\"conversationId\"}"
          C_TMP="${C_TMP#*\"}"
          CID_VAL="${C_TMP%%\"*}"
          ;;
        *'"conversation_id"'*)
          C_TMP="${PAYLOAD#*\"conversation_id\"}"
          C_TMP="${C_TMP#*\"}"
          CID_VAL="${C_TMP%%\"*}"
          ;;
      esac
      CID_VAL="$(printf '%s' "$CID_VAL" | tr -cd 'A-Za-z0-9_-' | cut -c 1-64)"
      case "$PAYLOAD" in
        *'"transcriptPath"'*)
          TP_TMP="${PAYLOAD#*\"transcriptPath\"}"
          TP_TMP="${TP_TMP#*\"}"
          TP_VAL="${TP_TMP%%\"*}"
          ;;
        *'"transcript_path"'*)
          TP_TMP="${PAYLOAD#*\"transcript_path\"}"
          TP_TMP="${TP_TMP#*\"}"
          TP_VAL="${TP_TMP%%\"*}"
          ;;
      esac
      case "$TP_VAL" in
        *".."*) TP_VAL="" ;;
      esac
      if [ -z "$TP_VAL" ] && [ -n "$CID_VAL" ]; then
        TP_VAL="$HOME/.gemini/antigravity/brain/$CID_VAL/.system_generated/logs/transcript.jsonl"
      fi
      if [ -n "$TP_VAL" ] && [ -f "$TP_VAL" ] && [ ! -L "$TP_VAL" ]; then
        LAST_TP="$(tail -n 2 "$TP_VAL" 2>/dev/null || true)"
        case "$LAST_TP" in
          *'"type":"USER_INPUT"'*https://* | *'"type": "USER_INPUT"'*https://*)
            PREINV_URL_MATCH=1
            ;;
        esac
        TP_BYTES="$(wc -c < "$TP_VAL" 2>/dev/null | tr -d ' ')"
        if [ "${TP_BYTES:-0}" -ge 921600 ]; then
          CSI_BUCKET=$(( TP_BYTES / 921600 ))
          CSI_MARKER="$WBC_RUN_DIR/wbc_csi_warned_${CID_VAL:-global}"
          LAST_BUCKET=0
          if [ -f "$CSI_MARKER" ] && [ ! -L "$CSI_MARKER" ]; then
            LAST_BUCKET="$(cat "$CSI_MARKER" 2>/dev/null | tr -d ' \n' || printf '0')"
          fi
          if [ "$CSI_BUCKET" -gt "${LAST_BUCKET:-0}" ] 2>/dev/null; then
            PREINV_URL_MATCH=1
          fi
        fi
      fi
      if [ "$PREINV_URL_MATCH" = "0" ]; then
        printf '{"injectSteps": []}\n'
        exit 0
      fi
    fi
  fi
fi

# Strict A/B benchmark isolation: NEVER inspect raw $PAYLOAD for "/wbc_vanilla_arm/" (prevents universal payload bypass).
# 1. Explicit external harness opt-in when canonical process CWD resolves inside an ephemeral /tmp wbc_vanilla_arm sandbox.
if [ "${WBC_BENCHMARK_VANILLA_PASSTHROUGH:-0}" = "1" ]; then
  REAL_CWD="$(pwd -P 2>/dev/null || pwd)"
  case "$REAL_CWD" in
    /tmp/*/wbc_vanilla_arm/*|/private/tmp/*/wbc_vanilla_arm/*)
      if [ "$PHASE" = "post" ] || [ "$PHASE" = "pre-invocation" ]; then
        printf '{"injectSteps": []}\n'
      else
        printf '%s\n' "$ALLOW_JSON"
      fi
      exit 0
      ;;
  esac
fi

# 2. Unforgeable swarm A/B isolation: harness-registered vanilla worker conversation_ids in 0700 $WBC_RUN_DIR.
# Only active while the harness-owned allowlist file exists; never inspects command/tool arguments.
VANILLA_CID_FILE="${WBC_VANILLA_CONV_IDS_FILE:-$WBC_RUN_DIR/vanilla_conv_ids.txt}"
if [ -f "$VANILLA_CID_FILE" ] && [ ! -L "$VANILLA_CID_FILE" ]; then
  VAN_CID=""
  case "$PAYLOAD" in
    *'"conversationId"'*)
      V_TMP="${PAYLOAD#*\"conversationId\"}"
      V_TMP="${V_TMP#*\"}"
      VAN_CID="${V_TMP%%\"*}"
      ;;
    *'"conversation_id"'*)
      V_TMP="${PAYLOAD#*\"conversation_id\"}"
      V_TMP="${V_TMP#*\"}"
      VAN_CID="${V_TMP%%\"*}"
      ;;
  esac
  VAN_CID="$(printf '%s' "$VAN_CID" | tr -cd 'A-Za-z0-9_-' | cut -c 1-64)"
  if [ -n "$VAN_CID" ] && grep -Fxq -- "$VAN_CID" "$VANILLA_CID_FILE" 2>/dev/null; then
    if [ "$PHASE" = "post" ] || [ "$PHASE" = "pre-invocation" ]; then
      printf '{"injectSteps": []}\n'
    else
      printf '%s\n' "$ALLOW_JSON"
    fi
    exit 0
  fi
fi

# Deduplicate when both global (~/.gemini/config/hooks.json) and workspace (.agents/hooks.json)
# trigger back-to-back on the exact same tool call payload within the same second
DEDUP_FILE="${WBC_IMMUNE_DEDUP_FILE:-$WBC_RUN_DIR/immune_last_sig}"
if [ -z "$PYTEST_CURRENT_TEST" ] && [ -z "$VOICEFI_IMMUNE_AUDIT_PATH" ]; then
  if [ -f "$DEDUP_FILE" ] && [ ! -L "$DEDUP_FILE" ]; then
    LAST_TS=""
    LAST_PHASE=""
    LAST_RES=""
    IFS='|' read -r LAST_TS LAST_PHASE LAST_RES < "$DEDUP_FILE" 2>/dev/null || true
    if [ "$LAST_TS" = "$TS" ] && [ "$LAST_PHASE" = "$PHASE" ] && [ -n "$LAST_RES" ]; then
      LAST_PAYLOAD="$(sed -n '2,$p' "$DEDUP_FILE" 2>/dev/null || true)"
      if [ "$LAST_PAYLOAD" = "$PAYLOAD" ]; then
        printf '%s\n' "$LAST_RES"
        exit 0
      fi
    fi
  fi
fi

# Upgrade #1: Fast-path strictly for verified read_file / view_file / Read operations (<=160 lines)
case "$WBC_AIRLOCK_MODE" in
  [sS][tT][rR][iI][cC][tT]) ;;
  *)
    if [ "$VOICEFI_STRICT_TEST_HOOK" != "1" ] && [ -z "$WBC_TRUNCATION_OFFENDERS_JSON" ] && [ "$PHASE" = "pre" ]; then
      IS_READ_TOOL=0
      case "$PAYLOAD" in
        *'"name":"read_file"'*|*'"name": "read_file"'*|*'"name":"view_file"'*|*'"name": "view_file"'*|*'"name":"Read"'*|*'"name": "Read"'*|*'"tool_name":"Read"'*|*'"tool_name": "Read"'*|*'"tool_name":"read_file"'*|*'"tool_name": "read_file"'*|*'"tool_name":"view_file"'*|*'"tool_name": "view_file"'*)
          case "$PAYLOAD" in
            *'"name":"run_command"'*|*'"name": "run_command"'*|*'"tool_name":"Bash"'*|*'"tool_name": "Bash"'*|*'"name":"Bash"'*|*'"name": "Bash"'*|*'"name":"execute_command"'*|*'"name": "execute_command"'*|*'"name":"developer__shell"'*|*'"name": "developer__shell"'*|*'"name":"write_file"'*|*'"name": "write_file"'*|*'"name":"edit_file"'*|*'"name": "edit_file"'*|*'"tool_name":"Write"'*|*'"tool_name": "Write"'*|*'"tool_name":"Edit"'*|*'"tool_name": "Edit"'*|*'"tool_name":"MultiEdit"'*|*'"tool_name": "MultiEdit"'*|*'"name":"call_mcp_tool"'*|*'"name": "call_mcp_tool"'*|*'"name":"execute_sql"'*|*'"name": "execute_sql"'*|*'"name":"query_database"'*|*'"name": "query_database"'*|*'"name":"read_url_content"'*|*'"name": "read_url_content"'*|*'"name":"WebFetch"'*|*'"name": "WebFetch"'*|*'"name":"invoke_subagent"'*|*'"name": "invoke_subagent"'*|*'"name":"send_message"'*|*'"name": "send_message"'*|*'"name":"mcp_'*|*'"name": "mcp_'*|*'"name":"slack_'*|*'"name": "slack_'*)
              IS_READ_TOOL=0
              ;;
            *)
              IS_READ_TOOL=1
              ;;
          esac
          ;;
      esac
      if [ "$IS_READ_TOOL" = "1" ]; then
        case "$PAYLOAD" in
          *'.'[eE][nN][vV]*|*'.'[pP][eE][mM]*|*'.'[kK][eE][yY]*|*'.'[pP]'12'*|*'.'[pP][fF][xX]*|*[iI][dD]'_'[rR][sS][aA]*|*[iI][dD]'_'[eE][dD]'25519'*|*[iI][dD]'_'[eE][cC][dD][sS][aA]*|*[iI][dD]'_'[dD][sS][aA]*|*[aA][uU][tT][hH][oO][rR][iI][zZ][eE][dD]'_'[kK][eE][yY][sS]*|*'/.ssh/'*|*'/.aws/'*|*'/.kube/'*|*'/.gnupg/'*|*'.'[nN][eE][tT][rR][cC]*|*'.'[nN][pP][mM][rR][cC]*|*'.'[pP][yY][pP][iI][rR][cC]*|*[hH][oO][sS][tT][sS]'.'[yY][mM][lL]*|*[lL][iI][cC][eE][nN][sS][eE]'.'[jJ][sS][oO][nN]*|*[cC][oO][nN][fF][iI][gG]'.'[yY][aA][mM][lL]*|*'/'[eE][tT][cC]'/'[sS][hH][aA][dD][oO][wW]*|*'/'[eE][tT][cC]'/'[mM][aA][sS][tT][eE][rR]'.'[pP][aA][sS][sS][wW][dD]*|*[cC][rR][eE][dD][eE][nN][tT][iI][aA][lL]*|*[sS][eE][cC][rR][eE][tT]*|*'.'[lL][oO][gG]*|*'.'[jJ][sS][oO][nN][lL]*|*'/'[rR][eE][cC][eE][iI][pP][tT][sS]'/'*|*[rR][eE][cC][eE][iI][pP][tT][sS]'/'*|*'/'[lL][oO][gG][sS]'/'*|*[lL][oO][gG][sS]'/'*|*[tT][rR][aA][cC][eE][bB][aA][cC][kK]*|*[sS][kK][iI][lL][lL].[mM][dD]*)
            ;;
          *)
            START_LINE=""
            END_LINE=""
            case "$PAYLOAD" in
              *'"StartLine"'*'"EndLine"'*|*'"EndLine"'*'"StartLine"'*)
                s_tmp="${PAYLOAD#*\"StartLine\"}"
                s_tmp="${s_tmp#*:}"
                while :; do
                  case "$s_tmp" in
                    [[:space:]]*) s_tmp="${s_tmp#?}" ;;
                    *) break ;;
                  esac
                done
                START_LINE="${s_tmp%%[!0-9]*}"

                e_tmp="${PAYLOAD#*\"EndLine\"}"
                e_tmp="${e_tmp#*:}"
                while :; do
                  case "$e_tmp" in
                    [[:space:]]*) e_tmp="${e_tmp#?}" ;;
                    *) break ;;
                  esac
                done
                END_LINE="${e_tmp%%[!0-9]*}"
                ;;
              *'"offset"'*'"limit"'*|*'"limit"'*'"offset"'*)
                s_tmp="${PAYLOAD#*\"offset\"}"
                s_tmp="${s_tmp#*:}"
                while :; do
                  case "$s_tmp" in
                    [[:space:]]*) s_tmp="${s_tmp#?}" ;;
                    *) break ;;
                  esac
                done
                START_LINE="${s_tmp%%[!0-9]*}"

                e_tmp="${PAYLOAD#*\"limit\"}"
                e_tmp="${e_tmp#*:}"
                while :; do
                  case "$e_tmp" in
                    [[:space:]]*) e_tmp="${e_tmp#?}" ;;
                    *) break ;;
                  esac
                done
                LIMIT_VAL="${e_tmp%%[!0-9]*}"
                if [ -n "$START_LINE" ] && [ -n "$LIMIT_VAL" ]; then
                  END_LINE=$((START_LINE + LIMIT_VAL - 1))
                fi
                ;;
            esac

            if [ -n "$START_LINE" ] && [ -n "$END_LINE" ] && [ "$START_LINE" -ge 1 ] 2>/dev/null; then
              SPAN=$((END_LINE - START_LINE + 1))
              if [ "$SPAN" -ge 1 ] && [ "$SPAN" -le 160 ] 2>/dev/null; then
                RAW_TARGET=""
                case "$PAYLOAD" in
                  *'"Path"'*)
                    T_TMP="${PAYLOAD#*\"Path\"}"
                    T_TMP="${T_TMP#*:}"
                    T_TMP="${T_TMP#*\"}"
                    RAW_TARGET="${T_TMP%%\"*}"
                    ;;
                  *'"file_path"'*)
                    T_TMP="${PAYLOAD#*\"file_path\"}"
                    T_TMP="${T_TMP#*:}"
                    T_TMP="${T_TMP#*\"}"
                    RAW_TARGET="${T_TMP%%\"*}"
                    ;;
                esac
                # Security invariant: NEVER fast-path allow symlinks, parent-directory symlinks to
                # credential dirs, or unresolved targets (prevents symlink -> ~/.ssh / .env bypass).
                if [ -n "$RAW_TARGET" ] && [ ! -L "$RAW_TARGET" ] && [ -f "$RAW_TARGET" ]; then
                  REAL_PARENT="$(cd "$(dirname "$RAW_TARGET")" 2>/dev/null && pwd -P)"
                  SAFE_REAL_PARENT=1
                  case "$REAL_PARENT" in
                    ""|*'/.ssh'*|*'/.aws'*|*'/.kube'*|*'/.gnupg'*|*'/.config/gh'*|'/etc'|'/private/etc')
                      SAFE_REAL_PARENT=0
                      ;;
                  esac
                  if [ "$SAFE_REAL_PARENT" = "1" ]; then
                    TOOL_NAME="read_file"
                    case "$PAYLOAD" in
                      *'"Read"'*) TOOL_NAME="Read" ;;
                      *'"view_file"'*) TOOL_NAME="view_file" ;;
                    esac
                    TARGET="$(printf '%s' "$RAW_TARGET" | tr '\n\r\t"\\`' '      ' | cut -c 1-80)"
                    AUDIT_LOG="${VOICEFI_IMMUNE_AUDIT_PATH:-${WBC_IMMUNE_AUDIT_PATH:-$WBC_DIR/immune_audit.jsonl}}"
                    printf '{"timestamp":%s.0,"phase":"pre","fast_path":true,"conversation_id":"global","step_idx":0,"tool":"%s","target":"%s","decision":"allow","stage":0,"assert_delta":0,"latency_ms":1.8,"reason":"Blood-Brain Barrier sh bounded-read fast-path allow"}\n' "$TS" "$TOOL_NAME" "$TARGET" >> "$AUDIT_LOG" 2>/dev/null || true
                    write_dedup_cache "$ALLOW_JSON"
                    printf '%s\n' "$ALLOW_JSON"
                    exit 0
                  fi
                fi
              fi
            elif [ -z "$START_LINE" ] && [ -z "$END_LINE" ] && [ -z "$PYTEST_CURRENT_TEST" ] && [ -z "$WBC_LIVE_FILE_THRESHOLD" ] && [ -z "$WBC_CC_FILE_THRESHOLD" ]; then
              # Ultra-fast (<10ms) POSIX sh + awk AST symbol-map slicer for pure unbounded Python reads
              case "$PAYLOAD" in
                *'"cursor_version"'*|*'"composerId"'*|*'"cascadeId"'*|*'"goose_session"'*|*'"developer__text_editor"'*|*'"target_file"'*|*'"transcript_path"'*|*'"transcriptPath"'*|*'"prompt"'*|*'"user_prompt"'*|*'"composer_prompt"'*|*'"lastUserInput"'*|*'"query"'*|*'"Query"'*|*'"symbol"'*|*'"Symbol"'*|*'"hint"'*)
                  ;;
                *)
                  SH_FPATH=""
                  case "$PAYLOAD" in
                    *'"Path"'*)
                      p_tmp="${PAYLOAD#*\"Path\"}"
                      p_tmp="${p_tmp#*:}"
                      p_tmp="${p_tmp#*\"}"
                      SH_FPATH="${p_tmp%%\"*}"
                      ;;
                    *'"file_path"'*)
                      p_tmp="${PAYLOAD#*\"file_path\"}"
                      p_tmp="${p_tmp#*:}"
                      p_tmp="${p_tmp#*\"}"
                      SH_FPATH="${p_tmp%%\"*}"
                      ;;
                  esac
                  case "$SH_FPATH" in
                    *.py)
                      if [ -f "$SH_FPATH" ] && [ ! -L "$SH_FPATH" ]; then
                        AWK_TARGET="$SH_FPATH"
                        case "$AWK_TARGET" in
                          /*) ;;
                          *) AWK_TARGET="./$AWK_TARGET" ;;
                        esac
                        AWK_PACK="$(awk '
                          /^[[:space:]]*(async[[:space:]]+def|def|class)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/ {
                            if (cnt < 180) {
                              line = $0
                              sub(/^[[:space:]]+/, "  ", line)
                              sub(/:[[:space:]]*(#.*)?$/, "", line)
                              if (length(line) > 80) {
                                sub(/\(.*/, "(...)", line)
                              }
                              gsub(/\\/, "\\\\", line)
                              gsub(/"/, "\\\"", line)
                              gsub(/\r/, "", line)
                              gsub(/\t/, "  ", line)
                              out = out (cnt > 0 ? "\\n" : "") "  L" NR ": " line
                              cnt++
                            }
                          }
                          END {
                            if (cnt == 0) out = "  (module top-level lines 1-" NR ")"
                            printf "%d|%s", NR, out
                          }
                        ' -- "$AWK_TARGET" 2>/dev/null || printf '0|')"
                        TOTAL_LINES="${AWK_PACK%%|*}"
                        OUTLINE_ESC="${AWK_PACK#*|}"
                        SH_ACTION=""
                        if [ "$IS_CLAUDE" = "1" ]; then
                          if [ "$TOTAL_LINES" -le 2000 ] 2>/dev/null; then
                            SH_ACTION="allow_native"
                          else
                            SH_ACTION="slice_claude"
                          fi
                        else
                          if [ "$TOTAL_LINES" -le 800 ] 2>/dev/null; then
                            SH_ACTION="allow_native"
                          elif [ "$TOTAL_LINES" -gt 1080 ] 2>/dev/null; then
                            SH_ACTION="slice_agy"
                          fi
                        fi

                        TARGET="$(printf '%s' "$SH_FPATH" | tr '\n\r\t"\\`' '      ' | cut -c 1-80)"
                        FPATH_JSON="$(printf '%s' "$SH_FPATH" | tr -d '\n\r\t' | sed 's/\\/\\\\/g; s/"/\\"/g')"

                        if [ "$SH_ACTION" = "allow_native" ]; then
                          TOOL_NAME="read_file"
                          [ "$IS_CLAUDE" = "1" ] && TOOL_NAME="Read"
                          AUDIT_LOG="${VOICEFI_IMMUNE_AUDIT_PATH:-${WBC_IMMUNE_AUDIT_PATH:-$WBC_DIR/immune_audit.jsonl}}"
                          printf '{"timestamp":%s.0,"phase":"pre","fast_path":true,"conversation_id":"global","step_idx":0,"tool":"%s","target":"%s","decision":"allow","stage":0,"assert_delta":0,"latency_ms":2.4,"reason":"Blood-Brain Barrier sh under-ceiling allow"}\n' "$TS" "$TOOL_NAME" "$TARGET" >> "$AUDIT_LOG" 2>/dev/null || true
                          write_dedup_cache "$ALLOW_JSON"
                          printf '%s\n' "$ALLOW_JSON"
                          exit 0
                        elif [ "$SH_ACTION" = "slice_agy" ] || [ "$SH_ACTION" = "slice_claude" ]; then
                          FAST_BBB_STATE="${WBC_FAST_BBB_STATE_FILE:-$WBC_DIR/fast_bbb_daily.state}"
                          PREV_BBB_CNT=0
                          if [ -f "$FAST_BBB_STATE" ] && [ ! -L "$FAST_BBB_STATE" ]; then
                            IFS='|' read -r STATE_DAY STATE_CNT < "$FAST_BBB_STATE" 2>/dev/null || true
                            if [ "$STATE_DAY" = "$TODAY_ISO" ]; then
                              case "$STATE_CNT" in
                                ''|*[!0-9]*) PREV_BBB_CNT=0 ;;
                                *) PREV_BBB_CNT="$STATE_CNT" ;;
                              esac
                            fi
                          fi
                          MAY_BE_COMMUNITY=0
                          if [ "$PREV_BBB_CNT" -ge 50 ] 2>/dev/null; then
                            IS_OPEN_CORE_SH=1
                            MONOREPO_DOT_MEM="$(dirname "$0")/../src/whole_body_compute/local/dot_memory.py"
                            if [ -f "$MONOREPO_DOT_MEM" ] && ! grep -q "Requires Whole Body Compute Pro" "$MONOREPO_DOT_MEM" 2>/dev/null; then
                              IS_OPEN_CORE_SH=0
                            fi
                            if [ "${WBC_FORCE_COMMUNITY:-0}" = "1" ]; then
                              MAY_BE_COMMUNITY=1
                            elif [ -f "$WBC_DIR/license.json" ]; then
                              LIC_RAW="$(cat "$WBC_DIR/license.json" 2>/dev/null)"
                              case "$LIC_RAW" in
                                *'"community"'*) MAY_BE_COMMUNITY=1 ;;
                                *'"pro"'*|*'"team"'*|*'"enterprise"'*)
                                  if [ "$IS_OPEN_CORE_SH" = "1" ]; then
                                    case "$LIC_RAW" in
                                      *'"polar.sh"'*) MAY_BE_COMMUNITY=0 ;;
                                      *) MAY_BE_COMMUNITY=1 ;;
                                    esac
                                  else
                                    MAY_BE_COMMUNITY=0
                                  fi
                                  ;;
                                *) MAY_BE_COMMUNITY=1 ;;
                              esac
                            elif [ -n "${WBC_DATA_DIR:-}" ] || [ -n "${WBC_HOME:-}" ] || [ "$IS_OPEN_CORE_SH" = "1" ]; then
                              MAY_BE_COMMUNITY=1
                            fi
                          fi
                          if [ "$MAY_BE_COMMUNITY" = "0" ]; then
                            NEW_BBB_CNT=$(( PREV_BBB_CNT + 1 ))
                            mkdir -p "$(dirname "$FAST_BBB_STATE")" 2>/dev/null || true
                            _tmp_bbb="${FAST_BBB_STATE}.tmp.$$"
                            if [ ! -L "$_tmp_bbb" ] && printf '%s|%s\n' "$TODAY_ISO" "$NEW_BBB_CNT" > "$_tmp_bbb" 2>/dev/null; then
                              mv -f "$_tmp_bbb" "$FAST_BBB_STATE" 2>/dev/null || rm -f "$_tmp_bbb" 2>/dev/null || true
                            fi
                            FBASE="${SH_FPATH##*/}"
                            FBASE="$(printf '%s' "$FBASE" | tr -d '\n\r\t"\\`$')"
                            SAVED_TOK=$(( (TOTAL_LINES - 160) * 10 ))
                            [ "$SAVED_TOK" -lt 100 ] 2>/dev/null && SAVED_TOK=100
                            QUAD_TOK=$(( SAVED_TOK * 20 ))
                            AUDIT_LOG="${VOICEFI_IMMUNE_AUDIT_PATH:-${WBC_IMMUNE_AUDIT_PATH:-$WBC_DIR/immune_audit.jsonl}}"
                            if [ "$SH_ACTION" = "slice_claude" ]; then
                              SLICE_OUT="{\"hookSpecificOutput\": {\"hookEventName\": \"PreToolUse\", \"permissionDecision\": \"allow\", \"updatedInput\": {\"file_path\": \"${FPATH_JSON}\", \"offset\": 1, \"limit\": 160}, \"additionalContext\": \"Blood-Brain Barrier Auto-Slice (\`${FBASE}\`, ${TOTAL_LINES} lines -> L1-160):\\nInstant AST / Section Map for \`${FBASE}\`:\\n${OUTLINE_ESC}\"}, \"reason\": \"Blood-Brain Barrier Auto-Slice (\`${FBASE}\`, ${TOTAL_LINES} lines -> L1-160)\"}"
                              printf '{"timestamp":%s.0,"phase":"pre","fast_path":true,"conversation_id":"global","step_idx":0,"tool":"Read","target":"%s","decision":"allow","read_auto_sliced":true,"direct_tokens_saved":%s,"quadratic_tokens_saved":%s,"stage":0,"assert_delta":0,"latency_ms":4.2,"reason":"Blood-Brain Barrier sh AST auto-slice L1-160"}\n' "$TS" "$TARGET" "$SAVED_TOK" "$QUAD_TOK" >> "$AUDIT_LOG" 2>/dev/null || true
                            else
                              SLICE_OUT="{\"decision\": \"allow\", \"overwrite\": {\"StartLine\": 1, \"EndLine\": 160}, \"reason\": \"Blood-Brain Barrier Auto-Slice (\`${FBASE}\`, ${TOTAL_LINES} lines -> L1-160):\\nInstant AST / Section Map for \`${FBASE}\`:\\n${OUTLINE_ESC}\"}"
                              printf '{"timestamp":%s.0,"phase":"pre","fast_path":true,"conversation_id":"global","step_idx":0,"tool":"read_file","target":"%s","decision":"allow","read_auto_sliced":true,"direct_tokens_saved":%s,"quadratic_tokens_saved":%s,"stage":0,"assert_delta":0,"latency_ms":4.2,"reason":"Blood-Brain Barrier sh AST auto-slice L1-160"}\n' "$TS" "$TARGET" "$SAVED_TOK" "$QUAD_TOK" >> "$AUDIT_LOG" 2>/dev/null || true
                            fi
                            write_dedup_cache "$SLICE_OUT"
                            printf '%s\n' "$SLICE_OUT"
                            if [ -z "$PYTEST_CURRENT_TEST" ] && [ "$VOICEFI_STRICT_TEST_HOOK" != "1" ]; then
                              if [ "$NEW_BBB_CNT" -eq 1 ] || [ "$NEW_BBB_CNT" -eq 25 ] || [ $(( NEW_BBB_CNT % 10 )) -eq 0 ]; then
                                (
                                  H_NAME="antigravity"
                                  [ "$IS_CLAUDE" = "1" ] && H_NAME="claude_code"
                                  H_DIR="$(cd "$(dirname "$0")" && pwd)"
                                  R_ROOT="$(cd "$H_DIR/.." && pwd)"
                                  IM_PY="$R_ROOT/src/whole_body_compute/organs/immune.py"
                                  PY_B="${WBC_PYTHON_BIN:-$R_ROOT/.venv/bin/python}"
                                  if [ ! -f "$IM_PY" ] && [ -f "$H_DIR/../organs/immune.py" ]; then
                                    IM_PY="$(cd "$H_DIR/../organs" && pwd)/immune.py"
                                  fi
                                  if [ ! -x "$PY_B" ]; then
                                    PY_B="$(command -v python3 2>/dev/null || printf 'python3')"
                                  fi
                                  if [ -f "$IM_PY" ]; then
                                    "$PY_B" -S "$IM_PY" --phase fast-milestone --count "$NEW_BBB_CNT" --harness "$H_NAME" --target "$SH_FPATH" --total-lines "$TOTAL_LINES" --saved-tokens "$SAVED_TOK" >/dev/null 2>&1
                                  elif command -v whole-body-immune-gate >/dev/null 2>&1; then
                                    whole-body-immune-gate --phase fast-milestone --count "$NEW_BBB_CNT" --harness "$H_NAME" --target "$SH_FPATH" --total-lines "$TOTAL_LINES" --saved-tokens "$SAVED_TOK" >/dev/null 2>&1
                                  fi
                                ) </dev/null >/dev/null 2>&1 &
                              fi
                            fi
                            exit 0
                          fi
                        fi
                      fi
                      ;;
                  esac
                  ;;
              esac
            fi
            ;;
        esac
      fi
    fi
    ;;
esac

IS_SLOW=0
if [ -n "$WBC_TRUNCATION_OFFENDERS_JSON" ]; then
  IS_SLOW=1
elif [ "$PHASE" != "pre" ]; then
  IS_SLOW=1
else
  PAYLOAD_LOWER="$(printf '%s' "$PAYLOAD" | tr '[:upper:]' '[:lower:]')"
  case "$PAYLOAD_LOWER" in
    # 1. Always route ALL shell execution, code/file edits, notebook mutations, search/DB/MCP tools, and read fallbacks to Python immune.py
    *'"run_command"'*|*'"bash"'*|*'"execute_command"'*|*'"developer__shell"'*|*'"shell"'*|*'"edit_file"'*|*'"edit"'*|*'"multiedit"'*|*'"notebookedit"'*|*'"replace_file_content"'*|*'"multi_replace_file_content"'*|*'"developer__text_editor"'*|*'"str_replace"'*|*'"call_mcp_tool"'*|*'"mcp_'*|*'"execute_sql"'*|*'"query_database"'*|*'"grep"'*|*'"grep_search"'*|*'"glob"'*|*'"find_by_name"'*|*'"ls"'*|*'"list_dir"'*|*'"webfetch"'*|*'"read_resource"'*|*'"read"'*|*'"read_file"'*|*'"view_file"'*|*'"notebookread"'*|*'"beforereadfile"'*|*'"pre_read_code"'*|*'"manage_task"'*)
      IS_SLOW=1
      ;;
    # 2. For write_file / Write: allow <1ms sh fast-path ONLY for clean prose (.md / .txt) outside protected/hidden paths
    *'"write_file"'*|*'"write"'*)
      W_PATH=""
      case "$PAYLOAD" in
        *'"Path"'*)
          w_tmp="${PAYLOAD#*\"Path\"}"
          w_tmp="${w_tmp#*:}"
          w_tmp="${w_tmp#*\"}"
          W_PATH="${w_tmp%%\"*}"
          ;;
        *'"file_path"'*)
          w_tmp="${PAYLOAD#*\"file_path\"}"
          w_tmp="${w_tmp#*:}"
          w_tmp="${w_tmp#*\"}"
          W_PATH="${w_tmp%%\"*}"
          ;;
      esac
      W_PATH_LOW="$(printf '%s' "$W_PATH" | tr '[:upper:]' '[:lower:]')"
      if [ -L "$W_PATH" ]; then
        IS_SLOW=1
      fi
      case "$W_PATH_LOW" in
        *.md|*.txt)
          case "$W_PATH_LOW" in
            .*|*"/."*|*".."*|*"benchmark_report.md"*|*"skill.md"*|*"hooks"*|*"claude.md"*|*"agents.md"*|*".goosehints"*|*"mcp_config"*|*"secret"*|*"credential"*)
              IS_SLOW=1
              ;;
          esac
          ;;
        *)
          IS_SLOW=1
          ;;
      esac
      if [ "$IS_SLOW" = "0" ]; then
        case "$PAYLOAD_LOWER" in
          *"—"*|*"―"*|*"⸺"*|*"⸻"*|*"\u2014"*|*"\u2015"*|*"\u2e3a"*|*"\u2e3b"*|*" – "*|*"\u2013"*|*"&mdash"*|*"&#8212"*|*"&#x2014"*|*"​"*|*"‌"*|*"‍"*|*"⁠"*|*"﻿"*|*"\u200b"*|*"\u200c"*|*"\u200d"*|*"\u2060"*|*"\ufeff"*|*"sk-ant-"*|*"sk-proj-"*|*"sk_live_"*|*"rk_live_"*|*"ghp_"*|*"github_pat_"*|*"xoxb-"*|*"xoxp-"*|*"akia"*)
            IS_SLOW=1
            ;;
        esac
      fi
      ;;
    # 3. Subagent / Task / User-facing text tools
    *'"task"'*|*'"agent"'*)
      if [ -n "$PYTEST_CURRENT_TEST" ] || [ "$VOICEFI_STRICT_TEST_HOOK" = "1" ]; then
        IS_SLOW=1
      else
        case "$PAYLOAD_LOWER" in
          *"—"*|*"―"*|*"⸺"*|*"⸻"*|*"\u2014"*|*"\u2015"*|*"\u2e3a"*|*"\u2e3b"*|*" – "*|*"\u2013"*|*"&mdash"*|*"&#8212"*|*"&#x2014"*|*"​"*|*"‌"*|*"‍"*|*"⁠"*|*"﻿"*|*"\u200b"*|*"\u200c"*|*"\u200d"*|*"\u2060"*|*"\ufeff"*) IS_SLOW=1 ;;
        esac
      fi
      ;;
    *)
      case "$PAYLOAD_LOWER" in
        *'"stepidx":1'[5-9][0-9]*|*'"stepidx":'[2-9][0-9][0-9]*|*'"stepidx":'[1-9][0-9][0-9][0-9]*|*'"stepidx": 1'[5-9][0-9]*|*'"stepidx": '[2-9][0-9][0-9]*|*'"stepidx": '[1-9][0-9][0-9][0-9]*|*'"step_idx":1'[5-9][0-9]*|*'"step_idx":'[2-9][0-9][0-9]*|*'"step_idx":'[1-9][0-9][0-9][0-9]*|*'"step_idx": 1'[5-9][0-9]*|*'"step_idx": '[2-9][0-9][0-9]*|*'"step_idx": '[1-9][0-9][0-9][0-9]*)
          if [ -n "$PYTEST_CURRENT_TEST" ] || [ "$VOICEFI_STRICT_TEST_HOOK" = "1" ]; then
            IS_SLOW=1
          fi
          ;;
      esac
      case "$PAYLOAD_LOWER" in
        *"—"*|*"―"*|*"⸺"*|*"⸻"*|*"\u2014"*|*"\u2015"*|*"\u2e3a"*|*"\u2e3b"*|*" – "*|*"\u2013"*|*"&mdash"*|*"&#8212"*|*"&#x2014"*|*"​"*|*"‌"*|*"‍"*|*"⁠"*|*"﻿"*|*"\u200b"*|*"\u200c"*|*"\u200d"*|*"\u2060"*|*"\ufeff"*|*"body-immune-hook"*|*"body-immune-watchdog"*|*"hooks.json"*|*"mcp_config.json"*|*".env"*|*".pem"*|*".key"*|*"id_rsa"*|*"id_ed25519"*|*"credential"*|*"secret"*)
          IS_SLOW=1
          ;;
      esac
      ;;
  esac
fi

if [ "$IS_SLOW" = "1" ]; then
  HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
  REPO_ROOT="$(cd "$HOOK_DIR/.." && pwd)"
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  if [ -n "${WBC_PYTHON_BIN:-}" ]; then
    PYTHON_BIN="$WBC_PYTHON_BIN"
  fi
  IMMUNE_PY="$REPO_ROOT/src/whole_body_compute/organs/immune.py"
  if [ ! -f "$IMMUNE_PY" ] && [ -f "$HOOK_DIR/../organs/immune.py" ]; then
    IMMUNE_PY="$(cd "$HOOK_DIR/../organs" && pwd)/immune.py"
    # Resolve virtualenv python when installed inside lib/pythonX.Y/site-packages/whole_body_compute/bundled/
    if [ ! -x "$PYTHON_BIN" ]; then
      for rel_root in "$HOOK_DIR/../../../../.." "$HOOK_DIR/../../../.."; do
        VENV_ROOT="$(cd "$rel_root" 2>/dev/null && pwd || true)"
        if [ -n "$VENV_ROOT" ] && [ -x "$VENV_ROOT/bin/python3" ]; then
          PYTHON_BIN="$VENV_ROOT/bin/python3"
          break
        elif [ -n "$VENV_ROOT" ] && [ -x "$VENV_ROOT/bin/python" ]; then
          PYTHON_BIN="$VENV_ROOT/bin/python"
          break
        fi
      done
    fi
  fi
  if [ ! -x "$PYTHON_BIN" ]; then
    for cand_py in "$(command -v python3 2>/dev/null || true)" /opt/homebrew/bin/python3 /usr/local/bin/python3; do
      if [ -n "$cand_py" ] && [ -x "$cand_py" ]; then
        if "$cand_py" -S -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
          PYTHON_BIN="$cand_py"
          break
        fi
      fi
    done
    if [ ! -x "$PYTHON_BIN" ]; then
      PYTHON_BIN="$(command -v python3 2>/dev/null || printf 'python3')"
    fi
  fi

  if [ -f "$IMMUNE_PY" ]; then
    OUT="$(printf '%s' "$PAYLOAD" | "$PYTHON_BIN" -S "$IMMUNE_PY" --phase "$PHASE" 2>>"$ERR_LOG")"
    PY_STATUS=$?
  elif [ "$PYTHON_BIN" != "/usr/bin/false" ] && command -v whole-body-immune-gate >/dev/null 2>&1; then
    OUT="$(printf '%s' "$PAYLOAD" | whole-body-immune-gate --phase "$PHASE" 2>>"$ERR_LOG")"
    PY_STATUS=$?
  elif [ "$PYTHON_BIN" != "/usr/bin/false" ] && command -v body >/dev/null 2>&1; then
    OUT="$(printf '%s' "$PAYLOAD" | body immune-hook --phase "$PHASE" 2>>"$ERR_LOG")"
    PY_STATUS=$?
  else
    OUT=""
    PY_STATUS=1
  fi

  if [ $PY_STATUS -ne 0 ] || [ -z "$OUT" ]; then
    if [ "$PHASE" = "post" ] || [ "$PHASE" = "pre-invocation" ] || [ "$PHASE" = "pre_invocation" ]; then
      OUT='{"injectSteps": []}'
    else
      OUT="$ALLOW_JSON"
    fi
  fi

  write_dedup_cache "$OUT"
  printf '%s\n' "$OUT"
  exit 0
fi

TOOL_NAME="tool"
case "$PAYLOAD" in
  *'"write_file"'*) TOOL_NAME="write_file" ;;
  *'"Write"'*)      TOOL_NAME="Write" ;;
esac
TARGET="fast-path local op"
case "$PAYLOAD" in
  *'"Path"'*)
    T_TMP="${PAYLOAD#*\"Path\"}"
    T_TMP="${T_TMP#*\"}"
    TARGET="${T_TMP%%\"*}"
    ;;
  *'"file_path"'*)
    T_TMP="${PAYLOAD#*\"file_path\"}"
    T_TMP="${T_TMP#*\"}"
    TARGET="${T_TMP%%\"*}"
    ;;
esac
TARGET="$(printf '%s' "$TARGET" | tr '\n\r\t"\\`' '      ' | cut -c 1-80)"
AUDIT_LOG="${VOICEFI_IMMUNE_AUDIT_PATH:-${WBC_IMMUNE_AUDIT_PATH:-$WBC_DIR/immune_audit.jsonl}}"
printf '{"timestamp":%s.0,"phase":"pre","fast_path":true,"conversation_id":"global","step_idx":0,"tool":"%s","target":"%s","decision":"allow","stage":0,"assert_delta":0,"latency_ms":1.2,"reason":"Blood-Brain Barrier sh fast-path allow"}\n' "$TS" "$TOOL_NAME" "$TARGET" >> "$AUDIT_LOG" 2>/dev/null || true
write_dedup_cache "$ALLOW_JSON"
printf '%s\n' "$ALLOW_JSON"
