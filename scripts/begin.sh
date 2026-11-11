#!/bin/sh
# ==============================================================================
# Whole Body Compute - 60-Second Sovereign Agent Runtime Installer
# Hosted at:
#   curl -fsSL https://wholebodycompute.com/begin | sh
#   curl -fsSL https://wholebodycompute.org/begin | sh
# ==============================================================================
set -e

BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

printf "\n${BOLD}${CYAN}🧬 Whole Body Compute - Sovereign Agent OS Awakening${RESET}\n"
printf "════════════════════════════════════════════════════════════════════\n"

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
printf "  • Platform: %s (%s)\n" "$OS_NAME" "$ARCH_NAME"

if [ "$OS_NAME" != "Darwin" ]; then
  printf "${YELLOW}  ! Note: Apple Silicon Metal organs degrade gracefully on non-macOS hosts.${RESET}\n"
fi

# 1. Locate or install Whole Body Compute CLI (`body`)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"

if command -v body >/dev/null 2>&1; then
  printf "${GREEN}  ✓ Found existing 'body' CLI: $(command -v body)${RESET}\n"
elif [ -x "$HOME/.local/bin/body" ]; then
  printf "${GREEN}  ✓ Found user '~/.local/bin/body' CLI${RESET}\n"
  PATH="$HOME/.local/bin:$PATH"
  export PATH
elif [ -x "./.venv/bin/body" ]; then
  printf "${GREEN}  ✓ Found workspace '.venv/bin/body' CLI${RESET}\n"
  PATH="$(pwd)/.venv/bin:$PATH"
  export PATH
else
  printf "  • Installing whole-body-compute CLI...\n"
  GIT_REPO_URL="git+https://github.com/wholebodycompute/whole-body-compute.git"
  LOCAL_SRC=""
  if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    LOCAL_SRC="$SCRIPT_DIR"
  fi

  INSTALL_LOG="$(mktemp /tmp/wbc_install.XXXXXX 2>/dev/null || printf '/tmp/wbc_install.log')"
  if command -v brew >/dev/null 2>&1 && [ "$OS_NAME" = "Darwin" ] && [ -z "$LOCAL_SRC" ] && [ "${WBC_PREFER_BREW:-0}" = "1" ]; then
    brew install wholebodycompute/tap/whole-body-compute >>"$INSTALL_LOG" 2>&1 || true
  fi

  if ! command -v body >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
      if [ -n "$LOCAL_SRC" ]; then
        uv tool install "$LOCAL_SRC" >>"$INSTALL_LOG" 2>&1 || uv tool install --force "$LOCAL_SRC" >>"$INSTALL_LOG" 2>&1 || true
      else
        uv tool install whole-body-compute >>"$INSTALL_LOG" 2>&1 || uv tool install "$GIT_REPO_URL" >>"$INSTALL_LOG" 2>&1 || true
      fi
    fi
  fi

  if ! command -v body >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/body" ] && command -v brew >/dev/null 2>&1 && [ "$OS_NAME" = "Darwin" ] && [ -z "$LOCAL_SRC" ]; then
    brew install wholebodycompute/tap/whole-body-compute >>"$INSTALL_LOG" 2>&1 || true
  fi

  if ! command -v body >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/body" ]; then
    if command -v pipx >/dev/null 2>&1; then
      if [ -n "$LOCAL_SRC" ]; then
        pipx install "$LOCAL_SRC" >>"$INSTALL_LOG" 2>&1 || true
      else
        pipx install whole-body-compute >>"$INSTALL_LOG" 2>&1 || pipx install "$GIT_REPO_URL" >>"$INSTALL_LOG" 2>&1 || true
      fi
    else
      if [ -n "$LOCAL_SRC" ]; then
        python3 -m pip install --user "$LOCAL_SRC" >>"$INSTALL_LOG" 2>&1 || true
      else
        python3 -m pip install --user whole-body-compute >>"$INSTALL_LOG" 2>&1 || python3 -m pip install --user "$GIT_REPO_URL" >>"$INSTALL_LOG" 2>&1 || true
      fi
      # PEP 668 externally-managed-environment fallback: create isolated user venv at ~/.whole_body_compute/venv
      if ! command -v body >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/body" ] && grep -qi "externally-managed-environment" "$INSTALL_LOG" 2>/dev/null; then
        WBC_VENV="$HOME/.whole_body_compute/venv"
        if python3 -m venv "$WBC_VENV" >>"$INSTALL_LOG" 2>&1; then
          if [ -n "$LOCAL_SRC" ]; then
            "$WBC_VENV/bin/pip" install "$LOCAL_SRC" >>"$INSTALL_LOG" 2>&1 || true
          else
            "$WBC_VENV/bin/pip" install whole-body-compute >>"$INSTALL_LOG" 2>&1 || "$WBC_VENV/bin/pip" install "$GIT_REPO_URL" >>"$INSTALL_LOG" 2>&1 || true
          fi
          if [ -x "$WBC_VENV/bin/body" ]; then
            mkdir -p "$HOME/.local/bin" 2>/dev/null || true
            ln -sf "$WBC_VENV/bin/body" "$HOME/.local/bin/body" 2>/dev/null || true
            ln -sf "$WBC_VENV/bin/wbc" "$HOME/.local/bin/wbc" 2>/dev/null || true
            ln -sf "$WBC_VENV/bin/body-immune-hook" "$HOME/.local/bin/body-immune-hook" 2>/dev/null || true
          fi
        fi
      fi
    fi
  fi

  if [ -x "$HOME/.local/bin/body" ]; then
    PATH="$HOME/.local/bin:$PATH"
    export PATH
  fi
fi

# 2. Awaken the current workspace (`body begin`)
if command -v body >/dev/null 2>&1; then
  rm -f "${INSTALL_LOG:-}" 2>/dev/null || true
  exec body begin "$@"
elif [ -x "./.venv/bin/body" ]; then
  rm -f "${INSTALL_LOG:-}" 2>/dev/null || true
  exec ./.venv/bin/body begin "$@"
elif python3 -c "import whole_body_compute" >/dev/null 2>&1; then
  rm -f "${INSTALL_LOG:-}" 2>/dev/null || true
  exec python3 -m whole_body_compute.cli begin "$@"
else
  printf "${YELLOW}  ! Automatic CLI install did not complete on this Python/OS setup.${RESET}\n" >&2
  if [ -n "${INSTALL_LOG:-}" ] && [ -s "$INSTALL_LOG" ]; then
    printf "  • Installer log tail:\n" >&2
    tail -n 6 "$INSTALL_LOG" >&2 || true
    rm -f "$INSTALL_LOG" 2>/dev/null || true
  fi
  printf "\n  Install explicitly with either:\n" >&2
  printf "    brew install wholebodycompute/tap/whole-body-compute && body begin\n" >&2
  printf "    uv tool install whole-body-compute && body begin\n\n" >&2
  exit 1
fi
