#!/usr/bin/env bash
# ==============================================================================
# scripts/build_wbc_eyes_app.sh
#
# Builds, packages, signs (Hardened Runtime + Library Validation ON), and
# optionally notarizes `Whole Body Compute.app` - the unified zero-Python macOS
# menu-bar TCC broker for Organ 03 Eyes, Hands & Ears (`com.wholebodycompute.app`).
#
# Usage:
#   # Local developer build (Hardened Runtime ON, ad-hoc signature):
#   ./scripts/build_wbc_eyes_app.sh --adhoc --native-only [--install]
#
#   # Production Developer ID signed + Apple Notary Service notarized release:
#   ./scripts/build_wbc_eyes_app.sh --notarize --install
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SWIFT_PKG_DIR="${REPO_ROOT}/macos/WBCEyes"
INFO_PLIST_SRC="${SWIFT_PKG_DIR}/Info.plist"
ENTITLEMENTS_SRC="${SWIFT_PKG_DIR}/WBCEyes.entitlements"

OUTPUT_DIR="${REPO_ROOT}/dist"
APP_NAME="Whole Body Compute.app"
EXEC_NAME="WholeBodyCompute"
BUNDLE_ID="com.wholebodycompute.app"

ADHOC_SIGN=0
NATIVE_ONLY=0
DO_NOTARIZE=0
INSTALL_MODE="none" # none | user | system | data-dir
SIGN_IDENTITY="${CODESIGN_IDENTITY:-}"
NOTARY_PROFILE="${NOTARY_KEYCHAIN_PROFILE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --adhoc)
            ADHOC_SIGN=1
            shift
            ;;
        --identity)
            SIGN_IDENTITY="${2:?Missing argument for --identity}"
            shift 2
            ;;
        --native-only)
            NATIVE_ONLY=1
            shift
            ;;
        --notarize)
            DO_NOTARIZE=1
            shift
            ;;
        --notary-profile)
            NOTARY_PROFILE="${2:?Missing argument for --notary-profile}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?Missing argument for --output-dir}"
            shift 2
            ;;
        --install)
            INSTALL_MODE="user"
            shift
            ;;
        --system)
            INSTALL_MODE="system"
            shift
            ;;
        --install-data-dir)
            INSTALL_MODE="data-dir"
            shift
            ;;
        -h|--help)
            sed -n '2,16p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Error: ${APP_NAME} can only be built on macOS (Darwin)." >&2
    exit 1
fi

if [[ "${ADHOC_SIGN}" -eq 0 && -z "${SIGN_IDENTITY}" ]]; then
    SIGN_IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null | sed -n 's/.*"\(Developer ID Application:[^"]*\)".*/\1/p' | head -n 1 || true)"
    if [[ -n "${SIGN_IDENTITY}" ]]; then
        echo "==> Auto-detected Developer ID identity: ${SIGN_IDENTITY}"
    else
        echo "Error: Provide CODESIGN_IDENTITY / --identity for Developer ID signing, or pass --adhoc for local dev builds." >&2
        exit 1
    fi
fi

if [[ "${DO_NOTARIZE}" -eq 1 && "${ADHOC_SIGN}" -eq 1 ]]; then
    echo "Error: --notarize requires a valid Developer ID Application identity (cannot notarize --adhoc builds)." >&2
    exit 1
fi

if [[ "${DO_NOTARIZE}" -eq 1 && -z "${NOTARY_PROFILE}" && -z "${APP_STORE_CONNECT_API_KEY_PATH:-}" && -z "${APPLE_ID:-}" ]]; then
    for candidate_profile in "wbc-notary" "WBC-Notary" "voicefi-notary"; do
        HIST_OUT="$(xcrun notarytool history --keychain-profile "${candidate_profile}" 2>&1 || true)"
        if ! echo "${HIST_OUT}" | grep -q "No Keychain password item found"; then
            NOTARY_PROFILE="${candidate_profile}"
            echo "==> Auto-detected notarytool keychain profile: ${NOTARY_PROFILE}"
            break
        fi
    done
fi

echo "==> [1/5] Compiling ${EXEC_NAME} (Swift Release, zero Python runtime)..."
pushd "${SWIFT_PKG_DIR}" >/dev/null
if [[ "${NATIVE_ONLY}" -eq 1 ]]; then
    swift build -c release --product "${EXEC_NAME}"
    BIN_SRC="$(swift build -c release --show-bin-path)/${EXEC_NAME}"
else
    swift build -c release --arch arm64 --arch x86_64 --product "${EXEC_NAME}"
    BIN_SRC="${SWIFT_PKG_DIR}/.build/apple/Products/Release/${EXEC_NAME}"
    if [[ ! -f "${BIN_SRC}" ]]; then
        BIN_SRC="$(swift build -c release --show-bin-path)/${EXEC_NAME}"
    fi
fi
popd >/dev/null

if [[ ! -x "${BIN_SRC}" ]]; then
    echo "Error: Compiled binary not found at ${BIN_SRC}" >&2
    exit 1
fi

APP_BUNDLE="${OUTPUT_DIR}/${APP_NAME}"
CONTENTS_DIR="${APP_BUNDLE}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"

echo "==> [2/5] Assembling app bundle at ${APP_BUNDLE}..."
rm -rf "${APP_BUNDLE}" "${OUTPUT_DIR}/WBC Eyes.app"
mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

cp "${BIN_SRC}" "${MACOS_DIR}/${EXEC_NAME}"
chmod 0755 "${MACOS_DIR}/${EXEC_NAME}"
cp "${INFO_PLIST_SRC}" "${CONTENTS_DIR}/Info.plist"
printf 'APPL????' > "${CONTENTS_DIR}/PkgInfo"

# Build AppIcon.icns from canonical WBC brand mark if available
ICON_SRC=""
for candidate in \
    "${REPO_ROOT}/assets/wbc-github-avatar-dark.png" \
    "${REPO_ROOT}/assets/wbc-github-avatar-light.png"; do
    if [[ -f "${candidate}" ]]; then
        ICON_SRC="${candidate}"
        break
    fi
done

if [[ -n "${ICON_SRC}" ]] && command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
    ICONSET_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wbc_app_iconset.XXXXXX")/AppIcon.iconset"
    mkdir -p "${ICONSET_DIR}"
    for sz in 16 32 128 256 512; do
        sips -z "${sz}" "${sz}" "${ICON_SRC}" --out "${ICONSET_DIR}/icon_${sz}x${sz}.png" >/dev/null 2>&1 || true
        sz2=$((sz * 2))
        sips -z "${sz2}" "${sz2}" "${ICON_SRC}" --out "${ICONSET_DIR}/icon_${sz}x${sz}@2x.png" >/dev/null 2>&1 || true
    done
    iconutil -c icns "${ICONSET_DIR}" -o "${RESOURCES_DIR}/AppIcon.icns" >/dev/null 2>&1 || true
    rm -rf "$(dirname "${ICONSET_DIR}")"
fi

echo "==> [3/5] Verifying zero external / Python dylib dependencies (Library Validation safety)..."
OTOOL_OUT="$(otool -L "${MACOS_DIR}/${EXEC_NAME}")"
if echo "${OTOOL_OUT}" | grep -E -i "python|homebrew|/opt/|/usr/local/|@rpath" >/dev/null 2>&1; then
    echo "Error: Unexpected non-system dylib linked into ${EXEC_NAME}:" >&2
    echo "${OTOOL_OUT}" >&2
    exit 1
fi

echo "==> [4/5] Signing ${APP_NAME} with Hardened Runtime (Library Validation ON)..."
if [[ "${ADHOC_SIGN}" -eq 1 ]]; then
    codesign --force --options runtime \
        --entitlements "${ENTITLEMENTS_SRC}" \
        --identifier "${BUNDLE_ID}" \
        --sign "-" \
        "${APP_BUNDLE}"
else
    codesign --force --options runtime --timestamp \
        --entitlements "${ENTITLEMENTS_SRC}" \
        --sign "${SIGN_IDENTITY}" \
        "${APP_BUNDLE}"
fi

# Strict verification of signature, Hardened Runtime flag (0x10000), and Library Validation ON
codesign --verify --deep --strict --verbose=2 "${APP_BUNDLE}"
CODESIGN_DV="$(codesign -dv --verbose=4 "${APP_BUNDLE}" 2>&1)"
if ! echo "${CODESIGN_DV}" | grep -q "runtime"; then
    echo "Error: Hardened Runtime flag ('runtime') missing from signed bundle!" >&2
    echo "${CODESIGN_DV}" >&2
    exit 1
fi

ENTITLEMENTS_DUMP="$(codesign -d --entitlements :- "${APP_BUNDLE}" 2>/dev/null || true)"
for forbidden_ent in \
    "com.apple.security.cs.disable-library-validation" \
    "com.apple.security.cs.allow-dyld-environment-variables" \
    "com.apple.security.cs.allow-unsigned-executable-memory"; do
    if echo "${ENTITLEMENTS_DUMP}" | grep -q "${forbidden_ent}"; then
        echo "Error: Forbidden entitlement '${forbidden_ent}' detected - Library Validation must remain ON!" >&2
        exit 1
    fi
done

if [[ "${DO_NOTARIZE}" -eq 1 ]]; then
    echo "==> [5/5] Submitting ${APP_NAME} to Apple Notary Service via xcrun notarytool..."
    ZIP_PATH="${OUTPUT_DIR}/Whole-Body-Compute-notary.zip"
    rm -f "${ZIP_PATH}"
    ditto -c -k --keepParent "${APP_BUNDLE}" "${ZIP_PATH}"

    if [[ -n "${NOTARY_PROFILE}" ]]; then
        xcrun notarytool submit "${ZIP_PATH}" \
            --keychain-profile "${NOTARY_PROFILE}" \
            --wait
    elif [[ -n "${APP_STORE_CONNECT_API_KEY_PATH:-}" && -n "${APP_STORE_CONNECT_KEY_ID:-}" && -n "${APP_STORE_CONNECT_ISSUER_ID:-}" ]]; then
        xcrun notarytool submit "${ZIP_PATH}" \
            --key "${APP_STORE_CONNECT_API_KEY_PATH}" \
            --key-id "${APP_STORE_CONNECT_KEY_ID}" \
            --issuer "${APP_STORE_CONNECT_ISSUER_ID}" \
            --wait
    elif [[ -n "${APPLE_ID:-}" && -n "${APPLE_TEAM_ID:-}" && -n "${APPLE_APP_PASSWORD:-}" ]]; then
        xcrun notarytool submit "${ZIP_PATH}" \
            --apple-id "${APPLE_ID}" \
            --team-id "${APPLE_TEAM_ID}" \
            --password "${APPLE_APP_PASSWORD}" \
            --wait
    else
        rm -f "${ZIP_PATH}"
        echo "Error: --notarize requires NOTARY_KEYCHAIN_PROFILE (or APP_STORE_CONNECT_* / APPLE_ID credentials)." >&2
        exit 1
    fi

    rm -f "${ZIP_PATH}"
    xcrun stapler staple "${APP_BUNDLE}"
    xcrun stapler validate "${APP_BUNDLE}"
    spctl --assess --type execute --verbose=4 "${APP_BUNDLE}"

    APP_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "${CONTENTS_DIR}/Info.plist" 2>/dev/null || echo '0.1.6')"
    RELEASE_ZIP="${OUTPUT_DIR}/Whole-Body-Compute-${APP_VERSION}.zip"
    rm -f "${RELEASE_ZIP}" "${RELEASE_ZIP}.sha256"
    ditto -c -k --keepParent "${APP_BUNDLE}" "${RELEASE_ZIP}"
    ZIP_SHA256="$(shasum -a 256 "${RELEASE_ZIP}" | awk '{print $1}')"
    printf '%s  %s\n' "${ZIP_SHA256}" "$(basename "${RELEASE_ZIP}")" > "${RELEASE_ZIP}.sha256"

    DAEMON_PY="${REPO_ROOT}/src/whole_body_compute/daemon.py"
    CASK_RB="${REPO_ROOT}/packaging/homebrew/Casks/whole-body-compute.rb"
    if [[ -f "${DAEMON_PY}" || -f "${CASK_RB}" ]]; then
        python3 - "${ZIP_SHA256}" "${DAEMON_PY}" "${CASK_RB}" <<'PYEOF'
import pathlib, re, sys
sha, daemon_path, cask_path = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
if daemon_path.is_file():
    txt = daemon_path.read_text(encoding="utf-8")
    new_txt = re.sub(r'^WBC_EYES_PINNED_SHA256\s*=\s*"[^"]*"', f'WBC_EYES_PINNED_SHA256 = "{sha}"', txt, count=1, flags=re.MULTILINE)
    if new_txt != txt:
        daemon_path.write_text(new_txt, encoding="utf-8")
if cask_path.is_file():
    txt = cask_path.read_text(encoding="utf-8")
    new_txt = re.sub(r'^\s*sha256\s+"[^"]*"', f'  sha256 "{sha}"', txt, count=1, flags=re.MULTILINE)
    if new_txt != txt:
        cask_path.write_text(new_txt, encoding="utf-8")
PYEOF
        echo "==> Auto-pinned SHA-256 in daemon.py and packaging/homebrew/Casks/whole-body-compute.rb"
    fi

    echo "==> Notarized & stapled release archive: ${RELEASE_ZIP}"
    echo "==> SHA-256: ${ZIP_SHA256}"
else
    echo "==> [5/5] Notarization skipped (pass --notarize with Developer ID + notarytool credentials for distribution)."
fi

case "${INSTALL_MODE}" in
    user)
        DEST_DIR="${HOME}/Applications"
        mkdir -p "${DEST_DIR}"
        rm -rf "${DEST_DIR}/${APP_NAME}" "${DEST_DIR}/WBC Eyes.app" "${HOME}/.whole_body_compute/Whole Body Compute.app"
        ditto "${APP_BUNDLE}" "${DEST_DIR}/${APP_NAME}"
        echo "==> Installed ${DEST_DIR}/${APP_NAME}"
        ;;
    system)
        DEST_DIR="/Applications"
        rm -rf "${DEST_DIR}/${APP_NAME}" "${DEST_DIR}/WBC Eyes.app" "${HOME}/.whole_body_compute/Whole Body Compute.app"
        ditto "${APP_BUNDLE}" "${DEST_DIR}/${APP_NAME}"
        echo "==> Installed ${DEST_DIR}/${APP_NAME}"
        ;;
    data-dir)
        WBC_DIR="${WBC_DATA_DIR:-${HOME}/.whole_body_compute}"
        mkdir -p "${WBC_DIR}"
        chmod 0700 "${WBC_DIR}" || true
        rm -rf "${WBC_DIR}/${APP_NAME}" "${WBC_DIR}/WBC Eyes.app"
        ditto "${APP_BUNDLE}" "${WBC_DIR}/${APP_NAME}"
        echo "==> Installed ${WBC_DIR}/${APP_NAME}"
        ;;
    none)
        ;;
esac

echo "Built & verified: ${APP_BUNDLE}"
