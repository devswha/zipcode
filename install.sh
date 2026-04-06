#!/bin/bash
# install.sh — Clone-user bootstrap for zipcode source checkouts
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/lib/install_common.sh"

usage() {
    cat <<'EOF'
Usage: ./install.sh [options]

Install zipcode from this source checkout into ~/.zipcode and ~/.local/bin.
If a model, tokenizer, or llama-server binary is already available, pass
their paths here or place them in ./models before running the installer.

Options:
  --binary PATH         Use an existing zipcode binary instead of building one.
  --model PATH          Copy PATH (.gguf) into ~/.zipcode/models.
  --tokenizer PATH      Copy PATH (tokenizer.json) into ~/.zipcode/models/tokenizer.json.
  --llama-server PATH   Install PATH as ~/.zipcode/bin/llama-server.
  --skip-build          Reuse ./target/release/zipcode instead of rebuilding.
  --skip-setup          Skip the post-install `zipcode setup --skip-smoke` run.
  -h, --help            Show this help text.
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    local label="$2"
    [ -f "${path}" ] || fail "${label} not found: ${path}"
}

prompt_for_path() {
    local prompt="$1"
    local current="${2:-}"
    local answer=""

    if [ -n "${current}" ] || [ ! -t 0 ]; then
        printf '%s' "${current}"
        return 0
    fi

    read -r -p "${prompt}" answer || true
    printf '%s' "${answer}"
}

discover_single_model() {
    local search_dir="$1"
    local matches=()
    local candidate

    [ -d "${search_dir}" ] || return 1

    while IFS= read -r -d '' candidate; do
        matches+=("${candidate}")
    done < <(find "${search_dir}" -maxdepth 1 -type f -name '*.gguf' -print0 | sort -z)

    [ "${#matches[@]}" -eq 1 ] || return 1
    printf '%s\n' "${matches[0]}"
}

resolve_helper_source() {
    local explicit_path="${1:-}"
    if [ -n "${explicit_path}" ]; then
        require_file "${explicit_path}" "llama-server binary"
        printf '%s\n' "${explicit_path}"
        return 0
    fi

    local env_path="${ZIPCODE_LLAMA_SERVER_BIN:-}"
    if [ -n "${env_path}" ]; then
        require_file "${env_path}" "ZIPCODE_LLAMA_SERVER_BIN"
        printf '%s\n' "${env_path}"
        return 0
    fi

    env_path="${LLAMA_SERVER_BIN:-}"
    if [ -n "${env_path}" ]; then
        require_file "${env_path}" "LLAMA_SERVER_BIN"
        printf '%s\n' "${env_path}"
        return 0
    fi

    if command -v llama-server >/dev/null 2>&1; then
        command -v llama-server
        return 0
    fi

    return 1
}

copy_if_needed() {
    local source_path="$1"
    local target_path="$2"
    require_file "${source_path}" "source file"

    if [ "${source_path}" = "${target_path}" ]; then
        echo "Using existing file: ${target_path}"
        return 0
    fi

    mkdir -p "$(dirname -- "${target_path}")"
    cp "${source_path}" "${target_path}"
    echo "Installed $(basename -- "${target_path}") to ${target_path}"
}

BINARY_SOURCE=""
MODEL_SOURCE=""
TOKENIZER_SOURCE=""
LLAMA_SERVER_SOURCE=""
SKIP_BUILD=0
SKIP_SETUP=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --binary)
            [ "$#" -gt 1 ] || fail "--binary requires a path"
            BINARY_SOURCE="$2"
            shift
            ;;
        --model)
            [ "$#" -gt 1 ] || fail "--model requires a path"
            MODEL_SOURCE="$2"
            shift
            ;;
        --tokenizer)
            [ "$#" -gt 1 ] || fail "--tokenizer requires a path"
            TOKENIZER_SOURCE="$2"
            shift
            ;;
        --llama-server)
            [ "$#" -gt 1 ] || fail "--llama-server requires a path"
            LLAMA_SERVER_SOURCE="$2"
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            ;;
        --skip-setup)
            SKIP_SETUP=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
    shift
done

INSTALL_DIR="$(zipcode_install_dir)"
INSTALL_BIN_DIR="$(zipcode_install_bin_dir)"
MODEL_DIR="$(zipcode_model_dir)"
SETUP_ENV="$(zipcode_setup_env_path)"
CONFIG_FILE="$(zipcode_config_path)"
USER_BIN_DIR="$(zipcode_user_bin_dir)"
NEXT_STEPS_FILE="$(zipcode_next_steps_path)"
INSTALL_HELPER="${INSTALL_DIR}/install_llama_server.sh"

if [ -z "${BINARY_SOURCE}" ]; then
    BINARY_SOURCE="${SCRIPT_DIR}/target/release/zipcode"
    if [ "${SKIP_BUILD}" -eq 1 ]; then
        [ -x "${BINARY_SOURCE}" ] || fail "--skip-build was set but ${BINARY_SOURCE} does not exist"
    else
        command -v cargo >/dev/null 2>&1 || fail "cargo is required for clone installs. Install Rust or use a release bundle."
        echo "Building zipcode from source..."
        (cd "${SCRIPT_DIR}" && cargo build --release -p zipcode)
    fi
else
    require_file "${BINARY_SOURCE}" "zipcode binary"
fi

MODEL_SOURCE="${MODEL_SOURCE:-$(discover_single_model "${SCRIPT_DIR}/models" || true)}"
if [ -z "${TOKENIZER_SOURCE}" ] && [ ! -f "${MODEL_DIR}/tokenizer.json" ]; then
    TOKENIZER_SOURCE="${TOKENIZER_SOURCE:-${SCRIPT_DIR}/models/tokenizer.json}"
fi
if [ ! -f "${TOKENIZER_SOURCE:-}" ]; then
    TOKENIZER_SOURCE=""
fi

MODEL_SOURCE="$(prompt_for_path "Path to a .gguf model (leave blank to skip for now): " "${MODEL_SOURCE}")"
TOKENIZER_SOURCE="$(prompt_for_path "Path to tokenizer.json (leave blank to skip for now): " "${TOKENIZER_SOURCE}")"
LLAMA_SERVER_SOURCE="$(prompt_for_path "Path to llama-server (leave blank to skip for now): " "$(resolve_helper_source "${LLAMA_SERVER_SOURCE}" || true)")"

echo "Installing zipcode into ${INSTALL_DIR}..."
ensure_install_dirs
copy_executable "${BINARY_SOURCE}" "${INSTALL_DIR}/zipcode"
ln -sf "../zipcode" "${INSTALL_BIN_DIR}/zipcode"
copy_executable "${SCRIPT_DIR}/scripts/install_llama_server.sh" "${INSTALL_HELPER}"

if [ -n "${MODEL_SOURCE}" ]; then
    copy_if_needed "${MODEL_SOURCE}" "${MODEL_DIR}/$(basename -- "${MODEL_SOURCE}")"
fi

if [ -n "${TOKENIZER_SOURCE}" ]; then
    copy_if_needed "${TOKENIZER_SOURCE}" "${MODEL_DIR}/tokenizer.json"
fi

if [ -n "${LLAMA_SERVER_SOURCE}" ]; then
    "${INSTALL_HELPER}" "${LLAMA_SERVER_SOURCE}" "${INSTALL_DIR}" >/dev/null
fi

LLAMA_SERVER_INSTALLED=""
if [ -x "${INSTALL_BIN_DIR}/llama-server" ]; then
    LLAMA_SERVER_INSTALLED="${INSTALL_BIN_DIR}/llama-server"
fi

write_default_config "${CONFIG_FILE}" "${MODEL_DIR}"
write_setup_env "${SETUP_ENV}" "${INSTALL_BIN_DIR}" "${USER_BIN_DIR}" "${LLAMA_SERVER_INSTALLED}"
USER_LAUNCHER_INSTALLED="$(ensure_user_launcher "${INSTALL_BIN_DIR}/zipcode" "${USER_BIN_DIR}")"

{
    echo "Next steps:"
    echo "  • zipcode installs into ${INSTALL_DIR}"
    echo "  • launcher installed at ${USER_LAUNCHER_INSTALLED}"
    if ! path_contains_dir "${USER_BIN_DIR}"; then
        echo "  • ${USER_BIN_DIR} is not on PATH in this shell yet"
        echo "    Run: source \"${SETUP_ENV}\""
    fi
    if [ -z "${MODEL_SOURCE}" ] && ! discover_single_model "${MODEL_DIR}" >/dev/null 2>&1; then
        echo "  • add a .gguf model with: ./install.sh --model /path/to/model.gguf"
        echo "    or download one with: ./scripts/download_model.sh"
    fi
    if [ ! -f "${MODEL_DIR}/tokenizer.json" ]; then
        echo "  • add tokenizer.json with: ./install.sh --tokenizer /path/to/tokenizer.json"
    fi
    if [ -z "${LLAMA_SERVER_INSTALLED}" ]; then
        echo "  • optional Gemma 4 fallback helper: ./install.sh --llama-server /path/to/llama-server"
    fi
} > "${NEXT_STEPS_FILE}"

if [ "${SKIP_SETUP}" -eq 1 ]; then
    echo "Skipped zipcode setup (--skip-setup)."
elif discover_single_model "${MODEL_DIR}" >/dev/null 2>&1; then
    echo
    echo "Running zipcode setup --skip-smoke..."
    "${INSTALL_BIN_DIR}/zipcode" setup --skip-smoke
else
    echo
    echo "Skipping zipcode setup for now because no .gguf model is installed yet."
fi

echo
echo "zipcode doctor"
"${INSTALL_BIN_DIR}/zipcode" doctor || true

echo
cat "${NEXT_STEPS_FILE}"
