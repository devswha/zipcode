#!/bin/bash
# install.sh — Clone-user bootstrap for zipcode source checkouts
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/lib/install_common.sh"

usage() {
    cat <<'EOF'
Usage: ./install.sh [options]

Install zipcode from this source checkout into ~/.zipcode and ~/.local/bin.
If model assets are missing, the installer walks you through the next step:
show the download links, ask for local paths after you fetch the files, or
skip setup for now.

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

prompt_value() {
    local prompt="$1"
    local default_value="${2:-}"
    local answer=""

    if [ -n "${default_value}" ]; then
        printf '%s [%s] ' "${prompt}" "${default_value}" >&2
    else
        printf '%s ' "${prompt}" >&2
    fi

    IFS= read -r answer || true
    if [ -n "${answer}" ]; then
        printf '%s' "${answer}"
    else
        printf '%s' "${default_value}"
    fi
}

prompt_choice() {
    local prompt="$1"
    local default_choice="$2"
    local answer=""

    printf '%s [%s] ' "${prompt}" "${default_choice}" >&2
    IFS= read -r answer || true
    printf '%s' "${answer:-${default_choice}}"
}

default_choice_for_stdin() {
    local interactive_default="$1"
    local noninteractive_default="$2"

    if [ -t 0 ] || [ -p /dev/stdin ]; then
        printf '%s' "${interactive_default}"
    else
        printf '%s' "${noninteractive_default}"
    fi
}

prompt_press_enter() {
    local prompt="$1"

    printf '%s' "${prompt}" >&2
    IFS= read -r _ || true
}

collect_model_assets_from_prompt() {
    local mode="$1"
    local discovered_model="$2"
    local discovered_tokenizer="$3"
    local model_input=""
    local tokenizer_input=""

    echo
    if [ "${mode}" = "links" ]; then
        cat <<EOF
Model setup:
  1. Download a GGUF model from:
     https://huggingface.co/google/gemma-4-27b-it-GGUF
  2. Download tokenizer.json from:
     https://huggingface.co/google/gemma-4-27b-it/raw/main/tokenizer.json

After the files are on this machine, paste their local paths below.
If you already copied them into ${MODEL_DIR}, you can just press Enter.
EOF
        echo
        prompt_press_enter "Press Enter once the files are ready on this machine. "
    else
        cat <<EOF
Model setup:
  Paste the local paths to the model files you already downloaded.
  If you already copied them into ${MODEL_DIR}, you can just press Enter.
EOF
    fi

    model_input="$(prompt_value "Local path to the .gguf model:" "${discovered_model}")"
    tokenizer_input="$(prompt_value "Local path to tokenizer.json:" "${discovered_tokenizer}")"

    MODEL_SOURCE="${model_input}"
    TOKENIZER_SOURCE="${tokenizer_input}"
}

discover_model_assets() {
    local model_in_install=""
    local model_in_repo=""

    model_in_install="$(discover_single_model "${MODEL_DIR}" || true)"
    if [ -n "${model_in_install}" ]; then
        MODEL_SOURCE="${MODEL_SOURCE:-${model_in_install}}"
    fi

    model_in_repo="$(discover_single_model "${SCRIPT_DIR}/models" || true)"
    if [ -n "${model_in_repo}" ] && [ -z "${MODEL_SOURCE}" ]; then
        MODEL_SOURCE="${model_in_repo}"
    fi

    if [ -z "${TOKENIZER_SOURCE}" ] && [ -f "${MODEL_DIR}/tokenizer.json" ]; then
        TOKENIZER_SOURCE="${MODEL_DIR}/tokenizer.json"
    elif [ -z "${TOKENIZER_SOURCE}" ] && [ -f "${SCRIPT_DIR}/models/tokenizer.json" ]; then
        TOKENIZER_SOURCE="${SCRIPT_DIR}/models/tokenizer.json"
    fi
}

guided_model_setup() {
    local choice=""

    discover_model_assets
    if [ -n "${MODEL_SOURCE}" ] && [ -n "${TOKENIZER_SOURCE}" ]; then
        return 0
    fi

    echo
    cat <<EOF
Model assets are still needed before zipcode can run the full setup.

Choose one:
  1. Show the download links, then I will paste the file paths
  2. I already downloaded the files; ask me for the paths now
  3. Skip model setup for now
EOF

    choice="$(prompt_choice "Selection:" "$(default_choice_for_stdin "1" "3")")"
    case "${choice}" in
        1)
            collect_model_assets_from_prompt "links" "${MODEL_SOURCE}" "${TOKENIZER_SOURCE}"
            ;;
        2)
            collect_model_assets_from_prompt "paths" "${MODEL_SOURCE}" "${TOKENIZER_SOURCE}"
            ;;
        3)
            echo "Skipping model setup for now."
            ;;
        *)
            echo "Unknown selection '${choice}'. Skipping model setup for now."
            ;;
    esac
}

guided_helper_setup() {
    local helper_choice=""
    local helper_input=""
    local model_name=""

    LLAMA_SERVER_SOURCE="$(resolve_helper_source "${LLAMA_SERVER_SOURCE}" || true)"
    if [ -n "${LLAMA_SERVER_SOURCE}" ]; then
        return 0
    fi

    if [ -n "${MODEL_SOURCE}" ]; then
        model_name="$(basename -- "${MODEL_SOURCE}")"
    elif discover_single_model "${MODEL_DIR}" >/dev/null 2>&1; then
        model_name="$(basename -- "$(discover_single_model "${MODEL_DIR}")")"
    fi

    case "${model_name}" in
        *gemma-4*) ;;
        *) return 0 ;;
    esac

    echo
    cat <<EOF
Optional Gemma 4 compatibility helper:
  1. I have a llama-server binary; ask me for the path
  2. Skip helper setup for now
EOF

    helper_choice="$(prompt_choice "Selection:" "$(default_choice_for_stdin "2" "2")")"
    case "${helper_choice}" in
        1)
            helper_input="$(prompt_value "Local path to llama-server:" "")"
            if [ -n "${helper_input}" ]; then
                require_file "${helper_input}" "llama-server binary"
                LLAMA_SERVER_SOURCE="${helper_input}"
            else
                echo "No llama-server path provided; skipping helper setup."
            fi
            ;;
        2)
            echo "Skipping helper setup for now."
            ;;
        *)
            echo "Unknown selection '${helper_choice}'. Skipping helper setup for now."
            ;;
    esac
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

echo "Installing zipcode into ${INSTALL_DIR}..."
ensure_install_dirs
copy_executable "${BINARY_SOURCE}" "${INSTALL_DIR}/zipcode"
ln -sf "../zipcode" "${INSTALL_BIN_DIR}/zipcode"
copy_executable "${SCRIPT_DIR}/scripts/install_llama_server.sh" "${INSTALL_HELPER}"

guided_model_setup

if [ -n "${MODEL_SOURCE}" ]; then
    copy_if_needed "${MODEL_SOURCE}" "${MODEL_DIR}/$(basename -- "${MODEL_SOURCE}")"
fi

if [ -n "${TOKENIZER_SOURCE}" ]; then
    copy_if_needed "${TOKENIZER_SOURCE}" "${MODEL_DIR}/tokenizer.json"
fi

guided_helper_setup

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
        echo "    or download one with: ./scripts/download_model.sh --yes --dir ${MODEL_DIR}"
    fi
    if [ ! -f "${MODEL_DIR}/tokenizer.json" ]; then
        echo "  • add tokenizer.json with: ./install.sh --tokenizer /path/to/tokenizer.json"
        echo "    or let ./scripts/download_model.sh fetch it into ${MODEL_DIR}"
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
