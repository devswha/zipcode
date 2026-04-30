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
download in the terminal now, show the download links, ask for local paths
after you fetch the files, or skip setup for now.

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

# Probe the C build deps that `cargo build -p zipcode` needs to compile
# `llama-cpp-sys-2`. Surfaces missing tooling at the top of the install
# instead of letting bindgen panic 80 lines deep into cargo output.
#
# Tracked as #148. Without this, fresh hosts fail with
#   thread 'main' panicked at bindgen-*/lib.rs: Unable to find libclang
# which is opaque for non-Rust users.
ensure_build_deps_available() {
    local missing=()

    command -v cmake >/dev/null 2>&1 || missing+=("cmake")
    command -v pkg-config >/dev/null 2>&1 || missing+=("pkg-config")

    if ! probe_libclang; then
        missing+=("libclang")
    fi

    [ "${#missing[@]}" -eq 0 ] && return 0

    echo "Missing native build prerequisites: ${missing[*]}" >&2
    if command -v apt-get >/dev/null 2>&1; then
        local apt_pkgs="libclang-dev cmake pkg-config build-essential"
        echo "" >&2
        echo "On Debian/Ubuntu, install with:" >&2
        echo "  sudo apt-get install -y ${apt_pkgs}" >&2
    elif command -v dnf >/dev/null 2>&1; then
        echo "" >&2
        echo "On Fedora/RHEL, install with:" >&2
        echo "  sudo dnf install -y clang-devel cmake pkgconf-pkg-config gcc-c++" >&2
    elif command -v brew >/dev/null 2>&1; then
        echo "" >&2
        echo "On macOS (Homebrew), install with:" >&2
        echo "  brew install cmake llvm pkg-config" >&2
    else
        echo "" >&2
        echo "Install your platform's libclang/cmake/pkg-config equivalents and re-run." >&2
    fi
    fail "build dependencies missing: ${missing[*]}"
}

probe_libclang() {
    [ -n "${LIBCLANG_PATH:-}" ] && [ -e "${LIBCLANG_PATH}" ] && return 0
    if command -v ldconfig >/dev/null 2>&1; then
        ldconfig -p 2>/dev/null | grep -q "libclang" && return 0
    fi
    local candidate
    for candidate in /usr/lib/x86_64-linux-gnu/libclang.so \
                     /usr/lib/x86_64-linux-gnu/libclang-*.so* \
                     /usr/lib/libclang.so* \
                     /usr/local/lib/libclang.so* \
                     /opt/homebrew/opt/llvm/lib/libclang.dylib \
                     /usr/local/opt/llvm/lib/libclang.dylib; do
        [ -e "${candidate}" ] && return 0
    done
    return 1
}

list_model_candidates() {
    local search_dir="$1"
    [ -d "${search_dir}" ] || return 0
    find "${search_dir}" -maxdepth 1 -type f -name '*.gguf' ! -name 'mmproj*.gguf' -print0 | sort -z
}

discover_single_model() {
    local search_dir="$1"
    local matches=()
    local candidate

    [ -d "${search_dir}" ] || return 1

    while IFS= read -r -d '' candidate; do
        matches+=("${candidate}")
    done < <(list_model_candidates "${search_dir}")

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

    local installed_path="$(zipcode_install_bin_dir)/llama-server"
    if [ -f "${installed_path}" ]; then
        printf '%s\n' "${installed_path}"
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

model_preset_name() {
    case "$1" in
        e2b) printf '%s\n' "Gemma 4 E2B IT (recommended smaller download)" ;;
        31b) printf '%s\n' "Gemma 4 31B IT (larger, higher VRAM)" ;;
        *) fail "unknown model preset: $1" ;;
    esac
}

model_repo_url_for_preset() {
    case "$1" in
        e2b) printf '%s\n' "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF" ;;
        31b) printf '%s\n' "https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF" ;;
        *) fail "unknown model preset: $1" ;;
    esac
}

tokenizer_url_for_preset() {
    case "$1" in
        e2b) printf '%s\n' "https://huggingface.co/google/gemma-4-E2B-it/raw/main/tokenizer.json" ;;
        31b) printf '%s\n' "https://huggingface.co/google/gemma-4-31B-it/raw/main/tokenizer.json" ;;
        *) fail "unknown model preset: $1" ;;
    esac
}

preferred_model_for_preset() {
    local preset="$1"
    local search_dir="$2"
    local needle=""
    local candidate=""
    local lower_name=""
    local matches=()
    local fallback=""

    case "${preset}" in
        e2b) needle="gemma-4-e2b" ;;
        31b) needle="gemma-4-31b" ;;
        *) fail "unknown model preset: $preset" ;;
    esac

    while IFS= read -r -d '' candidate; do
        lower_name="$(basename -- "${candidate}" | tr '[:upper:]' '[:lower:]')"
        case "${lower_name}" in
            *"${needle}"*)
                matches+=("${candidate}")
                ;;
        esac
    done < <(list_model_candidates "${search_dir}")

    if [ "${#matches[@]}" -eq 0 ]; then
        return 1
    fi

    for candidate in "${matches[@]}"; do
        lower_name="$(basename -- "${candidate}" | tr '[:upper:]' '[:lower:]')"
        case "${lower_name}" in
            *q8_0*)
                printf '%s\n' "${candidate}"
                return 0
                ;;
        esac
    done

    fallback="${matches[0]}"
    printf '%s\n' "${fallback}"
}

prompt_model_choice() {
    local default_index="${1:-1}"
    shift
    local candidates=("$@")
    local i=0
    local answer=""

    echo >&2
    echo "Multiple AI models are available. Choose which one zipcode should use:" >&2
    for candidate in "${candidates[@]}"; do
        i=$((i + 1))
        printf '  %d. %s\n' "${i}" "$(basename -- "${candidate}")" >&2
    done

    answer="$(prompt_choice "Selection:" "${default_index}")"
    if [[ "${answer}" =~ ^[0-9]+$ ]] && [ "${answer}" -ge 1 ] && [ "${answer}" -le "${#candidates[@]}" ]; then
        printf '%s\n' "${candidates[$((answer - 1))]}"
        return 0
    fi

    printf '%s\n' "${candidates[$((default_index - 1))]}"
}

choose_model_preset() {
    local preset_choice=""

    echo >&2
    cat >&2 <<EOF
Choose a model profile:
  1. $(model_preset_name e2b)
  2. $(model_preset_name 31b)
EOF

    preset_choice="$(prompt_choice "Selection:" "1")"
    case "${preset_choice}" in
        1) printf '%s' "e2b" ;;
        2) printf '%s' "31b" ;;
        *) echo "Unknown selection '${preset_choice}'. Using the recommended E2B model." >&2
           printf '%s' "e2b" ;;
    esac
}

run_terminal_model_download() {
    local preset="$1"
    local downloader="${ZIPCODE_DOWNLOAD_MODEL_SCRIPT:-${SCRIPT_DIR}/scripts/download_model.sh}"

    [ -f "${downloader}" ] || fail "download helper not found: ${downloader}"

    echo
    echo "Downloading in this terminal now:"
    echo "  • Model profile: $(model_preset_name "${preset}")"
    echo "  • GGUF page:     $(model_repo_url_for_preset "${preset}")"
    echo "  • Tokenizer:     $(tokenizer_url_for_preset "${preset}")"
    echo

    bash "${downloader}" --yes --dir "${MODEL_DIR}" --preset "${preset}"

    MODEL_SOURCE="$(preferred_model_for_preset "${preset}" "${MODEL_DIR}" || true)"
    if [ -f "${MODEL_DIR}/tokenizer.json" ]; then
        TOKENIZER_SOURCE="${MODEL_DIR}/tokenizer.json"
    fi
}

collect_model_assets_from_prompt() {
    local preset="$1"
    local mode="$2"
    local discovered_model="$3"
    local discovered_tokenizer="$4"
    local model_input=""
    local tokenizer_input=""

    echo
    if [ "${mode}" = "links" ]; then
        cat <<EOF
Model setup:
  1. Download a GGUF model from:
     $(model_repo_url_for_preset "${preset}")
  2. Download tokenizer.json from:
     $(tokenizer_url_for_preset "${preset}")

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
    local saved_model=""

    saved_model="$(saved_model_from_config "${CONFIG_FILE}" || true)"
    if [ -n "${saved_model}" ]; then
        MODEL_SOURCE="${MODEL_SOURCE:-${saved_model}}"
    fi

    model_in_install="$(discover_single_model "${MODEL_DIR}" || true)"
    if [ -n "${model_in_install}" ] && [ -z "${MODEL_SOURCE}" ]; then
        MODEL_SOURCE="${model_in_install}"
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

saved_model_from_config() {
    local config_path="$1"

    [ -f "${config_path}" ] || return 1
    command -v python3 >/dev/null 2>&1 || return 1

    python3 - "$config_path" <<'PY'
import json, pathlib, sys
config_path = pathlib.Path(sys.argv[1])
try:
    data = json.loads(config_path.read_text())
except Exception:
    raise SystemExit(1)
model_dir = data.get("model_dir")
model_file = data.get("model_file")
if not model_dir or not model_file:
    raise SystemExit(1)
path = pathlib.Path(model_dir) / model_file
if path.is_file():
    print(path)
else:
    raise SystemExit(1)
PY
}

resolve_setup_model() {
    local candidate=""
    local candidates=()
    local saved_model=""

    if [ -n "${MODEL_SOURCE}" ]; then
        candidate="${MODEL_DIR}/$(basename -- "${MODEL_SOURCE}")"
        if [ -f "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
        if [ -f "${MODEL_SOURCE}" ]; then
            printf '%s\n' "${MODEL_SOURCE}"
            return 0
        fi
    fi

    saved_model="$(saved_model_from_config "${CONFIG_FILE}" || true)"
    if [ -n "${saved_model}" ]; then
        printf '%s\n' "${saved_model}"
        return 0
    fi

    while IFS= read -r -d '' candidate; do
        candidates+=("${candidate}")
    done < <(list_model_candidates "${MODEL_DIR}")

    case "${#candidates[@]}" in
        0) return 1 ;;
        1)
            printf '%s\n' "${candidates[0]}"
            return 0
            ;;
        *)
            if [ -t 0 ] || [ -p /dev/stdin ]; then
                prompt_model_choice "1" "${candidates[@]}"
                return 0
            fi
            return 1
            ;;
    esac
}

run_terminal_helper_build() {
    local builder="${ZIPCODE_LLAMA_SERVER_BUILD_SCRIPT:-${SCRIPT_DIR}/scripts/build_llama_server.sh}"

    [ -f "${builder}" ] || fail "llama-server build helper not found: ${builder}"

    echo
    echo "Building llama-server in this terminal now..."
    bash "${builder}" "${INSTALL_DIR}"
}

guided_model_setup() {
    local choice=""
    local preset=""

    discover_model_assets
    if [ -n "${MODEL_SOURCE}" ] && [ -n "${TOKENIZER_SOURCE}" ]; then
        return 0
    fi

    echo
    cat <<EOF
Model assets are still needed before zipcode can run the full setup.

Choose one:
  1. Download in this terminal now (recommended)
  2. Show the download links, then I will paste the file paths
  3. I already downloaded the files; ask me for the paths now
  4. Skip model setup for now
EOF

    choice="$(prompt_choice "Selection:" "$(default_choice_for_stdin "1" "4")")"
    case "${choice}" in
        1)
            preset="$(choose_model_preset)"
            run_terminal_model_download "${preset}" || {
                echo "Terminal download did not finish. You can paste local paths instead."
                collect_model_assets_from_prompt "${preset}" "paths" "${MODEL_SOURCE}" "${TOKENIZER_SOURCE}"
            }
            ;;
        2)
            preset="$(choose_model_preset)"
            collect_model_assets_from_prompt "${preset}" "links" "${MODEL_SOURCE}" "${TOKENIZER_SOURCE}"
            ;;
        3)
            collect_model_assets_from_prompt "e2b" "paths" "${MODEL_SOURCE}" "${TOKENIZER_SOURCE}"
            ;;
        4)
            echo "Skipping model setup for now."
            ;;
        *)
            echo "Unknown selection '${choice}'. Skipping model setup for now."
            ;;
    esac
}

guided_helper_setup() {
    local selected_model="${1:-}"
    local helper_choice=""
    local helper_input=""
    local model_name=""

    LLAMA_SERVER_SOURCE="$(resolve_helper_source "${LLAMA_SERVER_SOURCE}" || true)"
    if [ -n "${LLAMA_SERVER_SOURCE}" ]; then
        return 0
    fi

    if [ -n "${selected_model}" ]; then
        model_name="$(basename -- "${selected_model}")"
    elif [ -n "${MODEL_SOURCE}" ]; then
        model_name="$(basename -- "${MODEL_SOURCE}")"
    fi

    if [ -z "${model_name}" ]; then
        while IFS= read -r -d '' candidate; do
            case "$(basename -- "${candidate}" | tr '[:upper:]' '[:lower:]')" in
                *gemma-4*)
                    model_name="$(basename -- "${candidate}")"
                    break
                    ;;
            esac
        done < <(list_model_candidates "${MODEL_DIR}")
    fi

    case "$(printf '%s' "${model_name}" | tr '[:upper:]' '[:lower:]')" in
        *gemma-4*) ;;
        *) return 0 ;;
    esac

    echo
    cat <<'EOF'
Gemma 4 compatibility helper is required for zipcode to run this model:
  1. Build/install llama-server in this terminal now (recommended)
  2. I already have a llama-server binary; ask me for the path
  3. Skip helper setup for now
EOF

    helper_choice="$(prompt_choice "Selection:" "$(default_choice_for_stdin "1" "3")")"
    case "${helper_choice}" in
        1)
            run_terminal_helper_build
            ;;
        2)
            helper_input="$(prompt_value "Local path to llama-server:" "")"
            if [ -n "${helper_input}" ]; then
                require_file "${helper_input}" "llama-server binary"
                LLAMA_SERVER_SOURCE="${helper_input}"
            else
                echo "No llama-server path provided; skipping helper setup."
            fi
            ;;
        3)
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
        ensure_build_deps_available
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

SETUP_MODEL="$(resolve_setup_model || true)"
guided_helper_setup "${SETUP_MODEL}"

if [ -n "${LLAMA_SERVER_SOURCE}" ]; then
    "${INSTALL_HELPER}" "${LLAMA_SERVER_SOURCE}" "${INSTALL_DIR}" >/dev/null
fi

LLAMA_SERVER_INSTALLED=""
if [ -x "${INSTALL_BIN_DIR}/llama-server" ]; then
    LLAMA_SERVER_INSTALLED="${INSTALL_BIN_DIR}/llama-server"
fi

MODEL_REQUIRES_HELPER=0
if [ -n "${SETUP_MODEL}" ]; then
    case "$(basename -- "${SETUP_MODEL}" | tr '[:upper:]' '[:lower:]')" in
        *gemma-4*) MODEL_REQUIRES_HELPER=1 ;;
    esac
fi

write_default_config "${CONFIG_FILE}" "${MODEL_DIR}" "${SETUP_MODEL}" "${LLAMA_SERVER_INSTALLED}"
write_setup_env "${SETUP_ENV}" "${INSTALL_BIN_DIR}" "${USER_BIN_DIR}" "${LLAMA_SERVER_INSTALLED}"
USER_LAUNCHER_INSTALLED="$(ensure_user_launcher "${INSTALL_BIN_DIR}/zipcode" "${USER_BIN_DIR}")"

{
    echo "Next steps:"
    echo "  • zipcode installs into ${INSTALL_DIR}"
    echo "  • launcher installed at ${USER_LAUNCHER_INSTALLED}"
    if [ -n "${SETUP_MODEL}" ]; then
        echo "  • selected model: ${SETUP_MODEL}"
    fi
    if ! path_contains_dir "${USER_BIN_DIR}"; then
        echo "  • ${USER_BIN_DIR} is not on PATH in this shell yet"
        echo "    Run: source \"${SETUP_ENV}\""
    fi
    if [ -z "${SETUP_MODEL}" ]; then
        echo "  • add a .gguf model with: ./install.sh --model /path/to/model.gguf"
        echo "    or download one with: ./scripts/download_model.sh --yes --dir ${MODEL_DIR}"
    fi
    if [ ! -f "${MODEL_DIR}/tokenizer.json" ]; then
        echo "  • add tokenizer.json with: ./install.sh --tokenizer /path/to/tokenizer.json"
        echo "    or let ./scripts/download_model.sh fetch it into ${MODEL_DIR}"
    fi
    if [ "${MODEL_REQUIRES_HELPER}" -eq 1 ] && [ -z "${LLAMA_SERVER_INSTALLED}" ]; then
        echo "  • install the required Gemma 4 helper with: ./scripts/build_llama_server.sh ${INSTALL_DIR}"
        echo "    or provide an existing binary with: ./install.sh --llama-server /path/to/llama-server"
    elif [ -z "${LLAMA_SERVER_INSTALLED}" ]; then
        echo "  • optional helper: ./install.sh --llama-server /path/to/llama-server"
    fi
} > "${NEXT_STEPS_FILE}"

if [ "${SKIP_SETUP}" -eq 1 ]; then
    echo "Skipped zipcode setup (--skip-setup)."
elif [ -n "${SETUP_MODEL}" ]; then
    echo
    echo "Running zipcode setup --skip-smoke..."
    "${INSTALL_BIN_DIR}/zipcode" setup --skip-smoke --model "${SETUP_MODEL}"
else
    echo
    echo "Skipping zipcode setup for now because no usable .gguf model was selected yet."
fi

echo
echo "zipcode doctor"
"${INSTALL_BIN_DIR}/zipcode" doctor || true

echo
cat "${NEXT_STEPS_FILE}"
