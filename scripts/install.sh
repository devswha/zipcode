#!/bin/bash
# scripts/install.sh — Install zipcode from an extracted bundle
set -euo pipefail

SYSTEM_BIN_DIR="/usr/local/bin"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON_LIB="${SCRIPT_DIR}/lib/install_common.sh"
if [ ! -f "${COMMON_LIB}" ]; then
    COMMON_LIB="${SCRIPT_DIR}/scripts/lib/install_common.sh"
fi
source "${COMMON_LIB}"

INSTALL_DIR="$(zipcode_install_dir)"
INSTALL_BIN_DIR="$(zipcode_install_bin_dir)"
MODEL_DIR="$(zipcode_model_dir)"
SETUP_ENV="$(zipcode_setup_env_path)"
NEXT_STEPS_FILE="$(zipcode_next_steps_path)"
CONFIG_FILE="$(zipcode_config_path)"
USER_BIN_DIR="$(zipcode_user_bin_dir)"

write_next_steps() {
    local llama_server_path="$1"
    local helper_path="$2"
    local user_launcher_path="$3"
    local system_launcher_path="$4"

    {
        echo "Next steps:"
        echo "  1. Copy your model into ${MODEL_DIR}:"
        echo "     cp /path/to/model.gguf \"${MODEL_DIR}/\""
        echo "  2. Copy the matching tokenizer into ${MODEL_DIR}:"
        echo "     cp /path/to/tokenizer.json \"${MODEL_DIR}/\""
        echo "  3. Start zipcode:"
        echo "     zipcode"
        echo "  4. If first run says setup or repair is needed:"
        echo "     zipcode doctor"
        echo "     zipcode setup --skip-smoke"
        echo ""
        echo "Installed launchers:"
        echo "  • ${user_launcher_path}"
        if [ -n "${system_launcher_path}" ]; then
            echo "  • ${system_launcher_path}"
        fi

        if ! path_contains_dir "${USER_BIN_DIR}"; then
            echo ""
            echo "${USER_BIN_DIR} is not on PATH in this shell."
            echo "Either run:"
            echo "  source \"${SETUP_ENV}\""
            echo "or add this to your shell profile:"
            echo "  export PATH=\"${USER_BIN_DIR}:\$PATH\""
        fi

        if [ -n "${llama_server_path}" ]; then
            echo ""
            echo "Bundled llama-server installed at ${llama_server_path}."
            echo "The generated ${SETUP_ENV} file already exports ZIPCODE_LLAMA_SERVER_BIN."
            echo "First run and repair flows will pick it up automatically."
        elif [ -n "${helper_path}" ]; then
            echo ""
            echo "No bundled llama-server was found."
            echo "If first run later says Gemma 4 needs the fallback server, install it with:"
            echo "  \"${helper_path}\" /path/to/llama-server \"${INSTALL_DIR}\""
        else
            echo ""
            echo "No bundled llama-server was found."
            echo "If first run later says Gemma 4 needs the fallback server, set:"
            echo "  export ZIPCODE_LLAMA_SERVER_BIN=/path/to/llama-server"
        fi
    } > "${NEXT_STEPS_FILE}"
}

echo "Installing zipcode into ${INSTALL_DIR}..."

ensure_install_dirs

ZIPCODE_SOURCE="$(find_first_existing \
    "${SCRIPT_DIR}/zipcode" \
    "${PWD}/zipcode")" || {
    echo "Error: zipcode binary not found next to install.sh or in the current directory" >&2
    exit 1
}

copy_executable "${ZIPCODE_SOURCE}" "${INSTALL_DIR}/zipcode"
ln -sf "../zipcode" "${INSTALL_BIN_DIR}/zipcode"

HELPER_SOURCE="$(find_first_existing \
    "${SCRIPT_DIR}/install_llama_server.sh" \
    "${SCRIPT_DIR}/scripts/install_llama_server.sh" \
    "${PWD}/install_llama_server.sh" || true)"
HELPER_INSTALLED=""
if [ -n "${HELPER_SOURCE}" ]; then
    HELPER_INSTALLED="${INSTALL_DIR}/install_llama_server.sh"
    copy_executable "${HELPER_SOURCE}" "${HELPER_INSTALLED}"
fi

LLAMA_SERVER_SOURCE="$(find_first_existing \
    "${SCRIPT_DIR}/llama-server" \
    "${SCRIPT_DIR}/bin/llama-server" \
    "${PWD}/llama-server" \
    "${PWD}/bin/llama-server" || true)"
LLAMA_SERVER_INSTALLED=""
if [ -n "${LLAMA_SERVER_SOURCE}" ]; then
    echo "Detected bundled llama-server: ${LLAMA_SERVER_SOURCE}"
    if [ -n "${HELPER_INSTALLED}" ]; then
        "${HELPER_INSTALLED}" "${LLAMA_SERVER_SOURCE}" "${INSTALL_DIR}" >/dev/null
    else
        LLAMA_SERVER_INSTALLED="${INSTALL_BIN_DIR}/llama-server"
        copy_executable "${LLAMA_SERVER_SOURCE}" "${LLAMA_SERVER_INSTALLED}"
    fi
    LLAMA_SERVER_INSTALLED="${INSTALL_BIN_DIR}/llama-server"
fi

write_default_config
write_setup_env "${SETUP_ENV}" "${INSTALL_BIN_DIR}" "${USER_BIN_DIR}" "${LLAMA_SERVER_INSTALLED}"
USER_LAUNCHER_INSTALLED="$(ensure_user_launcher "${INSTALL_BIN_DIR}/zipcode" "${USER_BIN_DIR}")"
SYSTEM_LAUNCHER_INSTALLED=""

if [ "${ZIPCODE_INSTALL_SKIP_SYSTEM_BIN:-0}" != "1" ] && [ -d "${SYSTEM_BIN_DIR}" ] && [ -w "${SYSTEM_BIN_DIR}" ]; then
    ln -sf "${INSTALL_BIN_DIR}/zipcode" "${SYSTEM_BIN_DIR}/zipcode"
    echo "Installed zipcode into ${SYSTEM_BIN_DIR}/zipcode"
    SYSTEM_LAUNCHER_INSTALLED="${SYSTEM_BIN_DIR}/zipcode"
fi

write_next_steps \
    "${LLAMA_SERVER_INSTALLED}" \
    "${HELPER_INSTALLED}" \
    "${USER_LAUNCHER_INSTALLED}" \
    "${SYSTEM_LAUNCHER_INSTALLED}"

echo
cat "${NEXT_STEPS_FILE}"
