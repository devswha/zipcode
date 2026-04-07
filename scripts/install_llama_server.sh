#!/bin/bash
# scripts/install_llama_server.sh — Install a llama-server binary into ~/.zipcode/bin
set -euo pipefail

INSTALL_DIR="${2:-${HOME}/.zipcode}"
INSTALL_BIN_DIR="${INSTALL_DIR}/bin"
INSTALL_LIB_DIR="${INSTALL_DIR}/lib/llama-server"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 <path-to-llama-server> [install-dir]" >&2
}

resolve_source() {
    local requested="${1:-}"
    if [ -n "${requested}" ]; then
        if [ -f "${requested}" ]; then
            printf '%s\n' "${requested}"
            return 0
        fi

        echo "Error: llama-server binary not found: ${requested}" >&2
        exit 1
    fi

    local candidates=(
        "${SCRIPT_DIR}/llama-server"
        "${SCRIPT_DIR}/bin/llama-server"
        "${PWD}/llama-server"
        "${PWD}/bin/llama-server"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -f "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    usage
    echo "Error: no llama-server binary was provided or detected." >&2
    exit 1
}

SOURCE_BIN="$(resolve_source "${1:-}")"
TARGET_BIN="${INSTALL_BIN_DIR}/llama-server"
TARGET_REAL_BIN="${INSTALL_LIB_DIR}/llama-server-real"

copy_runtime_libs() {
    local source_bin="$1"
    local root1 root2 root3 root
    local found=0

    root1="$(dirname -- "${source_bin}")"
    root2="$(dirname -- "${root1}")"
    root3="$(dirname -- "${root2}")"

    mkdir -p "${INSTALL_LIB_DIR}"

    for root in "${root1}" "${root2}" "${root3}"; do
        [ -d "${root}" ] || continue
        while IFS= read -r lib_path; do
            [ -n "${lib_path}" ] || continue
            cp -L "${lib_path}" "${INSTALL_LIB_DIR}/$(basename -- "${lib_path}")"
            found=1
        done < <(find "${root}" -maxdepth 3 -type f \( \
            -name 'libllama.so*' -o \
            -name 'libggml.so*' -o \
            -name 'libggml-*.so*' -o \
            -name 'libmtmd.so*' \
        \) | sort -u)
    done

    return "${found}"
}

write_wrapper() {
    cat > "${TARGET_BIN}" <<EOF
#!/bin/sh
DIR="\$(cd -- "\$(dirname -- "\$0")" && pwd)"
LIB_DIR="\${DIR}/../lib/llama-server"
BIN="\${LIB_DIR}/llama-server-real"
if [ -d "\${LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="\${LIB_DIR}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
fi
exec "\${BIN}" "\$@"
EOF
    chmod 755 "${TARGET_BIN}"
}

mkdir -p "${INSTALL_BIN_DIR}" "${INSTALL_LIB_DIR}"
cp -L "${SOURCE_BIN}" "${TARGET_REAL_BIN}"
chmod 755 "${TARGET_REAL_BIN}"
copy_runtime_libs "${SOURCE_BIN}" >/dev/null || true
write_wrapper

echo "Installed llama-server to ${TARGET_BIN}"
