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

normalize_source_bin() {
    local source_bin="$1"
    local sibling_real=""

    sibling_real="$(cd -- "$(dirname -- "${source_bin}")" && pwd)/../lib/llama-server/llama-server-real"
    if [ -f "${sibling_real}" ] && grep -q 'BIN="${LIB_DIR}/llama-server-real"' "${source_bin}" 2>/dev/null; then
        printf '%s
' "${sibling_real}"
        return 0
    fi

    printf '%s
' "${source_bin}"
}

SOURCE_BIN="$(normalize_source_bin "${SOURCE_BIN}")"

copy_runtime_libs() {
    local source_bin="$1"
    local root1 root2 root3 root
    local found=0
    local copied_lib=""
    local soname=""
    local fallback_soname=""

    root1="$(dirname -- "${source_bin}")"
    root2="$(dirname -- "${root1}")"
    root3="$(dirname -- "${root2}")"

    mkdir -p "${INSTALL_LIB_DIR}"

    for root in "${root1}" "${root2}" "${root3}"; do
        [ -d "${root}" ] || continue
        while IFS= read -r lib_path; do
            [ -n "${lib_path}" ] || continue
            copied_lib="${INSTALL_LIB_DIR}/$(basename -- "${lib_path}")"
            cp -L "${lib_path}" "${copied_lib}"
            soname="$(readelf -d "${copied_lib}" 2>/dev/null | awk -F'[][]' '/SONAME/ {print $2; exit}' || true)"
            if [ -z "${soname}" ]; then
                fallback_soname="$(basename -- "${copied_lib}" | sed -E 's/(.*\.so\.[0-9]+).*/\1/')"
                if [ "${fallback_soname}" != "$(basename -- "${copied_lib}")" ]; then
                    soname="${fallback_soname}"
                fi
            fi
            if [ -n "${soname}" ] && [ "${soname}" != "$(basename -- "${copied_lib}")" ]; then
                ln -sf "$(basename -- "${copied_lib}")" "${INSTALL_LIB_DIR}/${soname}"
            fi
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
if [ -e "${TARGET_REAL_BIN}" ] && [ "${SOURCE_BIN}" -ef "${TARGET_REAL_BIN}" ]; then
    chmod 755 "${TARGET_REAL_BIN}"
else
    cp -L "${SOURCE_BIN}" "${TARGET_REAL_BIN}"
    chmod 755 "${TARGET_REAL_BIN}"
fi
copy_runtime_libs "${SOURCE_BIN}" >/dev/null || true
write_wrapper

echo "Installed llama-server to ${TARGET_BIN}"
