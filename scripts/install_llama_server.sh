#!/bin/bash
# scripts/install_llama_server.sh — Install a llama-server binary into ~/.zipcode/bin
set -euo pipefail

INSTALL_DIR="${2:-${HOME}/.zipcode}"
INSTALL_BIN_DIR="${INSTALL_DIR}/bin"
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

mkdir -p "${INSTALL_BIN_DIR}"
cp "${SOURCE_BIN}" "${TARGET_BIN}"
chmod 755 "${TARGET_BIN}"

echo "Installed llama-server to ${TARGET_BIN}"
