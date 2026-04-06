#!/bin/bash
# scripts/package.sh — Build release binary and create ZIP archive
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/package.sh [--with-llama-server[=PATH]]

Options:
  --with-llama-server         Bundle llama-server from ZIPCODE_LLAMA_SERVER_BIN,
                              LLAMA_SERVER_BIN, or PATH so Gemma 4 first-run
                              and recovery stay offline.
  --with-llama-server=PATH    Bundle the llama-server binary at PATH.
  -h, --help                  Show this help text.
EOF
}

resolve_llama_server_bin() {
    local candidate="${1:-}"

    if [ -n "${candidate}" ]; then
        if [ ! -f "${candidate}" ]; then
            echo "Error: llama-server binary not found at ${candidate}" >&2
            exit 1
        fi
        if [ ! -x "${candidate}" ]; then
            echo "Error: llama-server binary at ${candidate} is not executable" >&2
            exit 1
        fi
        printf '%s\n' "${candidate}"
        return
    fi

    for env_var in ZIPCODE_LLAMA_SERVER_BIN LLAMA_SERVER_BIN; do
        local env_value="${!env_var:-}"
        if [ -n "${env_value}" ]; then
            if [ ! -f "${env_value}" ]; then
                echo "Error: ${env_var} points to a missing file: ${env_value}" >&2
                exit 1
            fi
            if [ ! -x "${env_value}" ]; then
                echo "Error: ${env_var} points to a non-executable file: ${env_value}" >&2
                exit 1
            fi
            printf '%s\n' "${env_value}"
            return
        fi
    done

    if command -v llama-server >/dev/null 2>&1; then
        command -v llama-server
        return
    fi

    cat >&2 <<'EOF'
Error: --with-llama-server could not resolve a llama-server binary.
Set ZIPCODE_LLAMA_SERVER_BIN, set LLAMA_SERVER_BIN, put llama-server on PATH,
or pass --with-llama-server=/path/to/llama-server.
EOF
    exit 1
}

WITH_LLAMA_SERVER=0
LLAMA_SERVER_OVERRIDE=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --with-llama-server)
            WITH_LLAMA_SERVER=1
            if [ "$#" -gt 1 ] && [[ "$2" != -* ]]; then
                LLAMA_SERVER_OVERRIDE="$2"
                shift
            fi
            ;;
        --with-llama-server=*)
            WITH_LLAMA_SERVER=1
            LLAMA_SERVER_OVERRIDE="${1#*=}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

VERSION=$(cargo metadata --no-deps --format-version 1 | python3 -c "import sys,json;print(json.load(sys.stdin)['packages'][0]['version'])")
ARCHIVE="zipcode-v${VERSION}-linux-x86_64-cuda"
ARCHIVE_DIR="dist/${ARCHIVE}"
ARCHIVE_ZIP="dist/${ARCHIVE}.zip"

echo "Building release binary..."
cargo build --release -p zipcode

echo "Creating archive..."
rm -rf "${ARCHIVE_DIR}"
rm -f "${ARCHIVE_ZIP}"
mkdir -p "${ARCHIVE_DIR}"
mkdir -p "${ARCHIVE_DIR}/scripts/lib"

cp target/release/zipcode "${ARCHIVE_DIR}/"
cp scripts/install.sh "${ARCHIVE_DIR}/"
cp scripts/install_llama_server.sh "${ARCHIVE_DIR}/"
cp scripts/download_model.sh "${ARCHIVE_DIR}/"
cp scripts/lib/install_common.sh "${ARCHIVE_DIR}/scripts/lib/"
cp README.md "${ARCHIVE_DIR}/" 2>/dev/null || true

if [ "${WITH_LLAMA_SERVER}" -eq 1 ]; then
    RESOLVED_LLAMA_SERVER_BIN="$(resolve_llama_server_bin "${LLAMA_SERVER_OVERRIDE}")"
    cp "${RESOLVED_LLAMA_SERVER_BIN}" "${ARCHIVE_DIR}/llama-server"
    chmod +x "${ARCHIVE_DIR}/llama-server"
    echo "Bundled llama-server from ${RESOLVED_LLAMA_SERVER_BIN}"
fi

mkdir -p "${ARCHIVE_DIR}/models"
echo "Place your .gguf model file here." > "${ARCHIVE_DIR}/models/PLACE_MODEL_HERE.txt"
echo "Place tokenizer.json next to the model file here." \
    > "${ARCHIVE_DIR}/models/PLACE_TOKENIZER_HERE.txt"

(
    cd dist
    zip -r "${ARCHIVE}.zip" "${ARCHIVE}/"
)

echo ""
echo "Archive created: ${ARCHIVE_ZIP}"
echo "Size: $(du -h "${ARCHIVE_ZIP}" | cut -f1)"
echo "First run after install: zipcode"
