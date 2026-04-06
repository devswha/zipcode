#!/bin/bash
# scripts/download_model.sh — Download Gemma 4 GGUF model
set -euo pipefail

MODEL_DIR="${HOME}/.zipcode/models"
AUTO_YES=0
HF_REPO="google/gemma-4-27b-it-GGUF"
TOKENIZER_URL="https://huggingface.co/google/gemma-4-27b-it/raw/main/tokenizer.json"

usage() {
    cat <<EOF
Usage: ./scripts/download_model.sh [--dir PATH] [--yes]

Options:
  --dir PATH   Download into PATH instead of ~/.zipcode/models
  --yes        Skip the interactive confirmation prompt
  -h, --help   Show this help text
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dir)
            [ "$#" -gt 1 ] || {
                echo "Error: --dir requires a path" >&2
                exit 1
            }
            MODEL_DIR="$2"
            shift
            ;;
        --yes)
            AUTO_YES=1
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

mkdir -p "${MODEL_DIR}"

echo "This script helps you download a Gemma 4 GGUF model."
echo ""
echo "For air-gapped environments, download these files on an internet-connected machine:"
echo ""
echo "  1. Model:     https://huggingface.co/google/gemma-4-27b-it-GGUF"
echo "  2. Tokenizer: https://huggingface.co/google/gemma-4-27b-it/raw/main/tokenizer.json"
echo ""
echo "Then copy both files to: ${MODEL_DIR}/"
echo "After that, run ./install.sh (from a git clone) or zipcode setup --skip-smoke."
echo ""

download_tokenizer() {
    local target_path="${MODEL_DIR}/tokenizer.json"

    if [ -f "${target_path}" ]; then
        echo "Tokenizer already exists: ${target_path}"
        return 0
    fi

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${TOKENIZER_URL}" -o "${target_path}"
        return 0
    fi

    if command -v wget >/dev/null 2>&1; then
        wget -qO "${target_path}" "${TOKENIZER_URL}"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        python3 - <<PY
from urllib.request import urlopen
target = r"""${target_path}"""
url = r"""${TOKENIZER_URL}"""
with urlopen(url) as resp, open(target, "wb") as out:
    out.write(resp.read())
PY
        return 0
    fi

    echo "Could not download tokenizer.json automatically (need curl, wget, or python3)." >&2
    return 1
}

download_model_repo() {
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "${HF_REPO}" --local-dir "${MODEL_DIR}"
        return 0
    fi

    echo "huggingface-cli not found. Install with: pip install huggingface-hub" >&2
    return 1
}

if [ "${AUTO_YES}" -ne 1 ]; then
    read -r -p "Download now? (requires internet + huggingface-cli) [y/N] " REPLY
    if [[ ! "${REPLY}" =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

echo "Downloading model snapshot from ${HF_REPO}..."
download_model_repo
echo "Downloading tokenizer.json..."
download_tokenizer
echo "Done! Model assets saved to ${MODEL_DIR}"
