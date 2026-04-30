#!/bin/bash
# scripts/download_model.sh — Download a recommended Gemma 4 GGUF model
set -euo pipefail

MODEL_DIR="${HOME}/.zipcode/models"
AUTO_YES=0
MODEL_PRESET="e2b"

set_model_preset() {
    case "$1" in
        e2b)
            HF_REPO="ggml-org/gemma-4-E2B-it-GGUF"
            TOKENIZER_URL="https://huggingface.co/google/gemma-4-E2B-it/raw/main/tokenizer.json"
            MODEL_LABEL="Gemma 4 E2B IT"
            ;;
        31b)
            HF_REPO="ggml-org/gemma-4-31B-it-GGUF"
            TOKENIZER_URL="https://huggingface.co/google/gemma-4-31B-it/raw/main/tokenizer.json"
            MODEL_LABEL="Gemma 4 31B IT"
            ;;
        *)
            echo "Error: unknown preset: $1 (expected e2b or 31b)" >&2
            exit 1
            ;;
    esac
}

set_model_preset "${MODEL_PRESET}"

usage() {
    cat <<EOF
Usage: ./scripts/download_model.sh [--dir PATH] [--preset e2b|31b] [--yes]

Options:
  --dir PATH   Download into PATH instead of ~/.zipcode/models
  --preset     Choose the recommended model family (default: e2b)
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
        --preset)
            [ "$#" -gt 1 ] || {
                echo "Error: --preset requires a value" >&2
                exit 1
            }
            MODEL_PRESET="$2"
            set_model_preset "${MODEL_PRESET}"
            shift
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

echo "This script helps you download a recommended ${MODEL_LABEL} GGUF model."
echo ""
echo "For air-gapped environments, download these files on an internet-connected machine:"
echo ""
echo "  1. Model:     https://huggingface.co/${HF_REPO}"
echo "  2. Tokenizer: ${TOKENIZER_URL}"
echo ""
echo "Note: the model repository may require Hugging Face login / access approval."
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

# Ensure `hf` or `huggingface-cli` is available before attempting a
# Hugging Face download. Mirrors the cargo / build-deps bootstrap in
# install.sh: probe PATH first, also probe `~/.local/bin/` (where
# `pip install --user` lands), and fall back to prompting the user
# to install `huggingface_hub` via pipx or pip. Default Y interactive,
# n non-interactive.
ensure_hf_cli_available() {
    if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
        return 0
    fi

    # `pip install --user` puts CLI shims under ~/.local/bin which fresh
    # shells often don't have on PATH yet. Probe explicitly so an
    # already-installed huggingface_hub just works.
    if [ -x "${HOME}/.local/bin/hf" ] || [ -x "${HOME}/.local/bin/huggingface-cli" ]; then
        export PATH="${HOME}/.local/bin:${PATH}"
        if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
            return 0
        fi
    fi

    echo "Neither hf nor huggingface-cli was found on PATH." >&2

    local installer=""
    local install_cmd=""
    if command -v pipx >/dev/null 2>&1; then
        installer="pipx"
        install_cmd="pipx install huggingface_hub"
    elif command -v pip3 >/dev/null 2>&1; then
        installer="pip3"
        install_cmd="pip3 install --user huggingface_hub"
    elif command -v pip >/dev/null 2>&1; then
        installer="pip"
        install_cmd="pip install --user huggingface_hub"
    fi

    if [ -z "${install_cmd}" ]; then
        echo "Install pip3 or pipx, then run: pip install --user huggingface_hub" >&2
        return 1
    fi

    echo "" >&2
    echo "Suggested install (${installer}):" >&2
    echo "  ${install_cmd}" >&2

    local default
    if [ -t 0 ] && [ -t 2 ]; then
        default="Y"
    else
        default="n"
    fi
    local answer=""
    if [ "${AUTO_YES}" -eq 1 ]; then
        answer="${default}"
    else
        printf 'Run the command above now? (y/N) [%s] ' "${default}" >&2
        IFS= read -r answer || true
        answer="${answer:-${default}}"
    fi

    case "${answer}" in
        [Yy]|[Yy][Ee][Ss])
            echo "Running: ${install_cmd}" >&2
            sh -c "${install_cmd}" || {
                echo "huggingface_hub install failed; install manually and rerun." >&2
                return 1
            }
            # Refresh PATH so the new ~/.local/bin/hf shim is reachable.
            if [ -d "${HOME}/.local/bin" ]; then
                export PATH="${HOME}/.local/bin:${PATH}"
            fi
            if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
                return 0
            fi
            echo "huggingface_hub installed but no hf/huggingface-cli on PATH; open a new shell and rerun." >&2
            return 1
            ;;
        *)
            echo "Install huggingface_hub manually and rerun: ${install_cmd}" >&2
            return 1
            ;;
    esac
}

download_model_repo() {
    local output_file
    output_file="$(mktemp)"
    local status=0

    run_download_with_visibility() {
        if [ -t 1 ] && [ -t 2 ]; then
            "$@"
            return $?
        fi

        set +e
        "$@" >"${output_file}" 2>&1
        status=$?
        set -e

        cat "${output_file}"
        return "${status}"
    }

    if command -v hf >/dev/null 2>&1; then
        if run_download_with_visibility hf download "${HF_REPO}" --local-dir "${MODEL_DIR}"; then
            rm -f "${output_file}"
            return 0
        fi
    elif command -v huggingface-cli >/dev/null 2>&1; then
        if run_download_with_visibility huggingface-cli download "${HF_REPO}" --local-dir "${MODEL_DIR}"; then
            rm -f "${output_file}"
            return 0
        fi
    else
        rm -f "${output_file}"
        echo "Neither hf nor huggingface-cli was found. Install one with: pip install huggingface_hub" >&2
        return 1
    fi

    if [ ! -t 1 ] || [ ! -t 2 ]; then
        cat "${output_file}" >&2
    fi
    rm -f "${output_file}"

    cat >&2 <<EOF

Download failed. This model repository often requires Hugging Face login or access approval.
Try:
  hf auth login
  hf download ${HF_REPO} --local-dir "${MODEL_DIR}"

If you are air-gapped, download the model + tokenizer on another machine and copy them into ${MODEL_DIR}.
EOF
    return 1
}

cleanup_partial_download() {
    local incomplete_snapshot="${MODEL_DIR}/.cache"
    if [ -d "${incomplete_snapshot}" ]; then
        echo "Keeping partial Hugging Face cache in ${incomplete_snapshot} for resume support."
    fi
}

if [ "${AUTO_YES}" -ne 1 ]; then
    read -r -p "Download now? (requires internet + hf or huggingface-cli) [y/N] " REPLY
    if [[ ! "${REPLY}" =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

ensure_hf_cli_available || exit 1

echo "Downloading model snapshot from ${HF_REPO}..."
download_model_repo || {
    cleanup_partial_download
    exit 1
}
echo "Downloading tokenizer.json..."
download_tokenizer
echo "Done! Model assets saved to ${MODEL_DIR}"
