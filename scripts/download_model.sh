#!/bin/bash
# scripts/download_model.sh — Download Gemma 4 GGUF model
set -e

MODEL_DIR="${HOME}/.zipcode/models"
mkdir -p "${MODEL_DIR}"

echo "This script helps you download a Gemma 4 GGUF model."
echo ""
echo "For air-gapped environments, download these files on an internet-connected machine:"
echo ""
echo "  1. Model:     https://huggingface.co/google/gemma-4-27b-it-GGUF"
echo "  2. Tokenizer: https://huggingface.co/google/gemma-4-27b-it/raw/main/tokenizer.json"
echo ""
echo "Then copy both files to: ${MODEL_DIR}/"
echo ""

read -p "Download now? (requires internet + huggingface-cli) [y/N] " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v huggingface-cli &> /dev/null; then
        echo "Downloading model..."
        huggingface-cli download google/gemma-4-27b-it-GGUF --local-dir "${MODEL_DIR}"
        echo "Done! Model saved to ${MODEL_DIR}"
    else
        echo "huggingface-cli not found. Install with: pip install huggingface-hub"
        exit 1
    fi
fi
