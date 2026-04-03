#!/bin/bash
# scripts/install.sh — Install zipcode from extracted ZIP
set -e

INSTALL_DIR="${HOME}/.zipcode"
BIN_DIR="/usr/local/bin"

echo "Installing zipcode..."

mkdir -p "${INSTALL_DIR}/models" "${INSTALL_DIR}/sessions"

if [ -f "./zipcode" ]; then
    cp ./zipcode "${INSTALL_DIR}/"
    chmod +x "${INSTALL_DIR}/zipcode"
else
    echo "Error: zipcode binary not found in current directory"
    exit 1
fi

# Try to symlink to PATH
if [ -w "${BIN_DIR}" ]; then
    ln -sf "${INSTALL_DIR}/zipcode" "${BIN_DIR}/zipcode"
    echo "Installed to ${BIN_DIR}/zipcode"
else
    echo "Cannot write to ${BIN_DIR}. Add ${INSTALL_DIR} to your PATH:"
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
fi

echo ""
echo "Done! Next steps:"
echo "  1. Place your .gguf model file in ${INSTALL_DIR}/models/"
echo "  2. Place the matching tokenizer.json alongside it"
echo "  3. Run: zipcode doctor"
