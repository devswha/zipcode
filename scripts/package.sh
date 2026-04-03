#!/bin/bash
# scripts/package.sh — Build release binary and create ZIP archive
set -e

VERSION=$(cargo metadata --no-deps --format-version 1 | python3 -c "import sys,json;print(json.load(sys.stdin)['packages'][0]['version'])")
ARCHIVE="zipcode-v${VERSION}-linux-x86_64-cuda"

echo "Building release binary..."
cargo build --release -p zipcode

echo "Creating archive..."
mkdir -p "dist/${ARCHIVE}"

cp target/release/zipcode "dist/${ARCHIVE}/"
cp scripts/install.sh "dist/${ARCHIVE}/"
cp scripts/download_model.sh "dist/${ARCHIVE}/"
cp README.md "dist/${ARCHIVE}/" 2>/dev/null || true

mkdir -p "dist/${ARCHIVE}/models"
echo "Place your .gguf model file here." > "dist/${ARCHIVE}/models/PLACE_MODEL_HERE.txt"

cd dist
zip -r "${ARCHIVE}.zip" "${ARCHIVE}/"
echo ""
echo "Archive created: dist/${ARCHIVE}.zip"
echo "Size: $(du -h "${ARCHIVE}.zip" | cut -f1)"
