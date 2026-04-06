#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BIN="${ROOT_DIR}/target/debug/zipcode"
REAL_MODEL_PATH="${ZIPCODE_TEST_MODEL_PATH:-${HOME}/.zipcode/models/gemma-4-e2b-it-Q8_0.gguf}"
REAL_LLAMA_SERVER_BIN="${ZIPCODE_LLAMA_SERVER_BIN:-}"
RUN_REAL_GEMMA4="${RUN_REAL_GEMMA4:-0}"

PASS_COUNT=0
FAIL_COUNT=0
TMP_DIRS=()

cleanup() {
  for dir in "${TMP_DIRS[@]}"; do
    [ -n "$dir" ] && rm -rf "$dir"
  done
}
trap cleanup EXIT

make_temp_home() {
  local dir
  dir="$(mktemp -d)"
  TMP_DIRS+=("$dir")
  printf '%s\n' "$dir"
}

build_bin_if_needed() {
  if [ ! -x "$BIN" ]; then
    echo "[build] compiling zipcode debug binary..."
    cargo build -p zipcode >/dev/null
  fi
}

expect_contains() {
  local haystack="$1"
  local needle="$2"
  local label="$3"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "[FAIL] $label"
    echo "  expected substring: $needle"
    echo "  actual output:"
    printf '%s\n' "$haystack"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    return 1
  fi
}

run_case() {
  local name="$1"
  shift
  echo
  echo "===== $name ====="
  local output
  if ! output="$($@ 2>&1)"; then
    echo "[FAIL] $name"
    printf '%s\n' "$output"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    return 1
  fi
  printf '%s\n' "$output"
  PASS_COUNT=$((PASS_COUNT + 1))
}

run_case_expect() {
  local name="$1"
  local expect1="$2"
  local expect2="${3:-}"
  shift 3
  echo
  echo "===== $name ====="
  local output
  if ! output="$($@ 2>&1)"; then
    echo "[FAIL] $name"
    printf '%s\n' "$output"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    return 1
  fi
  printf '%s\n' "$output"
  expect_contains "$output" "$expect1" "$name / check 1" || return 1
  if [ -n "$expect2" ]; then
    expect_contains "$output" "$expect2" "$name / check 2" || return 1
  fi
  PASS_COUNT=$((PASS_COUNT + 1))
}

run_empty_home_case() {
  local home_dir
  home_dir="$(make_temp_home)"
  run_case_expect \
    "bare zipcode on empty HOME" \
    "Setup needed before zipcode can start." \
    "Copy a .gguf AI model" \
    env -u ZIPCODE_LLAMA_SERVER_BIN -u LLAMA_SERVER_BIN HOME="$home_dir" "$BIN"
}

run_broken_config_case() {
  local home_dir
  home_dir="$(make_temp_home)"
  mkdir -p "$home_dir/.zipcode"
  cat > "$home_dir/.zipcode/config.json" <<'JSON'
{
  "model_dir": "/tmp/zipcode-missing-model-dir",
  "model_file": "missing.gguf",
  "permission_mode": "workspace-write"
}
JSON
  run_case_expect \
    "bare zipcode with broken saved model path" \
    "Repair needed before zipcode can start." \
    "Configured model path not found" \
    env -u ZIPCODE_LLAMA_SERVER_BIN -u LLAMA_SERVER_BIN HOME="$home_dir" "$BIN"
}

run_setup_then_doctor_case() {
  local home_dir model_dir bin_dir isolated_path setup_output doctor_output
  home_dir="$(make_temp_home)"
  model_dir="$home_dir/.zipcode/models"
  bin_dir="$home_dir/.zipcode/bin"
  isolated_path="$home_dir/empty-path"
  mkdir -p "$model_dir" "$bin_dir" "$isolated_path"
  printf 'gguf' > "$model_dir/gemma-4-test.gguf"
  printf '{}' > "$model_dir/tokenizer.json"
  cat > "$bin_dir/llama-server" <<'SCRIPT'
#!/bin/sh
echo fake llama-server
SCRIPT
  chmod +x "$bin_dir/llama-server"

  echo
  echo "===== setup --skip-smoke then doctor ====="
  setup_output="$(env -u ZIPCODE_LLAMA_SERVER_BIN -u LLAMA_SERVER_BIN HOME="$home_dir" PATH="$isolated_path" "$BIN" setup --skip-smoke 2>&1)"
  printf '%s\n' "$setup_output"
  expect_contains "$setup_output" "Status:         Ready" "setup readiness" || return 1
  expect_contains "$setup_output" "Smoke: skipped (--skip-smoke)" "setup skip smoke" || return 1

  doctor_output="$(env -u ZIPCODE_LLAMA_SERVER_BIN -u LLAMA_SERVER_BIN HOME="$home_dir" PATH="$isolated_path" "$BIN" doctor 2>&1)"
  printf '%s\n' "$doctor_output"
  expect_contains "$doctor_output" "Status: Ready" "doctor readiness" || return 1
  expect_contains "$doctor_output" "$model_dir/gemma-4-test.gguf" "doctor model path" || return 1
  expect_contains "$doctor_output" "$bin_dir/llama-server" "doctor helper path" || return 1

  PASS_COUNT=$((PASS_COUNT + 1))
}

run_empty_env_case() {
  local home_dir
  home_dir="$(make_temp_home)"
  run_case_expect \
    "empty helper env vars do not force repair" \
    "Setup needed before zipcode can start." \
    "Copy a .gguf AI model" \
    env HOME="$home_dir" ZIPCODE_LLAMA_SERVER_BIN= LLAMA_SERVER_BIN= "$BIN"
}

run_real_gemma4_case() {
  if [ "$RUN_REAL_GEMMA4" != "1" ]; then
    echo
    echo "===== real Gemma 4 prompt ====="
    echo "[SKIP] set RUN_REAL_GEMMA4=1 to enable this check"
    return 0
  fi

  if [ -z "$REAL_LLAMA_SERVER_BIN" ]; then
    echo
    echo "===== real Gemma 4 prompt ====="
    echo "[SKIP] ZIPCODE_LLAMA_SERVER_BIN is not set"
    return 0
  fi

  if [ ! -x "$REAL_LLAMA_SERVER_BIN" ]; then
    echo
    echo "===== real Gemma 4 prompt ====="
    echo "[SKIP] ZIPCODE_LLAMA_SERVER_BIN is not executable: $REAL_LLAMA_SERVER_BIN"
    return 0
  fi

  if [ ! -f "$REAL_MODEL_PATH" ]; then
    echo
    echo "===== real Gemma 4 prompt ====="
    echo "[SKIP] model not found: $REAL_MODEL_PATH"
    return 0
  fi

  local output
  echo
  echo "===== real Gemma 4 prompt ====="
  output="$(ZIPCODE_LLAMA_SERVER_BIN="$REAL_LLAMA_SERVER_BIN" "$BIN" --backend llama-cpp --model "$REAL_MODEL_PATH" prompt 'Say exactly: hello' 2>&1)"
  printf '%s\n' "$output"
  expect_contains "$output" "hello" "real Gemma 4 output" || return 1
  PASS_COUNT=$((PASS_COUNT + 1))
}

main() {
  build_bin_if_needed
  run_empty_home_case
  run_broken_config_case
  run_setup_then_doctor_case
  run_empty_env_case
  run_real_gemma4_case

  echo
  echo "===== summary ====="
  echo "PASS: $PASS_COUNT"
  echo "FAIL: $FAIL_COUNT"

  if [ "$FAIL_COUNT" -ne 0 ]; then
    exit 1
  fi
}

main "$@"
