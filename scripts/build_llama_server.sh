#!/bin/bash
# scripts/build_llama_server.sh — Build and install llama-server into ~/.zipcode/bin
set -euo pipefail

INSTALL_DIR="${1:-${HOME}/.zipcode}"
REPO_URL="${ZIPCODE_LLAMA_SERVER_REPO_URL:-https://github.com/ggml-org/llama.cpp.git}"
GIT_REF="${ZIPCODE_LLAMA_SERVER_GIT_REF:-}"
WORK_ROOT="${ZIPCODE_LLAMA_SERVER_WORKDIR:-$(mktemp -d)}"
SRC_DIR="${WORK_ROOT}/llama.cpp"
BUILD_DIR="${SRC_DIR}/build"
JOBS="${ZIPCODE_LLAMA_SERVER_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
INSTALL_HELPER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/install_llama_server.sh"
cleanup_workdir=0

if [ -z "${ZIPCODE_LLAMA_SERVER_WORKDIR:-}" ]; then
    cleanup_workdir=1
fi

cleanup() {
    if [ "${cleanup_workdir}" -eq 1 ]; then
        rm -rf "${WORK_ROOT}"
    fi
}
trap cleanup EXIT

command -v git >/dev/null 2>&1 || {
    echo "Error: git is required to build llama-server." >&2
    exit 1
}
command -v cmake >/dev/null 2>&1 || {
    echo "Error: cmake is required to build llama-server." >&2
    exit 1
}
command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || {
    echo "Error: a C++ compiler is required to build llama-server." >&2
    exit 1
}

echo "Preparing llama.cpp source in ${WORK_ROOT}..."
rm -rf "${SRC_DIR}"
git clone --depth 1 "${REPO_URL}" "${SRC_DIR}"

if [ -n "${GIT_REF}" ]; then
    git -C "${SRC_DIR}" fetch --depth 1 origin "${GIT_REF}"
    git -C "${SRC_DIR}" checkout FETCH_HEAD
fi

echo "Configuring llama-server build..."

# Auto-detect CUDA (requires nvcc from CUDA Toolkit, not just the driver)
cuda_available=0
if command -v nvcc >/dev/null 2>&1; then
    cuda_available=1
    echo "CUDA Toolkit detected (nvcc found), enabling GPU support."
elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: NVIDIA GPU detected but CUDA Toolkit (nvcc) not found."
    echo "  Install CUDA Toolkit for GPU-accelerated builds:"
    echo "    sudo apt install nvidia-cuda-toolkit"
    echo "  Building CPU-only for now."
fi

cmake_args=(-S "${SRC_DIR}" -B "${BUILD_DIR}" -DLLAMA_BUILD_SERVER=ON)
if [ "${cuda_available}" -eq 1 ]; then
    cmake_args+=(-DGGML_CUDA=ON)
fi
if [ -n "${ZIPCODE_LLAMA_SERVER_CMAKE_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    extra_args=(${ZIPCODE_LLAMA_SERVER_CMAKE_ARGS})
    cmake_args+=("${extra_args[@]}")
fi
cmake "${cmake_args[@]}"

echo "Building llama-server (jobs=${JOBS})..."
cmake --build "${BUILD_DIR}" --config Release -j "${JOBS}" --target llama-server

LLAMA_SERVER_BIN="$(find "${BUILD_DIR}" -type f -name 'llama-server' -perm -u+x | head -n 1)"
[ -n "${LLAMA_SERVER_BIN}" ] || {
    echo "Error: build finished but llama-server binary was not found." >&2
    exit 1
}

"${INSTALL_HELPER}" "${LLAMA_SERVER_BIN}" "${INSTALL_DIR}" >/dev/null
echo "Installed llama-server to ${INSTALL_DIR}/bin/llama-server"
