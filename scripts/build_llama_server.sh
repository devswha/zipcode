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

detect_nvcc_binary() {
    local candidate=""

    if [ -n "${CUDACXX:-}" ] && [ -x "${CUDACXX}" ]; then
        printf '%s\n' "${CUDACXX}"
        return 0
    fi

    if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        printf '%s\n' "${CUDA_HOME}/bin/nvcc"
        return 0
    fi

    for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        if [ -x "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    command -v nvcc 2>/dev/null || return 1
}

detect_nvcc_release() {
    local nvcc_bin="${CUDACXX:-$(command -v nvcc 2>/dev/null || true)}"
    [ -n "${nvcc_bin}" ] || return 1
    "${nvcc_bin}" --version 2>/dev/null \
        | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' \
        | tail -n 1
}

compiler_major_version() {
    local compiler="$1"
    "${compiler}" -dumpfullversion -dumpversion 2>/dev/null | awk -F. 'NR==1 {print $1}'
}

args_contain_prefix() {
    local prefix="$1"
    shift
    local arg
    for arg in "$@"; do
        case "${arg}" in
            "${prefix}"*)
                return 0
                ;;
        esac
    done
    return 1
}

filter_cuda_args() {
    local arg
    for arg in "$@"; do
        case "${arg}" in
            -DGGML_CUDA=*|-DCMAKE_CUDA_HOST_COMPILER=*|-DCMAKE_CUDA_ARCHITECTURES=*|-DCMAKE_CUDA_FLAGS=*)
                ;;
            *)
                printf '%s\n' "${arg}"
                ;;
        esac
    done
}

detect_cuda_architectures() {
    if [ -n "${ZIPCODE_LLAMA_SERVER_CUDA_ARCHITECTURES:-}" ]; then
        printf '%s\n' "${ZIPCODE_LLAMA_SERVER_CUDA_ARCHITECTURES}"
        return 0
    fi

    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local compute_cap
    compute_cap="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '. '
    )"
    [[ "${compute_cap}" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "${compute_cap}"
}

run_cmake_build() {
    local -n env_ref=$1
    shift
    local -a cmake_config_args=("$@")

    rm -rf "${BUILD_DIR}"
    env "${env_ref[@]}" cmake "${cmake_config_args[@]}"
    env "${env_ref[@]}" cmake --build "${BUILD_DIR}" --config Release -j "${JOBS}" --target llama-server
}

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
NVCC_BIN="$(detect_nvcc_binary || true)"
if [ -n "${NVCC_BIN}" ]; then
    cuda_available=1
    export CUDACXX="${NVCC_BIN}"
    echo "CUDA Toolkit detected (${NVCC_BIN}), enabling GPU support."
elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: NVIDIA GPU detected but CUDA Toolkit (nvcc) not found."
    echo "  Install CUDA Toolkit for GPU-accelerated builds:"
    echo "    sudo apt install nvidia-cuda-toolkit"
    echo "  Building CPU-only for now."
fi

cmake_args=(-S "${SRC_DIR}" -B "${BUILD_DIR}" -DLLAMA_BUILD_SERVER=ON)
cmake_env=()
user_cmake_args=()
cpu_fallback_enabled=1

if [ -n "${ZIPCODE_LLAMA_SERVER_CMAKE_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    user_cmake_args=(${ZIPCODE_LLAMA_SERVER_CMAKE_ARGS})
fi

if [ "${cuda_available}" -eq 1 ]; then
    cmake_args+=(-DGGML_CUDA=ON)
    if ! args_contain_prefix "-DCMAKE_CUDA_FLAGS=" "${user_cmake_args[@]}"; then
        cmake_args+=(-DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets)
    fi
    if ! args_contain_prefix "-DCMAKE_CUDA_ARCHITECTURES=" "${user_cmake_args[@]}"; then
        cuda_architectures="$(detect_cuda_architectures || true)"
        if [ -n "${cuda_architectures:-}" ]; then
            echo "Detected CUDA compute capability ${cuda_architectures}; restricting build targets."
            cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}")
        fi
    fi

    read -r nvcc_major nvcc_minor <<<"$(detect_nvcc_release || true)"
    host_cxx="${CXX:-$(command -v c++ || command -v g++)}"
    host_major="$(compiler_major_version "${host_cxx}" || true)"
    if [ -n "${nvcc_major:-}" ] && [ "${nvcc_major}" -le 11 ] && [ -n "${host_major:-}" ] && [ "${host_major}" -ge 11 ]; then
        gcc10_bin="$(command -v gcc-10 || true)"
        gpp10_bin="$(command -v g++-10 || true)"
        if [ -z "${CC:-}" ] && [ -z "${CXX:-}" ] \
            && [ -n "${gcc10_bin}" ] && [ -n "${gpp10_bin}" ] \
            && ! args_contain_prefix "-DCMAKE_CUDA_HOST_COMPILER=" "${user_cmake_args[@]}"; then
            echo "CUDA ${nvcc_major}.${nvcc_minor} with GCC ${host_major} is often incompatible; using gcc-10/g++-10 as CUDA host compiler."
            cmake_env=(CC="${gcc10_bin}" CXX="${gpp10_bin}")
            cmake_args+=(-DCMAKE_CUDA_HOST_COMPILER="${gpp10_bin}")
        else
            echo "Warning: CUDA ${nvcc_major}.${nvcc_minor} with GCC ${host_major} may fail to compile llama-server."
            echo "  If GPU build fails, zipcode will retry CPU-only automatically."
            echo "  For GPU builds, install gcc-10/g++-10 and set CC/CXX or ZIPCODE_LLAMA_SERVER_CMAKE_ARGS=-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10"
        fi
    fi
fi
cmake_args+=("${user_cmake_args[@]}")

echo "Building llama-server (jobs=${JOBS})..."
if ! run_cmake_build cmake_env "${cmake_args[@]}"; then
    if [ "${cuda_available}" -eq 1 ] && [ "${cpu_fallback_enabled}" -eq 1 ]; then
        echo "Warning: CUDA llama-server build failed. Retrying CPU-only build."
        mapfile -t cpu_user_args < <(filter_cuda_args "${user_cmake_args[@]}")
        cpu_cmake_args=(-S "${SRC_DIR}" -B "${BUILD_DIR}" -DLLAMA_BUILD_SERVER=ON)
        cpu_cmake_args+=("${cpu_user_args[@]}")
        cmake_env=()
        run_cmake_build cmake_env "${cpu_cmake_args[@]}"
    else
        exit 1
    fi
fi

LLAMA_SERVER_BIN="$(find "${BUILD_DIR}" -type f -name 'llama-server' -perm -u+x | head -n 1)"
[ -n "${LLAMA_SERVER_BIN}" ] || {
    echo "Error: build finished but llama-server binary was not found." >&2
    exit 1
}

"${INSTALL_HELPER}" "${LLAMA_SERVER_BIN}" "${INSTALL_DIR}" >/dev/null
echo "Installed llama-server to ${INSTALL_DIR}/bin/llama-server"
