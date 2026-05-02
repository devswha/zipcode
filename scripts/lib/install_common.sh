#!/bin/bash
# Shared helpers for zipcode installation entrypoints.

zipcode_install_dir() {
    printf '%s\n' "${ZIPCODE_INSTALL_DIR:-${HOME}/.zipcode}"
}

zipcode_install_bin_dir() {
    printf '%s\n' "$(zipcode_install_dir)/bin"
}

zipcode_model_dir() {
    printf '%s\n' "$(zipcode_install_dir)/models"
}

zipcode_session_dir() {
    printf '%s\n' "$(zipcode_install_dir)/sessions"
}

zipcode_setup_env_path() {
    printf '%s\n' "$(zipcode_install_dir)/setup.env"
}

zipcode_next_steps_path() {
    printf '%s\n' "$(zipcode_install_dir)/NEXT_STEPS.txt"
}

zipcode_config_path() {
    printf '%s\n' "$(zipcode_install_dir)/config.json"
}

zipcode_user_bin_dir() {
    printf '%s\n' "${HOME}/.local/bin"
}

find_first_existing() {
    local candidate
    for candidate in "$@"; do
        if [ -e "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

copy_executable() {
    local source_path="$1"
    local target_path="$2"
    cp "${source_path}" "${target_path}"
    chmod 755 "${target_path}"
}

ensure_install_dirs() {
    mkdir -p \
        "$(zipcode_install_dir)" \
        "$(zipcode_install_bin_dir)" \
        "$(zipcode_model_dir)" \
        "$(zipcode_session_dir)"
}

json_escape() {
    local value="${1:-}"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    printf '%s' "${value}"
}

cuda_available() {
    if [ -n "${CUDA_PATH:-}" ] || [ -n "${CUDA_HOME:-}" ]; then
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        return 0
    fi

    local cuda_libs=(
        "/usr/lib/x86_64-linux-gnu/libcuda.so"
        "/usr/lib/libcuda.so"
        "/usr/local/cuda/lib64/libcuda.so"
        "/usr/local/cuda/lib/libcuda.so"
    )
    local lib=""
    for lib in "${cuda_libs[@]}"; do
        if [ -e "${lib}" ]; then
            return 0
        fi
    done

    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q "libcuda.so"; then
        return 0
    fi

    return 1
}

detect_total_vram_mb() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local out
    out="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -n1 | tr -d ' [:space:]')"
    [ -n "${out}" ] && [ "${out}" -gt 0 ] 2>/dev/null || return 1
    printf '%s\n' "${out}"
}

# Recommend the value zipcode should write into config.json's `gpu_layers`.
# Returns 999 when the model fits cleanly on the detected GPU (or the model
# size is unknown), 0 when the model + KV-cache headroom won't fit (so the
# user gets a working CPU fallback instead of a CUDA OOM at first run).
#
# Args:
#   $1 helper_path  — llama-server binary (probed for GPU backend support)
#   $2 model_path   — optional; path to the .gguf so size can be checked
recommended_gpu_layers() {
    local helper_path="${1:-}"
    local model_path="${2:-}"
    [ -n "${helper_path}" ] || return 1
    cuda_available || return 1
    helper_supports_gpu_defaults "${helper_path}" || return 1

    if [ -n "${model_path}" ] && [ -f "${model_path}" ]; then
        local model_bytes model_mb vram_mb
        model_bytes="$(stat -c '%s' "${model_path}" 2>/dev/null \
            || stat -f '%z' "${model_path}" 2>/dev/null || echo 0)"
        model_mb="$((model_bytes / 1024 / 1024))"
        if vram_mb="$(detect_total_vram_mb 2>/dev/null)" && [ "${vram_mb}" -gt 0 ]; then
            # Need ~3 GiB headroom for KV cache, activations, and ctx growth.
            if [ "$((model_mb + 3072))" -gt "${vram_mb}" ]; then
                printf '%s\n' "0"
                return 0
            fi
        fi
    fi

    printf '%s\n' "999"
}

recommended_flash_attention() {
    local helper_path="${1:-}"
    [ -n "${helper_path}" ] || return 1
    cuda_available || return 1
    helper_supports_gpu_defaults "${helper_path}" || return 1
    printf '%s\n' "true"
}

helper_supports_gpu_defaults() {
    local helper_path="${1:-}"
    [ -n "${helper_path}" ] || return 1
    [ -x "${helper_path}" ] || return 1
    command -v timeout >/dev/null 2>&1 || return 1

    local probe_output=""
    if ! probe_output="$(timeout 2 "${helper_path}" --list-devices 2>/dev/null)"; then
        return 1
    fi

    printf '%s' "${probe_output}" | grep -Eiq 'cuda|gpu|nvidia|metal|vulkan|hip|rocm'
}

write_default_config() {
    local config_file="${1:-$(zipcode_config_path)}"
    local model_dir="${2:-$(zipcode_model_dir)}"
    local model_path="${3:-}"
    local helper_path="${4:-}"
    local model_file=""
    local gpu_layers=""
    local flash_attention="false"
    local entries=()
    local last_index=0
    local i=0

    if [ -f "${config_file}" ]; then
        echo "Preserving existing config: ${config_file}"
        return
    fi

    if [ -n "${model_path}" ]; then
        model_file="$(basename -- "${model_path}")"
    fi
    gpu_layers="$(recommended_gpu_layers "${helper_path}" "${model_path}" || true)"
    flash_attention="$(recommended_flash_attention "${helper_path}" || printf '%s' "false")"

    entries+=("  \"model_dir\": \"$(json_escape "${model_dir}")\"")
    entries+=("  \"permission_mode\": \"workspace-write\"")
    if [ -n "${model_file}" ]; then
        entries+=("  \"model_file\": \"$(json_escape "${model_file}")\"")
    fi
    if [ -n "${helper_path}" ]; then
        entries+=("  \"llama_server_bin\": \"$(json_escape "${helper_path}")\"")
    fi
    if [ -n "${gpu_layers}" ]; then
        entries+=("  \"gpu_layers\": ${gpu_layers}")
    fi
    entries+=("  \"flash_attention\": ${flash_attention}")

    mkdir -p "$(dirname -- "${config_file}")"
    last_index=$((${#entries[@]} - 1))
    {
        echo "{"
        for i in "${!entries[@]}"; do
            if [ "${i}" -lt "${last_index}" ]; then
                printf '%s,\n' "${entries[$i]}"
            else
                printf '%s\n' "${entries[$i]}"
            fi
        done
        echo "}"
    } > "${config_file}"
    echo "Wrote default config: ${config_file}"
}

write_setup_env() {
    local setup_env="$1"
    local install_bin_dir="$2"
    local user_bin_dir="$3"
    local llama_server_path="${4:-}"

    mkdir -p "$(dirname -- "${setup_env}")"
    cat > "${setup_env}" <<EOF2
# Generated by zipcode install
export PATH="${install_bin_dir}:${user_bin_dir}:\$PATH"
EOF2

    if [ -n "${llama_server_path}" ]; then
        cat >> "${setup_env}" <<EOF2
export ZIPCODE_LLAMA_SERVER_BIN="${llama_server_path}"
EOF2
    fi
}

ensure_user_launcher() {
    local launcher_source="$1"
    local user_bin_dir="${2:-$(zipcode_user_bin_dir)}"
    mkdir -p "${user_bin_dir}"
    ln -sf "${launcher_source}" "${user_bin_dir}/zipcode"
    printf '%s\n' "${user_bin_dir}/zipcode"
}

path_contains_dir() {
    local dir="$1"
    case ":${PATH:-}:" in
        *":${dir}:"*) return 0 ;;
        *) return 1 ;;
    esac
}
