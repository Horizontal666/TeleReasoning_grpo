#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

DEFAULT_MODEL="${PROJECT_ROOT}/model/Qwen2.5-7B-Instruct"
DEFAULT_OUT_DIR="${SCRIPT_DIR}/results/gsm8k_$(date +%Y%m%d_%H%M%S)"
DEFAULT_DATASET="${PROJECT_ROOT}/data/datasets/gsm8k/test.parquet"
DEFAULT_API_BASE="${VLLM_API_BASE:-}"
DEFAULT_API_KEY="${OPENAI_API_KEY:-EMPTY}"
DEFAULT_ATTEMPTS="${EVAL_ATTEMPTS:-1}"

MODEL="${DEFAULT_MODEL}"
OUT_DIR="${DEFAULT_OUT_DIR}"
DATASET="${DEFAULT_DATASET}"
GPU_SELECTION=""
API_BASE="${DEFAULT_API_BASE}"
API_KEY="${DEFAULT_API_KEY}"
ATTEMPTS="${DEFAULT_ATTEMPTS}"

if [[ $# -gt 0 ]]; then
    MODEL="$1"
    shift
fi

if [[ $# -gt 0 ]]; then
    OUT_DIR="$1"
    shift
fi

if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    DATASET="$1"
    shift
fi

if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    GPU_SELECTION="$1"
    shift
fi

PASS_ARGS=("$@")

has_flag() {
    local flag="$1"
    for ((i = 0; i < ${#PASS_ARGS[@]}; ++i)); do
        if [[ "${PASS_ARGS[i]}" == "${flag}" ]]; then
            return 0
        fi
    done
    return 1
}

append_flag_if_missing() {
    local flag="$1"
    local value="$2"
    if [[ -n "${value}" ]] && ! has_flag "${flag}"; then
        PASS_ARGS+=("${flag}" "${value}")
    fi
}

append_switch_if_missing() {
    local flag="$1"
    if ! has_flag "${flag}"; then
        PASS_ARGS+=("${flag}")
    fi
}

mkdir -p "${OUT_DIR}"

if [[ -n "${GPU_SELECTION}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_SELECTION}"
fi

if [[ -n "${GPU_SELECTION}" && "${GPU_SELECTION}" == *,* ]]; then
    has_device_map=false
    for arg in "${PASS_ARGS[@]}"; do
        if [[ "${arg}" == "--device-map" ]]; then
            has_device_map=true
            break
        fi
    done
    if [[ "${has_device_map}" == false ]]; then
        PASS_ARGS+=("--device-map" "auto")
    fi
fi

append_flag_if_missing "--api-base" "${API_BASE}"
append_flag_if_missing "--api-key" "${API_KEY}"
append_flag_if_missing "--max-attempts" "${ATTEMPTS}"
append_switch_if_missing "--apply-chat-template"

set -x
python "${SCRIPT_DIR}/eval_llm.py" \
    --model "${MODEL}" \
    --out-dir "${OUT_DIR}" \
    --dataset "${DATASET}" \
    "${PASS_ARGS[@]}"
