#!/usr/bin/env bash

# Fail fast on errors, unset vars, and pipeline issues.
set -euo pipefail

# Resolve script location and project root to build default paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Default model/out directory/dataset fallback to local assets.
DEFAULT_MODEL="${PROJECT_ROOT}/model/deepseek-math-7b-instruct"
DEFAULT_OUT_DIR="${SCRIPT_DIR}/results/$(date +%Y%m%d_%H%M%S)"
# DEFAULT_DATASET="${PROJECT_ROOT}/data/3GPP-TSG/3gpp_class_eval.json"
# DEFAULT_DATASET="${PROJECT_ROOT}/data/TeleQnA/TeleQnA.json"
# DEFAULT_DATASET="${PROJECT_ROOT}/data/TeleMath/test.json"
DEFAULT_DATASET="${PROJECT_ROOT}/data/TeleLogs/troubleshooting/test.json"
# DEFAULT_DATASET="${PROJECT_ROOT}/data/WirelessMathBench/Verl/wirelessMathBench_test.parquet"
# DEFAULT_DATASET="${PROJECT_ROOT}/data/WirelessMATHBench-XL/data/test-00000-of-00001.parquet"

DEFAULT_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:6005/v1}"
DEFAULT_API_KEY="${OPENAI_API_KEY:-EMPTY}"
DEFAULT_ATTEMPTS="${EVAL_ATTEMPTS:-3}"



# Allow overriding defaults via positional args.
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

# Third positional (if present) sets dataset unless caller already uses --dataset.
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    DATASET="$1"
    shift
fi

# Fourth positional (if present) toggles CUDA_VISIBLE_DEVICES for the session.
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

mkdir -p "${OUT_DIR}"

# Apply GPU selection if provided; leave existing env untouched otherwise.
if [[ -n "${GPU_SELECTION}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_SELECTION}"
fi

# Automatically enable device_map when multiple GPUs are exposed and user did not opt out.
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

# Forward remaining CLI options directly into the evaluator.
set -x
python "${SCRIPT_DIR}/eval_llm.py" \
    --model "${MODEL}" \
    --out-dir "${OUT_DIR}" \
    --dataset "${DATASET}" \
    "${PASS_ARGS[@]}"
