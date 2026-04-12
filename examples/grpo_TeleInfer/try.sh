#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# shellcheck source=/dev/null
. "${REPO_ROOT}/scripts/use_project_cache.sh"

python - << 'PY'
import os, ray
from vllm import envs

ray.init()

@ray.remote
def show_vllm_use_v1():
    import os
    from vllm import envs
    return {
        "os.environ[VLLM_USE_V1]": os.environ.get("VLLM_USE_V1"),
        "envs.VLLM_USE_V1": bool(envs.VLLM_USE_V1),
    }

print(ray.get(show_vllm_use_v1.remote()))
PY
