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

