# custom_reward.py

import math
import re

BOXED_RE = re.compile(r"\\boxed\{(.+?)\}")
MATH_ENV_RE = re.compile(r"^\$+|\\\(|\\\)|\)$")

def _strip_boxed(s: str) -> str:
    m = BOXED_RE.search(s)
    if m:
        return m.group(1)
    return s

def _cleanup_tex(s: str) -> str:
    s = s.strip()
    s = _strip_boxed(s)
    # 去掉最外层 $...$ 或 \( ... \) 之类
    s = re.sub(r"^\s*(?:\$\$?|\\\()\s*", "", s)
    s = re.sub(r"\s*(?:\$\$?|\\\))\s*$", "", s)
    # 去掉花括号包裹的一层
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s

def _try_parse_number(s: str):
    """尽量把字符串解析为数值（支持 a/b、sqrt()、√x），失败返回 None。"""
    s = s.strip()
    s = _cleanup_tex(s)
    # 去掉多余空格
    s = s.replace(" ", "")
    try:
        # 分数 a/b
        if "/" in s and all(part.strip() for part in s.split("/", 1)):
            a, b = s.split("/", 1)
            return float(a) / float(b)
        # 根号
        if s.startswith("√"):
            return math.sqrt(float(s[1:]))
        if s.lower().startswith("sqrt(") and s.endswith(")"):
            return math.sqrt(float(s[5:-1]))
        # 直接 float
        return float(s)
    except Exception:
        return None

def _normalize_expr(s: str) -> str:
    s = _cleanup_tex(s)
    # 去空格、小写；去掉无关括号
    s = s.replace(" ", "").lower()
    # 去掉外层括号
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return s

def _get_first_not_none(*candidates):
    for x in candidates:
        if x is not None:
            return x
    return None

def _extract_data_from_args_kwargs(args, kwargs):
    """
    兼容多种调用方式，尽量拿到 data/DataProto：
    - 位置参数优先：args[0]
    - 关键字参数里找：data / data_proto / batch
    """
    if args and args[0] is not None:
        return args[0]
    for k in ("data", "data_proto", "batch"):
        if k in kwargs and kwargs[k] is not None:
            return kwargs[k]
    return None

def _extract_non_tensor(batch_like):
    """
    从 DataProto 或 dict 中拿 non_tensor 部分。
    """
    if batch_like is None:
        return {}
    # DataProto: 有 non_tensor_batch
    nt = getattr(batch_like, "non_tensor_batch", None)
    if nt is not None:
        return nt
    # 如果本身就是 dict
    if isinstance(batch_like, dict):
        # 有的路径会把 non_tensor_batch 封在这个键里
        if "non_tensor_batch" in batch_like and isinstance(batch_like["non_tensor_batch"], dict):
            return batch_like["non_tensor_batch"]
        return batch_like
    return {}

def _extract_response(non_tensor):
    """
    模型输出可能在不同的 key：做个兜底查找。
    """
    # 常见命名
    resp_keys = [
        "response_str", "response", "generation", "output", "text",
        "pred", "answer", "model_answer", "chosen", "sample"
    ]
    for k in resp_keys:
        v = non_tensor.get(k)
        if isinstance(v, str) and v.strip():
            return v
        # 有些会是 list[str]
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            return v[0]
    # 有的会把多候选放在 "responses" 或 "generations"
    for k in ["responses", "generations"]:
        v = non_tensor.get(k)
        if isinstance(v, (list, tuple)) and v:
            # 取第一个字符串或其中的字段
            if isinstance(v[0], str):
                return v[0]
            if isinstance(v[0], dict):
                return _get_first_not_none(
                    v[0].get("text"),
                    v[0].get("response"),
                    v[0].get("generation"),
                )
    return None

def my_math_reward_fn(*args, **kwargs):
    """
    自定义奖励函数：支持 multiple_choice 和 fill_in_the_blank。
    兼容 VERL 不同调用路径：位置参数/关键字参数传入 data。
    允许额外 kwargs（例如 return_dict、tokenizer 等）无副作用。
    """
    eps = kwargs.get("eps", 1e-6)

    # 1) 拿到 data / DataProto
    data = _extract_data_from_args_kwargs(args, kwargs)

    # 2) 拿 non-tensor 部分
    non_tensor = _extract_non_tensor(data)

    # 3) reward 配置通常在 reward_model 节点
    reward_cfg = non_tensor.get("reward_model", non_tensor)
    style = reward_cfg.get("style") or non_tensor.get("style") or ""
    ground_truth = reward_cfg.get("ground_truth") or non_tensor.get("ground_truth") or ""

    # 4) 拿模型输出
    solution_str = _extract_response(non_tensor)

    # 5) 计算 reward
    reward = 0.0

    if style == "multiple_choice":
        # 只要选项字符串一致就给 1.0（可以再加更鲁棒的 A/B/C/D 提取）
        if solution_str and ground_truth:
            sol = solution_str.strip()
            gt = ground_truth.strip()
            # 从输出里抽末尾 "#### X" 的选项也可增强鲁棒性
            m = re.search(r"####\s*([A-D])\b", sol, re.IGNORECASE)
            if m:
                sol = m.group(1).upper()
                gt  = gt.upper()
                reward = 1.0 if sol == gt else 0.0
            else:
                reward = 1.0 if sol == gt else 0.0
        else:
            reward = 0.0

    elif style == "fill_in_the_blank":
        if solution_str and ground_truth:
            sol = solution_str.strip()
            gt  = ground_truth.strip()

            # 优先做表达式规范化的字符串精确匹配
            if _normalize_expr(sol) == _normalize_expr(gt):
                reward = 1.0
            else:
                # 再尝试数值比较
                sol_val = _try_parse_number(sol)
                gt_val  = _try_parse_number(gt)
                if sol_val is not None and gt_val is not None and abs(sol_val - gt_val) < eps:
                    reward = 1.0
                else:
                    reward = 0.0
        else:
            reward = 0.0

    else:
        # 未知样式，保守给 0
        reward = 0.0

    # 允许 reward manager 传多余的 kwargs（例如 return_dict），这里直接忽略
    return float(reward)
