# verl/utils/reward_score/telemath.py

import math
import re
from typing import Optional

# 匹配整数 / 小数 / 科学计数法
NUMBER_REGEX = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _strip_think(text: str) -> str:
    """去掉 Qwen3 思考模式中的 <think>...</think> 块。"""
    if not text:
        return ""
    return re.sub(
        r"<think>.*?</think>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _extract_answer_segment(full_text: str) -> Optional[str]:
    """
    从完整模型输出中抽取“答案片段”：
      1. 若存在 '####'，取最后一个 '####' 之后的部分；
      2. 然后取第一个非空行；
      3. 若没有 '####'，退化为整个文本的首个非空行。
    注意：这里不再做 <think> 处理，与 eval 脚本中的旧版 extract_answer_segment 保持一致。
    """
    if not full_text:
        return None

    segment = full_text
    if "####" in full_text:
        segment = full_text.split("####")[-1]

    lines = [ln.strip() for ln in segment.splitlines() if ln.strip()]
    if not lines:
        # segment 可能是空字符串或全空白，直接返回 strip 后的结果或 None
        segment = segment.strip()
        return segment if segment else None

    return lines[0]



def _try_parse_fraction(expr: str) -> Optional[float]:
    m = re.fullmatch(r"\s*([-+]?\d+)\s*/\s*(\d+)\s*", expr)
    if not m:
        return None
    num = int(m.group(1))
    den = int(m.group(2))
    if den == 0:
        return None
    return num / den


def _parse_answer_to_float(segment: str) -> Optional[float]:
    """和 eval 中的 parse_answer_to_float 同逻辑。"""
    if not segment:
        return None

    s = segment.strip().replace(",", " ")

    # 先试分数形式
    val = _try_parse_fraction(s)
    if val is not None:
        return val

    # 直接 float
    try:
        return float(s)
    except Exception:
        pass

    # 再从字符串中找最后一个数字
    matches = NUMBER_REGEX.findall(s)
    if not matches:
        return None

    last_num = matches[-1]
    try:
        return float(last_num)
    except Exception:
        return None


def _is_numeric_match(
    pred: float,
    target: float,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-6,
) -> bool:
    """数值匹配：|pred - target| <= max(abs_tol, rel_tol * |target|)"""
    return math.isclose(pred, target, rel_tol=rel_tol, abs_tol=abs_tol)


def compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 0.1,
    score: float = 1.0,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-6,
) -> float:
    """
    TeleMath 的规则奖励函数：
      - 能解析出数值且在容差范围内：给 score（默认 1.0 分）；
      - 能解析出数值但数值不匹配：给 format_score（默认 0.1 分）；
      - 完全解析不出数值：给 0 分。
    注意：签名风格和 gsm8k.compute_score 一致，便于在 __init__.py 中路由。
    """
    # ground_truth 原本就是字符串形式的数值（见你的 train.json）
    try:
        target = float(ground_truth)
    except Exception:
        # 再兜底试一次 parse
        target = _parse_answer_to_float(str(ground_truth))

    if target is None:
        # 标准答案都坏了，就直接不给分
        return 0.0

    seg = _extract_answer_segment(solution_str or "")
    pred = _parse_answer_to_float(seg) if seg is not None else None

    if pred is None:
        # 完全没答出数
        return 0.0

    if _is_numeric_match(pred, target, rel_tol=rel_tol, abs_tol=abs_tol):
        return float(score)
    else:
        # 格式/数值“接近”但不对，给一点点格式奖励
        return float(format_score)
