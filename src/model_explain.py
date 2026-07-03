"""Readable model explanations for the Streamlit demo."""

from __future__ import annotations

import pandas as pd


def market_regime(row: pd.Series) -> str:
    """Map market features into a simple product-facing regime label."""
    bias_20 = float(row.get("idx_bias_20d", 0.0))
    bias_60 = float(row.get("idx_bias_60d", 0.0))
    vol_20 = float(row.get("idx_vol_20d", 0.0))

    if vol_20 > 0.035:
        return "高波动"
    if bias_20 > 0 and bias_60 > 0:
        return "偏强"
    if bias_20 < 0 and bias_60 < 0:
        return "偏弱"
    return "震荡"


def gate_weights(row: pd.Series) -> tuple[float, float]:
    """
    Demo-only gate approximation.

    The production MoE gate should come from the trained PyTorch model. This
    function is only used to make the product demo explainable when saved MoE
    predictions are not available.
    """
    bias_signal = float(row.get("idx_bias_20d", 0.0)) * 8 + float(row.get("idx_bias_60d", 0.0)) * 4
    vol_penalty = max(float(row.get("idx_vol_mom", 0.0)), 0.0) * 2
    bull = 1 / (1 + pow(2.71828, -(bias_signal - vol_penalty)))
    bull = min(max(bull, 0.18), 0.82)
    bear = 1 - bull
    return round(bull, 3), round(bear, 3)


def stock_explanation(row: pd.Series) -> str:
    """Create a short natural-language reason for a stock recommendation."""
    regime = market_regime(row)
    bull_w = float(row.get("gate_bull", 0.5))
    parts: list[str] = []

    if float(row.get("mom_5d", 0.0)) > 0:
        parts.append("短期动量较强")
    else:
        parts.append("短期动量偏弱但估值/风险位置改善")

    if float(row.get("vol_20d", 0.0)) < float(row.get("vol_5d", 0.0)) * 1.25:
        parts.append("波动率可控")
    else:
        parts.append("近期波动偏高")

    if float(row.get("pv_corr", 0.0)) > 0:
        parts.append("价量配合较好")
    else:
        parts.append("价量信号仍需观察")

    expert = "牛市专家" if bull_w >= 0.5 else "熊市专家"
    return f"{'、'.join(parts)}；市场环境{regime}，{expert}权重较高，因此 Alpha 排名靠前。"


def moe_summary() -> str:
    return (
        "传统树模型学习的是相对静态的因子映射关系，而 MoE 模型可以根据市场状态"
        "动态切换专家，使模型更适应行情风格切换。"
    )
