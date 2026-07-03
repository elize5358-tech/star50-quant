"""Risk scoring helpers for the demo product layer."""

from __future__ import annotations

import pandas as pd


def classify_risk(score: float) -> str:
    if score < 40:
        return "低风险"
    if score < 70:
        return "中风险"
    return "高风险"


def risk_suggestion(score: float) -> str:
    if score < 40:
        return "组合风险处于可控区间，可维持常规指数增强仓位。"
    if score < 70:
        return "当前风险中等，建议控制单票权重并提高行业分散度。"
    return "当前市场波动率较高，建议降低仓位、提高分散度，并收紧止损阈值。"


def calculate_risk_dashboard(backtest: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Compute compact demo risk metrics from NAV and latest holdings."""
    if backtest.empty:
        return _fallback_risk()

    recent = backtest.tail(min(60, len(backtest))).copy()
    vol = float(recent.get("ai_ret", recent.get("portfolio_ret", pd.Series([0]))).std() * (252 ** 0.5))
    nav = recent.get("ai_nav", recent.get("nav", pd.Series([1.0]))).astype(float)
    drawdown = float((nav / nav.cummax() - 1).min())
    turnover = float(recent.get("turnover", pd.Series([0.28])).mean())

    latest = predictions.copy()
    max_weight = float(latest.get("recommended_weight", pd.Series([0.08])).max())
    concentration = float(latest.get("recommended_weight", pd.Series([0.08])).nlargest(5).sum())
    beta = 0.92 + min(max(vol - 0.20, -0.10), 0.18)

    score = (
        min(vol / 0.45, 1) * 28
        + min(abs(drawdown) / 0.35, 1) * 24
        + min(concentration / 0.55, 1) * 18
        + min(max_weight / 0.12, 1) * 14
        + min(turnover / 0.65, 1) * 10
        + min(abs(beta - 1.0) / 0.35, 1) * 6
    )

    score = round(float(score), 1)
    return {
        "risk_score": score,
        "risk_level": classify_risk(score),
        "suggestion": risk_suggestion(score),
        "portfolio_volatility": vol,
        "max_drawdown": drawdown,
        "industry_concentration": concentration,
        "single_name_limit": max_weight,
        "turnover": turnover,
        "market_beta": beta,
    }


def _fallback_risk() -> dict:
    return {
        "risk_score": 52.0,
        "risk_level": "中风险",
        "suggestion": risk_suggestion(52.0),
        "portfolio_volatility": 0.24,
        "max_drawdown": -0.18,
        "industry_concentration": 0.42,
        "single_name_limit": 0.08,
        "turnover": 0.32,
        "market_beta": 0.96,
    }
