"""Plotly charts used by the Streamlit demo."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


BLUE = "#1f5eff"
NAVY = "#0f2d52"
CYAN = "#22a6f2"
GREEN = "#10b981"
ORANGE = "#f59e0b"
RED = "#ef4444"


def nav_curve(backtest: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest["date"], y=backtest["ai_nav"], name="v5 MoE组合", line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=backtest["date"], y=backtest["index_nav"], name="科创50指数", line=dict(color=NAVY, width=2)))
    fig.add_trace(go.Scatter(x=backtest["date"], y=backtest["tree_nav"], name="XGBoost组合", line=dict(color=CYAN, width=2, dash="dash")))
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h"))
    return fig


def feature_distribution(features: pd.DataFrame, column: str) -> go.Figure:
    fig = px.histogram(features, x=column, nbins=40, color_discrete_sequence=[BLUE])
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="样本数")
    return fig


def top_predictions(predictions: pd.DataFrame) -> go.Figure:
    top = predictions.nsmallest(10, "rank").sort_values("pred_score")
    fig = px.bar(
        top,
        x="pred_score",
        y="ts_code",
        orientation="h",
        color="recommended_weight",
        color_continuous_scale=["#dbeafe", BLUE],
        labels={"pred_score": "Alpha Score", "ts_code": "股票代码"},
    )
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
    return fig


def gate_chart(bull_weight: float, bear_weight: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(name="牛市专家", x=["Gate 权重"], y=[bull_weight], marker_color=GREEN),
            go.Bar(name="熊市专家", x=["Gate 权重"], y=[bear_weight], marker_color=ORANGE),
        ]
    )
    fig.update_layout(template="plotly_white", barmode="stack", height=260, yaxis=dict(range=[0, 1]), margin=dict(l=20, r=20, t=20, b=20))
    return fig


def risk_radar(risk: dict) -> go.Figure:
    labels = ["组合波动率", "最大回撤", "行业集中度", "单票权重", "换手率", "Beta暴露"]
    values = [
        min(risk["portfolio_volatility"] / 0.45, 1) * 100,
        min(abs(risk["max_drawdown"]) / 0.35, 1) * 100,
        min(risk["industry_concentration"] / 0.55, 1) * 100,
        min(risk["single_name_limit"] / 0.12, 1) * 100,
        min(risk["turnover"] / 0.65, 1) * 100,
        min(abs(risk["market_beta"] - 1.0) / 0.35, 1) * 100,
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], fill="toself", name="风险暴露", line_color=BLUE))
    fig.update_layout(template="plotly_white", polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=420, margin=dict(l=30, r=30, t=30, b=30))
    return fig
