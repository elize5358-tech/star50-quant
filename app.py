#!/usr/bin/env python3
"""星火 Alpha：AI 指数增强量化平台 Streamlit demo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.demo_data import MARKET_FEATURES, STOCK_FEATURES, ensure_demo_outputs
from src.model_explain import moe_summary
from src.risk_control import calculate_risk_dashboard
from src.visualization import feature_distribution, gate_chart, nav_curve, risk_radar, top_predictions


st.set_page_config(page_title="星火 Alpha", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <meta name="google" content="notranslate">
    <script>
      document.documentElement.setAttribute("translate", "no");
      document.documentElement.classList.add("notranslate");
      document.body.setAttribute("translate", "no");
      document.body.classList.add("notranslate");
    </script>
    <style>
    .stApp { background: #f6f9ff; color: #0f172a; }
    section[data-testid="stSidebar"] { background: #0f2d52; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    .hero {
        padding: 28px 32px; border-radius: 8px;
        background: linear-gradient(135deg, #0f2d52 0%, #1f5eff 100%);
        color: white; margin-bottom: 18px;
    }
    .hero h1 { margin: 0 0 8px 0; font-size: 38px; letter-spacing: 0; }
    .hero p { margin: 0; font-size: 18px; opacity: .94; }
    .metric-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 16px 18px; min-height: 112px;
        box-shadow: 0 8px 24px rgba(15, 45, 82, .06);
    }
    .metric-card b { color: #0f2d52; font-size: 15px; }
    .metric-card span { display: block; color: #475569; margin-top: 8px; font-size: 14px; }
    .flow-box {
        background: white; border: 1px solid #dbeafe; color: #0f2d52;
        padding: 14px; border-radius: 8px; text-align: center; font-weight: 700;
    }
    .business-note {
        background: #eaf2ff; border-left: 4px solid #1f5eff; padding: 12px 14px;
        border-radius: 6px; color: #0f2d52; margin: 10px 0 18px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="正在读取项目数据与 demo 输出...")
def load_demo():
    return ensure_demo_outputs()


data = load_demo()
features = data["features"]
predictions = data["predictions"]
backtest = data["backtest"]
metrics = data["metrics"]
comparison = data["comparison"]
risk = calculate_risk_dashboard(backtest, predictions)


pages = ["首页 / 项目概览", "数据与特征工程", "模型", "预测结果", "回测评估", "风险管理"]
st.sidebar.title("星火 Alpha")
page = st.sidebar.radio("产品模块", pages)
st.sidebar.markdown("---")
st.sidebar.caption("AI量化金融 · 初创组 Demo")
st.sidebar.caption("真实数据优先读取，缺失时自动启用 demo mock。")


def business_note(text: str) -> None:
    st.markdown(f"<div class='business-note'>{text}</div>", unsafe_allow_html=True)


def metric_card(title: str, body: str) -> None:
    st.markdown(f"<div class='metric-card'><b>{title}</b><span>{body}</span></div>", unsafe_allow_html=True)


if page == "首页 / 项目概览":
    st.markdown(
        """
        <div class="hero">
          <h1>星火 Alpha：AI 指数增强量化平台</h1>
          <p>基于深度学习 MoE 与传统树模型融合的科创50指数增强系统</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    business_note("业务解释：平台把数据、模型、组合和风险控制串成一条投研生产线，帮助指数增强产品在不同市场状态下更稳地寻找超额收益。")

    cols = st.columns(5)
    cards = [
        ("AI Alpha 预测", "预测剥离市场 Beta 后的真实 Alpha，而不只是涨跌方向。"),
        ("市场状态识别", "识别偏强、偏弱、震荡和高波动环境。"),
        ("动态因子加权", "通过 Gate 网络让不同专家在不同行情中发挥作用。"),
        ("组合优化", "把股票得分转换成分散、可执行的推荐权重。"),
        ("风险监控", "跟踪波动、回撤、集中度、换手率与 Beta 暴露。"),
    ]
    for col, (title, body) in zip(cols, cards):
        with col:
            metric_card(title, body)

    st.markdown("### 系统流程图")
    flow_cols = st.columns(8)
    flow = ["数据输入", "特征工程", "XGBoost 对照", "v5 MoE", "股票打分", "组合构建", "回测评估", "风险监控"]
    for i, item in enumerate(flow):
        with flow_cols[i]:
            st.markdown(f"<div class='flow-box'>{item}</div>", unsafe_allow_html=True)
    st.caption("流程从真实行情数据出发，先建立可解释的树模型基准，再引入 MoE 适应行情切换，最后落到组合与风险指标。")

elif page == "数据与特征工程":
    st.title("数据与特征工程")
    business_note("业务解释：个股特征判断股票自身强弱；环境特征判断当前市场处于牛市、熊市、震荡市或高波动状态。")

    c1, c2, c3 = st.columns(3)
    c1.metric("样本记录", f"{len(features):,}")
    c2.metric("股票数量", f"{features['ts_code'].nunique():,}")
    c3.metric("日期范围", f"{features['trade_date'].min().date()} 至 {features['trade_date'].max().date()}")

    st.markdown("#### 个股特征")
    st.dataframe(features[["ts_code", "trade_date", *STOCK_FEATURES]].tail(20), use_container_width=True, hide_index=True)
    st.markdown("#### 市场环境特征")
    st.dataframe(features[["trade_date", *MARKET_FEATURES]].drop_duplicates("trade_date").tail(20), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(feature_distribution(features, "mom_5d"), use_container_width=True)
    with right:
        st.plotly_chart(feature_distribution(features, "idx_vol_20d"), use_container_width=True)

elif page == "模型":
    st.title("模型")
    business_note("业务解释：当前主基准线切换为 v5 MoE 动态 Loss 权重版本，XGBoost 作为传统树模型对照。")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("v5 MoE 主基准")
        st.markdown(
            """
            - v5 采用 **动态 Loss 权重**，是当前 demo 的主模型基准线。
            - v5 的 IC 为 0.0212，ICIR 为 0.1580。
            - v5 多空组合年化收益为 64.66%，夏普比率为 1.5410。
            - XGBoost 作为传统树模型对照，用来衡量深度学习 MoE 的相对位置。
            - 训练目标为 `label_5d / residual_ret`：个股未来收益减去指数收益后的真实 Alpha。
            """
        )

        st.subheader("v5 MoE vs XGBoost")
        show_cols = ["model", "setup", "ic", "icir", "annual_return", "sharpe_ratio", "max_drawdown", "evaluation"]
        st.dataframe(
            comparison[show_cols].assign(
                annual_return=lambda x: x["annual_return"].map(lambda v: f"{v:.2%}"),
                max_drawdown=lambda x: x["max_drawdown"].map(lambda v: f"{v:.2%}"),
                sharpe_ratio=lambda x: x["sharpe_ratio"].map(lambda v: f"{v:.2f}"),
                ic=lambda x: x["ic"].map(lambda v: f"{v:.4f}"),
                icir=lambda x: x["icir"].map(lambda v: f"{v:.3f}"),
            ).rename(
                columns={
                    "model": "模型",
                    "setup": "配置",
                    "ic": "IC",
                    "icir": "ICIR",
                    "annual_return": "年化收益",
                    "sharpe_ratio": "夏普",
                    "max_drawdown": "最大回撤",
                    "evaluation": "评价",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Deep Learning MoE")
        gate_pool = predictions.nsmallest(10, "rank")
        bull_weight = float(gate_pool["gate_bull"].mean())
        bear_weight = 1 - bull_weight
        st.plotly_chart(gate_chart(bull_weight, bear_weight), use_container_width=True)
        st.caption(
            f"Top 10 推荐池平均 Gate 权重：牛市专家 {bull_weight:.1%} / 熊市专家 {bear_weight:.1%}。"
            "当前为 demo 近似，真实生产应读取 MoE 训练模型的 gate 输出。"
        )

    st.markdown("#### MoE 结构")
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("expert_bull", "偏强市场专家：更关注动量延续、价量配合和风险承受能力。")
    with m2:
        metric_card("expert_bear", "偏弱市场专家：更关注低波动、回撤控制和防御性 Alpha。")
    with m3:
        metric_card("gate", "门控网络：根据市场环境动态分配专家权重。")
    st.info(moe_summary())

elif page == "预测结果":
    st.title("预测结果")
    business_note("业务解释：平台把模型输出转换成投资经理能直接使用的 Top 股票、推荐权重、风险标记和一句话解释。")

    latest_date = pd.to_datetime(predictions["trade_date"].max()).date()
    st.caption(f"展示期：{latest_date}")
    top10 = predictions.nsmallest(10, "rank")
    st.plotly_chart(top_predictions(top10), use_container_width=True)
    st.dataframe(
        top10[["ts_code", "trade_date", "pred_score", "rank", "recommended_weight", "risk_flag", "explanation"]],
        use_container_width=True,
        hide_index=True,
    )

elif page == "回测评估":
    st.title("回测评估")
    business_note("业务解释：深度学习模型不是单纯预测涨跌，而是预测剥离市场 Beta 后的真实 Alpha，再通过组合构建验证是否能形成相对指数的增强收益。")

    st.plotly_chart(nav_curve(backtest), use_container_width=True)
    st.markdown("#### v5 MoE 基准线对比")
    st.dataframe(
        comparison.assign(
            annual_return=lambda x: x["annual_return"].map(lambda v: "" if pd.isna(v) else f"{v:.2%}"),
            max_drawdown=lambda x: x["max_drawdown"].map(lambda v: "" if pd.isna(v) else f"{v:.2%}"),
            sharpe_ratio=lambda x: x["sharpe_ratio"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
            ic=lambda x: x["ic"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}"),
            icir=lambda x: x["icir"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}"),
            monthly_win_rate=lambda x: x["monthly_win_rate"].map(lambda v: "" if pd.isna(v) else f"{v:.1%}"),
        ).rename(
            columns={
                "model": "模型",
                "setup": "配置",
                "ic": "IC",
                "icir": "ICIR",
                "annual_return": "年化收益",
                "max_drawdown": "最大回撤",
                "sharpe_ratio": "夏普比率",
                "monthly_win_rate": "月度胜率",
                "evaluation": "评价",
                "source": "数据来源",
            }
        )[["模型", "配置", "IC", "ICIR", "年化收益", "夏普比率", "最大回撤", "月度胜率", "评价", "数据来源"]],
        use_container_width=True,
        hide_index=True,
    )
    cols = st.columns(7)
    metric_items = [
        ("年化收益", f"{metrics['年化收益']:.2%}"),
        ("超额收益", f"{metrics['超额收益']:.2%}"),
        ("最大回撤", f"{metrics['最大回撤']:.2%}"),
        ("夏普比率", f"{metrics['夏普比率']:.2f}"),
        ("Alpha IC", f"{metrics['IC']:.4f}"),
        ("ICIR", f"{metrics['ICIR']:.3f}"),
        ("月度胜率", f"{metrics['月度胜率']:.1%}"),
    ]
    for col, (name, value) in zip(cols, metric_items):
        col.metric(name, value)
    st.caption(metrics["label"])

elif page == "风险管理":
    st.title("风险管理")
    business_note("业务解释：指数增强不是只追求高收益，还要控制相对指数的波动、回撤、集中度和 Beta 暴露，让组合更适合真实资管落地。")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("风险评分", f"{risk['risk_score']:.1f} / 100", risk["risk_level"])
        st.info(risk["suggestion"])
        st.metric("组合波动率", f"{risk['portfolio_volatility']:.2%}")
        st.metric("最大回撤", f"{risk['max_drawdown']:.2%}")
        st.metric("市场 Beta 暴露", f"{risk['market_beta']:.2f}")
    with c2:
        st.plotly_chart(risk_radar(risk), use_container_width=True)

    risk_table = pd.DataFrame(
        [
            ("组合波动率", f"{risk['portfolio_volatility']:.2%}", "控制净值波动，避免策略收益不稳定。"),
            ("最大回撤", f"{risk['max_drawdown']:.2%}", "衡量极端亏损风险。"),
            ("行业集中度", f"{risk['industry_concentration']:.2%}", "避免组合过度暴露在单一行业。"),
            ("单票权重上限", f"{risk['single_name_limit']:.2%}", "防止单一股票影响组合。"),
            ("换手率", f"{risk['turnover']:.2%}", "控制交易成本和执行冲击。"),
            ("市场 Beta 暴露", f"{risk['market_beta']:.2f}", "保持指数增强组合相对基准的稳定性。"),
        ],
        columns=["风险指标", "当前值", "含义"],
    )
    st.dataframe(risk_table, use_container_width=True, hide_index=True)
