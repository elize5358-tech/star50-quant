# 星火 Alpha Demo 运行说明

## 一句话定位

“星火 Alpha：AI 指数增强量化平台”是一个面向科创50指数增强场景的本地 Web Demo，展示从数据输入、特征工程、v5 MoE 主模型、XGBoost 对照、股票打分、组合构建、回测评估到风险监控的完整产品闭环。

## 启动方式

```bash
cd /Users/elize/Desktop/量化/蚂蚁量化比赛/star50-quant
pip install -r requirements.txt
streamlit run app.py
```

如果你在真实 repo 中运行：

```bash
cd /Users/elize/Documents/金融量化/star50-quant
pip install -r requirements.txt
streamlit run app.py
```

## 数据读取策略

Demo 会优先读取项目已有文件：

- `data/processed/star50_features_stage1_preprocessed.parquet`
- `results/xgboost_results_20260611_102029.json`
- `data/raw/star50_daily_hfq_data_6yrs.parquet`
- `data/raw/star50_index_daily_6yrs.parquet`

如果复制到比赛资料目录后缺少上述文件，程序会继续搜索上级目录中的 parquet 文件。仍找不到时，会自动生成字段完整的 mock 数据，保证现场演示不中断。代码中已用注释标明 mock 数据仅用于 demo 展示。

## 页面模块

- 首页 / 项目概览：项目定位、核心能力、系统流程图
- 数据与特征工程：个股特征、市场环境特征、因子分布
- 模型：v5 动态 Loss 权重、MoE 专家结构、Gate 权重
- 模型对比：v5 MoE 与 XGBoost、业界优秀参考线对照
- 预测结果：Top 10 股票、Alpha Score、推荐权重、风险标记、解释字段
- 回测评估：AI增强组合 vs 科创50指数 vs 树模型组合、核心指标
- 风险管理：波动率、回撤、集中度、单票权重、换手率、Beta 暴露和风险评分

## 自动生成输出

启动应用或调用 `src/demo_data.py` 后会生成：

- `outputs/demo_predictions.csv`
- `outputs/demo_backtest.csv`
- `outputs/demo_metrics.json`
- `outputs/demo_model_comparison.csv`

这些文件可作为 MVP/POC 实验数据附件上传。
