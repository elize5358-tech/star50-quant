# 比赛平台上传材料清单

## 可运行产品原型

- `app.py`
- `src/demo_data.py`
- `src/model_explain.py`
- `src/risk_control.py`
- `src/visualization.py`
- `requirements.txt`
- `README_DEMO.md`

## 3分钟演示视频材料

- `demo_materials/demo_script_3min.md`

## 高保真交互说明图 / 逻辑流程图

- `demo_materials/logic_flow.md`

## MVP / POC 实验数据说明

- `demo_materials/poc_report.md`
- `outputs/demo_predictions.csv`
- `outputs/demo_backtest.csv`
- `outputs/demo_metrics.json`
- `outputs/demo_model_comparison.csv`

## 启动命令

```bash
cd /Users/elize/Desktop/量化/蚂蚁量化比赛/star50-quant
pip install -r requirements.txt
streamlit run app.py
```

## 注意事项

- 若真实 parquet 和回测输出存在，Demo 会优先读取真实文件。
- 若演示环境缺少真实数据，Demo 会自动启用 mock 数据兜底，避免现场启动失败。
- mock 数据仅用于产品交互展示，不代表正式收益承诺。
