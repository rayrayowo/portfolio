# B1战法选股器 v2.0

## 功能
- 多数据源: `Tushare` + `Yahoo Finance(yfinance)`
- 知行趋势线:
  - `EMA(EMA(C,10),10) + (MA(C,14)+MA(C,28)+MA(C,57)+MA(C,114))/4`
- 指标: `KDJ`, `MACD`, `RSI`, `BOLL`, `VOL`
- Streamlit 图形界面 (`app.py`)

## 运行

```bash
cd b1_scanner_v2.0
pip install -r requirements.txt
streamlit run app.py
```

## 说明
- Tushare 需要 token（可在侧边栏输入）。
- 批量扫描可直接读取 Tushare 主板股票池（600/601/603/000）。
- 单票分析会显示多指标图表和 B1 条件判定。
