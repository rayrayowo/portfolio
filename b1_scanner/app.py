#!/usr/bin/env python3
"""Streamlit app for B1 scanner v2.0."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_sources import DEFAULT_TUSHARE_TOKEN, get_tushare_mainboard_stocks
from scanner_core import B1Config, flatten_result_for_table, scan_batch, scan_symbol

st.set_page_config(page_title="B1战法选股器 v2.0", page_icon="📈", layout="wide")


def parse_symbol_input(raw: str) -> List[Dict[str, str]]:
    """解析股票列表，支持多种格式"""
    rows: List[Dict[str, str]] = []
    
    # 常用股票名称映射
    name_map = {
        "贵州茅台": "600519.SH",
        "恒瑞医药": "600276.SH",
        "药明康德": "603259.SH",
        "百济神州": "688235.SH",
        "荣昌生物": "688331.SH",
        "平安银行": "000001.SZ",
        "青岛港": "601298.SH",
        "新乳业": "002946.SZ",
        "民生银行": "600016.SH",
        "格力电器": "000651.SZ",
        "中国平安": "601318.SH",
        "招商银行": "600036.SH",
        "比亚迪": "002594.SZ",
        "宁德时代": "300750.SZ",
        "AAPL": "AAPL",
        "TSLA": "TSLA",
        "MSFT": "MSFT",
    }
    
    for line in raw.replace("，", ",").splitlines():
        line = line.strip()
        if not line:
            continue
            
        # 尝试解析: 代码,名称 或 只写代码 或 只写名称
        if "," in line:
            parts = [x.strip() for x in line.split(",") if x.strip()]
            if len(parts) >= 2:
                # 可能是 代码,名称 或 名称,代码
                if parts[0].replace(".", "").replace("SH", "").replace("SZ", "").isdigit():
                    code, name = parts[0], parts[1]
                else:
                    code, name = parts[1], parts[0]
            elif len(parts) == 1:
                code, name = parts[0], ""
        else:
            parts = line.split()
            if len(parts) >= 2:
                code, name = parts[0], parts[1]
            else:
                code = parts[0]
                name = ""
        
        # 如果只写了名称，尝试查找代码
        if code.upper() not in ["SH", "SZ", "SS"] and not any(c.isdigit() for c in code):
            # 可能是中文名称
            code = name_map.get(code, code)
            name = code if not name else name
        
        # 标准化代码格式
        code = code.upper().strip()
        if code.isdigit() and len(code) == 6:
            if code.startswith("6"):
                code = f"{code}.SH"
            elif code.startswith(("0", "3")):
                code = f"{code}.SZ"
        
        if code:
            rows.append({"symbol": code, "name": name})
    
    return rows


def build_figure(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.12, 0.14, 0.13, 0.13, 0.13],
        subplot_titles=(
            f"{symbol} K线 + 知行趋势线 + BOLL",
            "VOL",
            "MACD",
            "KDJ",
            "RSI",
            "砖型图 (白色=买点)",
        ),
    )

    x = df["date"]

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=x, y=df["zhixing_white"], name="白线(短期)", line={"width": 2, "color": "white"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["zhixing_yellow"], name="黄线(长期)", line={"width": 2, "color": "yellow"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["boll_upper"], name="BOLL上轨", line={"width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["boll_mid"], name="BOLL中轨", line={"width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["boll_lower"], name="BOLL下轨", line={"width": 1}), row=1, col=1)

    fig.add_trace(go.Bar(x=x, y=df["volume"], name="成交量", opacity=0.5), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["vol_ma5"], name="VOL MA5", line={"width": 1}), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["vol_ma10"], name="VOL MA10", line={"width": 1}), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["vol_ma20"], name="VOL MA20", line={"width": 1}), row=2, col=1)

    fig.add_trace(go.Bar(x=x, y=df["macd_hist"], name="MACD柱", opacity=0.6), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["macd_diff"], name="DIFF", line={"width": 1.5}), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["macd_dea"], name="DEA", line={"width": 1.5}), row=3, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["kdj_k"], name="K", line={"width": 1}), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["kdj_d"], name="D", line={"width": 1}), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["kdj_j"], name="J", line={"width": 1.5}), row=4, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["rsi14"], name="RSI14", line={"width": 1.5}), row=5, col=1)
    fig.add_hline(y=70, line_dash="dot", line_width=1, row=5, col=1)
    fig.add_hline(y=30, line_dash="dot", line_width=1, row=5, col=1)

    # 砖型图
    brick_chart = df["brick_chart"].fillna(0)
    brick_white = df["brick_white"].fillna(0)
    
    # 普通砖头 (红/绿)
    brick_colors = ["red" if brick_chart.iloc[i] > 0 else "green" for i in range(len(brick_chart))]
    # 白色砖头标记
    brick_colors = ["white" if brick_white.iloc[i] == 1 else brick_colors[i] for i in range(len(brick_white))]
    
    fig.add_trace(go.Bar(x=x, y=brick_chart, name="砖型图", marker_color=brick_colors, opacity=0.8), row=6, col=1)
    
    # 标记白色砖头
    white_indices = df[brick_white == 1].index
    if len(white_indices) > 0:
        white_x = [df.loc[i, "date"] for i in white_indices if i in df.index]
        white_y = [brick_chart.loc[i] for i in white_indices if i in df.index and i in brick_chart.index]
        fig.add_trace(go.Scatter(x=white_x, y=white_y, mode="markers", name="白色砖头(买点)", 
                                   marker=dict(symbol="triangle-up", size=12, color="white", line=dict(color="black", width=2))), row=6, col=1)

    fig.update_layout(height=1200, xaxis_rangeslider_visible=False, legend_orientation="h")
    return fig


def render_single_result(result: dict):
    if result.get("error"):
        st.error(result["error"])
        return

    m = result["metrics"]
    c = result["conditions"]

    cols = st.columns(8)
    cols[0].metric("收盘", f"{m['close']:.2f}" if m["close"] is not None else "-")
    cols[1].metric("涨幅", f"{m.get('price_change_pct', 0):.2f}%" if m.get('price_change_pct') is not None else "-")
    cols[2].metric("量比", f"{m.get('volume_ratio', 0):.2f}" if m.get('volume_ratio') is not None else "-")
    cols[3].metric("MACD DEA", f"{m['macd_dea']:.4f}" if m["macd_dea"] is not None else "-")
    cols[4].metric("KDJ J", f"{m['kdj_j']:.2f}" if m["kdj_j"] is not None else "-")
    cols[5].metric("RSI14", f"{m['rsi14']:.2f}" if m["rsi14"] is not None else "-")
    cols[6].metric("白线", f"{m['zhixing_white']:.2f}" if m.get('zhixing_white') is not None else "-")
    cols[7].metric("是否通过", "✅ 是" if result["passed"] else "❌ 否")

    st.caption(f"最新交易日: {m['date']}")

    # 根据战法显示不同条件
    if result.get("strategy", "B1") == "B1":
        cond_table = pd.DataFrame(
            [
                {"条件": "主板(600/601/603/000)", "结果": "✅" if c["mainboard_ok"] else "❌"},
                {"条件": "周线 MA30>MA60>MA120>MA240", "结果": "✅" if c["weekly_ok"] else "❌"},
                {"条件": "MACD DEA > 0", "结果": "✅" if c["macd_dea_ok"] else "❌"},
                {"条件": "KDJ J < 13", "结果": "✅" if c["kdj_j_ok"] else "❌"},
                {"条件": "白线 > 黄线 (黄线在白线下)", "结果": "✅" if c.get("zhixing_bullish") else "❌"},
                {"条件": "金叉后第一个B1", "结果": "✅" if c.get("golden_cross") else "⚪"},
                {"条件": "白色砖头 (买点)", "结果": "✅ 买点!" if c.get("brick_white") else "⚪"},
            ]
        )
    else:  # B2
        cond_table = pd.DataFrame(
            [
                {"条件": "主板(600/601/603/000)", "结果": "✅" if c["mainboard_ok"] else "❌"},
                {"条件": "周线 MA30>MA60>MA120>MA240", "结果": "✅" if c["weekly_ok"] else "❌"},
                {"条件": "MACD DEA > 0", "结果": "✅" if c["macd_dea_ok"] else "❌"},
                {"条件": "KDJ J < 55", "结果": "✅" if c["kdj_j_ok"] else "❌"},
                {"条件": "涨幅 >= 4%", "结果": "✅" if c.get("price_change_ok") else "❌"},
                {"条件": "量比 >= 1.1", "结果": "✅" if c.get("volume_ratio_ok") else "❌"},
                {"条件": "白色砖头 (买点)", "结果": "✅ 买点!" if c.get("brick_white") else "⚪"},
            ]
        )
    st.dataframe(cond_table, use_container_width=True, hide_index=True)
    if not c["weekly_ok"]:
        st.warning(f"周线检查: {result.get('weekly_reason', '未通过')}")

    fig = build_figure(result["daily_df"].tail(260), result["symbol"])
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("B1战法选股器 v2.0")
    st.write("支持 Tushare + Yahoo Finance，内置知行趋势线与 KDJ/MACD/RSI/BOLL/VOL 指标。")

    with st.sidebar:
        st.header("参数设置")
        
        # 战法选择
        strategy = st.selectbox(
            "选择战法",
            ["B1", "B2"],
            help="B1: 超跌反弹 | B2: 强势追涨"
        )
        
        st.markdown("---")
        
        if strategy == "B1":
            st.info("📌 B1战法: 超跌反弹\n- KDJ J < 13\n- 白线 > 黄线\n- 适合回调买入")
        else:
            st.info("📌 B2战法: 强势追涨\n- KDJ J < 55\n- 涨幅 >= 4%\n- 量比 >= 1.1\n- 适合追强势股")

        st.markdown("---")
        
        st.header("数据源")
        source = st.selectbox("数据源", ["tushare", "yahoo"], index=0)
        token = ""
        if source == "tushare":
            token = st.text_input("Tushare Token", value=DEFAULT_TUSHARE_TOKEN, type="password")

        mode = st.radio("模式", ["单票分析", "批量选股"], index=0)

        end_date = st.date_input("结束日期", value=date.today())
        start_date = st.date_input("开始日期", value=date.today() - timedelta(days=365 * 3))

        pause_sec = st.slider("请求间隔(秒)", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
        
        # 知行趋势线选项
        require_golden_cross = st.checkbox("仅显示金叉后第一个B1 (可选)", value=False, 
            help="开启后只显示白线向上穿透黄线(金叉)之后的第一个符合B1条件的股票")
        
        # 行业板块筛选
        sector = st.selectbox(
            "行业板块 (可选)",
            ["全部", "医药生物", "电子", "计算机", "机械设备", "化工", "有色金属", "电力设备", "汽车", "食品饮料", "传媒", "通信"],
            help="选择行业板块进行筛选"
        )
        
        # 市值筛选 (单位: 亿元)
        st.markdown("**市值筛选 (亿元)**")
        mc1, mc2 = st.columns(2)
        with mc1:
            market_cap_min = st.number_input("最小市值", min_value=0, value=0, help="单位: 亿元, 0表示不限制")
        with mc2:
            market_cap_max = st.number_input("最大市值", min_value=0, value=0, help="单位: 亿元, 0表示不限制")
        
        # 砖型图选项
        require_brick_white = st.checkbox("仅显示白色砖头 (买点)", value=False,
            help="砖型图白色砖头 = 买点信号")

    config = B1Config(
        source=source,
        tushare_token=token or None,
        start=start_date,
        end=end_date,
        request_pause_sec=pause_sec,
        require_golden_cross=require_golden_cross,
        require_brick_white=require_brick_white,
        strategy=strategy,
    )

    if mode == "单票分析":
        c1, c2 = st.columns([2, 1])
        with c1:
            symbol = st.text_input("股票代码", value="600519.SH")
        with c2:
            name = st.text_input("股票名称(可选)", value="")

        if st.button("开始分析", type="primary", use_container_width=True):
            with st.spinner("正在计算指标并检查 B1 条件..."):
                result = scan_symbol(symbol=symbol, name=name, config=config)
            render_single_result(result)

    else:
        st.subheader("批量选股")
        
        # 自选股管理
        with st.expander("📌 自选股管理", expanded=True):
            # 默认自选股列表
            default_stocks = """600519.SH,贵州茅台
600276.SH,恒瑞医药
603259.SH,药明康德
688235.SH,百济神州
688331.SH,荣昌生物
000001.SZ,平安银行
601298.SH,青岛港
002946.SZ,新乳业
600016.SH,民生银行
000651.SZ,格力电器"""
            
            watchlist = st.text_area(
                "📌 自选股列表 (每行: 代码,名称)",
                value=default_stocks,
                height=150,
                help="格式: 代码,名称 或 只写代码"
            )
            
            # 搜索添加股票
            st.markdown("**🔍 搜索添加股票**")
            search_col1, search_col2 = st.columns([2, 1])
            with search_col1:
                search_keyword = st.text_input("输入股票代码或名称搜索", placeholder="如: 600519 或 茅台")
            with search_col2:
                search_btn = st.button("搜索", use_container_width=True)
            
            # 使用Tushare搜索股票
            if search_btn and search_keyword:
                search_success = False
                
                # 尝试Tushare搜索
                if token:
                    try:
                        import tushare as ts
                        pro = ts.pro_api(token)
                        # 搜索股票
                        df = pro.stock_basic(fields='ts_code,name,area,industry,list_date')
                        # 模糊匹配
                        mask = df['ts_code'].str.contains(search_keyword, na=False) | df['name'].str.contains(search_keyword, na=False)
                        results = df[mask].head(10)
                        if not results.empty:
                            st.success(f"找到 {len(results)} 只股票:")
                            for _, row in results.iterrows():
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.code(row['ts_code'])
                                with col2:
                                    st.write(f"{row['name']} ({row.get('industry', '')})")
                            search_success = True
                    except Exception as e:
                        error_msg = str(e)
                        if "每小时" in error_msg or "每天" in error_msg or "权限" in error_msg:
                            st.warning("⚠️ Tushare 配额已用完，使用备用搜索...")
                            search_success = False
                        else:
                            st.error(f"搜索失败: {error_msg[:50]}")
                
                # Tushare失败时使用本地备用列表
                if not search_success:
                    # 备用股票列表
                    backup_stocks = {
                        "长江通信": "600345.SH",
                        "青岛港": "601298.SH",
                        "新乳业": "002946.SZ",
                        "亚盛医药": "06855.HK",
                        "阿里": "9988.HK",
                        "腾讯": "00700.HK",
                        "美团": "03690.HK",
                        "小米": "01810.HK",
                    }
                    matches = {k: v for k, v in backup_stocks.items() if search_keyword in k}
                    if matches:
                        st.info("📋 热门股票 (Tushare配额用完):")
                        for name, code in matches.items():
                            st.write(f"  {code} - {name}")
                    else:
                        st.info("📝 请直接在自选股列表中添加，或稍后再试搜索")
            
            # 常用股票快捷添加
            st.markdown("**⚡ 快捷添加**")
            quick_add_cols = st.columns(4)
            quick_stocks = [
                ("600519.SH", "贵州茅台"),
                ("600276.SH", "恒瑞医药"),
                ("000001.SZ", "平安银行"),
                ("601318.SH", "中国平安"),
                ("000651.SZ", "格力电器"),
                ("600036.SH", "招商银行"),
                ("002594.SZ", "比亚迪"),
                ("300750.SZ", "宁德时代"),
            ]
            for i, (code, name) in enumerate(quick_stocks):
                with quick_add_cols[i % 4]:
                    if st.button(f"{code}", key=f"add_{code}", use_container_width=True):
                        st.success(f"已添加 {name}")

        st.markdown("---")
        
        # 选择数据源
        scan_source = st.radio("扫描数据源", ["使用自选股列表", "使用Tushare主板列表"], horizontal=True)
        
        if scan_source == "使用自选股列表":
            rows = parse_symbol_input(watchlist)
            limit = len(rows)
            use_default = False
        else:
            use_default = True
            if source == "tushare":
                limit = st.slider("扫描数量", min_value=10, max_value=500, value=100, step=10)
                raw_symbols = ""
            else:
                limit = 0
                raw_symbols = ""

        if st.button("开始批量扫描", type="primary", use_container_width=True):
            with st.spinner("准备股票池..."):
                if source == "tushare" and use_default:
                    stock_df = get_tushare_mainboard_stocks(limit=limit, token=token or None)
                    rows = [
                        {"symbol": r["ts_code"], "name": r["name"]}
                        for _, r in stock_df.iterrows()
                    ]
                else:
                    rows = parse_symbol_input(watchlist) if scan_source == "使用自选股列表" else parse_symbol_input(raw_symbols)

            if not rows:
                st.error("股票池为空，请输入至少一只股票。")
                return

            progress = st.progress(0)
            status = st.empty()
            results = []

            for idx, row in enumerate(rows, start=1):
                status.write(f"扫描中: {row['symbol']} ({idx}/{len(rows)})")
                batch_res = scan_batch([row], config=config)
                results.extend(batch_res)
                progress.progress(idx / len(rows))

            table = pd.DataFrame([flatten_result_for_table(x) for x in results])
            table = table.sort_values(["passed", "symbol"], ascending=[False, True]).reset_index(drop=True)

            passed_df = table[table["passed"] == True]
            st.success(f"扫描完成: 总数 {len(table)}，符合B1 {len(passed_df)}")

            st.subheader("符合 B1 条件")
            st.dataframe(passed_df if not passed_df.empty else pd.DataFrame(columns=table.columns), use_container_width=True)

            st.subheader("全部结果")
            st.dataframe(table, use_container_width=True)

            csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下载结果 CSV",
                data=csv_bytes,
                file_name=f"b1_scan_{date.today().isoformat()}.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
