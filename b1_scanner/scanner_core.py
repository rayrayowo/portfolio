"""B1 strategy scan logic for v2.0."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from data_sources import FetchParams, fetch_data
from indicators import add_all_indicators


@dataclass
class B1Config:
    source: str = "tushare"  # tushare | yahoo
    tushare_token: Optional[str] = None
    start: date = date.today() - timedelta(days=365 * 3)
    end: date = date.today()
    weekly_lookback_days: int = 365 * 8
    min_daily_bars: int = 150
    min_weekly_bars: int = 240
    request_pause_sec: float = 0.2
    
    # 战法选择
    strategy: str = "B1"  # B1, B2
    
    # 知行趋势线选项
    require_golden_cross: bool = False  # 是否要求金叉后第一个B1
    require_brick_white: bool = False  # 是否要求白色砖头 (买点)
    # 市值筛选 (单位: 亿元)
    market_cap_min: float = 0  # 最小市值, 0表示不限制
    market_cap_max: float = 0  # 最大市值, 0表示不限制
    # 行业板块
    sector: str = "全部"


def is_cn_mainboard(symbol: str) -> bool:
    """Main board filter: 600/601/603/000xxxx."""
    symbol = symbol.strip().upper()
    m = re.search(r"(\d{6})", symbol)
    if not m:
        # non-CN symbols are treated as not-mainboard for strict B1
        return False

    code = m.group(1)
    return code.startswith(("600", "601", "603", "000"))


def _weekly_ma_check(weekly_df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "weekly_ok": False,
        "weekly_reason": "周线数据不足",
        "ma30": None,
        "ma60": None,
        "ma120": None,
        "ma240": None,
    }

    if weekly_df is None or weekly_df.empty or len(weekly_df) < 240:
        return out

    wk = weekly_df.copy()
    wk["ma30"] = wk["close"].rolling(30).mean()
    wk["ma60"] = wk["close"].rolling(60).mean()
    wk["ma120"] = wk["close"].rolling(120).mean()
    wk["ma240"] = wk["close"].rolling(240).mean()

    latest = wk.iloc[-1]
    ma30 = latest["ma30"]
    ma60 = latest["ma60"]
    ma120 = latest["ma120"]
    ma240 = latest["ma240"]

    out.update(
        {
            "ma30": float(ma30) if pd.notna(ma30) else None,
            "ma60": float(ma60) if pd.notna(ma60) else None,
            "ma120": float(ma120) if pd.notna(ma120) else None,
            "ma240": float(ma240) if pd.notna(ma240) else None,
        }
    )

    if pd.isna(ma30) or pd.isna(ma60) or pd.isna(ma120) or pd.isna(ma240):
        out["weekly_reason"] = "周线均线尚未形成"
        return out

    out["weekly_ok"] = bool(ma30 > ma60 > ma120 > ma240)
    out["weekly_reason"] = "OK" if out["weekly_ok"] else "周线均线未多头排列"
    return out


def scan_symbol(symbol: str, name: str = "", config: Optional[B1Config] = None) -> Dict[str, Any]:
    cfg = config or B1Config()
    symbol = symbol.strip().upper()

    daily = fetch_data(
        FetchParams(
            symbol=symbol,
            start=cfg.start,
            end=cfg.end,
            source=cfg.source,
            period="daily",
            tushare_token=cfg.tushare_token,
        )
    )

    weekly_start = cfg.end - timedelta(days=cfg.weekly_lookback_days)
    weekly = fetch_data(
        FetchParams(
            symbol=symbol,
            start=weekly_start,
            end=cfg.end,
            source=cfg.source,
            period="weekly",
            tushare_token=cfg.tushare_token,
        )
    )

    if daily.empty or len(daily) < cfg.min_daily_bars:
        return {
            "symbol": symbol,
            "name": name or symbol,
            "passed": False,
            "error": f"日线数据不足: {len(daily)} 条",
        }

    daily_ind = add_all_indicators(daily)
    latest = daily_ind.iloc[-1]

    weekly_info = _weekly_ma_check(weekly)

    # 知行趋势线: 白线(短期) vs 黄线(长期)
    zhixing_white = latest.get("zhixing_white")  # 白线: EMA(EMA(C,10),10)
    zhixing_yellow = latest.get("zhixing_yellow")  # 黄线: (MA14+MA28+MA57+MA114)/4
    
    # 核心条件: 黄线在白线之下 = 上涨趋势 (white > yellow)
    zhixing_bullish = bool(
        pd.notna(zhixing_white) and pd.notna(zhixing_yellow) and 
        zhixing_white > zhixing_yellow
    )
    
    # 金叉检测: 白线从下方穿越到上方 (可选)
    golden_cross = False
    if cfg.require_golden_cross and len(daily_ind) >= 2:
        prev = daily_ind.iloc[-2]
        prev_white = prev.get("zhixing_white")
        prev_yellow = prev.get("zhixing_yellow")
        if pd.notna(prev_white) and pd.notna(prev_yellow):
            # 昨天白线 <= 黄线, 今天白线 > 黄线 = 金叉
            golden_cross = bool(prev_white <= prev_yellow and zhixing_white > zhixing_yellow)
    
    # 砖型图检测
    brick_white = latest.get("brick_white", 0)  # 白色砖头 = 1 (买点)
    brick_chart = latest.get("brick_chart", 0)
    
    mainboard_ok = is_cn_mainboard(symbol)
    macd_dea = latest.get("macd_dea")
    kdj_j = latest.get("kdj_j")
    close = latest.get("close")
    volume = latest.get("volume", 0)
    prev_close = daily_ind.iloc[-2]["close"] if len(daily_ind) >= 2 else close
    
    # 计算涨跌幅
    price_change_pct = ((close - prev_close) / prev_close * 100) if pd.notna(prev_close) and prev_close != 0 else 0
    
    # 计算量比 (成交量 / 5日平均成交量)
    vol_ma5 = daily_ind["volume"].rolling(5).mean().iloc[-1] if len(daily_ind) >= 5 else volume
    volume_ratio = (volume / vol_ma5) if pd.notna(vol_ma5) and vol_ma5 != 0 else 0

    # B1 条件
    macd_ok = bool(pd.notna(macd_dea) and macd_dea > 0)
    kdj_ok_b1 = bool(pd.notna(kdj_j) and kdj_j < 13)  # B1: J < 13
    
    # B2 条件
    kdj_ok_b2 = bool(pd.notna(kdj_j) and kdj_j < 55)  # B2: J < 55
    price_ok = price_change_pct >= 4  # 涨幅 >= 4%
    vol_ok = volume_ratio >= 1.1  # 量比 >= 1.1

    # 根据战法选择条件
    if cfg.strategy == "B1":
        kdj_ok = kdj_ok_b1
        zhixing_required = True  # B1需要知行趋势线
    else:  # B2
        kdj_ok = kdj_ok_b2
        zhixing_required = False  # B2不需要知行趋势线

    conditions = {
        "mainboard_ok": mainboard_ok,
        "weekly_ok": weekly_info["weekly_ok"],
        "macd_dea_ok": macd_ok,
        "kdj_j_ok": kdj_ok,
        "zhixing_bullish": zhixing_bullish if zhixing_required else True,  # B1需要
        "golden_cross": golden_cross if cfg.require_golden_cross else True,
        "brick_white": bool(brick_white == 1),
        # B2特有
        "price_change_ok": price_ok if cfg.strategy == "B2" else True,
        "volume_ratio_ok": vol_ok if cfg.strategy == "B2" else True,
    }

    passed = all([
        conditions["mainboard_ok"],
        conditions["weekly_ok"],
        conditions["macd_dea_ok"],
        conditions["kdj_j_ok"],
        conditions["zhixing_bullish"],
        conditions["golden_cross"],
    ])
    
    # B2额外条件
    if cfg.strategy == "B2":
        passed = passed and conditions["price_change_ok"] and conditions["volume_ratio_ok"]
    
    # 如果开启白色砖头筛选
    if cfg.require_brick_white and not conditions["brick_white"]:
        passed = False

    return {
        "symbol": symbol,
        "name": name or symbol,
        "passed": passed,
        "error": "",
        "conditions": conditions,
        "weekly_reason": weekly_info["weekly_reason"],
        "metrics": {
            "date": latest["date"].strftime("%Y-%m-%d") if pd.notna(latest["date"]) else "",
            "close": float(close) if pd.notna(close) else None,
            "zhixing_white": float(zhixing_white) if pd.notna(zhixing_white) else None,
            "zhixing_yellow": float(zhixing_yellow) if pd.notna(zhixing_yellow) else None,
            "macd_dea": float(macd_dea) if pd.notna(macd_dea) else None,
            "kdj_j": float(kdj_j) if pd.notna(kdj_j) else None,
            "rsi14": float(latest.get("rsi14")) if pd.notna(latest.get("rsi14")) else None,
            "boll_upper": float(latest.get("boll_upper")) if pd.notna(latest.get("boll_upper")) else None,
            "boll_mid": float(latest.get("boll_mid")) if pd.notna(latest.get("boll_mid")) else None,
            "boll_lower": float(latest.get("boll_lower")) if pd.notna(latest.get("boll_lower")) else None,
            "volume": float(latest.get("volume")) if pd.notna(latest.get("volume")) else None,
            "vol_ma20": float(latest.get("vol_ma20")) if pd.notna(latest.get("vol_ma20")) else None,
            # B2 额外指标
            "price_change_pct": round(price_change_pct, 2),
            "volume_ratio": round(volume_ratio, 2),
            # 砖型图
            "brick_chart": float(brick_chart) if pd.notna(brick_chart) else None,
            "brick_white": bool(brick_white == 1),
            "ma30_w": weekly_info["ma30"],
            "ma60_w": weekly_info["ma60"],
            "ma120_w": weekly_info["ma120"],
            "ma240_w": weekly_info["ma240"],
        },
        "daily_df": daily_ind,
        "weekly_df": weekly,
    }


def scan_batch(symbol_rows: Iterable[Dict[str, str]], config: Optional[B1Config] = None) -> List[Dict[str, Any]]:
    cfg = config or B1Config()
    results: List[Dict[str, Any]] = []

    for row in symbol_rows:
        symbol = row.get("symbol", "").strip()
        name = row.get("name", "").strip()
        if not symbol:
            continue

        try:
            result = scan_symbol(symbol=symbol, name=name, config=cfg)
        except Exception as exc:
            result = {
                "symbol": symbol,
                "name": name or symbol,
                "passed": False,
                "error": str(exc),
            }

        results.append(result)

        if cfg.request_pause_sec > 0:
            time.sleep(cfg.request_pause_sec)

    return results


def flatten_result_for_table(result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("error"):
        return {
            "symbol": result.get("symbol"),
            "name": result.get("name"),
            "passed": False,
            "error": result.get("error", ""),
        }

    metrics = result.get("metrics", {})
    cond = result.get("conditions", {})

    return {
        "symbol": result.get("symbol"),
        "name": result.get("name"),
        "passed": result.get("passed", False),
        "date": metrics.get("date"),
        "close": metrics.get("close"),
        "zhixing_white": metrics.get("zhixing_white"),
        "zhixing_yellow": metrics.get("zhixing_yellow"),
        "macd_dea": metrics.get("macd_dea"),
        "kdj_j": metrics.get("kdj_j"),
        "rsi14": metrics.get("rsi14"),
        "mainboard_ok": cond.get("mainboard_ok"),
        "weekly_ok": cond.get("weekly_ok"),
        "macd_dea_ok": cond.get("macd_dea_ok"),
        "kdj_j_ok": cond.get("kdj_j_ok"),
        "zhixing_bullish": cond.get("zhixing_bullish"),
        "golden_cross": cond.get("golden_cross"),
        "weekly_reason": result.get("weekly_reason", ""),
        "error": result.get("error", ""),
    }
