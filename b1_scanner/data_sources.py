"""Data source adapters for Tushare and Yahoo Finance."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
import tushare as ts
import yfinance as yf

DEFAULT_TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")  # Set via environment variable


@dataclass
class FetchParams:
    symbol: str
    start: date
    end: date
    period: str = "daily"  # daily | weekly
    source: str = "tushare"  # tushare | yahoo
    tushare_token: Optional[str] = None


def _normalize_symbol_raw(symbol: str) -> str:
    if not symbol:
        raise ValueError("symbol 不能为空")
    return symbol.strip().upper()


def to_tushare_code(symbol: str) -> str:
    """Convert symbol to ts_code format (e.g. 600519.SH)."""
    symbol = _normalize_symbol_raw(symbol)

    if re.fullmatch(r"\d{6}\.(SH|SZ)", symbol):
        return symbol

    if re.fullmatch(r"\d{6}", symbol):
        if symbol.startswith(("6", "9")):
            return f"{symbol}.SH"
        return f"{symbol}.SZ"

    if re.fullmatch(r"\d{6}\.(SS|SZ)", symbol):
        return symbol.replace(".SS", ".SH")

    raise ValueError(f"无法转换为 Tushare 代码: {symbol}")


def to_yahoo_code(symbol: str) -> str:
    """Convert symbol to Yahoo format (e.g. 600519.SS)."""
    symbol = _normalize_symbol_raw(symbol)

    if re.fullmatch(r"\d{6}\.SH", symbol):
        return symbol.replace(".SH", ".SS")

    if re.fullmatch(r"\d{6}\.SZ", symbol):
        return symbol

    if re.fullmatch(r"\d{6}\.SS", symbol):
        return symbol

    if re.fullmatch(r"\d{6}", symbol):
        if symbol.startswith(("6", "9")):
            return f"{symbol}.SS"
        return f"{symbol}.SZ"

    # Keep non-CN tickers as they are (AAPL, TSLA, etc.)
    return symbol


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV columns into a shared schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    out = df.copy()

    rename_map = {
        "trade_date": "date",
        "vol": "volume",
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out = out.rename(columns=rename_map)

    # Flatten yfinance MultiIndex columns if needed.
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]
        out = out.rename(columns=rename_map)

    required = ["date", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[required]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _get_tushare_pro(token: Optional[str] = None):
    token = token or os.getenv("TUSHARE_TOKEN") or DEFAULT_TUSHARE_TOKEN
    if not token:
        raise ValueError("请提供 Tushare Token")
    return ts.pro_api(token)


def fetch_tushare_data(symbol: str, start: date, end: date, period: str = "daily", token: Optional[str] = None) -> pd.DataFrame:
    ts_code = to_tushare_code(symbol)
    pro = _get_tushare_pro(token)

    start_date = start.strftime("%Y%m%d")
    end_date = end.strftime("%Y%m%d")

    if period == "daily":
        raw = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    elif period == "weekly":
        raw = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
    else:
        raise ValueError("period 仅支持 daily/weekly")

    return normalize_ohlcv(raw)


def fetch_yahoo_data(symbol: str, start: date, end: date, period: str = "daily") -> pd.DataFrame:
    yf_code = to_yahoo_code(symbol)
    interval = "1d" if period == "daily" else "1wk"

    raw = yf.download(
        tickers=yf_code,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    raw = raw.reset_index()
    return normalize_ohlcv(raw)


def fetch_data(params: FetchParams) -> pd.DataFrame:
    source = params.source.lower().strip()
    if source == "tushare":
        return fetch_tushare_data(
            symbol=params.symbol,
            start=params.start,
            end=params.end,
            period=params.period,
            token=params.tushare_token,
        )

    if source == "yahoo":
        return fetch_yahoo_data(
            symbol=params.symbol,
            start=params.start,
            end=params.end,
            period=params.period,
        )

    raise ValueError(f"不支持的数据源: {params.source}")


def get_tushare_mainboard_stocks(limit: int = 100, token: Optional[str] = None) -> pd.DataFrame:
    """Fetch CN main-board symbols from Tushare."""
    pro = _get_tushare_pro(token)

    sse = pro.stock_basic(exchange="SSE", list_status="L", fields="ts_code,symbol,name")
    szse = pro.stock_basic(exchange="SZSE", list_status="L", fields="ts_code,symbol,name")

    stocks = pd.concat([sse, szse], ignore_index=True)
    stocks = stocks[stocks["ts_code"].str.match(r"^(600|601|603|000)\d{3}\.(SH|SZ)$", na=False)]
    stocks = stocks[~stocks["name"].str.contains("ST", na=False)]

    if limit > 0:
        stocks = stocks.head(limit)

    return stocks.reset_index(drop=True)
