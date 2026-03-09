"""Technical indicators used in B1 scanner v2.0."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_kdj(df: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    low_n = df["low"].rolling(window=n, min_periods=n).min()
    high_n = df["high"].rolling(window=n, min_periods=n).max()

    denom = (high_n - low_n).replace(0, np.nan)
    rsv = ((df["close"] - low_n) / denom * 100).fillna(50)

    k = rsv.ewm(alpha=1 / k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1 / d_smooth, adjust=False).mean()
    j = 3 * k - 2 * d

    return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(df["close"], fast)
    ema_slow = ema(df["close"], slow)

    diff = ema_fast - ema_slow
    dea = diff.ewm(span=signal, adjust=False).mean()
    hist = (diff - dea) * 2

    return pd.DataFrame({"macd_diff": diff, "macd_dea": dea, "macd_hist": hist})


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_boll(df: pd.DataFrame, period: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = ma(df["close"], period)
    std = df["close"].rolling(window=period, min_periods=period).std(ddof=0)

    upper = mid + n_std * std
    lower = mid - n_std * std

    return pd.DataFrame({"boll_mid": mid, "boll_upper": upper, "boll_lower": lower})


def calc_vol(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vol_ma5": df["volume"].rolling(5, min_periods=5).mean(),
            "vol_ma10": df["volume"].rolling(10, min_periods=10).mean(),
            "vol_ma20": df["volume"].rolling(20, min_periods=20).mean(),
        }
    )


def calc_zhixing_white(df: pd.DataFrame) -> pd.Series:
    """知行白线(短期趋势): EMA(EMA(C,10),10)."""
    return ema(ema(df["close"], 10), 10)


def calc_zhixing_yellow(df: pd.DataFrame) -> pd.Series:
    """知行黄线(长期趋势): (MA14 + MA28 + MA57 + MA114) / 4."""
    return (
        ma(df["close"], 14)
        + ma(df["close"], 28)
        + ma(df["close"], 57)
        + ma(df["close"], 114)
    ) / 4


def calc_zhixing_trend(df: pd.DataFrame) -> pd.DataFrame:
    """知行趋势线组件: 白线 + 黄线."""
    white = calc_zhixing_white(df)
    yellow = calc_zhixing_yellow(df)
    return pd.DataFrame({"zhixing_white": white, "zhixing_yellow": yellow})


def calc_brick_chart(df: pd.DataFrame) -> pd.DataFrame:
    """
    砖型图 (Brick Chart) - 富途副图版
    
    用法:
    - 白色砖头 = 买点 (昨绿今红，且今天红砖高度 > 昨天绿砖高度的 2/3)
    - 红砖 = 持有
    - 绿砖 = 空仓
    
    公式:
    DEN := HHV(H,4) - LLV(L,4);
    VAR1A := IF(DEN=0, 0, (HHV(H,4)-C)/DEN*100 - 90);
    VAR2A := SMA(VAR1A,4,1) + 100;
    VAR3A := IF(DEN=0, 0, (C-LLV(L,4))/DEN*100);
    VAR4A := SMA(VAR3A,6,1);
    VAR5A := SMA(VAR4A,6,1) + 100;
    VAR6A := VAR5A - VAR2A;
    砖型图 := IF(VAR6A>4, VAR6A-4, 0);
    """
    
    # 计算基础数据
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # DEN := HHV(H,4) - LLV(L,4)
    den = high.rolling(window=4, min_periods=4).max() - low.rolling(window=4, min_periods=4).min()
    
    # VAR1A := IF(DEN=0, 0, (HHV(H,4)-C)/DEN*100 - 90)
    hh = high.rolling(window=4, min_periods=4).max()
    var1a = pd.Series(np.where(den == 0, 0, (hh - close) / den * 100 - 90), index=df.index)
    
    # VAR2A := SMA(VAR1A,4,1) + 100
    var2a = var1a.rolling(window=4, min_periods=4).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True) + 100
    
    # VAR3A := IF(DEN=0, 0, (C-LLV(L,4))/DEN*100)
    ll = low.rolling(window=4, min_periods=4).min()
    var3a = pd.Series(np.where(den == 0, 0, (close - ll) / den * 100), index=df.index)
    
    # VAR4A := SMA(VAR3A,6,1)
    var4a = var3a.rolling(window=6, min_periods=6).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
    
    # VAR5A := SMA(VAR4A,6,1) + 100
    var5a = var4a.rolling(window=6, min_periods=6).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True) + 100
    
    # VAR6A := VAR5A - VAR2A
    var6a = var5a - var2a
    
    # 砖型图 := IF(VAR6A>4, VAR6A-4, 0)
    brick = pd.Series(np.where(var6a > 4, var6a - 4, 0), index=df.index)
    
    # 计算信号
    # PRE := REF(砖型图,1)
    # PRE2 := REF(砖型图,2)
    prev_brick = brick.shift(1)
    prev2_brick = brick.shift(2)
    
    # TODAY_RED := (PRE < 砖型图)  今天红砖
    today_red = prev_brick < brick
    
    # YEST_GREEN := (PRE2 > PRE)  昨天绿
    yest_green = prev2_brick > prev_brick
    
    # RED_LEN := 砖型图 - PRE
    red_len = brick - prev_brick
    
    # GREEN_LEN := PRE2 - PRE
    green_len = prev2_brick - prev_brick
    
    # WHITE_SIG := TODAY_RED AND YEST_GREEN AND (RED_LEN > GREEN_LEN * 2 / 3)
    # 白色砖头信号 = 买点!
    white_sig = today_red & yest_green & (red_len > green_len * 2 / 3)
    
    return pd.DataFrame({
        "brick_chart": brick,
        "brick_white": white_sig.astype(int),  # 白色砖头 = 1 (买点)
        "brick_red": today_red.astype(int),     # 红砖
    })


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    kdj = calc_kdj(out)
    macd = calc_macd(out)
    boll = calc_boll(out)
    vol = calc_vol(out)

    out = pd.concat([out, kdj, macd, boll, vol], axis=1)
    out["rsi14"] = calc_rsi(out, period=14)
    zhixing = calc_zhixing_trend(out)
    out = pd.concat([out, zhixing], axis=1)
    
    # 添加砖型图
    brick = calc_brick_chart(out)
    out = pd.concat([out, brick], axis=1)

    return out
