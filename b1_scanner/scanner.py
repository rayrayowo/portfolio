#!/usr/bin/env python3
"""
B1战法选股器 v1.0

B1战法条件:
1. 沪深主板 (排除创业板、科创板)
2. 周线均线多头: MA30>MA60>MA120>MA240
3. 日线MACD(12,26,9) DEA > 0
4. 日线KDJ(9,3,3) J线 < 13

Author: BOSS (OpenClaw)
Date: 2026-03-08
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Tushare Token
TOKEN = "3a870845a82bc2a522a1b9dbc324df8b0be58390ac0088804243a615"

pro = ts.pro_api(TOKEN)


def get_daily_data(ts_code, days=250):
    """获取日线数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        return df
    except Exception as e:
        print(f"  ❌ 获取 {ts_code} 失败: {e}")
        return None


def get_weekly_data(ts_code, days=500):
    """获取周线数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    
    try:
        df = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        return df
    except:
        return None


def calculate_ma(data, window):
    """计算移动平均线"""
    return data['close'].rolling(window=window).mean()


def calculate_ema(data, span):
    """计算指数移动平均线"""
    return data['close'].ewm(span=span, adjust=False).mean()


def calculate_kdj(data):
    """计算KDJ指标"""
    low_n = data['low'].rolling(window=9).min()
    high_n = data['high'].rolling(window=9).max()
    
    rsv = (data['close'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    
    k = rsv.ewm(com=3, adjust=False).mean()
    d = k.ewm(com=3, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j


def calculate_macd(data, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=signal, adjust=False).mean()
    
    return diff, dea


def check_b1_criteria(ts_code, name):
    """检查是否符合B1战法条件"""
    print(f"\n🔍 检查 {name} ({ts_code})...")
    
    # 获取日线数据
    daily_df = get_daily_data(ts_code, days=250)
    if daily_df is None or len(daily_df) < 100:
        print(f"  ⚠️ 数据不足，跳过")
        return False
    
    # 获取周线数据
    weekly_df = get_weekly_data(ts_code, days=500)
    
    # 1. 检查周线均线多头排列
    if weekly_df is not None and len(weekly_df) >= 240:
        # 计算周线均线
        weekly_df['ma30'] = weekly_df['close'].rolling(30).mean()
        weekly_df['ma60'] = weekly_df['close'].rolling(60).mean()
        weekly_df['ma120'] = weekly_df['close'].rolling(120).mean()
        weekly_df['ma240'] = weekly_df['close'].rolling(240).mean()
        
        latest = weekly_df.iloc[-1]
        
        # 检查均线多头排列
        ma_check = (latest['ma30'] > latest['ma60'] > latest['ma120'] > latest['ma240'])
        
        if not ma_check:
            print(f"  ❌ 周线均线不满足多头排列")
            return False
        print(f"  ✅ 周线均线多头排列")
    else:
        print(f"  ⚠️ 周线数据不足，跳过周线检查")
    
    # 2. 检查日线MACD DEA > 0
    daily_df['diff'], daily_df['dea'] = calculate_macd(daily_df)
    latest_daily = daily_df.iloc[-1]
    
    if latest_daily['dea'] <= 0:
        print(f"  ❌ MACD DEA <= 0")
        return False
    print(f"  ✅ MACD DEA = {latest_daily['dea']:.4f} > 0")
    
    # 3. 检查日线KDJ J线 < 13
    daily_df['k'], daily_df['d'], daily_df['j'] = calculate_kdj(daily_df)
    latest_kdj = daily_df.iloc[-1]
    
    if latest_kdj['j'] >= 13:
        print(f"  ❌ KDJ J = {latest_kdj['j']:.2f} >= 13")
        return False
    print(f"  ✅ KDJ J = {latest_kdj['j']:.2f} < 13")
    
    # 4. 检查是否为沪深主板
    # 主板: 600xxx, 601xxx, 603xxx (上海), 000xxx (深圳)
    # 排除: 688xxx (科创板), 300xxx (创业板)
    
    if ts_code.startswith('688') or ts_code.startswith('300'):
        print(f"  ❌ 科创板/创业板，不符合")
        return False
    
    print(f"  ✅ 符合B1战法!")
    return True


def run_scanner(stock_list=None, limit=50):
    """运行选股器"""
    print("=" * 50)
    print("🔬 B1战法选股器 v1.0")
    print("=" * 50)
    
    # 如果没有指定股票列表，获取主板股票
    if stock_list is None:
        print("\n📥 获取股票列表...")
        try:
            df = pro.stock_basic(exchange='SSE', list_status='L', fields='ts_code,name')
            df2 = pro.stock_basic(exchange='SZSE', list_status='L', fields='ts_code,name')
            stock_list = pd.concat([df, df2])
            
            # 过滤: 主板 (600, 601, 603, 000)
            stock_list = stock_list[
                stock_list['ts_code'].str.match(r'^(600|601|603|000)\d{3}')
            ]
            
            stock_list = stock_list.head(limit)
            print(f"  获取到 {len(stock_list)} 只主板股票")
        except Exception as e:
            print(f"  ❌ 获取失败: {e}")
            return []
    
    # 筛选
    results = []
    for idx, row in stock_list.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        
        if check_b1_criteria(ts_code, name):
            results.append({'ts_code': ts_code, 'name': name})
        
        time.sleep(0.5)  # 避免请求过快
    
    return results


if __name__ == "__main__":
    # 测试: 先测试3只股票
    test_stocks = pd.DataFrame([
        {'ts_code': '600276.SH', 'name': '恒瑞医药'},
        {'ts_code': '600519.SH', 'name': '贵州茅台'},
        {'ts_code': '000001.SH', 'name': '平安银行'},
    ])
    
    results = run_scanner(test_stocks)
    
    print("\n" + "=" * 50)
    print(f"🎉 选股结果: {len(results)} 只符合B1战法")
    for r in results:
        print(f"  ✅ {r['name']} ({r['ts_code']})")
