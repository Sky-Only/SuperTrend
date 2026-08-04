"""
MA + SuperTrend 组合策略
=======================
做多 = ST 翻多信号 AND MA 多头区间 (快线 > 慢线)
做空 = ST 翻空信号 AND MA 空头区间 (快线 < 慢线)
两个条件同时满足才执行，否则忽略信号。

复用了 backtest.py 的全部绘图模块，同时输出纯 ST 策略和组合策略两套报告。
"""

import numpy as np
import pandas as pd
from backtest import (
    load_data, calc_super_trend, run_backtest, calc_metrics,
    generate_report,
)


# ============================================================
# 1. MA 均线计算
# ============================================================

def calc_ma(df, fast=5, slow=20):
    """
    计算快慢均线及 MA 趋势方向

    返回: df 增加 ma_fast, ma_slow, ma_trend (1=多头, -1=空头, 0=无)
    """
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(fast, min_periods=1).mean()
    df["ma_slow"] = df["close"].rolling(slow, min_periods=1).mean()

    df["ma_trend"] = 0
    valid = df["ma_fast"].notna() & df["ma_slow"].notna()
    df.loc[valid & (df["ma_fast"] > df["ma_slow"]), "ma_trend"] = 1
    df.loc[valid & (df["ma_fast"] < df["ma_slow"]), "ma_trend"] = -1
    return df


# ============================================================
# 2. MA + SuperTrend 组合回测引擎
# ============================================================

def run_backtest_combined(df, capital=1_000_000):
    """
    回测 MA + SuperTrend 组合策略

    与 run_backtest 的唯一区别：
      纯 ST:  ST 趋势翻转 → 无条件产生信号
      组合:   ST 趋势翻转 AND MA 区间同向 → 才产生信号

    交易执行规则不变：
      - t 日收盘触发信号，t+1 日执行
      - 信号=1: t+1 HIGH 平空 + 开多
      - 信号=-1: t+1 LOW 平多 + 开空
      - 始终持仓（多或空）
    """
    df = df.copy()

    # ---- 信号生成 ----
    df["trend_prev"] = df["trend"].shift(1)
    df["signal"] = 0

    # ST 翻转条件
    st_flip_long = (df["trend_prev"] == -1) & (df["trend"] == 1)
    st_flip_short = (df["trend_prev"] == 1) & (df["trend"] == -1)

    # 组合策略: ST 翻转 + MA 确认
    ma_long = df["ma_trend"] == 1
    ma_short = df["ma_trend"] == -1
    df.loc[st_flip_long & ma_long, "signal"] = 1
    df.loc[st_flip_short & ma_short, "signal"] = -1

    # 首个趋势确立日: 也需要 MA 确认
    first_valid = df[df["trend"] != 0].index[0]
    if first_valid < len(df):
        st_dir = df.at[first_valid, "trend"]
        ma_dir = df.at[first_valid, "ma_trend"]
        if st_dir == 1 and ma_dir == 1:
            df.at[first_valid, "signal"] = 1
        elif st_dir == -1 and ma_dir == -1:
            df.at[first_valid, "signal"] = -1

    # ---- 模拟交易（t 日信号 → t+1 日执行）----
    trades = []
    equity = capital
    position = 0
    entry_price = 0
    entry_date = None
    direction = None
    pending_signal = None

    equity_curve = []
    daily_returns = []

    for i in range(len(df)):
        row = df.iloc[i]
        date = row["trade_date"]

        # Step 1: 执行前一日产生的信号
        if pending_signal is not None:
            signal_date = pending_signal_date

            if pending_signal == 1:
                exec_price = row["high"]
                if position < 0:
                    pnl = (entry_price - exec_price) * abs(position)
                    pnl_pct = (entry_price - exec_price) / entry_price
                    trades.append({
                        "signal_date": signal_date,
                        "exec_date": date,
                        "entry_date": entry_date,
                        "direction": "short",
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exec_price, 4),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 6),
                    })
                    equity += pnl
                position = equity / exec_price
                entry_price = exec_price
                entry_date = date
                direction = "long"

            elif pending_signal == -1:
                exec_price = row["low"]
                if position > 0:
                    pnl = (exec_price - entry_price) * position
                    pnl_pct = (exec_price - entry_price) / entry_price
                    trades.append({
                        "signal_date": signal_date,
                        "exec_date": date,
                        "entry_date": entry_date,
                        "direction": "long",
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exec_price, 4),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 6),
                    })
                    equity += pnl
                position = -equity / exec_price
                entry_price = exec_price
                entry_date = date
                direction = "short"

            pending_signal = None

        # Step 2: 记录今日信号
        sig = row["signal"]
        if sig != 0:
            pending_signal = sig
            pending_signal_date = date

        # Step 3: 按收盘价估值
        if direction == "long" and position > 0:
            mark_value = position * row["close"]
        elif direction == "short" and position < 0:
            mark_value = equity + (entry_price - row["close"]) * abs(position)
        else:
            mark_value = equity

        equity_curve.append({"trade_date": date, "equity": round(mark_value, 2)})

        if i > 0:
            prev_val = equity_curve[-2]["equity"]
            daily_returns.append((mark_value - prev_val) / prev_val if prev_val > 0 else 0)

    # 强制平仓
    if position != 0:
        last_row = df.iloc[-1]
        exit_px = last_row["close"]
        final_date = last_row["trade_date"]
        if position > 0:
            pnl = (exit_px - entry_price) * position
        else:
            pnl = (entry_price - exit_px) * abs(position)
        pnl_pct = pnl / (entry_price * abs(position))
        trades.append({
            "signal_date": None,
            "exec_date": final_date,
            "entry_date": entry_date,
            "direction": direction,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_px, 4),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 6),
        })
        equity += pnl
        equity_curve[-1]["equity"] = round(equity, 2)

    daily_returns.insert(0, 0)

    return trades, equity_curve, daily_returns


# ============================================================
# 3. 入口
# ============================================================

if __name__ == "__main__":
    # ==================== 在这里修改配置 ====================
    DATA_FILE = "沪深300_10年日线.parquet"   # 数据文件
    DATA_NAME = "沪深300"

    # SuperTrend 参数
    ST_PERIOD = 43
    ST_MULT   = 4.9

    # MA 参数
    MA_FAST   = 5
    MA_SLOW   = 20

    CAPITAL   = 1_000_000
    # ========================================================

    # 1. 加载数据
    print(f"加载数据: {DATA_FILE}")
    df_raw = load_data(DATA_FILE)
    print(f"  数据: {len(df_raw)} 条, {df_raw['trade_date'].iloc[0].date()} ~ {df_raw['trade_date'].iloc[-1].date()}")

    # 2. 计算指标
    df = calc_super_trend(df_raw, period=ST_PERIOD, multiplier=ST_MULT)
    df = calc_ma(df, fast=MA_FAST, slow=MA_SLOW)

    # 3. 纯 ST 策略（对比基准）
    trades_st, eq_st, rets_st = run_backtest(df, capital=CAPITAL)
    metrics_st = calc_metrics(trades_st, eq_st, rets_st, CAPITAL)

    # 4. ST + MA 组合策略
    trades_cb, eq_cb, rets_cb = run_backtest_combined(df, capital=CAPITAL)
    metrics_cb = calc_metrics(trades_cb, eq_cb, rets_cb, CAPITAL)

    # 5. 控制台对比
    print(f"\n{'='*70}")
    print(f"  策略对比")
    print(f"{'='*70}")
    print(f"  {'指标':<16} {'纯ST (N={ST_PERIOD},M={ST_MULT})':>24} {'ST+MA (MA{MA_FAST}/{MA_SLOW})':>24}")
    print(f"  {'-'*66}")
    keys = ["总收益率", "年化收益率", "最大回撤", "夏普比率", "交易次数", "胜率", "盈亏比"]
    for k in keys:
        print(f"  {k:<16} {metrics_st[k]:>24} {metrics_cb[k]:>24}")

    # 6. 生成两套报告
    print(f"\n{'='*70}")
    print(f"  生成纯ST策略报告")
    print(f"{'='*70}")
    generate_report(df, trades_st, eq_st, rets_st, CAPITAL,
                    DATA_NAME, ST_PERIOD, ST_MULT,
                    output_dir="output_st")

    print(f"\n{'='*70}")
    print(f"  生成ST+MA组合策略报告")
    print(f"{'='*70}")
    generate_report(df, trades_cb, eq_cb, rets_cb, CAPITAL,
                    f"{DATA_NAME}_MA{MA_FAST}/{MA_SLOW}", ST_PERIOD, ST_MULT,
                    output_dir="output_ma_st")
