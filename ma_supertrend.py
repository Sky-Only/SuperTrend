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
import metrics as mt
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
# 3. 研报专用综合图
# ============================================================

def plot_combined_report(df, trades, equity_curve, name, st_period, st_mult,
                         ma_fast, ma_slow, capital, output_dir,
                         output_file="研报_MA_ST组合策略全览.png"):
    """
    研报专用图：MA + ST 组合策略价格全览图

    内容:
      - 收盘价 / MA快线 / MA慢线 / SuperTrend线
      - 策略转折点标记（▲绿=买入开多, ▼红=卖出开空）
      - 趋势背景色（多头=淡绿, 空头=淡红）→ 直观看出何时处于什么策略
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, ax_price = plt.subplots(figsize=(18, 9))

    dates = df["trade_date"]
    close = df["close"].values
    ma_f = df["ma_fast"].values
    ma_s = df["ma_slow"].values
    trend = df["trend"].values
    n = len(df)

    # 趋势背景色（ST 多头/空头区间）
    for i in range(n - 1):
        if trend[i] == 1:
            ax_price.axvspan(dates.iloc[i], dates.iloc[i + 1], color="green", alpha=0.05, lw=0)
        elif trend[i] == -1:
            ax_price.axvspan(dates.iloc[i], dates.iloc[i + 1], color="red", alpha=0.05, lw=0)

    ax_price.plot(dates, close, color="#2c3e50", linewidth=0.7, alpha=0.75, label="标的收盘价")
    ax_price.plot(dates, ma_f, color="#0099ff", linewidth=1.3, label=f"MA快线 ({ma_fast})")
    ax_price.plot(dates, ma_s, color="#ff7700", linewidth=1.3, label=f"MA慢线 ({ma_slow})")
    ax_price.plot(dates, df["super_trend"].values, color="#e74c3c", linewidth=0.9, alpha=0.55, label="SuperTrend")

    # 策略转折点标记
    # direction=="short" → 平空开多 → 买入信号（绿▲）
    # direction=="long"  → 平多开空 → 卖出信号（红▼）
    buy_plotted = False
    sell_plotted = False
    for t in trades:
        ex_date = t["exec_date"]
        if ex_date is None:
            continue
        mask = df["trade_date"] == ex_date
        if not mask.any():
            continue
        px_at = df.loc[mask, "close"].values[0]
        if t["direction"] == "short":
            ax_price.scatter(ex_date, px_at, marker="^", color="#27ae60", s=110, zorder=6,
                             edgecolors="white", linewidth=0.8,
                             label="买入 (平空开多)" if not buy_plotted else None)
            buy_plotted = True
        else:
            ax_price.scatter(ex_date, px_at, marker="v", color="#c0392b", s=110, zorder=6,
                             edgecolors="white", linewidth=0.8,
                             label="卖出 (平多开空)" if not sell_plotted else None)
            sell_plotted = True

    ax_price.set_ylabel("价格", fontsize=12)
    ax_price.set_xlabel("日期", fontsize=12)
    ax_price.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax_price.grid(True, alpha=0.3)
    ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax_price.set_title(
        f"{name} — MA + SuperTrend 组合策略 (N={st_period}, M={st_mult}, MA{ma_fast}/{ma_slow})",
        fontsize=14, fontweight="bold"
    )

    fig.tight_layout()
    fig.savefig(f"{output_dir}/{output_file}", dpi=300, bbox_inches="tight")
    print(f"  研报图已保存: {output_dir}/{output_file}")
    plt.close(fig)


# ============================================================
# 3. 入口
# ============================================================

if __name__ == "__main__":
    # ==================== 在这里修改配置 ====================
    DATA_NAME = "标普500"
    DATA_FILE = f"{DATA_NAME}_10年日线.parquet"   # 数据文件
    

    # SuperTrend 参数
    ST_PERIOD = 18
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

    # 4. ST + MA 组合策略
    trades_cb, eq_cb, rets_cb = run_backtest_combined(df, capital=CAPITAL)

    # 5. 计算原始指标用于对比
    def _raw_metrics(trades, eq, rets, capital):
        eq_arr = np.array([e["equity"] for e in eq])
        rets_arr = np.array(rets)
        pnl_pcts = np.array([t["pnl_pct"] for t in trades]) if trades else np.array([])
        hd = None
        if trades:
            hd_list = []
            for t in trades:
                ed = t.get("entry_date")
                xd = t.get("exec_date")
                if ed is not None and xd is not None:
                    hd_list.append((pd.Timestamp(xd) - pd.Timestamp(ed)).days)
            if hd_list:
                hd = np.array(hd_list, dtype=float)
        return mt.calc_all_metrics(eq_arr, rets_arr, pnl_pcts, capital, hold_days=hd)

    rm_st = _raw_metrics(trades_st, eq_st, rets_st, CAPITAL)
    rm_cb = _raw_metrics(trades_cb, eq_cb, rets_cb, CAPITAL)

    # 控制台对比
    print(f"\n{'='*90}")
    print(f"  {DATA_NAME} — 策略对比: 纯ST vs ST+MA")
    print(f"{'='*90}")
    print(f"  ST: N={ST_PERIOD}, M={ST_MULT}    MA: 快线={MA_FAST}, 慢线={MA_SLOW}")
    print(f"  {'指标':<16} {'纯ST':>16} {'ST+MA':>16} {'差值':>16} {'方向':>8}")

    rows = [
        ("总收益率",   "total_return",  "pct",  "↑"),
        ("年化收益率", "annual_return",  "pct",  "↑"),
        ("最终净值",   "final_equity",   "int",  "↑"),
        ("最大回撤",   "max_dd",         "pct",  "↑"),
        ("回撤持续天数","max_dd_days",   "int",  "↓"),
        ("年化波动率", "daily_vol",      "pct",  "↓"),
        ("夏普比率",   "sharpe",         "num",  "↑"),
        ("卡尔玛比率", "calmar",         "num",  "↑"),
        ("交易次数",   "n_trades",       "int",  "—"),
        ("胜率",       "win_rate",       "pct",  "↑"),
        ("平均盈利",   "avg_win",        "pct",  "↑"),
        ("平均亏损",   "avg_loss",       "pct",  "↓"),
        ("盈亏比",     "profit_factor",  "num",  "↑"),
        ("平均持仓天数","avg_hold_days",  "num",  "—"),
        ("回测天数",   "n_days",         "int",  "—"),
    ]

    def _fmt(val, kind):
        if kind == "pct":
            return f"{val:>15.2%}"
        elif kind == "int":
            return f"{val:>15,.0f}" if val >= 1000 else f"{val:>15.0f}"
        else:
            return f"{val:>15.2f}"

    for label, key, kind, better in rows:
        v_st = rm_st[key]
        v_cb = rm_cb[key]
        diff = v_cb - v_st
        if kind in ("pct",):
            diff_str = f"{diff:>+15.2%}"
        elif kind == "int":
            diff_str = f"{diff:>+15,.0f}" if abs(diff) >= 1000 else f"{diff:>+15.0f}"
        else:
            diff_str = f"{diff:>+15.2f}"

        # 判断哪个更好
        if better == "↑":
            winner = "ST" if v_st > v_cb else ("MA" if v_cb > v_st else "  —")
        elif better == "↓":
            winner = "ST" if v_st < v_cb else ("MA" if v_cb < v_st else "  —")
        else:
            winner = "  —"

        print(f"  {label:<16}{_fmt(v_st, kind)} {_fmt(v_cb, kind)} {diff_str}  {winner}")

    print(f"  {'-'*88}")
    print(f"  ↑ = 越大越好    ↓ = 越小越好    — = 中性指标")
    print(f"  方向列: 哪个策略在该指标上更优")

    # 6. 生成两套报告
    output_st = f"{DATA_NAME}_st{ST_PERIOD}x{ST_MULT}"
    output_cb = f"{DATA_NAME}_ma{MA_FAST}x{MA_SLOW}_st{ST_PERIOD}x{ST_MULT}"

    print(f"\n{'='*70}")
    print(f"  生成纯ST策略报告 → {output_st}/")
    print(f"{'='*70}")
    generate_report(df, trades_st, eq_st, rets_st, CAPITAL,
                    DATA_NAME, ST_PERIOD, ST_MULT,
                    output_dir=output_st)

    print(f"\n{'='*70}")
    print(f"  生成ST+MA组合策略报告 → {output_cb}/")
    print(f"{'='*70}")
    generate_report(df, trades_cb, eq_cb, rets_cb, CAPITAL,
                    f"{DATA_NAME}_MA{MA_FAST}/{MA_SLOW}", ST_PERIOD, ST_MULT,
                    output_dir=output_cb)

    # 7. 研报专用综合图（MA + ST 组合策略全览）
    print(f"\n{'='*70}")
    print(f"  生成研报综合图")
    print(f"{'='*70}")
    plot_combined_report(df, trades_cb, eq_cb, DATA_NAME, ST_PERIOD, ST_MULT,
                         MA_FAST, MA_SLOW, CAPITAL, output_dir=output_cb)
