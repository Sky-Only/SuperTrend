"""
ADX + SuperTrend 组合策略
========================
做多 = ST 翻多信号 AND ADX 多头区间 (+DI > -DI) AND ADX > 阈值
做空 = ST 翻空信号 AND ADX 空头区间 (-DI > +DI) AND ADX > 阈值
ADX < 阈值 (震荡市) → 平仓空仓, 不开新仓

ADX 只测趋势强度不带方向，方向由 +DI / -DI 给出：
  +DI > -DI → 多头区间 (类似 MA 快线 > 慢线)
  -DI > +DI → 空头区间 (类似 MA 快线 < 慢线)
  ADX > 阈值 → 趋势足够强才参与 (过滤震荡市)
  ADX < 阈值 → 震荡市, 空仓观望

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
# 1. ADX 方向性指标计算
# ============================================================

def _wilder(series, period):
    """
    Wilder 平滑 (与 backtest.py 的 ATR 一致)

    注意: 用 np.nanmean 跳过早期 NaN（numpy 的 .mean() 会传播 NaN，
    导致种子变 NaN 后一路传播，ADX 全变 NaN）。
    """
    raw = series.to_numpy(dtype=float)
    n = len(raw)
    out = np.full(n, np.nan)
    if n < period:
        return pd.Series(out, index=series.index)
    # 种子: 前 period 个值的均值 (跳过 NaN)
    out[period - 1] = np.nanmean(raw[:period])
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + raw[i]) / period
    return pd.Series(out, index=series.index)


def calc_adx(df, period=14):
    """
    计算 ADX 方向性指标

    返回: df 增加 plus_di, minus_di, adx, adx_trend (1=多头, -1=空头, 0=无)
    """
    df = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # ---- True Range ----
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]  # 第一条无前收盘

    # ---- Directional Movement ----
    up_move = high.diff()
    down_move = -low.diff()  # 前低 - 今低

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    # +DM: 今日上移 > 今日下移 且 上移 > 0
    cond_plus = (up_move > down_move) & (up_move > 0)
    plus_dm[cond_plus] = up_move[cond_plus]

    # -DM: 今日下移 > 今日上移 且 下移 > 0
    cond_minus = (down_move > up_move) & (down_move > 0)
    minus_dm[cond_minus] = down_move[cond_minus]

    # ---- Wilder 平滑 ----
    tr_smooth = _wilder(tr, period)
    plus_dm_smooth = _wilder(plus_dm, period)
    minus_dm_smooth = _wilder(minus_dm, period)

    # ---- 方向指标 ----
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    sum_di = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / sum_di.replace(0, np.nan)
    adx = _wilder(dx, period)

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx

    # 方向 (由 DI 决定，与 ADX 强度无关；强度阈值在回测中单独应用)
    df["adx_trend"] = 0
    valid = plus_di.notna() & minus_di.notna() & adx.notna()
    df.loc[valid & (plus_di > minus_di), "adx_trend"] = 1
    df.loc[valid & (minus_di > plus_di), "adx_trend"] = -1
    return df


# ============================================================
# 2. ADX + SuperTrend 组合回测引擎
# ============================================================

def run_backtest_combined(df, capital=1_000_000, adx_threshold=20.0, reenter=True):
    """
    回测 ADX + SuperTrend 组合策略

    信号规则 (t 日收盘决策, t+1 日执行):
      开多 = (ST 翻多 或 [reenter 且空仓时 ST 多头]) AND +DI > -DI AND ADX > 阈值
      开空 = (ST 翻空 或 [reenter 且空仓时 ST 空头]) AND -DI > +DI AND ADX > 阈值
      空仓 = ADX < 阈值 (震荡市) → 平掉现有仓位, 不开新仓

    参数:
      reenter: 空仓后若 ST 趋势仍有效且 ADX 恢复 > 阈值、DI 同向，
               立即重新入场 (不必等下一次 ST 翻转)。默认开启。
      reenter=False 时, 空仓后必须等下一次 ST 翻转才重新入场。

    交易执行规则：
      - 开多: t+1 HIGH 平空 + 开多
      - 开空: t+1 LOW 平多 + 开空
      - 空仓: t+1 平掉现有仓位 (平多用 LOW, 平空用 HIGH), 保持空仓
    """
    df = df.copy()

    # ---- 预提取数组 (运行时逐日决策, 感知仓位状态) ----
    trend = df["trend"].values
    trend_prev = np.roll(trend, 1)
    trend_prev[0] = 0
    adx = df["adx"].values
    adx_trend = df["adx_trend"].values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

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
        date = df["trade_date"].iloc[i]

        # Step 1: 执行前一日产生的信号
        if pending_signal is not None:
            signal_date = pending_signal_date

            if pending_signal == 1:
                exec_price = high[i]
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
                exec_price = low[i]
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

            elif pending_signal == 2:
                # 空仓: 平掉现有仓位, 不开新仓
                if position > 0:
                    # 平多 (卖出用 LOW, 保守)
                    exec_price = low[i]
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
                    position = 0
                    direction = None
                elif position < 0:
                    # 平空 (买回用 HIGH, 保守)
                    exec_price = high[i]
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
                    position = 0
                    direction = None

            pending_signal = None

        # Step 2: 根据今日数据 + 当前仓位, 决定今日信号 (明日执行)
        action = 0
        st_flip_long = (trend_prev[i] == -1) and (trend[i] == 1)
        st_flip_short = (trend_prev[i] == 1) and (trend[i] == -1)
        adx_strong = not np.isnan(adx[i]) and adx[i] > adx_threshold

        if adx_strong:
            # 强趋势: ST 翻转 + DI 同向 → 开仓
            if st_flip_long and adx_trend[i] == 1:
                action = 1
            elif st_flip_short and adx_trend[i] == -1:
                action = -1
            # 空仓重入: 空仓 + ADX 恢复 + ST 方向与 DI 一致
            elif reenter and position == 0 and trend[i] == adx_trend[i] and adx_trend[i] != 0:
                action = 1 if adx_trend[i] == 1 else -1
        else:
            # 震荡市: 有仓位则平仓空仓
            if position != 0:
                action = 2

        if action != 0:
            pending_signal = action
            pending_signal_date = date

        # Step 3: 按收盘价估值
        if direction == "long" and position > 0:
            mark_value = position * close[i]
        elif direction == "short" and position < 0:
            mark_value = equity + (entry_price - close[i]) * abs(position)
        else:
            mark_value = equity

        equity_curve.append({"trade_date": date, "equity": round(mark_value, 2)})

        if i > 0:
            prev_val = equity_curve[-2]["equity"]
            daily_returns.append((mark_value - prev_val) / prev_val if prev_val > 0 else 0)

    # 强制平仓
    if position != 0:
        exit_px = close[-1]
        final_date = df["trade_date"].iloc[-1]
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

def plot_combined_report(df, trades, name, st_period, st_mult,
                         adx_period, adx_threshold, output_dir,
                         output_file="研报_ADX_ST组合策略全览.png"):
    """
    研报专用图：ADX + ST 组合策略综合分析图（双面板）

    上图（价格面板）:
      - 收盘价 / SuperTrend线
      - 策略转折点标记（▲绿=买入开多, ▼红=卖出开空）
      - 趋势背景色（多头=淡绿, 空头=淡红）

    下图（ADX 面板）:
      - +DI / -DI（方向指标）与 ADX（强度指标）
      - ADX 阈值线 → 直观看出何时满足"强趋势"条件
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, (ax_price, ax_adx) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08}
    )

    dates = df["trade_date"]
    close = df["close"].values
    trend = df["trend"].values
    n = len(df)

    # ========== 上图: 价格 + 信号 ==========
    for i in range(n - 1):
        if trend[i] == 1:
            ax_price.axvspan(dates.iloc[i], dates.iloc[i + 1], color="green", alpha=0.05, lw=0)
        elif trend[i] == -1:
            ax_price.axvspan(dates.iloc[i], dates.iloc[i + 1], color="red", alpha=0.05, lw=0)

    ax_price.plot(dates, close, color="#2c3e50", linewidth=0.7, alpha=0.75, label="标的收盘价")
    ax_price.plot(dates, df["super_trend"].values, color="#e74c3c", linewidth=0.9, alpha=0.55, label="SuperTrend")

    # 策略转折点标记
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
    ax_price.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax_price.grid(True, alpha=0.3)
    ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax_price.set_title(
        f"{name} — ADX + SuperTrend 组合策略 (N={st_period}, M={st_mult}, ADX{adx_period}, 阈值={adx_threshold})",
        fontsize=14, fontweight="bold"
    )

    # ========== 下图: ADX 面板 ==========
    ax_adx.plot(dates, df["plus_di"].values, color="#27ae60", linewidth=1.0, label="+DI (多头强度)")
    ax_adx.plot(dates, df["minus_di"].values, color="#c0392b", linewidth=1.0, label="-DI (空头强度)")
    ax_adx.plot(dates, df["adx"].values, color="#2980b9", linewidth=1.3, label=f"ADX ({adx_period})")
    ax_adx.axhline(adx_threshold, color="gray", linestyle="--", linewidth=0.8,
                   label=f"ADX阈值={adx_threshold}")
    ax_adx.fill_between(dates, 0, df["adx"].values,
                        where=(df["adx"].values > adx_threshold),
                        color="blue", alpha=0.08)
    ax_adx.set_ylabel("指标值 (0-100)", fontsize=11)
    ax_adx.set_xlabel("日期", fontsize=12)
    ax_adx.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_adx.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{output_dir}/{output_file}", dpi=300, bbox_inches="tight")
    print(f"  研报图已保存: {output_dir}/{output_file}")
    plt.close(fig)


# ============================================================
# 4. 入口
# ============================================================

if __name__ == "__main__":
    # ==================== 在这里修改配置 ====================
    DATA_FILE = "沪深300_10年日线.parquet"   # 数据文件
    DATA_NAME = "沪深300"

    # SuperTrend 参数
    ST_PERIOD = 43
    ST_MULT   = 4.9

    # ADX 参数
    ADX_PERIOD   = 14       # ADX 计算周期
    ADX_THRESHOLD = 5.0    # 趋势强度阈值 (ADX > 阈值才算强趋势)
    REENTER     = False      # 空仓后 ADX 恢复且 ST/DI 同向时, 立即重新入场 (False = 等下一次 ST 翻转)

    CAPITAL   = 1_000_000
    # ========================================================

    # 1. 加载数据
    print(f"加载数据: {DATA_FILE}")
    df_raw = load_data(DATA_FILE)
    print(f"  数据: {len(df_raw)} 条, {df_raw['trade_date'].iloc[0].date()} ~ {df_raw['trade_date'].iloc[-1].date()}")

    # 2. 计算指标
    df = calc_super_trend(df_raw, period=ST_PERIOD, multiplier=ST_MULT)
    df = calc_adx(df, period=ADX_PERIOD)

    # 3. 纯 ST 策略（对比基准）
    trades_st, eq_st, rets_st = run_backtest(df, capital=CAPITAL)

    # 4. ADX + ST 组合策略
    trades_cb, eq_cb, rets_cb = run_backtest_combined(
        df, capital=CAPITAL, adx_threshold=ADX_THRESHOLD, reenter=REENTER)

    # 5. 计算原始指标用于对比
    def _flat_stats(rets):
        """空仓统计: 日收益为 0 的天数 = 无持仓天数 (首日补的 0 不算)"""
        arr = np.array(rets)
        flat = int((arr[1:] == 0).sum())
        total = max(len(arr) - 1, 1)
        return flat, flat / total

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
        m = mt.calc_all_metrics(eq_arr, rets_arr, pnl_pcts, capital, hold_days=hd)
        # 补充空仓统计
        flat_days, flat_pct = _flat_stats(rets)
        m["flat_days"] = flat_days
        m["flat_pct"] = flat_pct
        return m

    rm_st = _raw_metrics(trades_st, eq_st, rets_st, CAPITAL)
    rm_cb = _raw_metrics(trades_cb, eq_cb, rets_cb, CAPITAL)

    # 控制台对比
    print(f"\n{'='*90}")
    print(f"  {DATA_NAME} — 策略对比: 纯ST vs ST+ADX")
    print(f"{'='*90}")
    print(f"  ST: N={ST_PERIOD}, M={ST_MULT}    ADX: 周期={ADX_PERIOD}, 阈值={ADX_THRESHOLD}")
    print(f"  {'指标':<16} {'纯ST':>16} {'ST+ADX':>16} {'差值':>16} {'方向':>8}")

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
        ("空仓天数",   "flat_days",      "int",  "—"),
        ("空仓占比",   "flat_pct",       "pct",  "—"),
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

        if better == "↑":
            winner = "ST" if v_st > v_cb else ("ADX" if v_cb > v_st else "  —")
        elif better == "↓":
            winner = "ST" if v_st < v_cb else ("ADX" if v_cb < v_st else "  —")
        else:
            winner = "  —"

        print(f"  {label:<16}{_fmt(v_st, kind)} {_fmt(v_cb, kind)} {diff_str}  {winner}")

    print(f"  {'-'*88}")
    print(f"  ↑ = 越大越好    ↓ = 越小越好    — = 中性指标")
    print(f"  方向列: 哪个策略在该指标上更优")

    # 6. 生成两套报告
    output_st = f"{DATA_NAME}_st{ST_PERIOD}x{ST_MULT}"
    output_cb = f"{DATA_NAME}_adx{ADX_PERIOD}x{int(ADX_THRESHOLD)}_st{ST_PERIOD}x{ST_MULT}"

    print(f"\n{'='*70}")
    print(f"  生成纯ST策略报告 → {output_st}/")
    print(f"{'='*70}")
    generate_report(df, trades_st, eq_st, rets_st, CAPITAL,
                    DATA_NAME, ST_PERIOD, ST_MULT,
                    output_dir=output_st)

    print(f"\n{'='*70}")
    print(f"  生成ST+ADX组合策略报告 → {output_cb}/")
    print(f"{'='*70}")
    generate_report(df, trades_cb, eq_cb, rets_cb, CAPITAL,
                    f"{DATA_NAME}_ADX{ADX_PERIOD}", ST_PERIOD, ST_MULT,
                    output_dir=output_cb)

    # 7. 研报专用综合图
    print(f"\n{'='*70}")
    print(f"  生成研报综合图")
    print(f"{'='*70}")
    plot_combined_report(df, trades_cb, DATA_NAME, ST_PERIOD, ST_MULT,
                         ADX_PERIOD, ADX_THRESHOLD, output_dir=output_cb)
