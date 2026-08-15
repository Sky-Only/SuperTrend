"""
吊灯止损 (Chandelier Exit) 只做多策略
====================================
吊灯止损线 = 近 N 日最高价 - K × ATR
  价格创新高时, 止损线像吊灯一样挂在最高价下方, 随价格上移;
  收盘跌破吊灯线 → 止盈/止损离场。

策略规则 (只做多, 不做空):
  入场: MA 金叉 (快线上穿慢线), 进入买入区间才开多
  离场: 收盘价跌破吊灯止损线 (近 N 日最高价 - K × ATR)

交易执行:
  - t 日收盘触发信号, t+1 日执行
  - 开多用 t+1 HIGH (保守), 平多用 t+1 LOW (保守)
  - 空仓期间现金按年化 1.5% 逐日复利计息
  - 单边手续费万三 (买入、卖出各收一次)

说明: 吊灯止损本质是「离场/止损」机制, 入场用的是标准突破规则。
      复用了 dynamic_atr.py 的只做多回测引擎 (与之前完全一致的交易规则)。
"""

import numpy as np
import pandas as pd
import metrics as mt
from backtest import load_data, generate_report
from dynamic_atr import run_backtest_long_only


# ============================================================
# 1. 吊灯止损信号计算
# ============================================================

def calc_chandelier(df, n=22, k=3.0, ma_fast=5, ma_slow=20):
    """
    计算吊灯止损信号 (只做多)

    参数:
        n: 回看天数 (最高价窗口 + ATR 周期), 业界常用 22
        k: ATR 倍数, 业界常用 3
        ma_fast: 快均线周期 (入场信号), 业界常用 5
        ma_slow: 慢均线周期 (入场信号), 业界常用 20

    入场: MA 金叉 (快线上穿慢线) → 进入买入区间
    离场: 收盘跌破吊灯止损线 (近 N 日最高价 - K × ATR)

    返回: df 增加 atr, hh, ma_fast, ma_slow, chandelier_stop, super_trend, trend(1=多, -1=空仓)
    """
    df = df.copy()
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    m = len(df)

    # ---- Wilder ATR (周期 = n) ----
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))
    atr = np.full(m, np.nan)
    atr[n - 1] = tr[:n].mean()
    for t in range(n, m):
        atr[t] = (atr[t - 1] * (n - 1) + tr[t]) / n

    # ---- 最高价 (用于吊灯止损线) ----
    hh = pd.Series(high).rolling(n, min_periods=n).max().values  # 近 N 日最高价 (含今日)

    # ---- 吊灯止损线 ----
    chandelier_stop = hh - k * atr

    # ---- MA 均线 (入场信号) ----
    ma_fast_vals = pd.Series(close).rolling(ma_fast, min_periods=1).mean().values
    ma_slow_vals = pd.Series(close).rolling(ma_slow, min_periods=1).mean().values

    # ---- 状态机: 1=多头, -1=空仓 ----
    trend = np.full(m, -1)
    state = -1
    for i in range(m):
        if state == -1:
            # 入场: MA 金叉 (快线上穿慢线) → 进入买入区间才开多
            if (i > 0
                    and ma_fast_vals[i] > ma_slow_vals[i]
                    and ma_fast_vals[i - 1] <= ma_slow_vals[i - 1]):
                state = 1
        else:
            # 离场: 收盘跌破吊灯止损线
            if (not np.isnan(chandelier_stop[i])) and close[i] < chandelier_stop[i]:
                state = -1
        trend[i] = state

    df["atr"] = atr
    df["hh"] = hh
    df["ma_fast"] = ma_fast_vals
    df["ma_slow"] = ma_slow_vals
    df["chandelier_stop"] = chandelier_stop
    df["super_trend"] = chandelier_stop  # 供 generate_report 的信号图展示吊灯线
    df["trend"] = trend
    return df


# ============================================================
# 2. 研报综合图
# ============================================================

def plot_chandelier_report(df, trades, eq, name, n, k, ma_fast, ma_slow, output_dir,
                           output_file="研报_吊灯止损策略全览.png"):
    """
    研报综合图（双面板）

    上图: 收盘价 + 吊灯止损线 + 交易信号
    下图: 策略净值 vs 标的 Buy&Hold
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, (ax_price, ax_eq) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08}
    )

    dates = df["trade_date"]
    close = df["close"].values
    trend = df["trend"].values
    nrows = len(df)

    # 多头持仓背景色
    for i in range(nrows - 1):
        if trend[i] == 1:
            ax_price.axvspan(dates.iloc[i], dates.iloc[i + 1], color="green", alpha=0.05, lw=0)

    ax_price.plot(dates, close, color="#2c3e50", linewidth=0.7, alpha=0.75, label="标的收盘价")
    ax_price.plot(dates, df["chandelier_stop"].values, color="#e74c3c", linewidth=0.9,
                  alpha=0.85, label=f"吊灯止损线 (HH{n} - {k}·ATR)")
    ax_price.plot(dates, df["ma_fast"].values, color="#0099ff", linewidth=1.1, label=f"MA快线 ({ma_fast})")
    ax_price.plot(dates, df["ma_slow"].values, color="#ff7700", linewidth=1.1, label=f"MA慢线 ({ma_slow})")

    # 交易信号
    buy_plotted = False
    sell_plotted = False
    for t in trades:
        en_date = t.get("entry_date")
        if en_date is not None:
            mask = df["trade_date"] == en_date
            if mask.any():
                px = df.loc[mask, "close"].values[0]
                ax_price.scatter(en_date, px, marker="^", color="#27ae60", s=110, zorder=6,
                                 edgecolors="white", linewidth=0.8,
                                 label="MA金叉买入开多" if not buy_plotted else None)
                buy_plotted = True
        ex_date = t.get("exec_date")
        if ex_date is not None:
            mask = df["trade_date"] == ex_date
            if mask.any():
                px = df.loc[mask, "close"].values[0]
                ax_price.scatter(ex_date, px, marker="v", color="#c0392b", s=110, zorder=6,
                                 edgecolors="white", linewidth=0.8,
                                 label="吊灯止损离场" if not sell_plotted else None)
                sell_plotted = True

    ax_price.set_ylabel("价格", fontsize=12)
    ax_price.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax_price.grid(True, alpha=0.3)
    ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax_price.set_title(
        f"{name} — 吊灯止损(Chandelier Exit) 只做多 (N={n}, K={k}, MA{ma_fast}/{ma_slow})",
        fontsize=14, fontweight="bold"
    )

    # ---- 下图: 净值 ----
    eq_dates = [e["trade_date"] for e in eq]
    eq_vals = np.array([e["equity"] for e in eq]) / eq[0]["equity"]
    bench = close / close[0]
    ax_eq.plot(eq_dates, eq_vals, color="#e74c3c", linewidth=1.0, label="策略净值")
    ax_eq.plot(dates, bench, color="#34495e", linewidth=0.8, linestyle="--",
               alpha=0.7, label="标的 Buy&Hold")
    ax_eq.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax_eq.set_ylabel("净值 (初始=1)", fontsize=11)
    ax_eq.set_xlabel("日期", fontsize=12)
    ax_eq.legend(loc="upper left", fontsize=9)
    ax_eq.grid(True, alpha=0.3)
    ax_eq.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    fig.savefig(f"{output_dir}/{output_file}", dpi=300, bbox_inches="tight")
    print(f"  研报图已保存: {output_dir}/{output_file}")
    plt.close(fig)


# ============================================================
# 3. 入口
# ============================================================

if __name__ == "__main__":
    # ==================== 在这里修改配置 ====================
    DATA_NAME = "沪深300"
    DATA_FILE = f"{DATA_NAME}_10年日线.parquet"   # 数据文件

    # 吊灯止损参数
    N = 22        # 回看天数 (最高价窗口 + ATR 周期)
    K = 3.0       # ATR 倍数

    # 入场信号 (MA 均线法)
    MA_FAST = 5    # 快均线周期
    MA_SLOW = 20   # 慢均线周期

    CAPITAL = 1_000_000

    # 交易规则 (只做多)
    FLAT_ANNUAL = 0.015    # 空仓期间现金年化收益
    FEE_RATE    = 0.0003   # 单边手续费 (万三 = 0.03%)
    # ========================================================

    # 1. 加载数据
    print(f"加载数据: {DATA_FILE}")
    df_raw = load_data(DATA_FILE)
    print(f"  数据: {len(df_raw)} 条, {df_raw['trade_date'].iloc[0].date()} ~ {df_raw['trade_date'].iloc[-1].date()}")

    # 2. 计算吊灯止损信号
    df = calc_chandelier(df_raw, n=N, k=K, ma_fast=MA_FAST, ma_slow=MA_SLOW)

    # 3. 回测 (复用只做多引擎)
    trades, eq, rets = run_backtest_long_only(
        df, capital=CAPITAL, flat_annual=FLAT_ANNUAL, commission=FEE_RATE)

    # 4. 绩效指标
    def _raw_metrics(trades, eq, rets, capital):
        eq_arr = np.array([e["equity"] for e in eq])
        rets_arr = np.array(rets)
        pnl_pcts = np.array([t["pnl_pct"] for t in trades]) if trades else np.array([])
        hd = None
        if trades:
            hd_list = [
                (pd.Timestamp(t["exec_date"]) - pd.Timestamp(t["entry_date"])).days
                for t in trades
                if t.get("entry_date") is not None and t.get("exec_date") is not None
            ]
            if hd_list:
                hd = np.array(hd_list, dtype=float)
        return mt.calc_all_metrics(eq_arr, rets_arr, pnl_pcts, capital, hold_days=hd)

    m = _raw_metrics(trades, eq, rets, CAPITAL)

    close = df_raw["close"].values
    bench_ret = close[-1] / close[0] - 1

    print(f"\n{'='*70}")
    print(f"  {DATA_NAME} — 吊灯止损只做多策略 (N={N}, K={K})")
    print(f"{'='*70}")
    print(f"  规则: MA金叉(快线{MA_FAST}上穿慢线{MA_SLOW})开多, 跌破吊灯线离场; 空仓现金年化 {FLAT_ANNUAL:.1%}, 单边手续费 {FEE_RATE:.2%}")
    print(f"  {'指标':<16} {'策略':>16} {'标的Buy&Hold':>16}")
    print(f"  {'总收益率':<16} {m['total_return']:>15.2%} {bench_ret:>15.2%}")
    print(f"  {'年化收益率':<16} {m['annual_return']:>15.2%}")
    print(f"  {'最大回撤':<16} {m['max_dd']:>15.2%}")
    print(f"  {'回撤持续天数':<16} {m['max_dd_days']:>15.0f}")
    print(f"  {'夏普比率':<16} {m['sharpe']:>15.2f}")
    print(f"  {'卡尔玛比率':<16} {m['calmar']:>15.2f}")
    print(f"  {'交易次数':<16} {m['n_trades']:>15.0f}")
    print(f"  {'胜率':<16} {m['win_rate']:>15.2%}")
    print(f"  {'盈亏比':<16} {m['profit_factor']:>15.2f}")
    print(f"  {'平均持仓天数':<16} {m['avg_hold_days']:>15.1f}")
    print(f"  {'回测天数':<16} {m['n_days']:>15.0f}")

    # 5. 生成报告
    output_dir = f"{DATA_NAME}_chandelier{N}x{K}_ma{MA_FAST}x{MA_SLOW}"
    print(f"\n{'='*70}")
    print(f"  生成吊灯止损策略报告 → {output_dir}/")
    print(f"{'='*70}")
    generate_report(df, trades, eq, rets, CAPITAL,
                    f"{DATA_NAME}_吊灯止损MA{MA_FAST}/{MA_SLOW}", N, K, output_dir=output_dir)

    # 6. 研报综合图
    print(f"\n{'='*70}")
    print(f"  生成研报综合图")
    print(f"{'='*70}")
    plot_chandelier_report(df, trades, eq, DATA_NAME, N, K, MA_FAST, MA_SLOW, output_dir=output_dir)
