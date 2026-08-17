"""
吊灯止损 (Chandelier Exit) 只做多策略
====================================
吊灯止损线 = 持仓期间最高价 (自入场起算, 回看上限 N 根) - K × ATR
  价格创新高时, 止损线像吊灯一样挂在最高价下方, 随价格上移;
  收盘跌破吊灯线 → 止盈/止损离场。
  每次卖出后重新计算持仓时间, N 是单笔交易的最大回看天数。

策略规则 (只做多, 不做空):
  入场: SuperTrend 翻多信号 (+ 可选 MA 多头区间确认)
  离场: 吊灯止损 (收盘跌破近 N 日最高价 - K×ATR) 或 SuperTrend 翻空 (可选)

交易执行:
  - t 日收盘触发信号, t+1 日执行
  - 开多用 t+1 HIGH (保守), 平多用 t+1 LOW (保守)
  - 空仓期间现金按年化 1.5% 逐日复利计息
  - 单边手续费万三 (买入、卖出各收一次)

说明: SuperTrend (period × multiplier) 负责入场翻多信号, MA 可选确认,
      离场可选吊灯止损或 SuperTrend 翻空。
      use_ma=False 且 use_chandelier=False 时即「只做多的纯 SuperTrend」。
      复用了 dynamic_atr.py 的只做多回测引擎。
"""

import numpy as np
import pandas as pd
import metrics as mt
from backtest import load_data, calc_super_trend, generate_report
from dynamic_atr import run_backtest_long_only
from hv_analysis import calc_hv


# ============================================================
# 1. 吊灯止损信号计算
# ============================================================

def calc_chandelier(df, st_period=10, st_mult=3.0, ma_fast=5, ma_slow=20, n=22, k=3.0,
                    atr_period=None, use_ma=True, use_chandelier=True,
                    use_volatility=False, hv_window=20, hv_threshold=0.30):
    """
    计算 SuperTrend入场 + 可选离场 (吊灯止损 / 波动率止盈) 的只做多策略信号

    参数:
        st_period: SuperTrend 周期 (入场翻多信号), 业界常用 10
        st_mult:   SuperTrend 乘数 M (入场翻多信号), 业界常用 3
        ma_fast:   快均线周期 (入场确认), 业界常用 5
        ma_slow:   慢均线周期 (入场确认), 业界常用 20
        n:         单笔交易最高价回看上限天数 (自入场起算的最大回看), 业界常用 22
        k:         吊灯止损 ATR 倍数, 业界常用 3
        atr_period: ATR 周期 (独立于 n); None 时回退到 n
        use_ma:          True=SuperTrend翻多+MA多头确认开仓; False=纯SuperTrend翻多开仓
        use_chandelier:  True=吊灯止损离场
        use_volatility:  True=波动率止盈 (HV 超过阈值离场)
        hv_window:       波动率 HV 计算窗口 (交易日)
        hv_threshold:    波动率止盈阈值 (年化, 如 0.30 = 30%)

    离场规则 (可单选或组合):
        - use_chandelier=True:   收盘跌破持仓期最高价 - K×ATR → 离场
        - use_volatility=True:   HV > hv_threshold → 离场
        - 两者都 False:           SuperTrend 翻空 → 离场
        - 两者都 True:            任一触发即离场 (吊灯止损 或 波动率止盈)

    返回: df 增加 atr, super_trend, st_trend, ma_fast, ma_slow, hh, chandelier_stop, trend(1=多, -1=空仓)
    """
    df = df.copy()
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    m = len(df)

    # ---- 1. SuperTrend (入场翻多信号) ----
    df = calc_super_trend(df, period=st_period, multiplier=st_mult)
    df = df.rename(columns={"trend": "st_trend"})   # SuperTrend 趋势改名, 避免与策略状态冲突
    st_trend = df["st_trend"].values

    # ---- 2. MA 均线 (入场确认) ----
    ma_fast_vals = pd.Series(close).rolling(ma_fast, min_periods=1).mean().values
    ma_slow_vals = pd.Series(close).rolling(ma_slow, min_periods=1).mean().values

    # ---- 2b. HV 波动率 (波动率止盈用) ----
    hv = calc_hv(close, hv_window)

    # ---- 3. 吊灯止损 ATR (Wilder, 周期 atr_period, 独立于回看上限 n) ----
    ap = n if atr_period is None else atr_period
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))
    atr = np.full(m, np.nan)
    atr[ap - 1] = tr[:ap].mean()
    for t in range(ap, m):
        atr[t] = (atr[t - 1] * (ap - 1) + tr[t]) / ap

    # ---- 4. 状态机: 1=多头, -1=空仓 ----
    trend = np.full(m, -1)
    hh_vals = np.full(m, np.nan)              # 持仓期间最高价 (自入场起算)
    chandelier_stop_vals = np.full(m, np.nan) # 动态吊灯止损线 (仅持仓时有效)
    exit_reason = np.zeros(m, dtype=int)      # 离场原因: 0=无,1=吊灯,2=波动率,3=ST翻空
    state = -1
    entry_idx = -1                            # 当前持仓入场 bar 索引
    st_trend_prev = np.roll(st_trend, 1)
    st_trend_prev[0] = 0
    for i in range(m):
        if state == -1:
            # 入场: SuperTrend 翻多 (st趋势 -1/0 → 1), 可选 MA 多头确认
            st_flip_long = (st_trend[i] == 1) and (st_trend_prev[i] != 1)
            ma_confirm = ma_fast_vals[i] > ma_slow_vals[i]
            if st_flip_long and (ma_confirm if use_ma else True):
                state = 1
                entry_idx = i
        else:
            # 离场: 吊灯止损 / 波动率止盈 / SuperTrend翻空 (可组合)
            exit_now = False
            reason = 0
            if use_chandelier:
                # 最高价自入场起算, 回看上限 n 根 (卖出后重新计算持仓时间)
                start = max(entry_idx, i - n + 1)
                hh_vals[i] = np.max(high[start:i + 1])
                chandelier_stop_vals[i] = hh_vals[i] - k * atr[i]
                if close[i] < chandelier_stop_vals[i]:
                    exit_now = True
                    reason = 1
            if use_volatility:
                # 波动率止盈: HV 超过阈值则离场
                if (not np.isnan(hv[i])) and hv[i] > hv_threshold:
                    exit_now = True
                    reason = 2
            if (not use_chandelier) and (not use_volatility):
                # SuperTrend 翻空
                st_flip_short = (st_trend[i] == -1) and (st_trend_prev[i] == 1)
                if st_flip_short:
                    exit_now = True
                    reason = 3
            if exit_now:
                state = -1
                exit_reason[i] = reason
        trend[i] = state

    df["ma_fast"] = ma_fast_vals
    df["ma_slow"] = ma_slow_vals
    df["hh"] = hh_vals
    df["chandelier_stop"] = chandelier_stop_vals
    df["trend"] = trend
    df["exit_reason"] = exit_reason
    return df


# ============================================================
# 2. 研报综合图
# ============================================================

def plot_chandelier_report(df, trades, eq, name, st_period, st_mult, ma_fast, ma_slow, n, k, use_ma, use_chandelier, output_dir,
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
    ax_price.plot(dates, df["super_trend"].values, color="#8e44ad", linewidth=0.9, alpha=0.8,
                  label=f"SuperTrend ({st_period}, {st_mult})")
    if use_chandelier:
        ax_price.plot(dates, df["chandelier_stop"].values, color="#e74c3c", linewidth=0.9,
                      alpha=0.85, label=f"吊灯止损线 (HH{n} - {k}·ATR)")
    if use_ma:
        ax_price.plot(dates, df["ma_fast"].values, color="#0099ff", linewidth=1.1, label=f"MA快线 ({ma_fast})")
        ax_price.plot(dates, df["ma_slow"].values, color="#ff7700", linewidth=1.1, label=f"MA慢线 ({ma_slow})")

    # 离场原因映射 (signal_date → exit_reason)
    reason_by_date = dict(zip(df["trade_date"], df["exit_reason"])) if "exit_reason" in df.columns else {}

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
                                 label=("SuperTrend翻多买入(MA确认)" if use_ma else "SuperTrend翻多买入") if not buy_plotted else None)
                buy_plotted = True
        ex_date = t.get("exec_date")
        if ex_date is not None:
            mask = df["trade_date"] == ex_date
            if mask.any():
                px = df.loc[mask, "close"].values[0]
                sig_date = t.get("signal_date")
                reason = reason_by_date.get(sig_date, 0) if sig_date is not None else 0
                label = {1: "吊灯止损离场", 2: "波动率止盈离场", 3: "SuperTrend翻空离场"}.get(reason, "期末平仓")
                ax_price.scatter(ex_date, px, marker="v", color="#c0392b", s=110, zorder=6,
                                 edgecolors="white", linewidth=0.8,
                                 label=label if not sell_plotted else None)
                sell_plotted = True

    ax_price.set_ylabel("价格", fontsize=12)
    ax_price.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax_price.grid(True, alpha=0.3)
    ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ma_str = f"MA{ma_fast}/{ma_slow}" if use_ma else "无MA过滤"
    exit_str = f"吊灯{n}x{k}" if use_chandelier else "ST翻空离场"
    ax_price.set_title(
        f"{name} — 只做多 (ST{st_period}x{st_mult}, {ma_str}, {exit_str})",
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

    # SuperTrend 参数 (入场翻多信号)
    ST_PERIOD = 34   # SuperTrend 周期
    ST_MULT   = 4.9  # SuperTrend 乘数 M

    # MA 过滤开关 (True=开启MA多头确认, False=纯SuperTrend开仓)
    USE_MA = False

    # MA 参数 (入场确认, USE_MA=True 时生效)
    MA_FAST = 5    # 快均线周期
    MA_SLOW = 20   # 慢均线周期

    # 吊灯止损开关 (True=吊灯止损离场)
    USE_CHANDELIER = True

    # 吊灯止损参数 (离场, USE_CHANDELIER=True 时生效)
    N = 5000        # 最高价回看上限天数 (自入场起算)
    ATR_PERIOD = 15  # ATR 周期 (独立于 N)
    K = 5.5       # ATR 倍数

    # 波动率止盈开关 (True=HV超阈值离场)
    USE_VOLATILITY = False

    # 波动率止盈参数 (USE_VOLATILITY=True 时生效)
    HV_WINDOW = 20       # HV 计算窗口 (交易日)
    HV_THRESHOLD = 0.5  # HV 止盈阈值 (年化, 0.30=30%)

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
    df = calc_chandelier(df_raw, st_period=ST_PERIOD, st_mult=ST_MULT,
                         ma_fast=MA_FAST, ma_slow=MA_SLOW, n=N, k=K, atr_period=ATR_PERIOD,
                         use_ma=USE_MA, use_chandelier=USE_CHANDELIER,
                         use_volatility=USE_VOLATILITY, hv_window=HV_WINDOW, hv_threshold=HV_THRESHOLD)

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

    # 标的 Buy&Hold 完整指标 (以收盘价归一化净值 + 日收益率计算)
    close = df_raw["close"].values
    bench_eq = close / close[0] * CAPITAL
    bench_rets = np.zeros(len(close))
    bench_rets[1:] = np.diff(close) / close[:-1]
    bench_m = mt.calc_all_metrics(bench_eq, bench_rets, np.array([]), CAPITAL)

    print(f"\n{'='*80}")
    ma_str = f"MA{MA_FAST}/{MA_SLOW}" if USE_MA else "无MA过滤"
    exit_parts = []
    if USE_CHANDELIER:
        exit_parts.append(f"吊灯{N}x{K}(ATR{ATR_PERIOD})")
    if USE_VOLATILITY:
        exit_parts.append(f"波动率止盈(HV>{HV_THRESHOLD:.0%})")
    exit_str = " + ".join(exit_parts) if exit_parts else "ST翻空离场"
    print(f"  {DATA_NAME} — 只做多 (ST{ST_PERIOD}x{ST_MULT}, {ma_str}, 离场={exit_str})")
    print(f"{'='*80}")
    entry_rule = (f"SuperTrend翻多+MA多头(快线{MA_FAST}>慢线{MA_SLOW})确认开多"
                  if USE_MA else "SuperTrend翻多开多")
    exit_rule = (" + ".join(exit_parts) if exit_parts else "SuperTrend翻空")
    print(f"  规则: {entry_rule}, {exit_rule}; 空仓现金年化 {FLAT_ANNUAL:.1%}, 单边手续费 {FEE_RATE:.2%}")
    print(f"  {'指标':<14} {'策略':>16} {'标的Buy&Hold':>16} {'差值':>14}")

    def _fmt(val, kind):
        if kind == "pct":
            return f"{val:>15.2%}"
        elif kind == "int":
            return f"{val:>15,.0f}" if val >= 1000 else f"{val:>15.0f}"
        else:
            return f"{val:>15.2f}"

    # (指标名, key, 格式, 标的是否也有该指标)
    rows = [
        ("总收益率",     "total_return",  "pct", True),
        ("年化收益率",   "annual_return",  "pct", True),
        ("最大回撤",     "max_dd",         "pct", True),
        ("回撤持续天数", "max_dd_days",    "int", True),
        ("年化波动率",   "daily_vol",      "pct", True),
        ("夏普比率",     "sharpe",         "num", True),
        ("卡尔玛比率",   "calmar",         "num", True),
        ("交易次数",     "n_trades",       "int", False),
        ("胜率",         "win_rate",       "pct", False),
        ("盈亏比",       "profit_factor",  "num", False),
        ("平均持仓天数", "avg_hold_days",  "num", False),
        ("回测天数",     "n_days",         "int", True),
    ]

    for label, key, kind, has_bench in rows:
        v_s = m[key]
        if has_bench:
            v_b = bench_m[key]
            diff = v_s - v_b
            if kind == "pct":
                diff_str = f"{diff:>+13.2%}"
            elif kind == "int":
                diff_str = f"{diff:>+13,.0f}" if abs(diff) >= 1000 else f"{diff:>+13.0f}"
            else:
                diff_str = f"{diff:>+13.2f}"
            print(f"  {label:<14}{_fmt(v_s, kind)} {_fmt(v_b, kind)} {diff_str}")
        else:
            print(f"  {label:<14}{_fmt(v_s, kind)} {'—':>16}")

    # 5. 生成报告
    ma_tag = f"ma{MA_FAST}x{MA_SLOW}" if USE_MA else "noma"
    exit_tag = f"ch{N}x{K}a{ATR_PERIOD}" if USE_CHANDELIER else "noch"
    output_dir = f"{DATA_NAME}_st{ST_PERIOD}x{ST_MULT}_{ma_tag}_{exit_tag}"
    print(f"\n{'='*70}")
    print(f"  生成吊灯止损策略报告 → {output_dir}/")
    print(f"{'='*70}")
    generate_report(df, trades, eq, rets, CAPITAL,
                    f"{DATA_NAME}_ST{ST_PERIOD}x{ST_MULT}", ST_PERIOD, ST_MULT, output_dir=output_dir)

    # 6. 研报综合图
    print(f"\n{'='*70}")
    print(f"  生成研报综合图")
    print(f"{'='*70}")
    plot_chandelier_report(df, trades, eq, DATA_NAME, ST_PERIOD, ST_MULT, MA_FAST, MA_SLOW, N, K, USE_MA, USE_CHANDELIER, output_dir=output_dir)
