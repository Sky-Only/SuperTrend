"""
动态 ATR 模块 (KAMA 效率比率自适应)
====================================
用 Kaufman 自适应移动平均 (KAMA) 的思路平滑 True Range，
让 ATR 的平滑速度随市场趋势效率 (Efficiency Ratio) 动态变化：

  - 趋势强 (ER 高): 平滑快 → ATR 反应更灵敏，带更贴合价格
  - 震荡市 (ER 低): 平滑慢 → ATR 更平稳，带更宽

公式:
  ER[t]  = |close[t] - close[t-N]| / Σ|close[t-i] - close[t-i-1]|   (i=0..N-1)
  fastSC = 2 / (fast + 1)
  slowSC = 2 / (slow + 1)
  SC[t]  = [ ER[t] * (fastSC - slowSC) + slowSC ]²
  ATR[t] = ATR[t-1] + SC[t] * (TR[t] - ATR[t-1])

独立模块：仅复用 backtest.py 的数据加载、固定 ATR 回测引擎与绘图模块，
动态 ATR 及其 SuperTrend 带构建逻辑全部在本文件内实现，不改动 backtest.py。
"""

import numpy as np
import pandas as pd
import metrics as mt
from backtest import (
    load_data, calc_super_trend, generate_report,
)


# ============================================================
# 1. 动态 ATR 计算 (KAMA 效率比率自适应)
# ============================================================

def calc_dynamic_atr(df, er_period=10, fast=2, slow=30):
    """
    计算动态 ATR（KAMA 自适应平滑 True Range）

    参数:
        er_period: 效率比率窗口 N（越大 ER 越平滑）
        fast:      快平滑周期（趋势强时趋近的有效周期）
        slow:      慢平滑周期（震荡市时趋近的有效周期）

    返回: df 增加 atr(动态), er(效率比率), sc(平滑系数) 三列
    """
    df = df.copy()
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    # ---- True Range ----
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))

    # ---- 效率比率 ER (基于收盘价) ----
    close_s = pd.Series(close)
    # 分母: er_period 内逐日变动的绝对值之和
    denom = close_s.diff().abs().rolling(er_period).sum()
    # 分子: er_period 内净变动
    numer = (close_s - close_s.shift(er_period)).abs()
    er = numer / denom.replace(0, np.nan)

    # ---- 平滑系数 SC ----
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # ---- KAMA 平滑 TR ----
    atr = np.full(n, np.nan)
    atr[er_period] = tr[:er_period + 1].mean()  # 种子: 前 er_period+1 条 TR 均值
    sc_vals = sc.to_numpy()
    for t in range(er_period + 1, n):
        atr[t] = atr[t - 1] + sc_vals[t] * (tr[t] - atr[t - 1])

    df["atr"] = atr
    df["er"] = er.to_numpy()
    df["sc"] = sc_vals
    return df


# ============================================================
# 2. 动态 ATR 版 SuperTrend
# ============================================================

def _build_super_trend(high, low, close, atr, multiplier, first=None):
    """
    由 ATR 数组构建 SuperTrend 带与趋势方向（逻辑与 backtest.py 一致）
    """
    n = len(close)
    if first is None:
        valid = np.where(~np.isnan(atr))[0]
        if len(valid) == 0:
            raise ValueError("ATR 全为 NaN, 无法构建 SuperTrend")
        first = int(valid[0])

    mid = (high + low) / 2
    upper_band = mid + multiplier * atr
    lower_band = mid - multiplier * atr

    super_trend = np.full(n, np.nan)
    trend = np.full(n, 0)  # 1=多, -1=空

    # 初始化：第一个有效 bar
    if close[first] > upper_band[first]:
        trend[first] = 1
        super_trend[first] = lower_band[first]
    elif close[first] < lower_band[first]:
        trend[first] = -1
        super_trend[first] = upper_band[first]
    else:
        trend[first] = 1
        super_trend[first] = lower_band[first]

    for i in range(first + 1, n):
        if trend[i - 1] == 1:
            st = max(lower_band[i], super_trend[i - 1])
            if close[i] < st:
                trend[i] = -1
                super_trend[i] = upper_band[i]
            else:
                trend[i] = 1
                super_trend[i] = st
        else:
            st = min(upper_band[i], super_trend[i - 1])
            if close[i] > st:
                trend[i] = 1
                super_trend[i] = lower_band[i]
            else:
                trend[i] = -1
                super_trend[i] = st

    return super_trend, trend, upper_band, lower_band


def calc_dynamic_super_trend(df, multiplier=3, er_period=10, fast=2, slow=30):
    """
    动态 ATR 版 SuperTrend

    返回: df 增加 atr(动态), er, sc, upper_band, lower_band, super_trend, trend
    """
    df = calc_dynamic_atr(df, er_period=er_period, fast=fast, slow=slow)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    atr = df["atr"].values

    super_trend, trend, upper_band, lower_band = _build_super_trend(
        high, low, close, atr, multiplier)

    df["upper_band"] = upper_band
    df["lower_band"] = lower_band
    df["super_trend"] = super_trend
    df["trend"] = trend
    return df


# ============================================================
# 2b. 只做多回测引擎 (空仓现金计息 + 手续费)
# ============================================================

FLAT_ANNUAL_RETURN = 0.015   # 空仓期间现金年化收益
COMMISSION = 0.0003          # 单边手续费 (万三 = 0.03%)
TRADING_DAYS_PER_YEAR = 252


def run_backtest_long_only(df, capital=1_000_000, flat_annual=FLAT_ANNUAL_RETURN,
                           commission=COMMISSION):
    """
    回测只做多策略 (SuperTrend 翻多做多, 翻空平仓空仓)

    规则:
      - t 日收盘触发信号, t+1 日执行
      - 趋势翻多 (trend -1→1 或首日=1): 全仓买入 (t+1 HIGH, 保守)
      - 趋势翻空 (trend 1→-1): 平仓空仓 (t+1 LOW, 保守)
      - 空仓期间现金按年化 flat_annual 逐日复利计息
      - 单边手续费 commission (万三): 买入、卖出各收一次
    """
    df = df.copy()

    # ---- 信号生成 (只做多) ----
    df["trend_prev"] = df["trend"].shift(1)
    df["signal"] = 0  # 1=开多, -1=平多空仓
    df.loc[(df["trend_prev"] == -1) & (df["trend"] == 1), "signal"] = 1
    df.loc[(df["trend_prev"] == 1) & (df["trend"] == -1), "signal"] = -1

    first_idx = df[df["trend"] != 0].index[0]
    if df.at[first_idx, "trend"] == 1:
        df.at[first_idx, "signal"] = 1
    # 首日 trend==-1 则保持空仓 (只做多, 不做空)

    daily_flat = (1 + flat_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

    trades = []
    equity = capital
    position = 0.0
    entry_price = 0.0
    entry_cash = 0.0
    entry_date = None
    pending_signal = None

    equity_curve = []
    daily_returns = []

    for i in range(len(df)):
        row = df.iloc[i]
        date = row["trade_date"]

        # Step 1: 执行前一日信号 (t+1)
        if pending_signal is not None:
            signal_date = pending_signal_date

            if pending_signal == 1 and position == 0:
                # 开多: 全仓买入
                exec_price = row["high"]
                invest = equity
                fee = invest * commission
                position = (invest - fee) / exec_price
                entry_price = exec_price
                entry_cash = invest
                entry_date = date

            elif pending_signal == -1 and position > 0:
                # 平多: 卖出空仓
                exec_price = row["low"]
                proceeds = position * exec_price
                fee = proceeds * commission
                equity = proceeds - fee
                pnl = equity - entry_cash
                pnl_pct = pnl / entry_cash if entry_cash > 0 else 0.0
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
                position = 0.0
                entry_price = 0.0
                entry_cash = 0.0

            pending_signal = None

        # Step 2: 记录今日信号 (明日执行)
        sig = row["signal"]
        if sig != 0:
            pending_signal = sig
            pending_signal_date = date

        # Step 3: 按收盘价估值 (空仓现金计息)
        if position > 0:
            equity = position * row["close"]
        else:
            equity = equity * (1.0 + daily_flat)

        equity_curve.append({"trade_date": date, "equity": round(equity, 2)})

        if i > 0:
            prev_val = equity_curve[-2]["equity"]
            daily_returns.append((equity - prev_val) / prev_val if prev_val > 0 else 0.0)

    # 强制平仓
    if position > 0:
        last_row = df.iloc[-1]
        exit_px = last_row["close"]
        proceeds = position * exit_px
        fee = proceeds * commission
        equity = proceeds - fee
        pnl = equity - entry_cash
        pnl_pct = pnl / entry_cash if entry_cash > 0 else 0.0
        trades.append({
            "signal_date": None,
            "exec_date": last_row["trade_date"],
            "entry_date": entry_date,
            "direction": "long",
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_px, 4),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 6),
        })
        equity_curve[-1]["equity"] = round(equity, 2)

    daily_returns.insert(0, 0.0)

    return trades, equity_curve, daily_returns


# ============================================================
# 3. 研报综合图：动态 vs 固定 ATR
# ============================================================

def plot_dynamic_report(df_dyn, df_fixed, trades, name, er_period, fast, slow,
                        multiplier, output_dir, output_file="研报_动态ATR策略全览.png"):
    """
    研报综合图（双面板）

    上图: 收盘价 + 动态 ATR SuperTrend(实线) + 固定 ATR SuperTrend(虚线) + 交易信号
    下图: 动态 ATR vs 固定 ATR + 效率比率 ER(右轴)
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, (ax_price, ax_atr) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08}
    )

    dates = df_dyn["trade_date"]
    close = df_dyn["close"].values
    trend = df_dyn["trend"].values
    n = len(df_dyn)

    # 趋势背景色
    for i in range(n - 1):
        if trend[i] == 1:
            ax_price.axvspan(dates.iloc[i], dates.iloc[i + 1], color="green", alpha=0.05, lw=0)
        elif trend[i] == -1:
            ax_price.axvspan(dates.iloc[i], dates.iloc[i + 1], color="red", alpha=0.05, lw=0)

    ax_price.plot(dates, close, color="#2c3e50", linewidth=0.7, alpha=0.75, label="标的收盘价")
    ax_price.plot(dates, df_dyn["super_trend"].values, color="#e74c3c", linewidth=0.9,
                  alpha=0.7, label="动态ATR SuperTrend")
    ax_price.plot(dates, df_fixed["super_trend"].values, color="#7f8c8d", linewidth=0.7,
                  alpha=0.5, linestyle="--", label="固定ATR SuperTrend")

    # 交易信号 (只做多: 买入开多 ▲ 绿色, 卖出平多 ▼ 红色)
    buy_plotted = False
    sell_plotted = False
    for t in trades:
        # 买入点 (开多)
        en_date = t.get("entry_date")
        if en_date is not None:
            mask = df_dyn["trade_date"] == en_date
            if mask.any():
                px_at = df_dyn.loc[mask, "close"].values[0]
                ax_price.scatter(en_date, px_at, marker="^", color="#27ae60", s=110, zorder=6,
                                 edgecolors="white", linewidth=0.8,
                                 label="买入开多" if not buy_plotted else None)
                buy_plotted = True
        # 卖出点 (平多空仓)
        ex_date = t.get("exec_date")
        if ex_date is not None:
            mask = df_dyn["trade_date"] == ex_date
            if mask.any():
                px_at = df_dyn.loc[mask, "close"].values[0]
                ax_price.scatter(ex_date, px_at, marker="v", color="#c0392b", s=110, zorder=6,
                                 edgecolors="white", linewidth=0.8,
                                 label="卖出平多" if not sell_plotted else None)
                sell_plotted = True

    ax_price.set_ylabel("价格", fontsize=12)
    ax_price.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax_price.grid(True, alpha=0.3)
    ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax_price.set_title(
        f"{name} — 动态ATR(KAMA) SuperTrend (ER={er_period}, fast={fast}, slow={slow}, M={multiplier})",
        fontsize=14, fontweight="bold"
    )

    # ---- 下图: ATR 对比 + ER ----
    ax_atr.plot(dates, df_dyn["atr"].values, color="#e74c3c", linewidth=1.2, label="动态 ATR (KAMA)")
    ax_atr.plot(dates, df_fixed["atr"].values, color="#2980b9", linewidth=1.0,
                alpha=0.7, label="固定 ATR (Wilder)")
    ax_atr.set_ylabel("ATR", fontsize=11)
    ax_atr.set_xlabel("日期", fontsize=12)
    ax_atr.grid(True, alpha=0.3)

    ax_er = ax_atr.twinx()
    ax_er.plot(dates, df_dyn["er"].values, color="#95a5a6", linewidth=0.6,
               alpha=0.7, label="效率比率 ER")
    ax_er.set_ylabel("效率比率 ER", fontsize=11)
    ax_er.set_ylim(0, 1)
    lines1, labels1 = ax_atr.get_legend_handles_labels()
    lines2, labels2 = ax_er.get_legend_handles_labels()
    ax_atr.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f"{output_dir}/{output_file}", dpi=300, bbox_inches="tight")
    print(f"  研报图已保存: {output_dir}/{output_file}")
    plt.close(fig)


# ============================================================
# 4. 入口
# ============================================================

if __name__ == "__main__":
    # ==================== 在这里修改配置 ====================
    DATA_NAME = "沪深300"
    DATA_FILE = f"{DATA_NAME}_10年日线.parquet"   # 数据文件

    # 固定 ATR SuperTrend 参数（对比基准）
    ST_PERIOD = 10
    ST_MULT   = 3

    # 动态 ATR (KAMA) 参数
    ER_PERIOD = 15      # 效率比率窗口 N
    FAST      = 5       # 快平滑周期
    SLOW      = 20      # 慢平滑周期
    DYN_MULT  = 3       # SuperTrend 乘数 M

    CAPITAL = 1_000_000

    # 交易规则 (只做多)
    FLAT_ANNUAL = 0.015    # 空仓期间现金年化收益
    FEE_RATE    = 0.0003   # 单边手续费 (万三 = 0.03%)
    # ========================================================

    # 1. 加载数据
    print(f"加载数据: {DATA_FILE}")
    df_raw = load_data(DATA_FILE)
    print(f"  数据: {len(df_raw)} 条, {df_raw['trade_date'].iloc[0].date()} ~ {df_raw['trade_date'].iloc[-1].date()}")

    # 2. 固定 ATR 基准
    df_fixed = calc_super_trend(df_raw, period=ST_PERIOD, multiplier=ST_MULT)
    trades_fixed, eq_fixed, rets_fixed = run_backtest_long_only(
        df_fixed, capital=CAPITAL, flat_annual=FLAT_ANNUAL, commission=FEE_RATE)

    # 3. 动态 ATR 策略
    df_dyn = calc_dynamic_super_trend(df_raw, multiplier=DYN_MULT,
                                      er_period=ER_PERIOD, fast=FAST, slow=SLOW)
    trades_dyn, eq_dyn, rets_dyn = run_backtest_long_only(
        df_dyn, capital=CAPITAL, flat_annual=FLAT_ANNUAL, commission=FEE_RATE)

    # 4. 计算原始指标用于对比
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

    rm_fixed = _raw_metrics(trades_fixed, eq_fixed, rets_fixed, CAPITAL)
    rm_dyn = _raw_metrics(trades_dyn, eq_dyn, rets_dyn, CAPITAL)

    # 5. 控制台对比
    print(f"\n{'='*90}")
    print(f"  {DATA_NAME} — 策略对比: 固定ATR vs 动态ATR(KAMA)   [只做多]")
    print(f"{'='*90}")
    print(f"  固定: N={ST_PERIOD}, M={ST_MULT}    动态: ER={ER_PERIOD}, fast={FAST}, slow={SLOW}, M={DYN_MULT}")
    print(f"  规则: 只做多, 空仓现金年化 {FLAT_ANNUAL:.1%}, 单边手续费 {FEE_RATE:.2%}")
    print(f"  {'指标':<16} {'固定ATR':>16} {'动态ATR':>16} {'差值':>16} {'方向':>8}")

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
        v_f = rm_fixed[key]
        v_d = rm_dyn[key]
        diff = v_d - v_f
        if kind == "pct":
            diff_str = f"{diff:>+15.2%}"
        elif kind == "int":
            diff_str = f"{diff:>+15,.0f}" if abs(diff) >= 1000 else f"{diff:>+15.0f}"
        else:
            diff_str = f"{diff:>+15.2f}"

        if better == "↑":
            winner = "固定" if v_f > v_d else ("动态" if v_d > v_f else "  —")
        elif better == "↓":
            winner = "固定" if v_f < v_d else ("动态" if v_d < v_f else "  —")
        else:
            winner = "  —"

        print(f"  {label:<16}{_fmt(v_f, kind)} {_fmt(v_d, kind)} {diff_str}  {winner}")

    print(f"  {'-'*88}")
    print(f"  ↑ = 越大越好    ↓ = 越小越好    — = 中性指标")
    print(f"  方向列: 哪个策略在该指标上更优")

    # 6. 生成两套报告
    output_fixed = f"{DATA_NAME}_st{ST_PERIOD}x{ST_MULT}"
    output_dyn = f"{DATA_NAME}_dynatr{ER_PERIOD}x{FAST}x{SLOW}_m{DYN_MULT}"

    print(f"\n{'='*70}")
    print(f"  生成固定ATR策略报告 → {output_fixed}/")
    print(f"{'='*70}")
    generate_report(df_fixed, trades_fixed, eq_fixed, rets_fixed, CAPITAL,
                    DATA_NAME, ST_PERIOD, ST_MULT, output_dir=output_fixed)

    print(f"\n{'='*70}")
    print(f"  生成动态ATR策略报告 → {output_dyn}/")
    print(f"{'='*70}")
    generate_report(df_dyn, trades_dyn, eq_dyn, rets_dyn, CAPITAL,
                    f"{DATA_NAME}_动态ATR", ER_PERIOD, DYN_MULT, output_dir=output_dyn)

    # 7. 研报综合图
    print(f"\n{'='*70}")
    print(f"  生成研报综合图")
    print(f"{'='*70}")
    plot_dynamic_report(df_dyn, df_fixed, trades_dyn, DATA_NAME,
                        ER_PERIOD, FAST, SLOW, DYN_MULT, output_dir=output_dyn)
