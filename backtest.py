import numpy as np
import pandas as pd
import metrics as mt


# ============================================================
# 1. 数据加载
# ============================================================

def load_data(filepath):
    """加载数据并统一列名"""
    df = pd.read_parquet(filepath)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    # 确保有统一的 OHLC 列
    required = ["trade_date", "open", "high", "low", "close"]
    for col in required:
        assert col in df.columns, f"缺少列: {col}"
    return df[required]


# ============================================================
# 2. SuperTrend 计算
# ============================================================

def calc_super_trend(df, period=10, multiplier=3):
    """
    计算 SuperTrend 指标
    返回: df 增加 atr, upper_band, lower_band, super_trend, trend(1=多, -1=空)
    """
    df = df.copy()
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(df)

    # --- ATR (Wilder's original smoothing) ---
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))
    # Wilder 原始公式: 第一条 ATR = 前 period 条 TR 的简单平均
    #                  后续 ATR[t] = (ATR[t-1] * (period-1) + TR[t]) / period
    atr = np.full(n, np.nan)
    atr[period - 1] = tr[:period].mean()
    for t in range(period, n):
        atr[t] = (atr[t - 1] * (period - 1) + tr[t]) / period

    # --- 基础带 ---
    mid = (high + low) / 2
    upper_band = mid + multiplier * atr
    lower_band = mid - multiplier * atr

    # --- 最终 SuperTrend ---
    super_trend = np.full(n, np.nan)
    trend = np.full(n, 0)  # 1=多, -1=空

    # 初始化：第一个有效 bar
    first = period  # 等 ATR 稳定
    if close[first] > upper_band[first]:
        trend[first] = 1
        super_trend[first] = lower_band[first]
    elif close[first] < lower_band[first]:
        trend[first] = -1
        super_trend[first] = upper_band[first]
    else:
        # 在两者之间，延续前一周期（这里简单默认为多头）
        trend[first] = 1
        super_trend[first] = lower_band[first]

    for i in range(first + 1, n):
        if trend[i - 1] == 1:
            # 多头中：SuperTrend 只能上移
            st = max(lower_band[i], super_trend[i - 1])
            if close[i] < st:
                # 翻转至空头
                trend[i] = -1
                super_trend[i] = upper_band[i]
            else:
                trend[i] = 1
                super_trend[i] = st
        else:
            # 空头中：SuperTrend 只能下移
            st = min(upper_band[i], super_trend[i - 1])
            if close[i] > st:
                # 翻转至多头
                trend[i] = 1
                super_trend[i] = lower_band[i]
            else:
                trend[i] = -1
                super_trend[i] = st

    df["atr"] = atr
    df["upper_band"] = upper_band
    df["lower_band"] = lower_band
    df["super_trend"] = super_trend
    df["trend"] = trend
    return df


# ============================================================
# 3. 回测引擎
# ============================================================

def run_backtest(df, capital=1_000_000):
    """
    回测 SuperTrend 双向策略
    规则：
      - t日收盘触发信号，t+1日执行
      - 多头入场：t+1 HIGH（买入）
      - 多头出场/空头入场：t+1 LOW（卖出/做空）
      - 空头出场/多头入场：t+1 HIGH（平空/买入）
      - 始终持仓（多或空）
    """
    df = df.copy()

    # ---- 找出趋势转折点（t日信号）----
    df["trend_prev"] = df["trend"].shift(1)
    df["signal"] = 0  # 0=无, 1=开多/平空, -1=开空/平多
    mask_long = (df["trend_prev"] == -1) & (df["trend"] == 1)
    mask_short = (df["trend_prev"] == 1) & (df["trend"] == -1)
    df.loc[mask_long, "signal"] = 1
    df.loc[mask_short, "signal"] = -1

    # 首个趋势确立日也作为入场信号
    first_sig = df[df["trend"] != 0].index[0]
    if df.at[first_sig, "trend"] == 1:
        df.at[first_sig, "signal"] = 1
    else:
        df.at[first_sig, "signal"] = -1

    # ---- 模拟交易（t日信号 → t+1日执行）----
    trades = []
    equity = capital
    position = 0       # >0=多头持仓(数量), <0=空头持仓(数量)
    entry_price = 0
    entry_date = None
    direction = None   # "long" or "short"
    pending_signal = None  # t日产生的信号，等待t+1日执行

    equity_curve = []
    daily_returns = []

    for i in range(len(df)):
        row = df.iloc[i]
        date = row["trade_date"]

        # ---- Step 1: 执行前一日产生的信号（t+1日成交）----
        if pending_signal is not None:
            signal_date = pending_signal_date  # t日（信号触发日）

            if pending_signal == 1:
                # 多头信号：平空 + 开多，都用 HIGH
                exec_price = row["high"]
                if position < 0:
                    # 平空（空头亏损 = 入场-出场）
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
                # 开多
                position = equity / exec_price
                entry_price = exec_price
                entry_date = date
                direction = "long"

            elif pending_signal == -1:
                # 空头信号：平多 + 开空，都用 LOW
                exec_price = row["low"]
                if position > 0:
                    # 平多
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
                # 开空
                position = -equity / exec_price
                entry_price = exec_price
                entry_date = date
                direction = "short"

            pending_signal = None

        # ---- Step 2: 记录今日信号，等待明日执行 ----
        sig = row["signal"]
        if sig != 0:
            pending_signal = sig
            pending_signal_date = date

        # ---- Step 3: 按收盘价估值 ----
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

    # 清理最后未平仓持仓（按最后收盘价强制平仓）
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

    # 补充第一天的日收益
    daily_returns.insert(0, 0)

    return trades, equity_curve, daily_returns


# ============================================================
# 4. 绩效统计
# ============================================================

def calc_metrics(trades, equity_curve, daily_returns, capital):
    """
    计算回测绩效指标（委托给独立的 metrics 模块）

    参数:
        trades: 交易记录列表 (list of dict)
        equity_curve: 净值曲线 (list of dict with "trade_date" and "equity")
        daily_returns: 日收益率列表
        capital: 初始资金

    返回:
        dict: 格式化后的指标字典（键为中文名，值为字符串或数字）
    """
    # 转换为 numpy 数组
    eq = np.array([e["equity"] for e in equity_curve])
    rets = np.array(daily_returns)

    # 提取交易 P&L 百分比
    trade_pnl_pcts = np.array([t["pnl_pct"] for t in trades]) if trades else np.array([])

    # 计算持仓天数（exec_date - entry_date）
    hold_days = None
    if trades:
        hd_list = []
        for t in trades:
            ed = t.get("entry_date")
            xd = t.get("exec_date")
            if ed is not None and xd is not None:
                hd_list.append((pd.Timestamp(xd) - pd.Timestamp(ed)).days)
        if hd_list:
            hold_days = np.array(hd_list, dtype=float)

    # 调用独立指标模块
    raw = mt.calc_all_metrics(eq, rets, trade_pnl_pcts, capital, hold_days=hold_days)

    # 格式化为展示用 dict（保持与之前版本的兼容性）
    return {
        "初始资金": capital,
        "最终净值": raw["final_equity"],
        "总收益率": f"{raw['total_return']:.2%}",
        "年化收益率": f"{raw['annual_return']:.2%}",
        "最大回撤": f"{raw['max_dd']:.2%}",
        "夏普比率": round(raw["sharpe"], 2),
        "卡尔玛比率": round(raw["calmar"], 2),
        "交易次数": raw["n_trades"],
        "胜率": f"{raw['win_rate']:.2%}",
        "平均盈利": f"{raw['avg_win']:.2%}",
        "平均亏损": f"{raw['avg_loss']:.2%}",
        "盈亏比": round(raw["profit_factor"], 2),
        "回测天数": raw["n_days"],
    }


# ============================================================
# 5. 绘图系统
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
import os

# ---- 字体 ----
_font_path = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansSC-Regular.ttf")
if os.path.exists(_font_path):
    fm.fontManager.addfont(_font_path)
    _cjk_font_name = fm.FontProperties(fname=_font_path).get_name()
else:
    _cjk_font_name = None

plt.rcParams["font.sans-serif"] = ([_cjk_font_name] if _cjk_font_name else []) + [
    "Microsoft YaHei", "SimHei", "Noto Sans SC",
    "Heiti SC", "STHeiti", "WenQuanYi Micro Hei",
    "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 颜色常量
COLOR_STRATEGY = "#e74c3c"
COLOR_BENCH = "#34495e"
COLOR_LONG = "#27ae60"
COLOR_SHORT = "#e74c3c"
COLOR_BANDS = ["#ecf0f1", "#bdc3c7"]


def _rolling_mean(a, window):
    """滚动均值"""
    return pd.Series(a).rolling(window).mean().values


def _rolling_std(a, window):
    """滚动标准差 (ddof=0, 与 numpy/matrics.py 一致)"""
    return pd.Series(a).rolling(window).std(ddof=0).values


def _rolling_sharpe(daily_returns, window=252):
    """滚动夏普比率"""
    roll_mean = _rolling_mean(daily_returns, window)
    roll_std = _rolling_std(daily_returns, window)
    rf_daily = mt.RISK_FREE_RATE / 252
    excess = roll_mean - rf_daily
    result = np.sqrt(252) * excess / roll_std
    result[roll_std == 0] = 0
    return result


def _rolling_vol(daily_returns, window=252):
    """滚动年化波动率"""
    return _rolling_std(daily_returns, window) * np.sqrt(252)


def _prepare_data(df, trades, equity_curve, daily_returns, capital):
    """从原始回测结果中提取所有画图需要的数组"""
    dates = np.array([e["trade_date"] for e in equity_curve])
    eq = np.array([e["equity"] for e in equity_curve])
    rets = np.array(daily_returns)

    # 基准 buy & hold
    close_raw = df["close"].values
    bench = close_raw / close_raw[0]

    # 回撤
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    bench_peak = np.maximum.accumulate(bench)
    bench_dd = (bench - bench_peak) / bench_peak * 100

    # 滚动指标 (252 日 ≈ 1 年)
    roll_sharpe = _rolling_sharpe(rets, 252)
    roll_vol = _rolling_vol(rets, 252)
    roll_ret = _rolling_mean(rets, 252) * 252  # 年化

    # 交易
    trade_pnl = np.array([t["pnl"] for t in trades]) if trades else np.array([])
    trade_pnl_pct = np.array([t["pnl_pct"] for t in trades]) if trades else np.array([])
    trade_dates = np.array([t["exec_date"] for t in trades]) if trades else np.array([])
    trade_dir = np.array([t["direction"] for t in trades]) if trades else np.array([])

    # 趋势方向（用于背景着色）
    trend = df["trend"].values

    # 持仓天数（exec_date - entry_date）
    hold_days = None
    if trades:
        hd_list = []
        for t in trades:
            ed = t.get("entry_date")
            xd = t.get("exec_date")
            if ed is not None and xd is not None:
                hd_list.append((pd.Timestamp(xd) - pd.Timestamp(ed)).days)
        if hd_list:
            hold_days = np.array(hd_list, dtype=float)

    # 原始指标值（未格式化的 dict）
    raw_metrics = mt.calc_all_metrics(
        eq, rets,
        trade_pnl_pct if len(trade_pnl_pct) > 0 else np.array([]),
        capital,
        hold_days=hold_days,
    )

    return {
        "dates": dates, "eq": eq, "rets": rets,
        "bench": bench, "bench_dd": bench_dd, "dd": dd,
        "roll_sharpe": roll_sharpe, "roll_vol": roll_vol, "roll_ret": roll_ret,
        "trade_pnl": trade_pnl, "trade_pnl_pct": trade_pnl_pct,
        "trade_dates": trade_dates, "trade_dir": trade_dir,
        "trend": trend, "close_raw": close_raw, "raw_metrics": raw_metrics,
    }


# ============================================================
# 5a. 单图函数
# ============================================================

def plot_equity_curve(d, name, period, multiplier, output_dir):
    """净值曲线：策略 vs 标的 buy & hold"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(d["dates"], d["eq"] / d["eq"][0], color=COLOR_STRATEGY, linewidth=1.2, label="策略净值")
    ax.plot(d["dates"], d["bench"], color=COLOR_BENCH, linewidth=0.8, linestyle="--", alpha=0.7, label="标的 Buy&Hold")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.fill_between(d["dates"], 1, d["bench"], alpha=0.05, color="gray")
    ax.set_title(f"{name} — 净值曲线 (N={period}, M={multiplier})", fontsize=14, fontweight="bold")
    ax.set_ylabel("净值 (初始=1)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_净值曲线.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(d, name, period, multiplier, output_dir):
    """回撤曲线：策略 vs 标的"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(d["dates"], 0, d["dd"], color="red", alpha=0.15, label="策略回撤")
    ax.plot(d["dates"], d["dd"], color="red", linewidth=0.5, alpha=0.8)
    ax.fill_between(d["dates"], 0, d["bench_dd"], color="gray", alpha=0.08, label="标的回撤")
    ax.plot(d["dates"], d["bench_dd"], color="gray", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_title(f"{name} — 回撤曲线 (N={period}, M={multiplier})", fontsize=14, fontweight="bold")
    ax.set_ylabel("回撤 %", fontsize=11)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    # 标记最大回撤点
    min_idx = np.argmin(d["dd"])
    ax.annotate(f'Max DD: {d["dd"][min_idx]:.1f}%',
                xy=(d["dates"][min_idx], d["dd"][min_idx]),
                xytext=(d["dates"][min_idx], d["dd"][min_idx] - 5),
                arrowprops=dict(arrowstyle="->", color="darkred"),
                fontsize=9, color="darkred", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_回撤曲线.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_sharpe(d, name, period, multiplier, output_dir):
    """滚动夏普比率 (252日)"""
    fig, ax = plt.subplots(figsize=(14, 5))
    valid = ~np.isnan(d["roll_sharpe"])
    ax.plot(d["dates"][valid], d["roll_sharpe"][valid], color="#2980b9", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(1, color="green", linestyle=":", linewidth=0.5, alpha=0.5, label="Sharpe=1")
    ax.fill_between(d["dates"][valid], 0, d["roll_sharpe"][valid],
                     where=(d["roll_sharpe"][valid] > 0), color="green", alpha=0.08)
    ax.fill_between(d["dates"][valid], 0, d["roll_sharpe"][valid],
                     where=(d["roll_sharpe"][valid] <= 0), color="red", alpha=0.08)
    ax.set_title(f"{name} — 滚动夏普比率 252日 (N={period}, M={multiplier})", fontsize=14, fontweight="bold")
    ax.set_ylabel("夏普比率", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_滚动夏普.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_returns(d, name, period, multiplier, output_dir):
    """滚动年化收益率 (252日) + 滚动波动率"""
    fig, ax1 = plt.subplots(figsize=(14, 5))
    valid = ~np.isnan(d["roll_ret"])
    ax1.plot(d["dates"][valid], d["roll_ret"][valid] * 100, color="#27ae60", linewidth=0.8, label="滚动年化收益")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("年化收益率 %", color="#27ae60", fontsize=11)
    ax1.tick_params(axis="y", colors="#27ae60")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(d["dates"][valid], d["roll_vol"][valid] * 100, color="#e74c3c", linewidth=0.5, alpha=0.7, label="滚动年化波动率")
    ax2.set_ylabel("年化波动率 %", color="#e74c3c", fontsize=11)
    ax2.tick_params(axis="y", colors="#e74c3c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    ax1.set_title(f"{name} — 滚动收益 & 波动率 252日 (N={period}, M={multiplier})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_滚动收益与波动率.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_annual_returns(d, name, period, multiplier, output_dir):
    """年度收益柱状图"""
    df_annual = pd.DataFrame({"date": d["dates"], "ret": d["rets"]})
    df_annual["year"] = df_annual["date"].dt.year
    annual = df_annual.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in annual.values]
    bars = ax.bar(annual.index.astype(str), annual.values * 100, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, annual.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (2 if val >= 0 else -4),
                f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{name} — 年度收益 (N={period}, M={multiplier})", fontsize=14, fontweight="bold")
    ax.set_ylabel("年度收益率", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_年度收益.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_heatmap(d, name, period, multiplier, output_dir):
    """月度收益热力图"""
    df_m = pd.DataFrame({"date": d["dates"], "ret": d["rets"]})
    df_m["year"] = df_m["date"].dt.year
    df_m["month"] = df_m["date"].dt.month
    monthly = df_m.groupby(["year", "month"])["ret"].apply(lambda x: (1 + x).prod() - 1).unstack()

    fig, ax = plt.subplots(figsize=(14, len(monthly) * 0.5 + 2))
    im = ax.imshow(monthly.values, aspect="auto", cmap="RdYlGn", vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"], fontsize=9)
    ax.set_yticks(range(len(monthly)))
    ax.set_yticklabels(monthly.index, fontsize=9)
    for i in range(len(monthly)):
        for j in range(12):
            val = monthly.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > 0.08 else "black")
    ax.set_title(f"{name} — 月度收益热力图 (N={period}, M={multiplier})", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, format=mticker.PercentFormatter(1.0), label="月度收益率")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_月度收益热力图.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trade_pnl(d, name, period, multiplier, output_dir):
    """逐笔交易盈亏"""
    if len(d["trade_pnl"]) == 0:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [COLOR_LONG if pnl > 0 else COLOR_SHORT for pnl in d["trade_pnl"]]
    x = range(1, len(d["trade_pnl"]) + 1)
    ax.bar(x, d["trade_pnl"], color=colors, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    cum = np.cumsum(d["trade_pnl"])
    ax2 = ax.twinx()
    ax2.plot(x, cum, color="#2980b9", linewidth=1.2, marker="o", markersize=3, label="累计盈亏")
    ax2.set_ylabel("累计盈亏", color="#2980b9", fontsize=10)
    ax2.tick_params(axis="y", colors="#2980b9")
    ax2.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{name} — 逐笔交易盈亏 (N={period}, M={multiplier})", fontsize=14, fontweight="bold")
    ax.set_xlabel("交易序号", fontsize=11)
    ax.set_ylabel("单笔盈亏", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_逐笔交易盈亏.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_signal_chart(d, df, trades, name, period, multiplier, output_dir, tail_bars=None):
    """SuperTrend 信号图，与 dashboard 一致风格"""
    n = len(df)
    if tail_bars is not None:
        start = max(0, n - tail_bars)
    else:
        start = 0
    df_tail = df.iloc[start:].copy()
    dates_t = df_tail["trade_date"]
    close_t = df_tail["close"].values
    st_t = df_tail["super_trend"].values
    trend_t = df_tail["trend"].values

    fig, ax = plt.subplots(figsize=(20, 8))

    # 背景着色：多头绿色 / 空头红色
    for i in range(len(dates_t) - 1):
        if trend_t[i] == 1:
            ax.axvspan(dates_t.iloc[i], dates_t.iloc[i + 1], alpha=0.04, color="green", linewidth=0)
        elif trend_t[i] == -1:
            ax.axvspan(dates_t.iloc[i], dates_t.iloc[i + 1], alpha=0.04, color="red", linewidth=0)

    ax.plot(dates_t, close_t, color="black", linewidth=0.6, alpha=0.7)
    ax.plot(dates_t, st_t, color=COLOR_STRATEGY, linewidth=0.8)

    # 标记信号
    # direction=="short" → 平空翻多 → 看涨 → 绿色 ▲
    # direction=="long"  → 平多翻空 → 看跌 → 红色 ▼
    tail_start = dates_t.iloc[0]
    for t in trades:
        if t["exec_date"] >= tail_start:
            ax.scatter(t["exec_date"], df.loc[df["trade_date"] == t["exec_date"], "close"].values[0]
                      if len(df.loc[df["trade_date"] == t["exec_date"], "close"].values) > 0 else np.nan,
                      marker="^" if t["direction"] == "short" else "v",
                      color="green" if t["direction"] == "short" else "red",
                      s=50, alpha=0.9, zorder=5, edgecolors="white", linewidth=0.5)

    if tail_bars is not None and tail_bars < n:
        label = f"最近{tail_bars}条"
    else:
        label = "全周期"
    ax.set_title(f"{name} — SuperTrend 信号 ({label}, N={period}, M={multiplier})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "08_SuperTrend信号图.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_return_distribution(d, name, period, multiplier, output_dir):
    """日收益率分布直方图"""
    fig, ax = plt.subplots(figsize=(14, 5))
    rets = d["rets"][d["rets"] != 0]
    ax.hist(rets * 100, bins=80, color=COLOR_STRATEGY, alpha=0.7, edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(rets.mean() * 100, color="#2980b9", linestyle="--", linewidth=1.2, label=f'均值: {rets.mean():.4%}')
    ax.axvline((rets.mean() + rets.std()) * 100, color="green", linestyle=":", linewidth=0.8, label=f'+1σ')
    ax.axvline((rets.mean() - rets.std()) * 100, color="red", linestyle=":", linewidth=0.8, label=f'-1σ')
    ax.set_title(f"{name} — 日收益率分布 (N={period}, M={multiplier})", fontsize=14, fontweight="bold")
    ax.set_xlabel("日收益率 %", fontsize=11)
    ax.set_ylabel("概率密度", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "09_日收益率分布.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(d, name, period, multiplier, output_dir):
    """指标汇总：纯文本面板"""
    m = d["raw_metrics"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    lines = [
        f"={' ' * 20}{name} — 策略绩效报告{' ' * 20}=",
        "",
        f"  参数:  ATR周期 = {period},  乘数 M = {multiplier}",
        "",
        "━━━━━━━━━━━━━━━━ 收益指标 ━━━━━━━━━━━━━━━━",
        f"  总收益率:        {m['total_return']:>10.2%}        最终净值:    {m['final_equity']:>12,.0f}",
        f"  年化收益率:      {m['annual_return']:>10.2%}        回测天数:    {m['n_days']:>10} 天",
        f"  年化波动率:      {m['daily_vol']:>10.2%}        数据年限:    {m['n_days']/252:>10.1f} 年",
        "",
        "━━━━━━━━━━━━━━━━ 风险指标 ━━━━━━━━━━━━━━━━",
        f"  最大回撤:        {m['max_dd']:>10.2%}        最长回撤持续: {m['max_dd_days']:>6} 天",
        f"  夏普比率:        {m['sharpe']:>10.2f}        卡尔玛比率:   {m['calmar']:>10.2f}",
        "",
        "━━━━━━━━━━━━━━━━ 交易指标 ━━━━━━━━━━━━━━━━",
        f"  交易次数:        {m['n_trades']:>10}        胜率:         {m['win_rate']:>10.2%}",
        f"  平均盈利:        {m['avg_win']:>10.2%}        平均亏损:     {m['avg_loss']:>10.2%}",
        f"  盈亏比:          {m['profit_factor']:>10.2f}        平均持仓天数: {m['avg_hold_days']:>8.1f} 天",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    text = "\n".join(lines)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=11, fontfamily="sans-serif",
            ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#dee2e6", pad=1.5))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "10_绩效汇总.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5b. 拼盘大图
# ============================================================

def plot_dashboard(d, df, trades, name, period, multiplier, output_dir):
    """综合仪表盘：一张大图包含所有关键图表"""
    fig = plt.figure(figsize=(22, 28))
    gs = GridSpec(6, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ---- Row 1: 净值曲线 (跨两列) + 指标面板 ----
    ax_eq = fig.add_subplot(gs[0, :2])
    ax_eq.plot(d["dates"], d["eq"] / d["eq"][0], color=COLOR_STRATEGY, linewidth=1.0, label="策略")
    ax_eq.plot(d["dates"], d["bench"], color=COLOR_BENCH, linewidth=0.6, linestyle="--", alpha=0.7, label="标的 Buy&Hold")
    ax_eq.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax_eq.set_title("净值曲线", fontsize=12, fontweight="bold")
    ax_eq.legend(loc="upper left", fontsize=8)
    ax_eq.grid(True, alpha=0.3)
    ax_eq.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax_m = fig.add_subplot(gs[0, 2])
    ax_m.axis("off")
    m = d["raw_metrics"]
    summary = (
        f"总收益: {m['total_return']:.2%}\n"
        f"年化: {m['annual_return']:.2%}\n"
        f"最大回撤: {m['max_dd']:.2%}\n"
        f"回撤天数: {m['max_dd_days']}\n"
        f"夏普: {m['sharpe']:.2f}\n"
        f"Calmar: {m['calmar']:.2f}\n"
        f"波动率: {m['daily_vol']:.2%}\n"
        f"交易: {m['n_trades']} 笔\n"
        f"胜率: {m['win_rate']:.2%}\n"
        f"盈亏比: {m['profit_factor']:.2f}\n"
        f"平均持仓: {m['avg_hold_days']:.1f}天"
    )
    ax_m.text(0.05, 0.95, summary, transform=ax_m.transAxes, fontsize=9, fontfamily="sans-serif",
              va="top", bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    # ---- Row 2: 回撤曲线 + 滚动夏普 ----
    ax_dd = fig.add_subplot(gs[1, :2])
    ax_dd.fill_between(d["dates"], 0, d["dd"], color="red", alpha=0.12)
    ax_dd.plot(d["dates"], d["dd"], color="red", linewidth=0.5)
    ax_dd.fill_between(d["dates"], 0, d["bench_dd"], color="gray", alpha=0.06)
    ax_dd.plot(d["dates"], d["bench_dd"], color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    ax_dd.set_title("回撤曲线 (%)", fontsize=12, fontweight="bold")
    ax_dd.grid(True, alpha=0.3)

    ax_sh = fig.add_subplot(gs[1, 2])
    valid = ~np.isnan(d["roll_sharpe"])
    ax_sh.plot(d["dates"][valid], d["roll_sharpe"][valid], color="#2980b9", linewidth=0.6)
    ax_sh.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax_sh.fill_between(d["dates"][valid], 0, d["roll_sharpe"][valid],
                        where=(d["roll_sharpe"][valid] > 0), color="green", alpha=0.06)
    ax_sh.fill_between(d["dates"][valid], 0, d["roll_sharpe"][valid],
                        where=(d["roll_sharpe"][valid] <= 0), color="red", alpha=0.06)
    ax_sh.set_title("滚动夏普 (252日)", fontsize=12, fontweight="bold")
    ax_sh.grid(True, alpha=0.3)

    # ---- Row 3: 滚动收益&波动 + 月度热力图 ----
    ax_rr = fig.add_subplot(gs[2, :2])
    valid2 = ~np.isnan(d["roll_ret"])
    ax_rr.plot(d["dates"][valid2], d["roll_ret"][valid2] * 100, color="#27ae60", linewidth=0.6, label="滚动年化收益")
    ax_rr.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax_rr2 = ax_rr.twinx()
    ax_rr2.plot(d["dates"][valid2], d["roll_vol"][valid2] * 100, color="#e74c3c", linewidth=0.4, alpha=0.6, label="波动率")
    ax_rr.set_title("滚动年化收益 & 波动率 (252日)", fontsize=12, fontweight="bold")
    ax_rr.grid(True, alpha=0.3)
    lines1, labels1 = ax_rr.get_legend_handles_labels()
    lines2, labels2 = ax_rr2.get_legend_handles_labels()
    ax_rr.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

    df_m = pd.DataFrame({"date": d["dates"], "ret": d["rets"]})
    df_m["year"] = df_m["date"].dt.year
    df_m["month"] = df_m["date"].dt.month
    monthly = df_m.groupby(["year", "month"])["ret"].apply(lambda x: (1 + x).prod() - 1).unstack()
    ax_mh = fig.add_subplot(gs[2, 2])
    im = ax_mh.imshow(monthly.values, aspect="auto", cmap="RdYlGn", vmin=-0.15, vmax=0.15)
    ax_mh.set_xticks(range(12))
    ax_mh.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"], fontsize=6)
    ax_mh.set_yticks(range(len(monthly)))
    ax_mh.set_yticklabels(monthly.index, fontsize=6)
    ax_mh.set_title("月度收益 %", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax_mh, format=mticker.PercentFormatter(1.0))

    # ---- Row 4: 年度收益 + 交易盈亏 ----
    df_a = pd.DataFrame({"date": d["dates"], "ret": d["rets"]})
    df_a["year"] = df_a["date"].dt.year
    annual = df_a.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)
    ax_ar = fig.add_subplot(gs[3, :2])
    bar_colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in annual.values]
    ax_ar.bar(annual.index.astype(str), annual.values * 100, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax_ar.axhline(0, color="black", linewidth=0.5)
    for xi, val in zip(annual.index.astype(str), annual.values):
        ax_ar.text(xi, (val * 100) + (3 if val >= 0 else -5), f"{val:.1%}", ha="center", fontsize=8, fontweight="bold")
    ax_ar.set_title("年度收益", fontsize=12, fontweight="bold")
    ax_ar.grid(True, alpha=0.3, axis="y")

    if len(d["trade_pnl"]) > 0:
        ax_tp = fig.add_subplot(gs[3, 2])
        colors_t = [COLOR_LONG if pnl > 0 else COLOR_SHORT for pnl in d["trade_pnl"]]
        ax_tp.bar(range(1, len(d["trade_pnl"]) + 1), d["trade_pnl"], color=colors_t,
                   edgecolor="white", linewidth=0.2)
        ax_tp.axhline(0, color="black", linewidth=0.5)
        ax_tp.set_title("逐笔交易盈亏", fontsize=12, fontweight="bold")
        ax_tp.grid(True, alpha=0.3, axis="y")

    # ---- Row 5: SuperTrend 信号图 (全周期) ----
    ax_sig = fig.add_subplot(gs[4, :])
    n_tail = len(df)  # 显示全周期
    df_tail = df.iloc[-n_tail:]
    dates_t = df_tail["trade_date"]
    close_t = df_tail["close"].values
    st_t = df_tail["super_trend"].values
    trend_t = df_tail["trend"].values
    for i in range(len(dates_t) - 1):
        if trend_t[i] == 1:
            ax_sig.axvspan(dates_t.iloc[i], dates_t.iloc[i + 1], alpha=0.04, color="green", linewidth=0)
        elif trend_t[i] == -1:
            ax_sig.axvspan(dates_t.iloc[i], dates_t.iloc[i + 1], alpha=0.04, color="red", linewidth=0)
    ax_sig.plot(dates_t, close_t, color="black", linewidth=0.6, alpha=0.7)
    ax_sig.plot(dates_t, st_t, color=COLOR_STRATEGY, linewidth=0.8)
    tail_start = dates_t.iloc[0]
    for t in trades:
        if t["exec_date"] >= tail_start:
            ax_sig.scatter(t["exec_date"], df.loc[df["trade_date"] == t["exec_date"], "close"].values[0]
                          if len(df.loc[df["trade_date"] == t["exec_date"], "close"].values) > 0 else np.nan,
                          marker="^" if t["direction"] == "short" else "v",
                          color="green" if t["direction"] == "short" else "red",
                          s=50, alpha=0.9, zorder=5, edgecolors="white", linewidth=0.5)
    ax_sig.set_title("SuperTrend 信号 (全周期)", fontsize=12, fontweight="bold")
    ax_sig.grid(True, alpha=0.3)

    # ---- Row 6: 日收益分布 ----
    ax_hist = fig.add_subplot(gs[5, :])
    rets_clean = d["rets"][d["rets"] != 0]
    ax_hist.hist(rets_clean * 100, bins=60, color=COLOR_STRATEGY, alpha=0.6, edgecolor="white", linewidth=0.3, density=True)
    ax_hist.axvline(0, color="black", linewidth=0.8)
    ax_hist.axvline(rets_clean.mean() * 100, color="#2980b9", linestyle="--", linewidth=1.0, label=f'均值={rets_clean.mean():.4%}')
    ax_hist.set_title("日收益率分布", fontsize=12, fontweight="bold")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    fig.suptitle(f"{name} · SuperTrend 策略回测报告 (N={period}, M={multiplier})",
                 fontsize=16, fontweight="bold", y=0.995)
    fig.savefig(os.path.join(output_dir, "dashboard.png"), dpi=200, bbox_inches="tight")
    print(f"  仪表盘已保存: {os.path.join(output_dir, 'dashboard.png')}")
    plt.close(fig)


# ============================================================
# 5c. 总入口
# ============================================================

def generate_report(df, trades, equity_curve, daily_returns, capital, name, period, multiplier, output_dir="backtest_output"):
    """
    生成完整回测报告：
      - 10 张独立分析图
      - 1 张综合仪表盘
      - 交易明细 CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  生成报告: {name} (N={period}, M={multiplier})")
    print(f"{'='*60}")

    d = _prepare_data(df, trades, equity_curve, daily_returns, capital)

    # 独立图
    plot_equity_curve(d, name, period, multiplier, output_dir)
    plot_drawdown(d, name, period, multiplier, output_dir)
    plot_rolling_sharpe(d, name, period, multiplier, output_dir)
    plot_rolling_returns(d, name, period, multiplier, output_dir)
    plot_annual_returns(d, name, period, multiplier, output_dir)
    plot_monthly_heatmap(d, name, period, multiplier, output_dir)
    plot_trade_pnl(d, name, period, multiplier, output_dir)
    plot_signal_chart(d, df, trades, name, period, multiplier, output_dir)
    plot_return_distribution(d, name, period, multiplier, output_dir)
    plot_metrics_summary(d, name, period, multiplier, output_dir)

    # 拼盘大图
    plot_dashboard(d, df, trades, name, period, multiplier, output_dir)

    # 交易明细 CSV
    if trades:
        trades_df = pd.DataFrame(trades)
        csv_path = os.path.join(output_dir, "trades.csv")
        trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  交易明细已保存: {csv_path} ({len(trades)} 笔)")

    print(f"\n  报告生成完毕 → {output_dir}/")
    print(f"    - 10 张独立分析图")
    print(f"    - 1 张仪表盘大图 (dashboard.png)")
    print(f"    - 1 份交易明细 (trades.csv)")
    return d


# ============================================================
# 6. 入口
# ============================================================

if __name__ == "__main__":
    # ==================== 在这里修改配置 ====================
    DATA_FILE = "标普500_10年日线.parquet"   # 数据文件路径
    DATA_NAME = "标普500"                    # 品种名称（用于图表标题）
    PERIOD = 56                              # SuperTrend ATR 周期
    MULTIPLIER = 7.2                          # SuperTrend 乘数 M
    CAPITAL = 1_000_000                      # 初始资金
    OUTPUT_DIR = f"标普500 {PERIOD},{MULTIPLIER}回测结果"           # 报告输出目录
    # ========================================================

    print(f"加载数据: {DATA_FILE}")
    df = load_data(DATA_FILE)
    print(f"  数据: {len(df)} 条, {df['trade_date'].iloc[0].date()} ~ {df['trade_date'].iloc[-1].date()}")

    df = calc_super_trend(df, period=PERIOD, multiplier=MULTIPLIER)
    trades, equity_curve, daily_returns = run_backtest(df, capital=CAPITAL)

    raw_metrics = calc_metrics(trades, equity_curve, daily_returns, CAPITAL)
    print(f"\n  === 绩效概览 ===")
    for k, v in raw_metrics.items():
        print(f"  {k}: {v}")

    generate_report(df, trades, equity_curve, daily_returns, CAPITAL,
                    DATA_NAME, PERIOD, MULTIPLIER, OUTPUT_DIR)
