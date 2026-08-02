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
# 5. 绘图
# ============================================================

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm
import os

# ---- 注册本地字体（跨平台中文支持）----
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


def plot_results(all_results, period=10, multiplier=3):
    """绘制回测结果：净值曲线 + 回撤 + SuperTrend信号"""
    names = list(all_results.keys())
    colors = ["#e74c3c", "#2980b9", "#f39c12"]
    n = len(names)

    fig = plt.figure(figsize=(18, 5 * n + 6))

    # ---- Panel 1: 净值曲线对比 ----
    ax1 = fig.add_subplot(n + 1, 2, 1)
    for name, color in zip(names, colors):
        eq = all_results[name]["equity"]
        dates = [e["trade_date"] for e in eq]
        vals = [e["equity"] / 1_000_000 for e in eq]
        ax1.plot(dates, vals, color=color, linewidth=0.8, label=name)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_title("净值曲线对比", fontsize=13, fontweight="bold")
    ax1.set_ylabel("净值 (初始=1)")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: 回撤曲线 ----
    ax2 = fig.add_subplot(n + 1, 2, 2)
    for name, color in zip(names, colors):
        eq = all_results[name]["equity"]
        dates = [e["trade_date"] for e in eq]
        vals = np.array([e["equity"] for e in eq])
        peak = np.maximum.accumulate(vals)
        dd = (vals - peak) / peak * 100
        ax2.fill_between(dates, 0, dd, color=color, alpha=0.3, label=name)
        ax2.plot(dates, dd, color=color, linewidth=0.5)
    ax2.set_title("回撤曲线 (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("回撤 %")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3~5: 每个品种的 SuperTrend + 价格 + 信号 ----
    for idx, (name, color) in enumerate(zip(names, colors)):
        ax = fig.add_subplot(n + 1, 2, 3 + idx * 2)
        res = all_results[name]
        df = res["df"]
        trades = res["trades"]

        # 画全部数据
        plot_df = df.copy()

        dates = plot_df["trade_date"]
        ax.plot(dates, plot_df["close"], color="black", linewidth=0.6, alpha=0.7, label="Close")
        ax.plot(dates, plot_df["super_trend"], color=color, linewidth=0.8, label="SuperTrend")

        # 标记买卖信号点
        for t in trades:
            if t["exec_date"] in dates.values:
                if t["direction"] == "long":
                    # 多头出场=卖出信号
                    ax.scatter(t["exec_date"], t["exit_price"], marker="v", color="green",
                              s=30, alpha=0.8, zorder=5)
                else:
                    # 空头出场=买入信号（平空=买回）
                    ax.scatter(t["exec_date"], t["exit_price"], marker="^", color="red",
                              s=30, alpha=0.8, zorder=5)

        ax.set_title(f"{name} — SuperTrend 信号 (全周期)", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 交易统计小面板
        ax_metric = fig.add_subplot(n + 1, 2, 4 + idx * 2)
        ax_metric.axis("off")
        m = res["metrics"]
        text = (
            f"总收益: {m['总收益率']}    年化: {m['年化收益率']}\n"
            f"最大回撤: {m['最大回撤']}    夏普: {m['夏普比率']}\n"
            f"交易: {m['交易次数']}笔    胜率: {m['胜率']}\n"
            f"平均盈利: {m['平均盈利']}    平均亏损: {m['平均亏损']}\n"
            f"盈亏比: {m['盈亏比']}    回测天数: {m['回测天数']}"
        )
        ax_metric.text(0.05, 0.5, text, transform=ax_metric.transAxes,
                       fontsize=10, verticalalignment="center",
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"SuperTrend 双向策略回测 (ATR={period}, M={multiplier})",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
    print("\n图表已保存: backtest_results.png")
    plt.close()

def backtest_symbol(name, filepath, period=10, multiplier=3, capital=1_000_000):
    """对单个品种执行完整回测流程"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    df = load_data(filepath)
    print(f"  数据: {len(df)} 条, {df['trade_date'].iloc[0].date()} ~ {df['trade_date'].iloc[-1].date()}")

    df = calc_super_trend(df, period=period, multiplier=multiplier)
    trades, equity_curve, daily_returns = run_backtest(df, capital=capital)
    metrics = calc_metrics(trades, equity_curve, daily_returns, capital)

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 导出交易明细
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(f"{name}_trades.csv", index=False, encoding="utf-8-sig")
        print(f"  交易明细已导出: {name}_trades.csv ({len(trades)} 笔)")

    return df, trades, equity_curve, metrics


if __name__ == "__main__":
    DATA = {
        "沪深300": "沪深300_10年日线.parquet",
        "标普500": "标普500_10年日线.parquet",
        "沪金期货": "沪金期货_10年日线.parquet",
    }

    # SuperTrend 参数
    PERIOD = 10
    MULTIPLIER = 3
    CAPITAL = 1_000_000

    all_results = {}
    for name, path in DATA.items():
        df, trades, eq_curve, metrics = backtest_symbol(
            name, path, period=PERIOD, multiplier=MULTIPLIER, capital=CAPITAL
        )
        all_results[name] = {"df": df, "trades": trades, "equity": eq_curve, "metrics": metrics}

    print(f"\n{'='*60}")
    print(f"  策略汇总 (ATR={PERIOD}, M={MULTIPLIER})")
    print(f"{'='*60}")
    for name, res in all_results.items():
        m = res["metrics"]
        print(f"  {name}: 总收益={m['总收益率']}, 年化={m['年化收益率']}, "
              f"最大回撤={m['最大回撤']}, 夏普={m['夏普比率']}, 胜率={m['胜率']}")

    plot_results(all_results, period=PERIOD, multiplier=MULTIPLIER)
