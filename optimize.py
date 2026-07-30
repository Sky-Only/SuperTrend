"""
SuperTrend 参数网格搜索优化（细粒度）
乘数 M: 0.1 ~ 10.0, 步长 0.1
周期 N: 1 ~ 100,   步长 1
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.sans-serif"] = ["Heiti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from backtest import load_data

# ============================================================
# 搜索范围
# ============================================================
PERIODS = list(range(1, 101))                     # 1 ~ 100
MULTIPLIERS = [round(x * 0.1, 1) for x in range(1, 101)]  # 0.1 ~ 10.0

DATA = {
    "沪深300": "沪深300_10年日线.parquet",
    "标普500": "标普500_10年日线.parquet",
    "沪金期货": "沪金期货_10年日线.parquet",
}
CAPITAL = 1_000_000


# ============================================================
# 快速 SuperTrend + 回测（纯 numpy，不做交易记录）
# ============================================================

def fast_supertrend_trend(high, low, close, period, multiplier):
    """
    快速计算 SuperTrend 趋势序列
    返回: trend (1=多, -1=空), 价格数据不变
    """
    n = len(close)
    # TR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))
    # ATR (Wilder)
    alpha = 1.0 / period
    atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    # 基础带
    mid = (high + low) / 2
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr
    # SuperTrend
    st = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    # 初始化
    start = max(period, 1)
    if close[start] > upper[start]:
        trend[start] = 1
        st[start] = lower[start]
    elif close[start] < lower[start]:
        trend[start] = -1
        st[start] = upper[start]
    else:
        trend[start] = 1
        st[start] = lower[start]
    for i in range(start + 1, n):
        if trend[i - 1] == 1:
            s = max(lower[i], st[i - 1])
            if close[i] < s:
                trend[i] = -1
                st[i] = upper[i]
            else:
                trend[i] = 1
                st[i] = s
        else:
            s = min(upper[i], st[i - 1])
            if close[i] > s:
                trend[i] = 1
                st[i] = lower[i]
            else:
                trend[i] = -1
                st[i] = s
    return trend


def fast_backtest_equity(high, low, close, trend, capital=CAPITAL):
    """
    快速回测，返回每日净值和日收益率（不做交易记录）
    规则: t日趋势变化 → t+1日执行, 买入用HIGH, 卖出用LOW
    """
    n = len(close)
    # 找出信号（趋势变化点 + 首个趋势）
    signal = np.zeros(n, dtype=int)
    trend_prev = np.roll(trend, 1)
    trend_prev[0] = 0
    signal[(trend_prev == -1) & (trend == 1)] = 1
    signal[(trend_prev == 1) & (trend == -1)] = -1
    # 首个非零趋势为入场信号
    first_valid = np.where(trend != 0)[0]
    if len(first_valid) > 0:
        fi = first_valid[0]
        signal[fi] = 1 if trend[fi] == 1 else -1

    equity = float(capital)
    position = 0.0      # >0 long, <0 short
    entry_price = 0.0
    direction = 0       # 1=long, -1=short
    pending = 0         # 待执行信号

    eq_curve = np.zeros(n)
    daily_ret = np.zeros(n)

    for i in range(n):
        # Step 1: 执行前日信号
        if pending != 0:
            px = high[i] if pending == 1 else low[i]
            if direction == -1 and pending == 1:
                pnl = (entry_price - px) * abs(position)
                equity += pnl
            elif direction == 1 and pending == -1:
                pnl = (px - entry_price) * position
                equity += pnl
            # 开新仓
            if pending == 1:
                position = equity / px
                direction = 1
            else:
                position = -equity / px
                direction = -1
            entry_price = px
            pending = 0

        # Step 2: 记录今日信号
        if signal[i] != 0:
            pending = signal[i]

        # Step 3: 按收盘价估值
        if direction == 1:
            eq_curve[i] = position * close[i]
        elif direction == -1:
            eq_curve[i] = equity + (entry_price - close[i]) * abs(position)
        else:
            eq_curve[i] = equity

        if i > 0 and eq_curve[i - 1] > 0:
            daily_ret[i] = (eq_curve[i] - eq_curve[i - 1]) / eq_curve[i - 1]

    # 强制平仓最后持仓
    if direction != 0:
        px_last = close[-1]
        if direction == 1:
            equity += (px_last - entry_price) * position
        else:
            equity += (entry_price - px_last) * abs(position)
        eq_curve[-1] = equity

    return eq_curve, daily_ret


def compute_metrics(eq_curve, daily_ret, capital=CAPITAL):
    """从净值和日收益计算绩效指标"""
    n = len(daily_ret)
    n_years = n / 252
    total_return = (eq_curve[-1] - capital) / capital
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 and total_return > -1 else total_return / n_years if n_years > 0 else 0
    peak = np.maximum.accumulate(eq_curve)
    max_dd = (eq_curve - peak).min() / peak[np.argmax(peak)] if peak.max() > 0 else 0
    rf_daily = 0.03 / 252
    excess = daily_ret[1:] - rf_daily  # skip first day
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    return total_return, annual_return, max_dd, sharpe, calmar


# ============================================================
# 网格搜索
# ============================================================

def grid_search_fast(name, filepath):
    """快速网格搜索"""
    print(f"\n{'='*60}")
    print(f"  {name} — 细粒度参数搜索")
    print(f"  周期 1~100 × 乘数 0.1~10.0 = 10,000 组")
    print(f"{'='*60}")

    df_raw = load_data(filepath)
    high = df_raw["high"].values
    low = df_raw["low"].values
    close = df_raw["close"].values
    n = len(close)

    # 预计算 TR（和周期无关）
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))

    # 预计算所有周期的 ATR（100个周期 × n条数据 = 100×2500 ≈ 250KB，内存友好）
    atr_cache = {}
    for period in PERIODS:
        alpha = 1.0 / period
        atr_cache[period] = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values

    results = []
    total = len(PERIODS) * len(MULTIPLIERS)
    t0 = time.time()

    for count, period in enumerate(PERIODS):
        atr = atr_cache[period]
        mid = (high + low) / 2
        for mult in MULTIPLIERS:
            upper = mid + mult * atr
            lower = mid - mult * atr
            trend = fast_supertrend_from_bands(high, low, close, upper, lower, period, n)
            eq_curve, daily_ret = fast_backtest_equity(high, low, close, trend, CAPITAL)
            total_ret, annual_ret, max_dd, sharpe, calmar = compute_metrics(eq_curve, daily_ret, CAPITAL)
            results.append({
                "period": period,
                "multiplier": mult,
                "total_return": total_ret,
                "annual_return": annual_ret,
                "max_dd": max_dd,
                "sharpe": sharpe,
                "calmar": calmar,
            })
        # 进度
        done = (count + 1) * len(MULTIPLIERS)
        if (count + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  周期 {period:3d}/100 完成, 已耗时 {elapsed:.0f}s, 预计剩余 {eta:.0f}s")

    print(f"  搜索完成! 总耗时 {time.time()-t0:.1f}s")
    return df_raw, pd.DataFrame(results)


def fast_supertrend_from_bands(high, low, close, upper, lower, period, n):
    """从预计算的 bands 快速计算 SuperTrend 趋势"""
    st = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    start = max(period, 1)
    if start >= n:
        return trend
    if close[start] > upper[start]:
        trend[start] = 1
        st[start] = lower[start]
    elif close[start] < lower[start]:
        trend[start] = -1
        st[start] = upper[start]
    else:
        trend[start] = 1
        st[start] = lower[start]
    for i in range(start + 1, n):
        if trend[i - 1] == 1:
            s = max(lower[i], st[i - 1])
            if close[i] < s:
                trend[i] = -1
                st[i] = upper[i]
            else:
                trend[i] = 1
                st[i] = s
        else:
            s = min(upper[i], st[i - 1])
            if close[i] > s:
                trend[i] = 1
                st[i] = lower[i]
            else:
                trend[i] = -1
                st[i] = s
    return trend


# ============================================================
# 可视化
# ============================================================

def plot_heatmaps(all_results):
    """绘制总收益率 + 夏普比率热力图"""
    names = list(all_results.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(20, 6 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for idx, name in enumerate(names):
        df_r = all_results[name]["results"]

        # --- 总收益率热力图 ---
        ax1 = axes[idx, 0]
        pivot_ret = df_r.pivot_table(index="period", columns="multiplier", values="total_return")
        # 对收益做裁剪，让色阶更好看
        vmax_ret = max(1.0, pivot_ret.values.max())
        vmin_ret = min(-1.0, pivot_ret.values.min())
        im1 = ax1.imshow(pivot_ret.values, aspect="auto", origin="lower",
                         cmap="RdYlGn", vmin=vmin_ret, vmax=vmax_ret)
        ax1.set_xlabel("乘数 M", fontsize=11)
        ax1.set_ylabel("ATR 周期 N", fontsize=11)
        ax1.set_title(f"{name} — 总收益率", fontsize=13, fontweight="bold")
        # 坐标刻度每隔一定步长显示
        xtick_step = 10
        ytick_step = 10
        ax1.set_xticks(range(0, len(MULTIPLIERS), xtick_step))
        ax1.set_xticklabels([f"{MULTIPLIERS[i]:.1f}" for i in range(0, len(MULTIPLIERS), xtick_step)])
        ax1.set_yticks(range(0, len(PERIODS), ytick_step))
        ax1.set_yticklabels([PERIODS[i] for i in range(0, len(PERIODS), ytick_step)])
        # 标注最佳
        best_idx = df_r["total_return"].idxmax()
        best = df_r.iloc[best_idx]
        ax1.text(0.02, 0.98,
                 f"最佳: N={int(best['period'])}, M={best['multiplier']:.1f}\n"
                 f"收益={best['total_return']:.1%}  夏普={best['sharpe']:.2f}\n"
                 f"回撤={best['max_dd']:.1%}",
                 transform=ax1.transAxes, fontsize=9, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        plt.colorbar(im1, ax=ax1, format=mticker.PercentFormatter(1.0))

        # --- 夏普比率热力图 ---
        ax2 = axes[idx, 1]
        pivot_sharpe = df_r.pivot_table(index="period", columns="multiplier", values="sharpe")
        vmax_s = max(0.5, pivot_sharpe.values.max())
        vmin_s = min(-0.5, pivot_sharpe.values.min())
        im2 = ax2.imshow(pivot_sharpe.values, aspect="auto", origin="lower",
                         cmap="RdYlGn", vmin=vmin_s, vmax=vmax_s)
        ax2.set_xlabel("乘数 M", fontsize=11)
        ax2.set_ylabel("ATR 周期 N", fontsize=11)
        ax2.set_title(f"{name} — 夏普比率", fontsize=13, fontweight="bold")
        ax2.set_xticks(range(0, len(MULTIPLIERS), xtick_step))
        ax2.set_xticklabels([f"{MULTIPLIERS[i]:.1f}" for i in range(0, len(MULTIPLIERS), xtick_step)])
        ax2.set_yticks(range(0, len(PERIODS), ytick_step))
        ax2.set_yticklabels([PERIODS[i] for i in range(0, len(PERIODS), ytick_step)])
        best_s_idx = df_r["sharpe"].idxmax()
        best_s = df_r.iloc[best_s_idx]
        ax2.text(0.02, 0.98,
                 f"最佳夏普: N={int(best_s['period'])}, M={best_s['multiplier']:.1f}\n"
                 f"夏普={best_s['sharpe']:.2f}  收益={best_s['total_return']:.1%}\n"
                 f"回撤={best_s['max_dd']:.1%}",
                 transform=ax2.transAxes, fontsize=9, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        plt.colorbar(im2, ax=ax2)

        # 打印 Top 10
        top10 = df_r.nlargest(10, "sharpe")
        print(f"\n  {name} Top 10 (按夏普):")
        for _, row in top10.iterrows():
            print(f"    N={int(row['period']):3d}  M={row['multiplier']:4.1f}  "
                  f"收益={row['total_return']:7.2%}  年化={row['annual_return']:7.2%}  "
                  f"回撤={row['max_dd']:7.2%}  夏普={row['sharpe']:7.3f}  "
                  f"Calmar={row['calmar']:6.2f}")

    fig.suptitle("SuperTrend 细粒度参数搜索 (N=1~100, M=0.1~10.0)", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("optimize_heatmap.png", dpi=150, bbox_inches="tight")
    print("\n热力图已保存: optimize_heatmap.png")
    plt.close()


def plot_best_equity(all_results, all_best):
    """最优参数净值曲线"""
    from backtest import calc_super_trend, run_backtest

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    colors = ["#e74c3c", "#2980b9", "#f39c12"]

    for idx, (name, color) in enumerate(zip(all_results.keys(), colors)):
        best = all_best[name]
        raw = all_results[name]["raw"]
        period, mult = int(best["period"]), float(best["multiplier"])

        df = calc_super_trend(raw, period=period, multiplier=mult)
        trades, eq_curve, _ = run_backtest(df, capital=CAPITAL)

        dates = [e["trade_date"] for e in eq_curve]
        vals = np.array([e["equity"] for e in eq_curve]) / CAPITAL
        peak = np.maximum.accumulate(vals)
        dd = (vals - peak) / peak * 100

        ax = axes[idx]
        ax.plot(dates, vals, color=color, linewidth=0.9, label="净值")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_ylabel("净值", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.fill_between(dates, 0, dd, color="red", alpha=0.12)
        ax2.plot(dates, dd, color="red", linewidth=0.3, alpha=0.4)
        ax2.set_ylabel("回撤 %", color="red", fontsize=9)
        ax2.tick_params(axis="y", colors="red")

        total_ret = (vals[-1] - 1)
        ax.set_title(f"{name} (N={period}, M={mult:.1f})  "
                     f"总收益={total_ret:.1%}  夏普={best['sharpe']:.2f}  "
                     f"最大回撤={best['max_dd']:.1%}  交易={int(best.get('n_trades', 0))}笔",
                     fontsize=12, fontweight="bold")

    fig.suptitle("最优参数净值曲线（按夏普比率）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("optimize_equity.png", dpi=150, bbox_inches="tight")
    print("净值曲线已保存: optimize_equity.png")
    plt.close()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    all_results = {}
    all_best = {}

    for name, path in DATA.items():
        raw, df_r = grid_search_fast(name, path)
        # 估算交易次数（用最优参数跑一次完整回测）
        best = df_r.iloc[df_r["sharpe"].idxmax()].to_dict()
        from backtest import calc_super_trend, run_backtest
        df_t = calc_super_trend(raw, period=int(best["period"]), multiplier=float(best["multiplier"]))
        trades, _, _ = run_backtest(df_t, capital=CAPITAL)
        best["n_trades"] = len(trades)
        all_best[name] = best
        all_results[name] = {"raw": raw, "results": df_r}

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"  最优参数汇总 (按夏普比率)")
    print(f"{'='*60}")
    for name in DATA:
        b = all_best[name]
        print(f"  {name}: N={int(b['period']):3d}, M={b['multiplier']:.1f}  "
              f"夏普={b['sharpe']:.3f}  收益={b['total_return']:.1%}  "
              f"回撤={b['max_dd']:.1%}  交易={int(b['n_trades'])}笔")

    # ---- 绘图 ----
    plot_heatmaps(all_results)
    plot_best_equity(all_results, all_best)
