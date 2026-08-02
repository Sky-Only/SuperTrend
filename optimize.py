"""
SuperTrend 参数网格搜索优化（Numba JIT 加速版）

乘数 M: 0.1 ~ 50.0, 步长 0.1  (500 个值)
周期 N: 1 ~ 500,   步长 1      (500 个值)
总组合: 500x500 = 250,000 / 资产

性能对比:
  纯 Python:  ~30-60 分钟 / 资产
  Numba JIT:  ~10-20 秒 / 资产  (100-200x 加速)
  + 多进程:    ~2-5  秒 / 资产  (额外 4-8x)
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm
import os
from multiprocessing import Pool, cpu_count

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

from backtest import load_data
from tqdm import tqdm
import metrics as mt

# ============================================================
# 搜索范围
# ============================================================
PERIODS = list(range(5, 100))                              # 1 ~ 500
MULTIPLIERS = [round(x * 0.1, 1) for x in range(5, 50)]   # 0.1 ~ 50.0

DATA = {
    "沪深300": "沪深300_10年日线.parquet",
    "标普500": "标普500_10年日线.parquet",
    "沪金期货": "沪金期货_10年日线.parquet",
}
CAPITAL = 1_000_000

# 是否使用多进程并行（通过环境变量或直接设置）
USE_PARALLEL = True
N_WORKERS = min(8, max(1, cpu_count() - 1))  # 最多 8 个进程，留一个核心给系统

# ============================================================
# Numba JIT 加速核心函数
# ============================================================

try:
    import numba as nb
    HAS_NUMBA = True

    @nb.jit(nopython=True, fastmath=True, cache=True)
    def _super_trend_backtest(high, low, close, atr, multiplier, period, capital):
        """
        JIT 编译的核心函数: SuperTrend 趋势 + 回测 + 绩效指标，一条链路完成。
        全部在机器码层执行，无 Python 解释器开销。

        注意: 本函数中绩效指标的计算逻辑与 metrics.py 模块保持一致。
        如需验证或修改指标算法，请参考 metrics.py 中的独立实现。

        参数:
            high, low, close: 价格序列 (float64 1D array)
            atr: 预计算的 ATR 序列 (float64 1D array)
            multiplier: SuperTrend 乘数
            period: ATR 周期
            capital: 初始资金
        返回:
            (total_return, annual_return, max_dd, sharpe, calmar)
        """
        n = len(close)

        # ---- SuperTrend 波段 ----
        mid = (high + low) * 0.5
        upper = mid + multiplier * atr
        lower = mid - multiplier * atr

        # ---- SuperTrend 趋势 ----
        st = np.full(n, np.nan)
        trend = np.zeros(n, dtype=np.int32)
        start_val = max(period, 1)

        if start_val >= n:
            return (0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)

        if close[start_val] > upper[start_val]:
            trend[start_val] = 1
            st[start_val] = lower[start_val]
        elif close[start_val] < lower[start_val]:
            trend[start_val] = -1
            st[start_val] = upper[start_val]
        else:
            trend[start_val] = 1
            st[start_val] = lower[start_val]

        for i in range(start_val + 1, n):
            if trend[i - 1] == 1:
                s = lower[i]
                if st[i - 1] > s:
                    s = st[i - 1]
                if close[i] < s:
                    trend[i] = -1
                    st[i] = upper[i]
                else:
                    trend[i] = 1
                    st[i] = s
            else:
                s = upper[i]
                if st[i - 1] < s:
                    s = st[i - 1]
                if close[i] > s:
                    trend[i] = 1
                    st[i] = lower[i]
                else:
                    trend[i] = -1
                    st[i] = s

        # ---- 信号生成 ----
        signal = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            prev = trend[i - 1]
            cur = trend[i]
            if prev == -1 and cur == 1:
                signal[i] = 1
            elif prev == 1 and cur == -1:
                signal[i] = -1

        # 首个非零趋势为入场信号
        for i in range(n):
            if trend[i] != 0:
                if trend[i] == 1:
                    signal[i] = 1
                else:
                    signal[i] = -1
                break

        # ---- 回测 ----
        equity = float(capital)
        position = 0.0        # >0 long, <0 short
        entry_price = 0.0
        direction = 0         # 0=flat, 1=long, -1=short
        pending = 0

        eq_curve = np.zeros(n)
        daily_ret = np.zeros(n)

        # ---- 交易跟踪 ----
        n_trades = 0
        n_wins = 0
        n_losses = 0
        sum_win_ret = 0.0
        sum_loss_ret = 0.0
        total_hold_days = 0
        entry_day = 0
        old_position = 0.0  # 记录平仓前的持仓量

        for i in range(n):
            # Step 1: 执行前一日信号
            if pending != 0:
                px = high[i] if pending == 1 else low[i]

                if direction == -1 and pending == 1:
                    # 平空: 空头 P&L = 入场 - 出场, position 为负数, abs() 取数量
                    pnl = (entry_price - px) * abs(position)
                    equity += pnl
                    # 记录交易
                    n_trades += 1
                    pnl_pct = pnl / (entry_price * abs(position))
                    hold_days = i - entry_day
                    total_hold_days += hold_days
                    if pnl > 0.0:
                        n_wins += 1
                        sum_win_ret += pnl_pct
                    else:
                        n_losses += 1
                        sum_loss_ret -= pnl_pct
                    old_position = position
                elif direction == 1 and pending == -1:
                    # 平多: 多头 P&L = 出场 - 入场, position 为正数
                    pnl = (px - entry_price) * position
                    equity += pnl
                    # 记录交易
                    n_trades += 1
                    pnl_pct = pnl / (entry_price * position)
                    hold_days = i - entry_day
                    total_hold_days += hold_days
                    if pnl > 0.0:
                        n_wins += 1
                        sum_win_ret += pnl_pct
                    else:
                        n_losses += 1
                        sum_loss_ret -= pnl_pct
                    old_position = position

                # 开新仓
                if pending == 1:
                    position = equity / px
                    direction = 1
                else:
                    position = -equity / px
                    direction = -1
                entry_price = px
                entry_day = i
                pending = 0

            # Step 2: 记录今日信号
            if signal[i] != 0:
                pending = signal[i]

            # Step 3: 按收盘价估值
            if direction == 1:
                eq_curve[i] = position * close[i]
            elif direction == -1:
                pos_abs = -position if position < 0 else position
                eq_curve[i] = equity + (entry_price - close[i]) * pos_abs
            else:
                eq_curve[i] = equity

            if i > 0 and eq_curve[i - 1] > 0.0:
                daily_ret[i] = (eq_curve[i] - eq_curve[i - 1]) / eq_curve[i - 1]

        # 强制平仓
        if direction != 0:
            px_last = close[-1]
            if direction == 1:
                pnl_close = (px_last - entry_price) * position
                equity += pnl_close
                # 记录最终交易
                n_trades += 1
                pnl_pct = pnl_close / (entry_price * position)
                hold_days = n - 1 - entry_day
                total_hold_days += hold_days
                if pnl_close > 0.0:
                    n_wins += 1
                    sum_win_ret += pnl_pct
                else:
                    n_losses += 1
                    sum_loss_ret -= pnl_pct
            else:
                pnl_close = (entry_price - px_last) * abs(position)
                equity += pnl_close
                # 记录最终交易
                n_trades += 1
                pnl_pct = pnl_close / (entry_price * abs(position))
                hold_days = n - 1 - entry_day
                total_hold_days += hold_days
                if pnl_close > 0.0:
                    n_wins += 1
                    sum_win_ret += pnl_pct
                else:
                    n_losses += 1
                    sum_loss_ret -= pnl_pct
            eq_curve[-1] = equity

        # ---- 绩效指标 ----
        n_years = n / 252.0
        total_return = (eq_curve[-1] - capital) / capital

        if n_years > 0.0 and total_return > -1.0:
            annual_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0
        elif n_years > 0.0:
            annual_return = total_return / n_years
        else:
            annual_return = 0.0

        # 最大回撤
        peak = eq_curve[0]
        max_dd_val = 0.0
        for i in range(n):
            if eq_curve[i] > peak:
                peak = eq_curve[i]
            d = (eq_curve[i] - peak) / peak if peak > 0.0 else 0.0
            if d < max_dd_val:
                max_dd_val = d

        # 夏普比率
        rf_daily = 0.03 / 252.0
        excess_sum = 0.0
        excess_sq_sum = 0.0
        count = 0
        for i in range(1, n):
            ex = daily_ret[i] - rf_daily
            excess_sum += ex
            excess_sq_sum += ex * ex
            count += 1

        if count > 0:
            mean_ex = excess_sum / count
            var_ex = excess_sq_sum / count - mean_ex * mean_ex
            if var_ex > 1e-15:
                sharpe = np.sqrt(252.0) * mean_ex / np.sqrt(var_ex)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        calmar = annual_return / (-max_dd_val) if abs(max_dd_val) > 1e-15 else 0.0

        # 最大回撤持续天数
        max_dd_days = 0
        in_dd = False
        cur_dd_days = 0
        peak2 = eq_curve[0]
        for i in range(n):
            if eq_curve[i] >= peak2:
                peak2 = eq_curve[i]
                in_dd = False
                cur_dd_days = 0
            else:
                if not in_dd:
                    in_dd = True
                    cur_dd_days = 1
                else:
                    cur_dd_days += 1
                if cur_dd_days > max_dd_days:
                    max_dd_days = cur_dd_days

        # 日波动率（年化）
        ret_sum = 0.0
        ret_sq_sum = 0.0
        ret_count = 0
        for i in range(1, n):
            if daily_ret[i] != 0.0 or i == 1:
                ret_sum += daily_ret[i]
                ret_sq_sum += daily_ret[i] * daily_ret[i]
                ret_count += 1
        if ret_count > 1:
            ret_mean = ret_sum / ret_count
            ret_var = ret_sq_sum / ret_count - ret_mean * ret_mean
            daily_vol = np.sqrt(ret_var * 252.0) if ret_var > 0.0 else 0.0
        else:
            daily_vol = 0.0

        # 胜率、盈亏比、平均持仓天数
        if n_trades > 0:
            win_rate = n_wins / n_trades
            avg_win = sum_win_ret / n_wins if n_wins > 0 else 0.0
            avg_loss = sum_loss_ret / n_losses if n_losses > 0 else 0.0
            profit_factor = avg_win / avg_loss if avg_loss > 1e-15 else 10.0
            avg_hold_days = total_hold_days / n_trades
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_hold_days = 0.0

        return (total_return, annual_return, max_dd_val, max_dd_days,
                daily_vol, sharpe, calmar, n_trades,
                win_rate, profit_factor, avg_hold_days)

except ImportError:
    HAS_NUMBA = False
    print("[!] Numba not installed, falling back to pure Python (very slow)")
    print("    Install: uv add numba  or  pip install numba")


# ============================================================
# 无 Numba 时的纯 Python 回退（逻辑相同）
# ============================================================

def _super_trend_backtest_py(high, low, close, atr, multiplier, period, capital):
    """
    纯 Python 版本的核心函数（Numba 不可用时的回退方案）
    使用独立的 metrics 模块计算绩效指标，与 JIT 版本逻辑一致但更易验证。
    """
    n = len(close)
    mid = (high + low) * 0.5
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr

    st = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    start_val = max(period, 1)

    if start_val >= n:
        return (0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)

    if close[start_val] > upper[start_val]:
        trend[start_val] = 1; st[start_val] = lower[start_val]
    elif close[start_val] < lower[start_val]:
        trend[start_val] = -1; st[start_val] = upper[start_val]
    else:
        trend[start_val] = 1; st[start_val] = lower[start_val]

    for i in range(start_val + 1, n):
        if trend[i - 1] == 1:
            s = max(lower[i], st[i - 1])
            if close[i] < s:
                trend[i] = -1; st[i] = upper[i]
            else:
                trend[i] = 1; st[i] = s
        else:
            s = min(upper[i], st[i - 1])
            if close[i] > s:
                trend[i] = 1; st[i] = lower[i]
            else:
                trend[i] = -1; st[i] = s

    signal = np.zeros(n, dtype=int)
    for i in range(1, n):
        if trend[i - 1] == -1 and trend[i] == 1: signal[i] = 1
        elif trend[i - 1] == 1 and trend[i] == -1: signal[i] = -1
    for i in range(n):
        if trend[i] != 0:
            signal[i] = 1 if trend[i] == 1 else -1
            break

    # ---- 回测模拟 ----
    equity = float(capital)
    position = 0.0; entry_price = 0.0; direction = 0; pending = 0
    eq_curve = np.zeros(n); daily_ret = np.zeros(n)
    entry_day = 0

    # 收集交易数据用于指标计算
    trade_pnl_pcts = []
    trade_hold_days = []

    for i in range(n):
        if pending != 0:
            px = high[i] if pending == 1 else low[i]
            if direction == -1 and pending == 1:
                pnl = (entry_price - px) * abs(position)
                equity += pnl
                pnl_pct = pnl / (entry_price * abs(position))
                trade_pnl_pcts.append(pnl_pct)
                trade_hold_days.append(i - entry_day)
            elif direction == 1 and pending == -1:
                pnl = (px - entry_price) * position
                equity += pnl
                pnl_pct = pnl / (entry_price * position)
                trade_pnl_pcts.append(pnl_pct)
                trade_hold_days.append(i - entry_day)
            if pending == 1:
                position = equity / px; direction = 1
            else:
                position = -equity / px; direction = -1
            entry_price = px; entry_day = i; pending = 0

        if signal[i] != 0: pending = signal[i]

        if direction == 1:
            eq_curve[i] = position * close[i]
        elif direction == -1:
            eq_curve[i] = equity + (entry_price - close[i]) * abs(position)
        else:
            eq_curve[i] = equity

        if i > 0 and eq_curve[i - 1] > 0:
            daily_ret[i] = (eq_curve[i] - eq_curve[i - 1]) / eq_curve[i - 1]

    # 强制平仓
    if direction != 0:
        px_last = close[-1]
        if direction == 1:
            pnl_close = (px_last - entry_price) * position
            equity += pnl_close
            pnl_pct = pnl_close / (entry_price * position)
        else:
            pnl_close = (entry_price - px_last) * abs(position)
            equity += pnl_close
            pnl_pct = pnl_close / (entry_price * abs(position))
        trade_pnl_pcts.append(pnl_pct)
        trade_hold_days.append(n - 1 - entry_day)
        eq_curve[-1] = equity

    # ---- 委托给独立的 metrics 模块计算绩效指标 ----
    raw = mt.calc_all_metrics(
        eq_curve,
        daily_ret,
        np.array(trade_pnl_pcts),
        capital,
        hold_days=np.array(trade_hold_days, dtype=float),
    )

    return (raw["total_return"], raw["annual_return"], raw["max_dd"], raw["max_dd_days"],
            raw["daily_vol"], raw["sharpe"], raw["calmar"], raw["n_trades"],
            raw["win_rate"], raw["profit_factor"], raw["avg_hold_days"])


# 根据 Numba 是否可用来选择实现
_compute_one = _super_trend_backtest if HAS_NUMBA else _super_trend_backtest_py


# ============================================================
# ATR 预计算
# ============================================================

def precompute_atr(high, low, close, periods):
    """
    预计算所有周期的 ATR（Wilder 原始公式）

    Wilder 原始 ATR:
      第一条 (index = period-1): TR[0..period-1] 的简单平均
      后续 (t >= period):       ATR[t] = (ATR[t-1] * (period-1) + TR[t]) / period

    这与 ewm(alpha=1/period, adjust=False) 在递推部分等价，
    但初始值使用简单平均而非从第 0 条开始 EMA。
    """
    n = len(close)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))

    atr_cache = {}
    for p in periods:
        atr = np.full(n, np.nan, dtype=np.float64)
        if n > p - 1:
            atr[p - 1] = tr[:p].mean()
            for t in range(p, n):
                atr[t] = (atr[t - 1] * (p - 1) + tr[t]) / p
        atr_cache[p] = atr
    return atr_cache


# ============================================================
# 核心搜索函数（单个 period 的处理，供并行调用）
# ============================================================

def _search_one_period(args):
    """
    对给定的一个 period，遍历所有 multiplier 并返回结果列表。
    设计为可被 multiprocessing 直接调用的顶层函数。
    """
    period, high, low, close, atr, multipliers, capital = args
    results = []
    for mult in multipliers:
        (total_ret, annual_ret, max_dd, max_dd_days,
         daily_vol, sharpe, calmar, n_trades,
         win_rate, profit_factor, avg_hold_days) = _compute_one(
            high, low, close, atr, mult, period, capital
        )
        results.append({
            "period": period,
            "multiplier": mult,
            "total_return": total_ret,
            "annual_return": annual_ret,
            "max_dd": max_dd,
            "max_dd_days": max_dd_days,
            "daily_vol": daily_vol,
            "sharpe": sharpe,
            "calmar": calmar,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_hold_days": avg_hold_days,
        })
    return results


# ============================================================
# 网格搜索
# ============================================================

def grid_search_fast(name, filepath):
    """快速网格搜索（Numba JIT + 可选多进程）"""
    total_combos = len(PERIODS) * len(MULTIPLIERS)
    print(f"\n{'='*60}")
    print(f"  {name} — 参数搜索")
    print(f"  周期 {PERIODS[0]}~{PERIODS[-1]} x 乘数 {MULTIPLIERS[0]}~{MULTIPLIERS[-1]}")
    print(f"  共 {len(PERIODS)}x{len(MULTIPLIERS)} = {total_combos:,} 组合")
    print(f"  {'Numba JIT 加速' if HAS_NUMBA else '纯 Python（慢）'}", end="")
    if USE_PARALLEL and HAS_NUMBA:
        print(f" + {N_WORKERS} 进程并行")
    else:
        print()
    print(f"{'='*60}")

    # 加载数据
    df_raw = load_data(filepath)
    high = df_raw["high"].values.astype(np.float64)
    low = df_raw["low"].values.astype(np.float64)
    close = df_raw["close"].values.astype(np.float64)
    n = len(close)
    print(f"  数据: {n} 条, {df_raw['trade_date'].iloc[0].date()} ~ {df_raw['trade_date'].iloc[-1].date()}")

    # 预计算 ATR
    print(f"  预计算 {len(PERIODS)} 个周期的 ATR...", end=" ", flush=True)
    atr_cache = precompute_atr(high, low, close, PERIODS)
    print("完成")

    multipliers_arr = np.array(MULTIPLIERS, dtype=np.float64)

    # 确保 Numba 函数在并行前被编译（触发 JIT 编译，避免子进程中重复编译）
    if HAS_NUMBA:
        print("  JIT 预热编译...", end=" ", flush=True)
        _compute_one(high, low, close, atr_cache[PERIODS[0]], MULTIPLIERS[0], PERIODS[0], CAPITAL)
        print("完成")

    results = []

    if USE_PARALLEL and HAS_NUMBA and len(PERIODS) > 4:
        # --- 多进程并行 ---
        print(f"  使用 {N_WORKERS} 个进程并行搜索...")
        tasks = []
        for period in PERIODS:
            tasks.append((period, high, low, close, atr_cache[period], multipliers_arr, CAPITAL))

        with Pool(processes=N_WORKERS) as pool:
            all_chunks = list(tqdm(
                pool.imap_unordered(_search_one_period, tasks),
                total=len(tasks),
                desc=f"  {name}",
                unit="period",
                ncols=80,
            ))
        for chunk in all_chunks:
            results.extend(chunk)

        # 按 period, multiplier 排序
        results.sort(key=lambda x: (x["period"], x["multiplier"]))
    else:
        # --- 单进程 ---
        pbar = tqdm(total=len(PERIODS), desc=f"  {name}", unit="period", ncols=80)
        for period in PERIODS:
            atr = atr_cache[period]
            res = _search_one_period((period, high, low, close, atr, multipliers_arr, CAPITAL))
            results.extend(res)
            pbar.update(1)
        pbar.close()

    print(f"  搜索完成! 共测试 {len(results):,} 组参数")
    return df_raw, pd.DataFrame(results)


# ============================================================
# 可视化（保持不变）
# ============================================================

def plot_heatmaps(all_results, output_dir="heatmaps"):
    """
    为每个资产 × 每个指标生成独立的热力图文件，保存到 output_dir 目录下。

    文件命名: {资产名} — {指标名}.png
    """
    names = list(all_results.keys())
    n_assets = len(names)

    # 定义所有要展示的指标: (列名, 标题, colormap, 是否百分比)
    metrics_list = [
        ("total_return",   "总收益率",   "RdYlGn", True),
        ("annual_return",  "年化收益率", "RdYlGn", True),
        ("max_dd",         "最大回撤",   "RdYlGn_r", True),
        ("max_dd_days",    "回撤持续天数","YlOrRd",  False),
        ("sharpe",         "夏普比率",   "RdYlGn",  False),
        ("calmar",         "Calmar比率", "RdYlGn",  False),
        ("daily_vol",      "年化波动率", "YlOrRd",  True),
        ("n_trades",       "总交易次数", "Blues",   False),
        ("win_rate",       "胜率",       "RdYlGn",  True),
        ("profit_factor",  "盈亏比",     "RdYlGn",  False),
        ("avg_hold_days",  "平均持仓天数","Blues",   False),
    ]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 预生成刻度标签（所有图共用）
    xtick_step = max(1, len(MULTIPLIERS) // 8)
    ytick_step = max(1, len(PERIODS) // 8)
    x_tick_indices = list(range(0, len(MULTIPLIERS), xtick_step))
    x_tick_labels = [f"{MULTIPLIERS[i]:.1f}" for i in x_tick_indices]
    y_tick_indices = list(range(0, len(PERIODS), ytick_step))
    y_tick_labels = [PERIODS[i] for i in y_tick_indices]

    total_files = 0

    for col_idx, name in enumerate(names):
        df_r = all_results[name]["results"]
        best_sharpe_idx = df_r["sharpe"].idxmax()
        best_sharpe = df_r.iloc[best_sharpe_idx]

        for col, title, cmap, is_pct in metrics_list:
            # 每个小图独立一个 figure
            fig, ax = plt.subplots(figsize=(8, 6))

            pivot = df_r.pivot_table(index="period", columns="multiplier", values=col)
            vals = pivot.values

            # 设定色阶范围
            if is_pct:
                vmax = max(0.3, vals.max()) if not np.isnan(vals.max()) else 0.3
                vmin = min(-0.3, vals.min()) if not np.isnan(vals.min()) else -0.3
            else:
                vmax = vals.max() if not np.isnan(vals.max()) else 1.0
                vmin = vals.min() if not np.isnan(vals.min()) else 0.0

            im = ax.imshow(vals, aspect="auto", origin="lower",
                          cmap=cmap, vmin=vmin, vmax=vmax)

            ax.set_xlabel("乘数 M", fontsize=10)
            ax.set_ylabel("周期 N", fontsize=10)
            ax.set_title(f"{name} — {title}", fontsize=13, fontweight="bold")
            ax.set_xticks(x_tick_indices)
            ax.set_xticklabels(x_tick_labels, fontsize=8)
            ax.set_yticks(y_tick_indices)
            ax.set_yticklabels(y_tick_labels, fontsize=8)

            # 标注最佳夏普参数
            best_val = best_sharpe[col]
            if isinstance(best_val, (int, np.integer)):
                val_str = str(int(best_val))
            elif isinstance(best_val, float):
                val_str = f"{best_val:.1%}" if is_pct else f"{best_val:.2f}"
            else:
                val_str = str(best_val)

            ax.text(0.98, 0.02,
                    f"最佳夏普 N={int(best_sharpe['period'])}, M={best_sharpe['multiplier']:.1f}\n{title}={val_str}",
                    transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            if is_pct:
                plt.colorbar(im, ax=ax, format=mticker.PercentFormatter(1.0))
            else:
                plt.colorbar(im, ax=ax)

            # 保存为独立文件
            filename = f"{name} — {title}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            total_files += 1

    print(f"热力图已保存到 '{output_dir}/' 目录，共 {total_files} 个文件")

    # ---- 同时生成大拼图 ----
    fig, axes = plt.subplots(len(metrics_list), n_assets, figsize=(6 * n_assets, 4 * len(metrics_list)))
    if n_assets == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, name in enumerate(names):
        df_r = all_results[name]["results"]
        best_sharpe_idx = df_r["sharpe"].idxmax()
        best_sharpe = df_r.iloc[best_sharpe_idx]

        for row_idx, (col, title, cmap, is_pct) in enumerate(metrics_list):
            ax = axes[row_idx, col_idx]
            pivot = df_r.pivot_table(index="period", columns="multiplier", values=col)
            vals = pivot.values

            if is_pct:
                vmax = max(0.3, vals.max()) if not np.isnan(vals.max()) else 0.3
                vmin = min(-0.3, vals.min()) if not np.isnan(vals.min()) else -0.3
            else:
                vmax = vals.max() if not np.isnan(vals.max()) else 1.0
                vmin = vals.min() if not np.isnan(vals.min()) else 0.0

            im = ax.imshow(vals, aspect="auto", origin="lower",
                          cmap=cmap, vmin=vmin, vmax=vmax)

            ax.set_xlabel("乘数 M", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("周期 N", fontsize=8)
            ax.set_title(f"{name} — {title}", fontsize=10, fontweight="bold")
            ax.set_xticks(x_tick_indices)
            ax.set_xticklabels(x_tick_labels, fontsize=7)
            ax.set_yticks(y_tick_indices)
            ax.set_yticklabels(y_tick_labels, fontsize=7)

            best_val = best_sharpe[col]
            if isinstance(best_val, (int, np.integer)):
                val_str = str(int(best_val))
            elif isinstance(best_val, float):
                val_str = f"{best_val:.1%}" if is_pct else f"{best_val:.2f}"
            else:
                val_str = str(best_val)

            ax.text(0.98, 0.02,
                    f"最佳夏普 N={int(best_sharpe['period'])}, M={best_sharpe['multiplier']:.1f}\n{title}={val_str}",
                    transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            if is_pct:
                plt.colorbar(im, ax=ax, format=mticker.PercentFormatter(1.0))
            else:
                plt.colorbar(im, ax=ax)

    fig.suptitle(f"SuperTrend 参数搜索 (N={PERIODS[0]}~{PERIODS[-1]}, M={MULTIPLIERS[0]}~{MULTIPLIERS[-1]})",
                 fontsize=16, fontweight="bold", y=1.001)
    plt.tight_layout()
    fig.savefig("optimize_heatmap.png", dpi=300, bbox_inches="tight")
    print("大拼图已保存: optimize_heatmap.png")
    plt.close(fig)


# 最优参数净值曲线

def plot_best_equity(all_results, all_best):
    """最优参数净值曲线（含标的自身 buy & hold 对比），含完整绩效指标"""
    from backtest import calc_super_trend, run_backtest

    fig, axes = plt.subplots(3, 1, figsize=(18, 15))
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

        # 标的自身 buy & hold 净值（按收盘价归一化）
        close_vals = raw["close"].values
        benchmark = close_vals / close_vals[0]

        ax = axes[idx]
        ax.plot(dates, benchmark, color="gray", linewidth=0.6, linestyle="--", alpha=0.7, label="标的 Buy&Hold")
        ax.plot(dates, vals, color=color, linewidth=0.9, label="策略净值")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.5)
        ax.set_ylabel("净值", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.fill_between(dates, 0, dd, color="red", alpha=0.12)
        ax2.plot(dates, dd, color="red", linewidth=0.3, alpha=0.4)
        ax2.set_ylabel("回撤 %", color="red", fontsize=9)
        ax2.tick_params(axis="y", colors="red")

        # 标的 buy & hold 收益
        bench_ret = benchmark[-1] - 1
        total_ret = (vals[-1] - 1)
        metrics_text = (
            f"策略: 总收益={total_ret:.1%}  年化={best['annual_return']:.1%}  夏普={best['sharpe']:.3f}  "
            f"最大回撤={best['max_dd']:.1%}  回撤持续={int(best.get('max_dd_days', 0))}天\n"
            f"标的: 总收益={bench_ret:.1%}  "
            f"波动率={best.get('daily_vol', 0):.1%}  交易={int(best.get('n_trades', 0))}笔  "
            f"胜率={best.get('win_rate', 0):.1%}  盈亏比={best.get('profit_factor', 0):.2f}  "
            f"平均持仓={best.get('avg_hold_days', 0):.1f}天"
        )
        ax.set_title(f"{name} (N={period}, M={mult:.1f})  {metrics_text}",
                     fontsize=11, fontweight="bold")

    fig.suptitle("最优参数净值曲线（按夏普比率）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("optimize_equity.png", dpi=300, bbox_inches="tight")
    print("净值曲线已保存: optimize_equity.png")
    plt.close(fig)


def plot_strategy_vs_benchmark(all_results, all_best):
    """策略 vs 标的对比图：左轴净值曲线，右轴回撤，独立一张图"""
    from backtest import calc_super_trend, run_backtest

    names = list(all_results.keys())
    n = len(names)
    colors = ["#e74c3c", "#2980b9", "#f39c12"]

    fig, axes = plt.subplots(n, 1, figsize=(18, 5 * n))

    for idx, (name, color) in enumerate(zip(names, colors)):
        best = all_best[name]
        raw = all_results[name]["raw"]
        period, mult = int(best["period"]), float(best["multiplier"])

        df = calc_super_trend(raw, period=period, multiplier=mult)
        trades, eq_curve, _ = run_backtest(df, capital=CAPITAL)

        dates = [e["trade_date"] for e in eq_curve]
        vals = np.array([e["equity"] for e in eq_curve]) / CAPITAL
        peak = np.maximum.accumulate(vals)
        strategy_dd = (vals - peak) / peak * 100

        # 标的 buy & hold
        close_vals = raw["close"].values
        benchmark = close_vals / close_vals[0]
        bench_peak = np.maximum.accumulate(benchmark)
        bench_dd = (benchmark - bench_peak) / bench_peak * 100

        ax = axes[idx]

        # 左轴：净值
        ax.plot(dates, benchmark, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="标的 Buy&Hold")
        ax.plot(dates, vals, color=color, linewidth=1.0, label=f"策略 (N={period}, M={mult:.1f})")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.5)
        ax.set_ylabel("净值", fontsize=10)
        ax.grid(True, alpha=0.3)

        # 标注最大回撤区域
        ax2 = ax.twinx()
        ax2.fill_between(dates, 0, strategy_dd, color="red", alpha=0.08, label="策略回撤")
        ax2.fill_between(dates, 0, bench_dd, color="gray", alpha=0.05, label="标的回撤")
        ax2.plot(dates, strategy_dd, color="red", linewidth=0.3, alpha=0.5)
        ax2.plot(dates, bench_dd, color="gray", linewidth=0.3, alpha=0.5, linestyle="--")
        ax2.set_ylabel("回撤 %", color="red", fontsize=9)
        ax2.tick_params(axis="y", colors="red")
        ax2.set_ylim(min(strategy_dd.min(), bench_dd.min()) * 1.1, 5)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

        # 绩效对比文本
        bench_ret = benchmark[-1] - 1
        strategy_ret = vals[-1] - 1
        bench_annual = (1 + bench_ret) ** (252 / len(benchmark)) - 1 if len(benchmark) > 252 else 0
        bench_dd_max = bench_dd.min()

        text = (
            f"策略: 收益={strategy_ret:.1%}  年化={best['annual_return']:.1%}  夏普={best['sharpe']:.3f}  "
            f"最大回撤={best['max_dd']:.1%}  交易={int(best.get('n_trades', 0))}笔\n"
            f"标的: 收益={bench_ret:.1%}  年化={bench_annual:.1%}  "
            f"最大回撤={bench_dd_max:.1%}  策略相对超额={strategy_ret - bench_ret:.1%}"
        )
        ax.set_title(f"{name} — 策略 vs 标的对比", fontsize=12, fontweight="bold")
        ax.text(0.01, -0.15, text, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("SuperTrend 策略 vs 标的 Buy & Hold 对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig("strategy_vs_benchmark.png", dpi=300, bbox_inches="tight")
    print("策略对比图已保存: strategy_vs_benchmark.png")
    plt.close(fig)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    if HAS_NUMBA:
        print(f"[OK] Numba {nb.__version__} loaded, JIT acceleration enabled")
    import time

    all_results = {}
    all_best = {}

    for name, path in DATA.items():
        t0 = time.perf_counter()
        raw, df_r = grid_search_fast(name, path)
        elapsed = time.perf_counter() - t0
        print(f"  耗时: {elapsed:.1f} 秒")

        best = df_r.iloc[df_r["sharpe"].idxmax()].to_dict()
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
              f"年化={b['annual_return']:.1%}  回撤={b['max_dd']:.1%}  "
              f"回撤持续={int(b.get('max_dd_days', 0))}天")
        print(f"         "
              f"波动率={b.get('daily_vol', 0):.1%}  交易={int(b.get('n_trades', 0))}笔  "
              f"胜率={b.get('win_rate', 0):.1%}  盈亏比={b.get('profit_factor', 0):.2f}  "
              f"平均持仓={b.get('avg_hold_days', 0):.1f}天")

    # ---- Top 10 打印 ----
    for name in DATA:
        df_r = all_results[name]["results"]
        top10 = df_r.nlargest(10, "sharpe")
        print(f"\n  {name} Top 10 (按夏普):")
        for _, row in top10.iterrows():
            print(f"    N={int(row['period']):3d}  M={row['multiplier']:4.1f}  "
                  f"收益={row['total_return']:7.2%}  年化={row['annual_return']:7.2%}  "
                  f"回撤={row['max_dd']:7.2%}  DD天={int(row.get('max_dd_days', 0)):3d}  "
                  f"夏普={row['sharpe']:7.3f}  胜率={row.get('win_rate', 0):5.1%}  "
                  f"盈亏比={row.get('profit_factor', 0):5.2f}")

    # ---- 绘图 ----
    plot_heatmaps(all_results)
    plot_best_equity(all_results, all_best)
    plot_strategy_vs_benchmark(all_results, all_best)
