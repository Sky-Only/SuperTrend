"""
SuperTrend入场 + 吊灯止损离场 — 5 参数网格扫描 (Numba JIT 加速版)

扫描参数:
    ST_PERIOD / ST_MULT   (SuperTrend 入场翻多)
    N         / K         (吊灯止损离场)
    ATR_PERIOD            (吊灯止损 ATR 周期)

固定常量 (可在下面配置区改):
    USE_MA        (是否 MA 确认入场, 默认 False)
    USE_CHANDELIER(是否吊灯止损离场, 默认 True)

入场: SuperTrend 翻多 (+ 可选 MA 确认)
离场: 收盘跌破吊灯止损线 (持仓期最高价自入场起算上限 N 根 - K × ATR)
只做多, 空仓现金年化 1.5%, 单边手续费万三

性能: 核心链路全部在 Numba JIT 内完成, 无 pandas/DataFrame, 支持多进程并行。
输出: CSV + Top 20 排名 + 三张热力图 (ST / 吊灯 / ATR 各一对)
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm

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

from backtest import load_data

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("[!] 未安装 numba, 将用纯 Python (很慢)。安装: pip install numba")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ============================================================
# 配置
# ============================================================
DATA_NAME = "沪深300"
DATA_FILE = f"{DATA_NAME}_10年日线.parquet"
CAPITAL = 1_000_000

FLAT_ANNUAL = 0.015      # 空仓现金年化
FEE_RATE    = 0.0003     # 单边手续费 (万三)

# ==================== 在这里修改搜索范围 ====================
# 每个参数: (起点, 终点, 步长)，闭区间 [起点, 终点]

# ST_PERIOD_RANGE = (7, 60, 1)        # SuperTrend 周期
# ST_MULT_RANGE   = (2.0, 7.0, 0.1)   # SuperTrend 乘数
# N_RANGE         = (10, 30, 2)      # 吊灯最高价回看上限天数
# K_RANGE         = (0.5, 6.5, 0.1)   # 吊灯 ATR 倍数
# ATR_PERIOD_RANGE = (5, 20, 1)     # 吊灯 ATR 周期
# HV_THRESHOLD_RANGE = (0.10, 0.40, 0.05)  # 波动率止盈阈值 (年化)

ST_PERIOD_RANGE = (5, 54, 2)        # SuperTrend 周期
ST_MULT_RANGE   = (0.9, 4.9, 0.2)   # SuperTrend 乘数
N_RANGE         = (12, 42, 2)      # 吊灯最高价回看上限天数
K_RANGE         = (0.5, 5.5, 0.2)   # 吊灯 ATR 倍数
ATR_PERIOD_RANGE = (4, 56, 1)     # 吊灯 ATR 周期
HV_THRESHOLD_RANGE = (0.01, 0.99, 0.05)  # 波动率止盈阈值 (年化)
# ========================================================


def _arange(start, end, step):
    """按 [start, end] 闭区间 + 步长生成参数列表 (处理浮点精度)"""
    vals = []
    x = start
    while x <= end + 1e-9:
        vals.append(x)
        x = round(x + step, 10)
    return vals


# 由上面的范围自动生成网格 (一般不用改下面)
ST_PERIODS = [int(v) for v in _arange(*ST_PERIOD_RANGE)]
ST_MULTS   = [round(v, 2) for v in _arange(*ST_MULT_RANGE)]
NS         = [int(v) for v in _arange(*N_RANGE)]
KS         = [round(v, 2) for v in _arange(*K_RANGE)]
ATR_PERIODS = [int(v) for v in _arange(*ATR_PERIOD_RANGE)]
HV_THRESHOLDS = [round(v, 2) for v in _arange(*HV_THRESHOLD_RANGE)]

# ---- 固定常量 ----
USE_MA = False         # True=MA确认入场; False=纯SuperTrend翻多
USE_CHANDELIER = True  # True=吊灯止损离场
USE_VOLATILITY = True  # True=波动率止盈离场 (HV超阈值)
HV_WINDOW = 20         # 波动率 HV 计算窗口 (交易日)
MA_FAST = 5            # MA 快线 (USE_MA=True 时生效)
MA_SLOW = 20           # MA 慢线 (USE_MA=True 时生效)

# 多进程并行 (充分利用所有逻辑核心)
USE_PARALLEL = True
N_WORKERS = 30   # 使用全部逻辑核心 (本机 32 核)
# N_WORKERS = os.cpu_count()   # 使用全部逻辑核心 (本机 32 核)

# 挑选"最优"时要求的最低交易笔数 (过滤"买一次拿十年"的退化结果)
MIN_TRADES = 8

# 热力图指标: (列名, 标题, colormap, 是否百分比)
METRICS = [
    ("total_return",  "总收益率",    "RdYlGn",   True),
    ("annual_return", "年化收益率",  "RdYlGn",   True),
    ("max_dd",        "最大回撤",    "RdYlGn_r", True),
    ("sharpe",        "夏普比率",    "RdYlGn",   False),
    ("calmar",        "Calmar比率",  "RdYlGn",   False),
    ("n_trades",      "交易次数",    "Blues",    False),
]


# ============================================================
# Numba JIT 核心: SuperTrend入场 + 吊灯止损离场 + 只做多回测 + 指标
# ============================================================

if HAS_NUMBA:
    @nb.jit(nopython=True, cache=True)
    def _backtest_one(high, low, close, st_period, st_mult, ma_fast, ma_slow, use_ma,
                      n, k, atr_period, use_chandelier, use_volatility, hv_window, hv_threshold,
                      capital, flat_annual, commission):
        """
        一次完整回测, 全部在 Numba 内完成, 返回 11 个指标 (全 float):
        total_return, annual_return, max_dd, max_dd_days, daily_vol,
        sharpe, calmar, n_trades, win_rate, profit_factor, avg_hold_days
        """
        m = len(close)
        rf_rate = 0.03
        trading_days = 252

        # ---- True Range ----
        tr = np.empty(m, dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, m):
            a = high[i] - low[i]
            b = abs(high[i] - close[i - 1])
            c = abs(low[i] - close[i - 1])
            v = a
            if b > v:
                v = b
            if c > v:
                v = c
            tr[i] = v

        # ---- SuperTrend ATR (Wilder) ----
        atr_st = np.empty(m, dtype=np.float64)
        atr_st[0:st_period - 1] = np.nan
        s = 0.0
        for i in range(st_period):
            s += tr[i]
        atr_st[st_period - 1] = s / st_period
        for i in range(st_period, m):
            atr_st[i] = (atr_st[i - 1] * (st_period - 1) + tr[i]) / st_period

        # ---- SuperTrend 趋势 ----
        st_trend = np.zeros(m, dtype=np.int64)
        st_line = np.empty(m, dtype=np.float64)
        st_line[:] = np.nan
        first = st_period
        if first < m:
            mid0 = (high[first] + low[first]) * 0.5
            up0 = mid0 + st_mult * atr_st[first]
            lo0 = mid0 - st_mult * atr_st[first]
            if close[first] > up0:
                st_trend[first] = 1
                st_line[first] = lo0
            elif close[first] < lo0:
                st_trend[first] = -1
                st_line[first] = up0
            else:
                st_trend[first] = 1
                st_line[first] = lo0
            for i in range(first + 1, m):
                mid = (high[i] + low[i]) * 0.5
                up = mid + st_mult * atr_st[i]
                lo = mid - st_mult * atr_st[i]
                if st_trend[i - 1] == 1:
                    s = lo
                    if st_line[i - 1] > s:
                        s = st_line[i - 1]
                    if close[i] < s:
                        st_trend[i] = -1
                        st_line[i] = up
                    else:
                        st_trend[i] = 1
                        st_line[i] = s
                else:
                    s = up
                    if st_line[i - 1] < s:
                        s = st_line[i - 1]
                    if close[i] > s:
                        st_trend[i] = 1
                        st_line[i] = lo
                    else:
                        st_trend[i] = -1
                        st_line[i] = s

        # ---- MA (部分窗口均线, 等价 pandas min_periods=1) ----
        ma_f = np.zeros(m, dtype=np.float64)
        ma_s = np.zeros(m, dtype=np.float64)
        if use_ma:
            sf = 0.0
            ss = 0.0
            for i in range(m):
                sf += close[i]
                ss += close[i]
                if i >= ma_fast:
                    sf -= close[i - ma_fast]
                if i >= ma_slow:
                    ss -= close[i - ma_slow]
                ma_f[i] = sf / (i + 1 if i < ma_fast else ma_fast)
                ma_s[i] = ss / (i + 1 if i < ma_slow else ma_slow)

        # ---- 吊灯止损 ATR (Wilder, atr_period) ----
        atr_ch = np.empty(m, dtype=np.float64)
        atr_ch[0:atr_period - 1] = np.nan
        s = 0.0
        for i in range(atr_period):
            s += tr[i]
        atr_ch[atr_period - 1] = s / atr_period
        for i in range(atr_period, m):
            atr_ch[i] = (atr_ch[i - 1] * (atr_period - 1) + tr[i]) / atr_period

        # ---- 波动率 HV (rolling std of 日收益率, 年化) ----
        hv = np.full(m, np.nan)
        if use_volatility:
            ret_arr = np.empty(m, dtype=np.float64)
            ret_arr[0] = 0.0
            for j in range(1, m):
                ret_arr[j] = close[j] / close[j - 1] - 1.0
            if hv_window >= 1 and hv_window < m:
                s_ret = 0.0
                s2_ret = 0.0
                for j in range(1, hv_window + 1):
                    s_ret += ret_arr[j]
                    s2_ret += ret_arr[j] * ret_arr[j]
                mean = s_ret / hv_window
                var = s2_ret / hv_window - mean * mean
                hv[hv_window] = np.sqrt(var) * np.sqrt(252.0) if var > 0.0 else 0.0
                for i in range(hv_window + 1, m):
                    s_ret += ret_arr[i] - ret_arr[i - hv_window]
                    s2_ret += ret_arr[i] * ret_arr[i] - ret_arr[i - hv_window] * ret_arr[i - hv_window]
                    mean = s_ret / hv_window
                    var = s2_ret / hv_window - mean * mean
                    hv[i] = np.sqrt(var) * np.sqrt(252.0) if var > 0.0 else 0.0

        # ---- 状态机: 1=多头, -1=空仓 ----
        trend = np.full(m, -1, dtype=np.int64)
        state = -1
        entry_idx = -1
        for i in range(m):
            prev_st = st_trend[i - 1] if i > 0 else 0
            if state == -1:
                st_flip_long = (st_trend[i] == 1) and (prev_st != 1)
                if st_flip_long:
                    if (not use_ma) or (ma_f[i] > ma_s[i]):
                        state = 1
                        entry_idx = i
            else:
                exit_now = False
                if use_chandelier:
                    start = entry_idx
                    if start < i - n + 1:
                        start = i - n + 1
                    hh = high[start]
                    for j in range(start + 1, i + 1):
                        if high[j] > hh:
                            hh = high[j]
                    stop = hh - k * atr_ch[i]
                    if close[i] < stop:
                        exit_now = True
                if use_volatility:
                    if (not np.isnan(hv[i])) and (hv[i] > hv_threshold):
                        exit_now = True
                if (not use_chandelier) and (not use_volatility):
                    st_flip_short = (st_trend[i] == -1) and (prev_st == 1)
                    if st_flip_short:
                        exit_now = True
                if exit_now:
                    state = -1
            trend[i] = state

        # ---- 只做多回测 ----
        daily_flat = (1.0 + flat_annual) ** (1.0 / trading_days) - 1.0
        equity = float(capital)
        position = 0.0
        entry_cash = 0.0
        entry_day = -1
        pending = 0

        eq_curve = np.empty(m, dtype=np.float64)
        daily_ret = np.zeros(m, dtype=np.float64)

        n_trades = 0
        n_wins = 0
        sum_win = 0.0
        sum_loss = 0.0
        total_hold = 0

        for i in range(m):
            # Step 1: 执行前一日信号
            if pending == 1:
                exec_price = high[i]
                invest = equity
                fee = invest * commission
                position = (invest - fee) / exec_price
                entry_cash = invest
                entry_day = i
                pending = 0
            elif pending == -1:
                if position > 0:
                    exec_price = low[i]
                    proceeds = position * exec_price
                    fee = proceeds * commission
                    equity = proceeds - fee
                    pnl = equity - entry_cash
                    pnl_pct = pnl / entry_cash if entry_cash > 0 else 0.0
                    n_trades += 1
                    if pnl_pct > 0.0:
                        n_wins += 1
                        sum_win += pnl_pct
                    else:
                        sum_loss += -pnl_pct
                    total_hold += (i - entry_day)
                    position = 0.0
                pending = 0

            # Step 2: 记录今日信号 (趋势翻转)
            if i > 0:
                if trend[i - 1] == -1 and trend[i] == 1:
                    pending = 1
                elif trend[i - 1] == 1 and trend[i] == -1:
                    pending = -1

            # Step 3: 按收盘估值
            if position > 0:
                equity = position * close[i]
            else:
                equity = equity * (1.0 + daily_flat)
            eq_curve[i] = equity
            if i > 0 and eq_curve[i - 1] > 0.0:
                daily_ret[i] = (eq_curve[i] - eq_curve[i - 1]) / eq_curve[i - 1]

        # 强制平仓
        if position > 0:
            exec_price = close[m - 1]
            proceeds = position * exec_price
            fee = proceeds * commission
            equity = proceeds - fee
            pnl = equity - entry_cash
            pnl_pct = pnl / entry_cash if entry_cash > 0 else 0.0
            n_trades += 1
            if pnl_pct > 0.0:
                n_wins += 1
                sum_win += pnl_pct
            else:
                sum_loss += -pnl_pct
            total_hold += (m - 1 - entry_day)
            eq_curve[m - 1] = equity

        # ---- 指标 ----
        n_years = m / trading_days
        total_return = (eq_curve[m - 1] - capital) / capital
        if n_years > 0.0 and total_return > -1.0:
            annual_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0
        else:
            annual_return = 0.0

        peak = eq_curve[0]
        max_dd = 0.0
        max_dd_days = 0
        cur_dd_days = 0
        for i in range(m):
            if eq_curve[i] > peak:
                peak = eq_curve[i]
            d = (eq_curve[i] - peak) / peak if peak > 0.0 else 0.0
            if d < max_dd:
                max_dd = d
            if eq_curve[i] >= peak:
                cur_dd_days = 0
            else:
                cur_dd_days += 1
                if cur_dd_days > max_dd_days:
                    max_dd_days = cur_dd_days

        rf_daily = rf_rate / trading_days
        ex_sum = 0.0
        ex_sq = 0.0
        for i in range(1, m):
            ex = daily_ret[i] - rf_daily
            ex_sum += ex
            ex_sq += ex * ex
        sharpe = 0.0
        if m > 1:
            mean_ex = ex_sum / (m - 1)
            var_ex = ex_sq / (m - 1) - mean_ex * mean_ex
            if var_ex > 1e-15:
                sharpe = np.sqrt(trading_days) * mean_ex / np.sqrt(var_ex)

        calmar = annual_return / (-max_dd) if abs(max_dd) > 1e-15 else 0.0

        ret_sum = 0.0
        ret_sq = 0.0
        for i in range(1, m):
            ret_sum += daily_ret[i]
            ret_sq += daily_ret[i] * daily_ret[i]
        daily_vol = 0.0
        if m > 1:
            ret_mean = ret_sum / (m - 1)
            ret_var = ret_sq / (m - 1) - ret_mean * ret_mean
            if ret_var > 0.0:
                daily_vol = np.sqrt(ret_var * trading_days)

        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        avg_win = sum_win / n_wins if n_wins > 0 else 0.0
        n_losses = n_trades - n_wins
        avg_loss = sum_loss / n_losses if n_losses > 0 else 0.0
        if avg_loss > 1e-15:
            profit_factor = avg_win / avg_loss
        else:
            profit_factor = 10.0 if avg_win > 0.0 else 0.0
        avg_hold = total_hold / n_trades if n_trades > 0 else 0.0

        return (total_return, annual_return, max_dd, float(max_dd_days), daily_vol,
                sharpe, calmar, float(n_trades), win_rate, profit_factor, avg_hold)


# ============================================================
# 纯 Python 回退 (numba 不可用时)
# ============================================================

def _backtest_one_py(high, low, close, st_period, st_mult, ma_fast, ma_slow, use_ma,
                     n, k, atr_period, use_chandelier, use_volatility, hv_window, hv_threshold,
                     capital, flat_annual, commission):
    from chandelier_exit import calc_chandelier
    from dynamic_atr import run_backtest_long_only
    import metrics as mt
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    df = calc_chandelier(df, st_period=st_period, st_mult=st_mult, ma_fast=ma_fast, ma_slow=ma_slow,
                         n=n, k=k, atr_period=atr_period, use_ma=use_ma, use_chandelier=use_chandelier,
                         use_volatility=use_volatility, hv_window=hv_window, hv_threshold=hv_threshold)
    trades, eq, rets = run_backtest_long_only(df, capital=capital, flat_annual=flat_annual, commission=commission)
    eq_arr = np.array([e["equity"] for e in eq])
    rets_arr = np.array(rets)
    pnl_pcts = np.array([t["pnl_pct"] for t in trades]) if trades else np.array([])
    m = mt.calc_all_metrics(eq_arr, rets_arr, pnl_pcts, capital)
    return (m["total_return"], m["annual_return"], m["max_dd"], float(m["max_dd_days"]), m["daily_vol"],
            m["sharpe"], m["calmar"], float(m["n_trades"]), m["win_rate"], m["profit_factor"], m["avg_hold_days"])


_compute_one = _backtest_one if HAS_NUMBA else _backtest_one_py


# ============================================================
# 单个 ST 周期的搜索 (供并行调用)
# ============================================================

def _search_one_st_period(args):
    (st_period, high, low, close, st_mults, ns, ks, atr_periods, hv_thresholds,
     ma_fast, ma_slow, use_ma, use_chandelier, use_volatility, hv_window,
     capital, flat_annual, commission) = args
    rows = []
    for st_m in st_mults:
        for n in ns:
            for k in ks:
                for atr_p in atr_periods:
                    for hvt in hv_thresholds:
                        res = _compute_one(high, low, close, st_period, st_m, ma_fast, ma_slow, use_ma,
                                           n, k, atr_p, use_chandelier, use_volatility, hv_window, hvt,
                                           capital, flat_annual, commission)
                        total_ret, annual_ret, max_dd, max_dd_days, daily_vol, sharpe, calmar, \
                            n_trades, win_rate, profit_factor, avg_hold = res
                        rows.append({
                            "st_period": st_period, "st_mult": st_m, "n": n, "k": k, "atr_period": atr_p,
                            "hv_threshold": hvt,
                            "total_return": total_ret, "annual_return": annual_ret,
                            "max_dd": max_dd, "max_dd_days": int(max_dd_days),
                            "sharpe": sharpe, "calmar": calmar, "daily_vol": daily_vol,
                            "n_trades": int(n_trades), "win_rate": win_rate,
                            "profit_factor": profit_factor, "avg_hold_days": avg_hold,
                        })
    return rows


# ============================================================
# 网格扫描
# ============================================================

def scan(df_raw):
    total = len(ST_PERIODS) * len(ST_MULTS) * len(NS) * len(KS) * len(ATR_PERIODS) * len(HV_THRESHOLDS)
    high = df_raw["high"].values.astype(np.float64)
    low = df_raw["low"].values.astype(np.float64)
    close = df_raw["close"].values.astype(np.float64)

    print(f"  网格: ST{ST_PERIODS}x{ST_MULTS} × 吊灯{NS}x{KS} × ATR{ATR_PERIODS} × HV阈值{HV_THRESHOLDS}")
    print(f"  固定: USE_MA={USE_MA}, USE_CHANDELIER={USE_CHANDELIER}, USE_VOLATILITY={USE_VOLATILITY}, HV窗口={HV_WINDOW}")
    print(f"  共 {total:,} 组合")
    print(f"  引擎: {'Numba JIT' if HAS_NUMBA else '纯Python(慢)'}", end="")
    if HAS_NUMBA and USE_PARALLEL and len(ST_PERIODS) > 1:
        print(f" + {N_WORKERS} 进程并行")
    else:
        print()

    # JIT 预热编译
    if HAS_NUMBA:
        print("  JIT 预热编译...", end=" ", flush=True)
        _compute_one(high, low, close, ST_PERIODS[0], ST_MULTS[0], MA_FAST, MA_SLOW, USE_MA,
                     NS[0], KS[0], ATR_PERIODS[0], USE_CHANDELIER, USE_VOLATILITY, HV_WINDOW, HV_THRESHOLDS[0],
                     CAPITAL, FLAT_ANNUAL, FEE_RATE)
        print("完成")

    tasks = [
        (st_p, high, low, close, ST_MULTS, NS, KS, ATR_PERIODS, HV_THRESHOLDS,
         MA_FAST, MA_SLOW, USE_MA, USE_CHANDELIER, USE_VOLATILITY, HV_WINDOW,
         CAPITAL, FLAT_ANNUAL, FEE_RATE)
        for st_p in ST_PERIODS
    ]

    if USE_PARALLEL and len(ST_PERIODS) > 1:
        from multiprocessing import Pool
        with Pool(processes=N_WORKERS) as pool:
            if HAS_TQDM:
                chunks = list(tqdm(pool.imap_unordered(_search_one_st_period, tasks),
                                   total=len(tasks), desc="  扫描", unit="ST", ncols=80))
            else:
                chunks = list(pool.imap_unordered(_search_one_st_period, tasks))
        rows = []
        for c in chunks:
            rows.extend(c)
        rows.sort(key=lambda x: (x["st_period"], x["st_mult"], x["n"], x["k"]))
    else:
        rows = []
        for st_p in ST_PERIODS:
            rows.extend(_search_one_st_period(tasks[ST_PERIODS.index(st_p)]))
            print(f"  ST周期={st_p:>2} 完成 ({len(rows)}/{total})")

    print(f"  扫描完成! 共测试 {len(rows):,} 组参数")
    return pd.DataFrame(rows)


# ============================================================
# 热力图
# ============================================================

def plot_grid(results, fix_dict, x_key, y_key, x_vals, y_vals, title, outpath):
    sub = results.copy()
    for kk, vv in fix_dict.items():
        sub = sub[sub[kk] == vv]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for ax, (col, label, cmap, pct) in zip(axes, METRICS):
        pivot = sub.pivot_table(index=y_key, columns=x_key, values=col)
        pivot = pivot.reindex(index=y_vals, columns=x_vals)
        vals = pivot.values

        if pct:
            vmin = min(0.0, np.nanmin(vals))
            vmax = max(0.0, np.nanmax(vals))
        else:
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 1.0

        im = ax.imshow(vals, aspect="auto", origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([f"{v:g}" for v in x_vals], fontsize=8)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([f"{v:g}" for v in y_vals], fontsize=8)
        ax.set_xlabel(x_key, fontsize=9)
        ax.set_ylabel(y_key, fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, format=mticker.PercentFormatter(1.0) if pct else None)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  热力图已保存: {outpath}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"  {DATA_NAME} — SuperTrend入场+吊灯止损/波动率止盈 6参数扫描")
    print(f"{'='*70}")
    print(f"  规则: ST翻多开多(USE_MA={USE_MA}), 吊灯止损离场(USE_CHANDELIER={USE_CHANDELIER}); "
          f"空仓现金年化 {FLAT_ANNUAL:.1%}, 单边手续费 {FEE_RATE:.2%}")

    df_raw = load_data(DATA_FILE)
    print(f"  数据: {len(df_raw)} 条, {df_raw['trade_date'].iloc[0].date()} ~ {df_raw['trade_date'].iloc[-1].date()}")

    results = scan(df_raw)

    # 保存明细
    csv_path = f"{DATA_NAME}_吊灯止损参数扫描.csv"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  明细已保存: {csv_path}")

    # ---- Top 20 (按夏普) ----
    top = results.nlargest(200, "sharpe")
    print(f"\n{'='*94}")
    print(f"  Top 200 参数组合 (按夏普比率)")
    print(f"{'='*94}")
    print(f"  {'STp':>4} {'STm':>4} {'N':>3} {'K':>4} {'ATR':>3} {'HV':>4} | "
          f"{'总收益':>8} {'年化':>8} {'回撤':>8} {'夏普':>7} {'Calmar':>7} {'交易':>4} {'胜率':>7} {'盈亏比':>6}")
    for _, r in top.iterrows():
        flag = "  <-- 退化" if r["n_trades"] < MIN_TRADES else ""
        print(f"  {int(r['st_period']):>4} {r['st_mult']:>4.1f} {int(r['n']):>3} {r['k']:>4.1f} {int(r['atr_period']):>3} {r['hv_threshold']:>4.0%} | "
              f"{r['total_return']:>8.1%} {r['annual_return']:>8.1%} {r['max_dd']:>8.1%} "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {int(r['n_trades']):>4} "
              f"{r['win_rate']:>7.1%} {r['profit_factor']:>6.2f}{flag}")

    # ---- 最优组合 (过滤退化) ----
    close = df_raw["close"].values
    bench_ret = close[-1] / close[0] - 1

    filtered = results[results["n_trades"] >= MIN_TRADES]
    print(f"\n  {'='*72}")
    print(f"  最优组合 (交易≥{MIN_TRADES}笔, 已过滤退化)    标的 Buy&Hold 总收益 = {bench_ret:.1%}")
    print(f"  {'='*72}")
    if len(filtered) == 0:
        print("  没有满足最低交易笔数的组合")
    else:
        for label, col in [("夏普最高", "sharpe"), ("Calmar最高", "calmar"), ("总收益最高", "total_return")]:
            b = filtered.loc[filtered[col].idxmax()]
            print(f"  {label}: ST{int(b['st_period'])}x{b['st_mult']:.1f}, 吊灯{int(b['n'])}x{b['k']:.1f}(ATR{int(b['atr_period'])}), HV阈值{b['hv_threshold']:.0%}  |  "
                  f"总收益={b['total_return']:.1%}  年化={b['annual_return']:.1%}  回撤={b['max_dd']:.1%}  "
                  f"夏普={b['sharpe']:.3f}  Calmar={b['calmar']:.3f}  交易={int(b['n_trades'])}笔")

    # ---- 热力图 ----
    if len(filtered) > 0:
        anchor = filtered.loc[filtered["sharpe"].idxmax()]
        ap, am = int(anchor["st_period"]), float(anchor["st_mult"])
        an, ak = int(anchor["n"]), float(anchor["k"])
        aa = int(anchor["atr_period"])
        ahv = float(anchor["hv_threshold"])

        print(f"\n  生成热力图 (锚点: ST{ap}x{am}, 吊灯{an}x{ak}, ATR{aa}, HV{ahv:.0%})...")
        plot_grid(
            results, {"n": an, "k": ak, "atr_period": aa, "hv_threshold": ahv},
            x_key="st_period", y_key="st_mult", x_vals=ST_PERIODS, y_vals=ST_MULTS,
            title=f"{DATA_NAME} — SuperTrend 参数 (吊灯{an}x{ak}, ATR{aa}, HV{ahv:.0%})",
            outpath=f"{DATA_NAME}_吊灯止损_热力图_ST.png",
        )
        plot_grid(
            results, {"st_period": ap, "st_mult": am, "atr_period": aa, "hv_threshold": ahv},
            x_key="n", y_key="k", x_vals=NS, y_vals=KS,
            title=f"{DATA_NAME} — 吊灯止损参数 (ST{ap}x{am}, ATR{aa}, HV{ahv:.0%})",
            outpath=f"{DATA_NAME}_吊灯止损_热力图_吊灯.png",
        )
        plot_grid(
            results, {"st_period": ap, "st_mult": am, "n": an, "hv_threshold": ahv},
            x_key="atr_period", y_key="k", x_vals=ATR_PERIODS, y_vals=KS,
            title=f"{DATA_NAME} — ATR周期 × 吊灯K (ST{ap}x{am}, N{an}, HV{ahv:.0%})",
            outpath=f"{DATA_NAME}_吊灯止损_热力图_ATR.png",
        )

    print(f"\n  扫描完成。把 Top 组合的参数填回 chandelier_exit.py 的 __main__ 配置区即可复现。")
