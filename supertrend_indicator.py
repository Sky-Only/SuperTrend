"""
SuperTrend 趋势划分 (状态识别) 代码集
====================================
用 SuperTrend 做趋势状态划分 (多头/空头), 离线测试, 数据源为 CSV 文件。

数据字段 (tushare index_daily 导出的 CSV):
    trade_date : 交易日期 (YYYYMMDD 或日期字符串, 自动解析)
    open       : 开盘价
    high       : 最高价
    low        : 最低价
    close      : 收盘价

测试目标:
    1. 测试标的与周期: 在 __main__ 改 DATA_FILE, 可看任意 CSV 指数
    2. 自定义指标参数: SuperTrend 周期/乘数, HV 窗口/年数
    3. 输出:
       - SuperTrend 指标叠加走势图 (价格 + SuperTrend线 + 趋势背景色)
       - 近 x 日历史波动率 HV + 近 y 年历史分位数 (标注 HV 90% 值和时段)
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd
import hv_analysis
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
    "Microsoft YaHei", "SimHei", "Noto Sans SC", "Heiti SC", "STHeiti",
    "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 数据加载
# ============================================================

def load_csv(filepath):
    """离线加载 CSV, 统一字段为 trade_date/open/high/low/close"""
    df = pd.read_csv(filepath)
    td = df["trade_date"].astype(str)
    if td.str.isdigit().all() and td.str.len().eq(8).all():
        df["trade_date"] = pd.to_datetime(td, format="%Y%m%d")
    else:
        df["trade_date"] = pd.to_datetime(td)
    df = df.sort_values("trade_date").reset_index(drop=True)
    required = ["trade_date", "open", "high", "low", "close"]
    for c in required:
        assert c in df.columns, f"缺少字段: {c}"
    return df[required]


# ============================================================
# 2. 指标计算
# ============================================================

def calc_atr(df, period=14):
    """Wilder ATR"""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))
    atr = np.full(n, np.nan)
    atr[period - 1] = tr[:period].mean()
    for t in range(period, n):
        atr[t] = (atr[t - 1] * (period - 1) + tr[t]) / period
    return atr


def calc_supertrend(df, period=10, multiplier=3.0):
    """SuperTrend 趋势划分, 返回 (trend, super_trend, upper, lower, atr)"""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    atr = calc_atr(df, period=period)
    mid = (high + low) / 2
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr
    trend = np.zeros(n, dtype=int)       # 1=多头, -1=空头, 0=未确立
    super_trend = np.full(n, np.nan)
    first = period
    if first < n:
        if close[first] > upper[first]:
            trend[first] = 1
            super_trend[first] = lower[first]
        elif close[first] < lower[first]:
            trend[first] = -1
            super_trend[first] = upper[first]
        else:
            trend[first] = 1
            super_trend[first] = lower[first]
        for i in range(first + 1, n):
            if trend[i - 1] == 1:
                st = max(lower[i], super_trend[i - 1])
                if close[i] < st:
                    trend[i] = -1
                    super_trend[i] = upper[i]
                else:
                    trend[i] = 1
                    super_trend[i] = st
            else:
                st = min(upper[i], super_trend[i - 1])
                if close[i] > st:
                    trend[i] = 1
                    super_trend[i] = lower[i]
                else:
                    trend[i] = -1
                    super_trend[i] = st
    return trend, super_trend, upper, lower, atr


# ============================================================
# 3. 绘图
# ============================================================

def plot_supertrend_overlay(df, period, multiplier, output_dir):
    """SuperTrend 指标叠加走势图 (价格 + SuperTrend线 + 趋势背景)"""
    dates = df["trade_date"]
    close = df["close"].values
    trend, st, upper, lower, _ = calc_supertrend(df, period, multiplier)
    n = len(df)
    fig, ax = plt.subplots(figsize=(16, 8))
    for i in range(n - 1):
        if trend[i] == 1:
            ax.axvspan(dates.iloc[i], dates.iloc[i + 1], color="green", alpha=0.06, lw=0)
        elif trend[i] == -1:
            ax.axvspan(dates.iloc[i], dates.iloc[i + 1], color="red", alpha=0.06, lw=0)
    ax.plot(dates, close, color="#2c3e50", linewidth=0.8, label="收盘价")
    ax.plot(dates, st, color="#e74c3c", linewidth=0.9, alpha=0.85,
            label=f"SuperTrend (N={period}, M={multiplier})")
    ax.set_title(f"SuperTrend 趋势划分 (N={period}, M={multiplier})", fontsize=14, fontweight="bold")
    ax.set_ylabel("价格", fontsize=11)
    ax.set_xlabel("日期", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_SuperTrend叠加走势图.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: 01_SuperTrend叠加走势图.png")


# ============================================================
# 4. 入口
# ============================================================

if __name__ == "__main__":
    # ==================== 在这里修改配置 ====================
    DATA_FILE = "沪深300_10年日线.csv"   # CSV 数据文件 (离线, 指数)
    DATA_NAME = "沪深300"

    # SuperTrend 参数
    ST_PERIOD = 10
    ST_MULT   = 3.0

    # HV 历史波动率参数
    HV_WINDOW = 20    # 近 x 日 HV
    HV_YEARS  = 3     # 近 y 年历史分位数
    HV_Q      = 90    # 分位数 (90 = 只标最极端的高波动)

    OUTPUT_DIR = f"{DATA_NAME}_指标分析"
    # ========================================================

    print(f"加载数据: {DATA_FILE}")
    df = load_csv(DATA_FILE)
    print(f"  数据: {len(df)} 条, {df['trade_date'].iloc[0].date()} ~ {df['trade_date'].iloc[-1].date()}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_supertrend_overlay(df, ST_PERIOD, ST_MULT, OUTPUT_DIR)
    hv_analysis.plot_hv(df, HV_WINDOW, HV_YEARS, HV_Q, OUTPUT_DIR, output_file="02_历史波动率HV.png")

    print(f"\n全部图表已保存到 {OUTPUT_DIR}/")
