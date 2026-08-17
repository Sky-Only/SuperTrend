"""
HV 历史波动率分析模块
================================================
计算标的的历史波动率 HV, 以及历史分位数 (用于识别高波动时段)。

核心函数:
    calc_hv(close, window)           → 近 window 日年化 HV 序列
    calc_hv_percentile(hv, years, q) → 近 years 年 HV 的 q 分位阈值
    analyze_hv(close, window, years, q) → 返回 (hv, hv_q, high_vol 布尔掩码)
    plot_hv(df, ...)                 → 画 HV 图 (含 90% 分位线 + 高波动时段标注)

使用方式:
    from hv_analysis import calc_hv, analyze_hv, plot_hv

公式:
    HV = std(近 window 日收益率, ddof=0) × √252   (年化, 单位 %)
    高波动时段 = HV > 近 years 年 HV 的 q 分位阈值

也可直接运行本文件做独立测试:
    python hv_analysis.py
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
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
    "Microsoft YaHei", "SimHei", "Noto Sans SC", "Heiti SC", "STHeiti",
    "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 默认年化交易日数
TRADING_DAYS_PER_YEAR = 252


# ============================================================
# 1. HV 计算
# ============================================================

def calc_hv(close, window=20):
    """
    计算近 window 日历史波动率 HV (年化)。

    参数:
        close: 收盘价序列 (numpy 数组或 pandas Series)
        window: 滚动窗口 (交易日数), 常用 10/20/60

    返回:
        hv: 年化历史波动率序列 (长度同 close, 前 window-1 个为 NaN)
    """
    rets = pd.Series(close).pct_change()
    hv = rets.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return hv.values


def calc_hv_percentile(hv, years=3, q=90):
    """
    计算 hv 序列在最近 years 年的 q 分位阈值。

    参数:
        hv: 历史波动率序列 (来自 calc_hv)
        years: 回看年数 (用于统计分位的历史长度)
        q: 分位数 (0~100), 默认 90

    返回:
        阈值 (float), 若数据不足则返回 NaN
    """
    n_recent = int(years * TRADING_DAYS_PER_YEAR)
    recent = hv[-n_recent:] if len(hv) > n_recent else hv
    recent = recent[~np.isnan(recent)]
    if len(recent) == 0:
        return np.nan
    return np.percentile(recent, q)


def analyze_hv(close, window=20, years=3, q=90):
    """
    HV、分位阈值、高波动掩码。

    返回:
        (hv, hv_q, high_vol)
        hv       : 年化 HV 序列
        hv_q     : 近 years 年 q 分位阈值
        high_vol : 布尔数组, True = HV 超过 q 分位 (高波动)
    """
    hv = calc_hv(close, window)
    hv_q = calc_hv_percentile(hv, years, q)
    high_vol = ~np.isnan(hv) & (hv > hv_q)
    return hv, hv_q, high_vol


# ============================================================
# 2. HV 绘图
# ============================================================

def plot_hv(df, window=20, years=3, q=90, output_dir=".", output_file="历史波动率HV.png"):
    """
    画 HV 历史波动率图 (含 q 分位线 + 高波动时段标注)。

    参数:
        df: 含 trade_date 和 close 列的 DataFrame
        window / years / q: 同 calc_hv / calc_hv_percentile
        output_dir: 输出目录
        output_file: 输出文件名

    返回:
        (hv, hv_q) 供调用方继续使用
    """
    dates = df["trade_date"]
    close = df["close"].values
    hv, hv_q, high_vol = analyze_hv(close, window, years, q)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, hv * 100, color="#2980b9", linewidth=0.9, label=f"HV ({window}日, 年化)")
    if not np.isnan(hv_q):
        ax.axhline(hv_q * 100, color="#e74c3c", linestyle="--", linewidth=1.0,
                   label=f"历史 {years}年 {q}% 分位 = {hv_q*100:.1f}%")

    # 标注 HV > q 分位的时段 (高波动)
    in_high = False
    start = None
    for i in range(len(dates)):
        if high_vol[i] and not in_high:
            in_high = True
            start = dates.iloc[i]
        elif not high_vol[i] and in_high:
            in_high = False
            ax.axvspan(start, dates.iloc[i], color="#e74c3c", alpha=0.15, lw=0)
    if in_high:
        ax.axvspan(start, dates.iloc[-1], color="#e74c3c", alpha=0.15, lw=0)

    ax.set_title(f"历史波动率 HV + {years}年分位数 (红区=HV>{q}% 高波动)", fontsize=14, fontweight="bold")
    ax.set_ylabel("HV %", fontsize=11)
    ax.set_xlabel("日期", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  HV 图已保存: {os.path.join(output_dir, output_file)}")

    return hv, hv_q


# ============================================================
# 3. 独立测试入口
# ============================================================

if __name__ == "__main__":
    # ==================== 在这里修改配置 ====================
    DATA_FILE = "沪深300_10年日线.csv"   # CSV 数据文件
    HV_WINDOW = 2    # 近 x 日 HV
    HV_YEARS  = 10     # 近 y 年历史分位数
    HV_Q      = 80    # 分位数 (90 = 只标最极端的高波动)
    OUTPUT_DIR = "HV分析"
    # ========================================================

    df = pd.read_csv(DATA_FILE)
    td = df["trade_date"].astype(str)
    if td.str.isdigit().all() and td.str.len().eq(8).all():
        df["trade_date"] = pd.to_datetime(td, format="%Y%m%d")
    else:
        df["trade_date"] = pd.to_datetime(td)
    df = df.sort_values("trade_date").reset_index(drop=True)

    print(f"数据: {len(df)} 条, {df['trade_date'].iloc[0].date()} ~ {df['trade_date'].iloc[-1].date()}")
    hv, hv_q = plot_hv(df, HV_WINDOW, HV_YEARS, HV_Q, OUTPUT_DIR)
    print(f"HV 90% 分位 = {hv_q*100:.1f}%")
