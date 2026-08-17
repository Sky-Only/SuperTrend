"""
ATR 移动止盈 代码集
===================
用 ATR 做移动止盈位 (滚动最高价 - n × ATR), 离线测试, 数据源为 CSV 文件。

数据字段 (tushare index_daily 导出的 CSV):
    trade_date : 交易日期 (YYYYMMDD 或日期字符串, 自动解析)
    open       : 开盘价
    high       : 最高价
    low        : 最低价
    close      : 收盘价
    (其余列如 pre_close/change/pct_chg/vol/amount 会被自动忽略)

移动止盈位 = 近 ATR_PERIOD 日最高价 - N × ATR(ATR_PERIOD)
  价格创新高时止盈位跟着上移; 收盘跌破止盈位 = 触发止盈/止损离场。

测试目标:
    1. 测试标的与周期: 在 __main__ 改 DATA_FILE
    2. 自定义指标参数: ATR 周期 / 倍数 N
    3. 输出: ATR 值 + 移动止盈位 (N × ATR) 叠加走势图
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


# ============================================================
# 1. 数据加载 (离线 CSV)
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


def calc_trailing_stop(df, atr_period=14, n=3.0):
    """移动止盈位 = 近 atr_period 日最高价 - n × ATR, 返回 (atr, trailing)"""
    high = df["high"].values
    atr = calc_atr(df, atr_period)
    hh = pd.Series(high).rolling(atr_period, min_periods=1).max().values
    trailing = hh - n * atr
    return atr, trailing


# ============================================================
# 3. 绘图
# ============================================================

def plot_atr_trailing(df, atr_period, n, output_dir):
    """ATR 值 + 移动止盈位 (N × ATR) 叠加走势图 (双面板)"""
    dates = df["trade_date"]
    close = df["close"].values
    atr, trailing = calc_trailing_stop(df, atr_period, n)

    fig, (ax_p, ax_a) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

    # 上图: 价格 + 移动止盈位
    ax_p.plot(dates, close, color="#2c3e50", linewidth=0.8, label="收盘价")
    ax_p.plot(dates, trailing, color="#e74c3c", linewidth=0.9, alpha=0.85,
              label=f"移动止盈位 (HH{atr_period} - {n}×ATR)")
    ax_p.set_ylabel("价格", fontsize=11)
    ax_p.legend(loc="upper left", fontsize=10)
    ax_p.grid(True, alpha=0.3)
    ax_p.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax_p.set_title(f"ATR 移动止盈 (ATR周期={atr_period}, 倍数 N={n})", fontsize=14, fontweight="bold")

    # 下图: ATR 值
    ax_a.plot(dates, atr, color="#e67e22", linewidth=0.9, label=f"ATR ({atr_period})")
    ax_a.set_ylabel("ATR", fontsize=11)
    ax_a.set_xlabel("日期", fontsize=11)
    ax_a.legend(loc="upper left", fontsize=10)
    ax_a.grid(True, alpha=0.3)

    fig.subplots_adjust(hspace=0.08, left=0.07, right=0.97, top=0.93, bottom=0.08)
    fig.savefig(os.path.join(output_dir, "01_ATR移动止盈.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: 01_ATR移动止盈.png")


# ============================================================
# 4. 入口
# ============================================================

if __name__ == "__main__":
    # ==================== 在这里修改配置 ====================
    DATA_FILE = "沪深300_10年日线.csv"   # CSV 数据文件 (离线, 指数)
    DATA_NAME = "沪深300"

    # ATR 移动止盈参数
    ATR_PERIOD = 14   # ATR 周期
    ATR_N      = 3.0  # 移动止盈倍数 N (N × ATR)

    OUTPUT_DIR = f"{DATA_NAME}_ATR移动止盈"
    # ========================================================

    print(f"加载数据: {DATA_FILE}")
    df = load_csv(DATA_FILE)
    print(f"  数据: {len(df)} 条, {df['trade_date'].iloc[0].date()} ~ {df['trade_date'].iloc[-1].date()}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_atr_trailing(df, ATR_PERIOD, ATR_N, OUTPUT_DIR)

    print(f"\n图表已保存到 {OUTPUT_DIR}/")
