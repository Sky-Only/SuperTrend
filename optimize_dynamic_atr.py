"""
动态 ATR 参数网格扫描 (沪深300)

对 4 个参数做网格搜索:
    ER_PERIOD  (效率比率窗口)
    FAST       (快平滑周期)
    SLOW       (慢平滑周期)
    MULTIPLIER (SuperTrend 乘数 M)

回测引擎复用 dynamic_atr.py 的只做多引擎:
    - 只做多, 空仓现金年化 1.5%, 单边手续费万三

输出:
    - 全部组合明细 CSV
    - Top 20 控制台排名 (按夏普)
    - 两张热力图: (慢周期 × 乘数) 和 (快周期 × 慢周期)
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
    "Microsoft YaHei", "SimHei", "Noto Sans SC",
    "Heiti SC", "STHeiti", "WenQuanYi Micro Hei",
    "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

import metrics as mt
from backtest import load_data
from dynamic_atr import calc_dynamic_atr, _build_super_trend, run_backtest_long_only


# ============================================================
# 配置
# ============================================================
DATA_NAME = "沪深300"
DATA_FILE = f"{DATA_NAME}_10年日线.parquet"
CAPITAL = 1_000_000

FLAT_ANNUAL = 0.015      # 空仓现金年化
FEE_RATE    = 0.0003     # 单边手续费 (万三)

# 搜索网格
ER_PERIODS  = [5, 10, 14, 20]
FASTS       = [2, 3, 4, 5]
SLOWS       = [6, 7, 8, 10, 15]                 # 等效慢周期 ~23~127 天
MULTIPLIERS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# 热力图展示的指标: (列名, 标题, colormap, 是否百分比)
METRICS = [
    ("total_return",  "总收益率",    "RdYlGn",   True),
    ("annual_return", "年化收益率",  "RdYlGn",   True),
    ("max_dd",        "最大回撤",    "RdYlGn_r", True),
    ("sharpe",        "夏普比率",    "RdYlGn",   False),
    ("calmar",        "Calmar比率",  "RdYlGn",   False),
    ("n_trades",      "交易次数",    "Blues",    False),
]


# ============================================================
# 单组合绩效计算
# ============================================================

def _metrics(trades, eq, rets, capital):
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


# ============================================================
# 网格扫描
# ============================================================

def scan(df_raw):
    high = df_raw["high"].values
    low = df_raw["low"].values
    close = df_raw["close"].values

    total = len(ER_PERIODS) * len(FASTS) * len(SLOWS) * len(MULTIPLIERS)
    rows = []
    done = 0

    print(f"  网格: ER{ER_PERIODS} × FAST{FASTS} × SLOW{SLOWS} × M{MULTIPLIERS}")
    print(f"  共 {total} 组合")

    for er in ER_PERIODS:
        for fast in FASTS:
            for slow in SLOWS:
                # 动态 ATR 只与 (er, fast, slow) 有关, 对每个乘数复用一次
                df_atr = calc_dynamic_atr(df_raw, er_period=er, fast=fast, slow=slow)
                atr = df_atr["atr"].values

                for mult in MULTIPLIERS:
                    st, trend, ub, lb = _build_super_trend(high, low, close, atr, mult)
                    df = df_raw.copy()
                    df["trend"] = trend
                    df["super_trend"] = st
                    df["atr"] = atr

                    trades, eq, rets = run_backtest_long_only(
                        df, capital=CAPITAL, flat_annual=FLAT_ANNUAL, commission=FEE_RATE)
                    m = _metrics(trades, eq, rets, CAPITAL)

                    rows.append({
                        "er_period": er, "fast": fast, "slow": slow, "multiplier": mult,
                        "total_return": m["total_return"],
                        "annual_return": m["annual_return"],
                        "max_dd": m["max_dd"],
                        "max_dd_days": m["max_dd_days"],
                        "sharpe": m["sharpe"],
                        "calmar": m["calmar"],
                        "daily_vol": m["daily_vol"],
                        "n_trades": m["n_trades"],
                        "win_rate": m["win_rate"],
                        "profit_factor": m["profit_factor"],
                        "avg_hold_days": m["avg_hold_days"],
                    })
                    done += 1
        print(f"  ER={er:>2} 完成 ({done}/{total})")

    return pd.DataFrame(rows)


# ============================================================
# 热力图
# ============================================================

def plot_grid(results, fix_dict, x_key, y_key, x_vals, y_vals, title, outpath):
    sub = results.copy()
    for k, v in fix_dict.items():
        sub = sub[sub[k] == v]

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
    print(f"  {DATA_NAME} — 动态 ATR 参数网格扫描")
    print(f"{'='*70}")
    print(f"  规则: 只做多, 空仓现金年化 {FLAT_ANNUAL:.1%}, 单边手续费 {FEE_RATE:.2%}")

    df_raw = load_data(DATA_FILE)
    print(f"  数据: {len(df_raw)} 条, {df_raw['trade_date'].iloc[0].date()} ~ {df_raw['trade_date'].iloc[-1].date()}")

    results = scan(df_raw)

    # 保存明细
    csv_path = f"{DATA_NAME}_动态ATR参数扫描.csv"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  明细已保存: {csv_path}")

    # ---- Top 20 (按夏普) ----
    top = results.nlargest(20, "sharpe")
    print(f"\n{'='*90}")
    print(f"  Top 20 参数组合 (按夏普比率)")
    print(f"{'='*90}")
    print(f"  {'ER':>3} {'FAST':>4} {'SLOW':>4} {'M':>4} | {'总收益':>8} {'年化':>8} {'回撤':>8} "
          f"{'夏普':>7} {'Calmar':>7} {'交易':>4} {'胜率':>7} {'盈亏比':>6} {'持仓天':>6}")
    for _, r in top.iterrows():
        print(f"  {int(r['er_period']):>3} {int(r['fast']):>4} {int(r['slow']):>4} "
              f"{r['multiplier']:>4.1f} | {r['total_return']:>8.1%} {r['annual_return']:>8.1%} "
              f"{r['max_dd']:>8.1%} {r['sharpe']:>7.3f} {r['calmar']:>7.3f} "
              f"{int(r['n_trades']):>4} {r['win_rate']:>7.1%} {r['profit_factor']:>6.2f} "
              f"{r['avg_hold_days']:>6.1f}")

    # ---- 最优组合 (各标准) ----
    best_sharpe = results.iloc[results["sharpe"].idxmax()]
    best_calmar = results.iloc[results["calmar"].idxmax()]
    best_ret = results.iloc[results["total_return"].idxmax()]
    print(f"\n  {'='*60}")
    print(f"  最优组合 (不同目标)")
    print(f"  {'='*60}")
    for label, b in [("夏普最高", best_sharpe), ("Calmar最高", best_calmar), ("总收益最高", best_ret)]:
        print(f"  {label}: ER={int(b['er_period'])}, FAST={int(b['fast'])}, SLOW={int(b['slow'])}, "
              f"M={b['multiplier']:.1f}  |  总收益={b['total_return']:.1%}  年化={b['annual_return']:.1%}  "
              f"回撤={b['max_dd']:.1%}  夏普={b['sharpe']:.3f}  Calmar={b['calmar']:.3f}  "
              f"交易={int(b['n_trades'])}笔")

    # ---- 热力图 ----
    be = int(best_sharpe["er_period"])
    bf = int(best_sharpe["fast"])
    bm = float(best_sharpe["multiplier"])

    print(f"\n  生成热力图 (最优 ER={be}, FAST={bf}, M={bm})...")
    plot_grid(
        results, {"er_period": be, "fast": bf},
        x_key="multiplier", y_key="slow",
        x_vals=MULTIPLIERS, y_vals=SLOWS,
        title=f"{DATA_NAME} — 动态ATR: 慢周期 × 乘数 (ER={be}, FAST={bf})",
        outpath=f"{DATA_NAME}_动态ATR_慢乘数热力图.png",
    )
    plot_grid(
        results, {"er_period": be, "multiplier": bm},
        x_key="fast", y_key="slow",
        x_vals=FASTS, y_vals=SLOWS,
        title=f"{DATA_NAME} — 动态ATR: 快周期 × 慢周期 (ER={be}, M={bm})",
        outpath=f"{DATA_NAME}_动态ATR_快慢热力图.png",
    )

    print(f"\n  扫描完成。把 Top 组合的参数填回 dynamic_atr.py 的 __main__ 配置区即可复现。")
