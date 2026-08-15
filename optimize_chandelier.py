"""
吊灯止损 (Chandelier Exit) 参数网格扫描 (沪深300)

对 4 个参数做网格搜索:
    N        (吊灯止损回看天数: 最高价窗口 + ATR 周期)
    K        (ATR 倍数)
    MA_FAST  (入场信号快均线周期)
    MA_SLOW  (入场信号慢均线周期)

入场: MA 金叉 (快线上穿慢线)
离场: 收盘跌破吊灯止损线 (近 N 日最高价 - K × ATR)
只做多, 空仓现金年化 1.5%, 单边手续费万三

输出:
    - 全部组合明细 CSV
    - Top 20 控制台排名 (按夏普)
    - 两张热力图: (N×K 吊灯参数) 和 (MA快×MA慢 入场参数)
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
from chandelier_exit import calc_chandelier
from dynamic_atr import run_backtest_long_only


# ============================================================
# 配置
# ============================================================
DATA_NAME = "沪深300"
DATA_FILE = f"{DATA_NAME}_10年日线.parquet"
CAPITAL = 1_000_000

FLAT_ANNUAL = 0.015      # 空仓现金年化
FEE_RATE    = 0.0003     # 单边手续费 (万三)

# 搜索网格
NS        = [10, 14, 20, 22, 30, 40, 60, 80]               # 吊灯止损回看天数
KS        = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]       # ATR 倍数
MA_FASTS  = [5, 10, 20]                                     # 入场快均线周期
MA_SLOWS  = [30, 60, 100]                                   # 入场慢均线周期

# 挑选"最优"时要求的最低交易笔数 (过滤"买一次拿十年"的退化结果)
MIN_TRADES = 5

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
    total = len(NS) * len(KS) * len(MA_FASTS) * len(MA_SLOWS)
    rows = []
    done = 0

    print(f"  网格: N{NS} × K{KS} × MA快{MA_FASTS} × MA慢{MA_SLOWS}")
    print(f"  共 {total} 组合")

    for n in NS:
        for k in KS:
            for mf in MA_FASTS:
                for ms in MA_SLOWS:
                    if mf >= ms:
                        continue  # 快线周期必须小于慢线
                    df = calc_chandelier(df_raw, n=n, k=k, ma_fast=mf, ma_slow=ms)
                    trades, eq, rets = run_backtest_long_only(
                        df, capital=CAPITAL, flat_annual=FLAT_ANNUAL, commission=FEE_RATE)
                    m = _metrics(trades, eq, rets, CAPITAL)

                    rows.append({
                        "n": n, "k": k, "ma_fast": mf, "ma_slow": ms,
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
        print(f"  N={n:>2} 完成 ({done}/{total})")

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
    print(f"  {DATA_NAME} — 吊灯止损 4 维参数网格扫描")
    print(f"{'='*70}")
    print(f"  规则: MA金叉开多, 跌破吊灯线离场; 空仓现金年化 {FLAT_ANNUAL:.1%}, 单边手续费 {FEE_RATE:.2%}")

    df_raw = load_data(DATA_FILE)
    print(f"  数据: {len(df_raw)} 条, {df_raw['trade_date'].iloc[0].date()} ~ {df_raw['trade_date'].iloc[-1].date()}")

    results = scan(df_raw)

    # 保存明细
    csv_path = f"{DATA_NAME}_吊灯止损参数扫描.csv"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  明细已保存: {csv_path}")

    # ---- Top 20 (按夏普) ----
    top = results.nlargest(20, "sharpe")
    print(f"\n{'='*100}")
    print(f"  Top 20 参数组合 (按夏普比率)")
    print(f"{'='*100}")
    print(f"  {'N':>3} {'K':>4} {'MA快':>4} {'MA慢':>5} | {'总收益':>8} {'年化':>8} {'回撤':>8} "
          f"{'夏普':>7} {'Calmar':>7} {'交易':>4} {'胜率':>7} {'盈亏比':>6}")
    for _, r in top.iterrows():
        flag = "  <-- 退化(交易太少)" if r["n_trades"] < MIN_TRADES else ""
        print(f"  {int(r['n']):>3} {r['k']:>4.1f} {int(r['ma_fast']):>4} {int(r['ma_slow']):>5} | "
              f"{r['total_return']:>8.1%} {r['annual_return']:>8.1%} {r['max_dd']:>8.1%} "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {int(r['n_trades']):>4} "
              f"{r['win_rate']:>7.1%} {r['profit_factor']:>6.2f}{flag}")

    # ---- 最优组合 (过滤退化) ----
    close = df_raw["close"].values
    bench_ret = close[-1] / close[0] - 1

    filtered = results[results["n_trades"] >= MIN_TRADES]
    print(f"\n  {'='*70}")
    print(f"  最优组合 (交易≥{MIN_TRADES}笔, 已过滤退化)    标的 Buy&Hold 总收益 = {bench_ret:.1%}")
    print(f"  {'='*70}")
    if len(filtered) == 0:
        print("  没有满足最低交易笔数的组合")
    else:
        for label, col in [("夏普最高", "sharpe"), ("Calmar最高", "calmar"), ("总收益最高", "total_return")]:
            b = filtered.iloc[filtered[col].idxmax()]
            print(f"  {label}: N={int(b['n'])}, K={b['k']:.1f}, MA快={int(b['ma_fast'])}, MA慢={int(b['ma_slow'])}  |  "
                  f"总收益={b['total_return']:.1%}  年化={b['annual_return']:.1%}  回撤={b['max_dd']:.1%}  "
                  f"夏普={b['sharpe']:.3f}  Calmar={b['calmar']:.3f}  交易={int(b['n_trades'])}笔")

    # ---- 热力图 (以过滤后的最优夏普为锚点) ----
    if len(filtered) > 0:
        anchor = filtered.iloc[filtered["sharpe"].idxmax()]
        be, bk = int(anchor["n"]), float(anchor["k"])
        bf, bs = int(anchor["ma_fast"]), int(anchor["ma_slow"])

        print(f"\n  生成热力图 (锚点: N={be}, K={bk}, MA{bf}/{bs})...")
        plot_grid(
            results, {"ma_fast": bf, "ma_slow": bs},
            x_key="n", y_key="k", x_vals=NS, y_vals=KS,
            title=f"{DATA_NAME} — 吊灯参数 N×K (MA快{bf}/慢{bs})",
            outpath=f"{DATA_NAME}_吊灯止损_热力图_NK.png",
        )
        plot_grid(
            results, {"n": be, "k": bk},
            x_key="ma_fast", y_key="ma_slow", x_vals=MA_FASTS, y_vals=MA_SLOWS,
            title=f"{DATA_NAME} — 入场MA参数 (N={be}, K={bk})",
            outpath=f"{DATA_NAME}_吊灯止损_热力图_MA.png",
        )

    print(f"\n  扫描完成。把 Top 组合的 N、K、MA快、MA慢 填回 chandelier_exit.py 的 __main__ 配置区即可复现。")
