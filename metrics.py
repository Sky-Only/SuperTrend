"""
Strategy performance metrics calculation module.

所有函数均为纯函数（无副作用），以 numpy 数组作为输入/输出。
本模块独立于回测引擎、SuperTrend 计算和绘图模块，可单独测试和验证。

使用方式:
    import metrics as mt
    result = mt.calc_all_metrics(equity_curve, daily_returns, trade_pnl_pcts, capital)
"""

import numpy as np

# 默认年化交易日数
TRADING_DAYS_PER_YEAR = 252
# 默认无风险利率
RISK_FREE_RATE = 0.03


# ============================================================
# 单指标计算函数
# ============================================================

def calc_total_return(final_equity: float, initial_capital: float) -> float:
    """
    总收益率
    公式: (最终净值 - 初始资金) / 初始资金
    """
    if initial_capital <= 0:
        return 0.0
    return (final_equity - initial_capital) / initial_capital


def calc_annual_return(total_return: float, n_days: int,
                       trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    年化收益率 (CAGR)
    公式: (1 + total_return)^(1 / n_years) - 1
    当 total_return <= -1 时（本金亏完），使用线性近似。
    """
    n_years = n_days / trading_days
    if n_years <= 0:
        return 0.0
    if total_return > -1.0:
        return (1.0 + total_return) ** (1.0 / n_years) - 1.0
    else:
        # 本金亏完或更差时，用线性近似
        return total_return / n_years


def calc_max_drawdown(equity_curve: np.ndarray) -> tuple:
    """
    最大回撤 & 最长回撤持续天数

    参数:
        equity_curve: 1D 净值曲线数组

    返回:
        (max_drawdown, max_drawdown_days)
        max_drawdown: 负值，例如 -0.25 表示 25% 回撤
        max_drawdown_days: 最长连续回撤天数
    """
    n = len(equity_curve)
    if n == 0:
        return 0.0, 0

    # 最大回撤
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = float(drawdown.min()) if peak.max() > 0 else 0.0

    # 最长回撤持续天数（连续低于前高的天数）
    max_dd_days = 0
    in_dd = False
    cur_dd_days = 0
    for i in range(n):
        if equity_curve[i] >= peak[i]:
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

    return max_dd, max_dd_days


def calc_sharpe_ratio(daily_returns: np.ndarray,
                      risk_free_rate: float = RISK_FREE_RATE,
                      trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    年化夏普比率
    公式: sqrt(252) * mean(daily_returns - rf_daily) / std(daily_returns - rf_daily)

    参数:
        daily_returns: 日收益率序列
        risk_free_rate: 年化无风险利率 (默认 3%)
        trading_days: 年交易日数 (默认 252)
    """
    if len(daily_returns) < 2:
        return 0.0

    rf_daily = risk_free_rate / trading_days
    excess = daily_returns - rf_daily
    std_excess = excess.std()
    if std_excess > 0:
        return float(np.sqrt(trading_days) * excess.mean() / std_excess)
    return 0.0


def calc_annual_volatility(daily_returns: np.ndarray,
                           trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    年化波动率
    公式: std(daily_returns) * sqrt(252)
    """
    if len(daily_returns) > 1:
        return float(daily_returns.std() * np.sqrt(trading_days))
    return 0.0


def calc_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """
    卡尔玛比率 (Calmar Ratio)
    公式: 年化收益率 / |最大回撤|
    衡量承受每单位回撤所获得的年化收益。
    """
    if abs(max_drawdown) > 1e-15:
        return annual_return / abs(max_drawdown)
    return 0.0


def calc_trade_stats(trade_pnl_pcts: np.ndarray,
                     hold_days: np.ndarray | None = None) -> dict:
    """
    交易层面统计指标

    参数:
        trade_pnl_pcts: 每笔交易的收益率序列 (正=盈利, 负/零=亏损)
        hold_days: 每笔交易的持仓天数 (可选)

    返回:
        dict 包含:
            n_trades:    交易次数
            win_rate:    胜率 (盈利交易数 / 总交易数)
            avg_win:     平均盈利 (盈利交易的平均收益率)
            avg_loss:    平均亏损 (亏损交易的平均收益率, 负值或零)
            profit_factor: 盈亏比 (|avg_win / avg_loss|)
            avg_hold_days: 平均持仓天数 (仅当 hold_days 传入时)
    """
    n_trades = len(trade_pnl_pcts)

    if n_trades == 0:
        result = {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }
        if hold_days is not None:
            result["avg_hold_days"] = 0.0
        return result

    wins = trade_pnl_pcts[trade_pnl_pcts > 0]
    losses = trade_pnl_pcts[trade_pnl_pcts <= 0]

    n_wins = len(wins)
    n_losses = len(losses)

    win_rate = n_wins / n_trades
    avg_win = float(wins.mean()) if n_wins > 0 else 0.0
    avg_loss = float(losses.mean()) if n_losses > 0 else 0.0
    profit_factor = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-15 else (
        10.0 if avg_win > 0 else 0.0
    )

    result = {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }

    if hold_days is not None and len(hold_days) > 0:
        result["avg_hold_days"] = float(hold_days.mean())
    elif hold_days is not None:
        result["avg_hold_days"] = 0.0

    return result


# ============================================================
# 聚合计算函数
# ============================================================

def calc_all_metrics(equity_curve: np.ndarray,
                     daily_returns: np.ndarray,
                     trade_pnl_pcts: np.ndarray,
                     initial_capital: float,
                     risk_free_rate: float = RISK_FREE_RATE,
                     trading_days: int = TRADING_DAYS_PER_YEAR,
                     hold_days: np.ndarray | None = None) -> dict:
    """
    计算全部策略绩效指标。

    参数:
        equity_curve:    1D 净值曲线数组 (长度 = n_days)
        daily_returns:   1D 日收益率数组 (长度 = n_days, 首日可为 0)
        trade_pnl_pcts:  1D 每笔交易收益率数组
        initial_capital: 初始资金
        risk_free_rate:  年化无风险利率 (默认 3%)
        trading_days:    年交易日数 (默认 252)
        hold_days:       每笔交易持仓天数数组 (可选)

    返回:
        dict 包含所有指标（值均为原始 float/int，未经格式化）
    """
    n_days = len(daily_returns)

    total_return = calc_total_return(float(equity_curve[-1]), initial_capital)
    annual_return = calc_annual_return(total_return, n_days, trading_days)
    max_dd, max_dd_days = calc_max_drawdown(equity_curve)
    sharpe = calc_sharpe_ratio(daily_returns, risk_free_rate, trading_days)
    annual_vol = calc_annual_volatility(daily_returns, trading_days)
    calmar = calc_calmar_ratio(annual_return, max_dd)
    trade_stats = calc_trade_stats(trade_pnl_pcts, hold_days)

    return {
        "initial_capital": initial_capital,
        "final_equity": round(float(equity_curve[-1]), 2),
        "total_return": total_return,
        "annual_return": annual_return,
        "max_dd": max_dd,
        "max_dd_days": max_dd_days,
        "daily_vol": annual_vol,
        "sharpe": sharpe,
        "calmar": calmar,
        "n_trades": trade_stats["n_trades"],
        "win_rate": trade_stats["win_rate"],
        "avg_win": trade_stats["avg_win"],
        "avg_loss": trade_stats["avg_loss"],
        "profit_factor": trade_stats["profit_factor"],
        "avg_hold_days": trade_stats.get("avg_hold_days", 0.0),
        "n_days": n_days,
    }


# ============================================================
# 辅助：格式化输出
# ============================================================

def format_metrics(metrics: dict) -> dict:
    """
    将原始指标值格式化为可读字符串，便于打印和展示。
    输入是 calc_all_metrics 返回的 dict。
    """
    return {
        "初始资金": metrics["initial_capital"],
        "最终净值": metrics["final_equity"],
        "总收益率": f"{metrics['total_return']:.2%}",
        "年化收益率": f"{metrics['annual_return']:.2%}",
        "最大回撤": f"{metrics['max_dd']:.2%}",
        "最大回撤持续天数": metrics["max_dd_days"],
        "年化波动率": f"{metrics['daily_vol']:.2%}",
        "夏普比率": round(metrics["sharpe"], 2),
        "卡尔玛比率": round(metrics["calmar"], 2),
        "交易次数": metrics["n_trades"],
        "胜率": f"{metrics['win_rate']:.2%}",
        "平均盈利": f"{metrics['avg_win']:.2%}",
        "平均亏损": f"{metrics['avg_loss']:.2%}",
        "盈亏比": round(metrics["profit_factor"], 2),
        "平均持仓天数": round(metrics["avg_hold_days"], 1),
        "回测天数": metrics["n_days"],
    }
