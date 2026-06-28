"""Z-score mean-reversion backtest for a cointegrated pair.

Trades the Kalman-filtered spread with the classic band strategy: enter when the
spread's z-score is stretched (|z| >= entry), exit as it reverts toward the mean
(|z| <= exit). Positions are dollar-neutral — long one leg, short the other in
equal dollar amounts — so the strategy return each day is

    r_t = position_{t-1} * (ret_y_t - ret_x_t)

where ``ret`` are simple daily returns. "Long the spread" (position +1) means
long Y / short X; it profits when Y outperforms X, i.e. when a depressed spread
mean-reverts upward. Positions are lagged one bar so signals never peek at the
same day's return, and a per-turnover transaction cost is charged on changes.

The equity curve compounds these returns from a starting capital (default
$100,000). This is a deliberately simple, transparent backtest for research and
illustration — no slippage model, borrow cost, or capital constraints beyond
dollar-neutral sizing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Trade:
    """A single round-trip in the spread."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int          # +1 long spread, -1 short spread
    pnl: float              # strategy return contribution over the holding period


@dataclass
class BacktestResult:
    """Output of :func:`backtest_pair`."""

    equity: pd.Series           # dollar equity curve, starts at initial_capital
    returns: pd.Series          # daily strategy returns (net of costs)
    position: pd.Series         # held position per day (-1/0/+1), lagged for execution
    trades: list[Trade]
    initial_capital: float

    # Summary metrics
    final_equity: float
    total_return: float
    cagr: float
    ann_volatility: float
    sharpe: float
    max_drawdown: float
    num_trades: int
    win_rate: float

    def metrics_dict(self) -> dict[str, str]:
        """Human-readable metrics for display."""
        return {
            "Starting capital": f"${self.initial_capital:,.0f}",
            "Final equity": f"${self.final_equity:,.0f}",
            "Total return": f"{self.total_return * 100:,.1f}%",
            "CAGR": f"{self.cagr * 100:,.1f}%",
            "Annualized volatility": f"{self.ann_volatility * 100:,.1f}%",
            "Sharpe ratio": f"{self.sharpe:,.2f}",
            "Max drawdown": f"{self.max_drawdown * 100:,.1f}%",
            "Round-trip trades": f"{self.num_trades}",
            "Win rate": f"{self.win_rate * 100:,.0f}%" if self.num_trades else "—",
        }


def _build_positions(z: np.ndarray, entry: float, exit: float) -> np.ndarray:
    """Stateful band logic -> target position (-1/0/+1) for each bar.

    A high spread (z >= +entry) is shorted; a low spread (z <= -entry) is bought.
    Positions are held until the z-score reverts inside the ``exit`` band.
    """
    pos = np.zeros(len(z))
    state = 0
    for t in range(len(z)):
        zt = z[t]
        if np.isnan(zt):
            pos[t] = state
            continue
        if state == 0:
            if zt >= entry:
                state = -1
            elif zt <= -entry:
                state = 1
        elif state == 1 and zt >= -exit:   # long spread reverted up
            state = 0
        elif state == -1 and zt <= exit:   # short spread reverted down
            state = 0
        pos[t] = state
    return pos


def _extract_trades(
    dates: pd.DatetimeIndex,
    exec_pos: np.ndarray,
    strat_ret: np.ndarray,
) -> list[Trade]:
    """Group consecutive held bars into round-trip trades with summed P&L."""
    trades: list[Trade] = []
    i = 0
    n = len(exec_pos)
    while i < n:
        if exec_pos[i] == 0:
            i += 1
            continue
        direction = int(exec_pos[i])
        start = i
        pnl = 0.0
        while i < n and exec_pos[i] == direction:
            pnl += strat_ret[i]
            i += 1
        trades.append(
            Trade(
                entry_date=dates[start],
                exit_date=dates[min(i, n - 1)],
                direction=direction,
                pnl=float(pnl),
            )
        )
    return trades


def backtest_pair(
    y: pd.Series,
    x: pd.Series,
    zscore: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    initial_capital: float = 100_000.0,
    cost_bps: float = 1.0,
) -> BacktestResult:
    """Backtest the z-score mean-reversion strategy on a pair.

    Parameters
    ----------
    y, x:
        Price series for the two legs (same as fed to the Kalman filter).
    zscore:
        The Kalman-filtered spread's rolling z-score (trading signal).
    entry, exit:
        Z-score bands for entering and exiting positions.
    initial_capital:
        Starting equity in dollars (default $100,000).
    cost_bps:
        Round-turn transaction cost in basis points, charged on the change in
        dollar-neutral exposure whenever the position flips or closes.

    Returns
    -------
    BacktestResult with the equity curve, daily returns, trades and metrics.
    """
    df = pd.concat(
        [y.rename("y"), x.rename("x"), zscore.rename("z")], axis=1
    ).dropna()
    if len(df) < 3:
        raise ValueError("Not enough overlapping data to backtest.")

    z = df["z"].to_numpy()
    target = _build_positions(z, entry=entry, exit=exit)

    # Lag the position by one bar: act on today's signal at tomorrow's prices.
    exec_pos = np.empty_like(target)
    exec_pos[0] = 0.0
    exec_pos[1:] = target[:-1]

    ret_y = df["y"].pct_change().fillna(0.0).to_numpy()
    ret_x = df["x"].pct_change().fillna(0.0).to_numpy()
    gross = exec_pos * (ret_y - ret_x)

    # Transaction cost on turnover (gross exposure changes by |Δposition|, and a
    # dollar-neutral pair trades both legs, hence the factor of 2).
    turnover = np.abs(np.diff(exec_pos, prepend=0.0))
    cost = turnover * 2.0 * (cost_bps / 1e4)
    strat_ret = gross - cost

    equity = initial_capital * np.cumprod(1.0 + strat_ret)
    equity_s = pd.Series(equity, index=df.index, name="equity")
    returns_s = pd.Series(strat_ret, index=df.index, name="returns")
    position_s = pd.Series(exec_pos, index=df.index, name="position")

    trades = _extract_trades(df.index, exec_pos, strat_ret)

    final_equity = float(equity_s.iloc[-1])
    total_return = final_equity / initial_capital - 1.0
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
    cagr = (final_equity / initial_capital) ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0

    ann_vol = float(returns_s.std(ddof=0) * np.sqrt(TRADING_DAYS))
    mean_ann = float(returns_s.mean() * TRADING_DAYS)
    sharpe = mean_ann / ann_vol if ann_vol > 0 else 0.0

    running_max = equity_s.cummax()
    drawdown = equity_s / running_max - 1.0
    max_dd = float(drawdown.min())

    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / len(trades) if trades else 0.0

    return BacktestResult(
        equity=equity_s,
        returns=returns_s,
        position=position_s,
        trades=trades,
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_return=total_return,
        cagr=cagr,
        ann_volatility=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        num_trades=len(trades),
        win_rate=win_rate,
    )
