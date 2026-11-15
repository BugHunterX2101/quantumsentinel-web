"""QuantumSentinel — the statistics behind the validation policy's gates (R3-3).

The validation policy (``validation_policy``) judges a candidate strategy
version against every variant tried in its research family. This module
holds the statistics its gates use, computed from the trial matrix
(``trial_matrix``): T sessions by N distinct return series, each return
stored as the integer round(r * 10^12).

* Pure functions: no database, no clock, no randomness.
* Per-period units. A Sharpe ratio here is the mean over the standard
  deviation (ddof=1) of the series as it is, one value per session. Nothing
  is annualised, so no family's trading calendar (252 or 365 sessions a
  year) enters a statistic.
* Fail closed. Inputs on which a statistic is not defined (too few
  sessions, a constant series, a Sharpe ratio that is not positive where a
  positive one is needed, values that are not integer quanta) raise
  ``StatsError``; a gate that catches it fails. Nothing returns a neutral
  default instead. One deliberate exception: inside ``pbo_cscv`` a block in
  which a strategy's returns are constant (a strategy flat for a stretch is
  common) is ranked by the sign of its sum rather than refused, because the
  ranking, not a Sharpe ratio, is all CSCV needs from it; refusing would make
  PBO undefined for most real families.

What each statistic is, with its source (the formulas were checked against
the typeset papers):

* ``pbo_cscv``: the probability of backtest overfitting by combinatorially
  symmetric cross-validation (Bailey, Borwein, López de Prado and Zhu, "The
  Probability of Backtest Overfitting", Algorithm 2.3 and Section 3.1).
* ``expected_max`` and ``min_backtest_length``: the expected maximum of N
  independent standard normal draws (eq. 2.4) and the minimum backtest length
  (Theorem 3.1, eq. 3.2) of Bailey, Borwein, López de Prado and Zhu,
  "Pseudo-Mathematics and Financial Charlatanism", Notices of the AMS 61(5),
  2014.
* ``adjusted_p_values`` and ``haircut``: the multiple-testing adjustments
  (Bonferroni, Holm, Benjamini-Hochberg-Yekutieli) and the haircut Sharpe
  ratio of Harvey and Liu, "Backtesting", Journal of Portfolio Management,
  Fall 2015.

The deflated Sharpe ratio with an empirical trial variance and the number of
effectively independent trials (ONC clustering) are a separate change. The
columns these functions take are a trial matrix's (``trial_matrix.build``):
Python integers, oldest session first.

Cost. ``pbo_cscv`` is pure Python over C(S, S/2) combinations (12,870 at the
recommended S = 16), seconds for a family of a hundred variants; it runs in
the research worker, never in the API process.

Floating point. ``pbo_cscv`` ranks Sharpe ratios computed from exact integer
sums with only correctly rounded IEEE operations (+, -, *, /, sqrt), so its
counts are the same on every platform. The other statistics go through
library functions (the normal and t quantiles), which may differ in the last
bits between platforms; a gate compares them with a threshold, never for
equality.
"""
from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np

EULER_GAMMA = 0.5772156649015329
METHODS = ("bonferroni", "holm", "bhy")


class StatsError(ValueError):
    """Inputs on which a statistic is not defined. A gate that meets one fails."""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def _quanta(column) -> list[int]:
    """A return series as integer quanta (round(r * 10^12)), checked."""
    if isinstance(column, (str, bytes)) or not isinstance(column, (Sequence, np.ndarray)):
        raise StatsError("a return series is a sequence of integer quanta")
    values = []
    for value in column:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise StatsError("returns must be integer quanta, round(r * 10^12)")
        values.append(int(value))
    return values


def _count(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StatsError(f"{name} must be an integer of at least {minimum}")
    return value


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StatsError(f"{name} must be a finite number")
    return float(value)


def _positive_sharpe(value) -> float:
    sr = _finite(value, "the Sharpe ratio")
    if sr <= 0:
        raise StatsError("the Sharpe ratio must be positive")
    return sr


# --------------------------------------------------------------------------
# Sharpe ratio
# --------------------------------------------------------------------------

def _dispersion(n: int, total: int, squares: int) -> int:
    """n * sum(q^2) - (sum q)^2 = n (n - 1) times the sample variance of the
    quanta: an exact integer, zero only for a constant series."""
    return n * squares - total * total


def sharpe(column) -> float:
    """The per-period Sharpe ratio of a return series given as quanta: its
    mean over its standard deviation (ddof=1). A series of fewer than two
    sessions, or a constant one, has none."""
    values = _quanta(column)
    n = len(values)
    if n < 2:
        raise StatsError("a Sharpe ratio needs at least two sessions")
    total, squares = sum(values), sum(v * v for v in values)
    dispersion = _dispersion(n, total, squares)
    if dispersion == 0:
        raise StatsError("a constant series has no Sharpe ratio")
    # mean / std = (total / n) / sqrt(dispersion / (n (n - 1)))
    return total / math.sqrt(dispersion) * math.sqrt((n - 1) / n)


# --------------------------------------------------------------------------
# Expected maximum and minimum backtest length
# --------------------------------------------------------------------------

def expected_max(n_trials: int) -> float:
    """The expected maximum of ``n_trials`` independent standard normal
    draws (Pseudo-Mathematics, eq. 2.4):

        E[max_N] ~ (1 - gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - 1/(N e)),

    gamma the Euler-Mascheroni constant. The approximation is for large N;
    a single draw's expected value is exactly 0, which is returned for N=1.
    """
    n = _count(n_trials, "the number of trials", 1)
    if n == 1:
        return 0.0
    inv = NormalDist().inv_cdf
    return (1 - EULER_GAMMA) * inv(1 - 1 / n) + EULER_GAMMA * inv(1 - 1 / (n * math.e))


def min_backtest_length(n_trials: int, sharpe_per_period: float) -> float:
    """The minimum backtest length, in sessions, below which the best of
    ``n_trials`` skill-less strategies is expected to show a Sharpe ratio of
    ``sharpe_per_period`` (Pseudo-Mathematics, Theorem 3.1).

    The paper states it in years for an annualised Sharpe ratio, whose
    estimate has standard deviation y^(-1/2) after y years when the true
    ratio is zero: MinBTL = (E[max_N] / SR_annual)^2. A per-period estimate
    over T sessions has standard deviation T^(-1/2), so the same bound in
    sessions is (E[max_N] / SR_per_period)^2: the paper's figure times the
    sessions in a year, without choosing that number. It is a necessary, not
    a sufficient, condition (the paper's words) and needs a positive Sharpe
    ratio.
    """
    sr = _positive_sharpe(sharpe_per_period)
    return (expected_max(n_trials) / sr) ** 2


# --------------------------------------------------------------------------
# Multiple testing and the haircut Sharpe ratio
# --------------------------------------------------------------------------

def _p_values(p_values: Iterable) -> list[float]:
    values = []
    for p in p_values:
        if isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, float, np.floating)) or not 0 <= p <= 1:
            raise StatsError("a p-value is a number in [0, 1]")
        values.append(float(p))
    if not values:
        raise StatsError("there are no p-values to adjust")
    return values


def adjusted_p_values(p_values: Sequence[float], method: str) -> list[float]:
    """The p-values adjusted for testing them together, in their order
    (Harvey and Liu, "P-value adjustment under FWER" and "under FDR"); M is
    the number of p-values and p(1) <= ... <= p(M) their order.

    * ``bonferroni``: min(M p(i), 1).
    * ``holm``: min(max_{j<=i} (M - j + 1) p(j), 1).
    * ``bhy`` (Benjamini and Yekutieli 2001, valid under any dependence):
      min(min_{j>=i} M c(M) p(j) / j, 1), c(M) = sum_{j=1..M} 1/j. This is
      the published procedure. Harvey and Liu's version leaves the largest
      p-value unscaled (their worked example's last two values are 0.06,
      where this gives 0.1323 and 0.147); this one is never smaller, so it
      is the stricter of the two.
    """
    ps = _p_values(p_values)
    m = len(ps)
    if method == "bonferroni":
        return [min(m * p, 1.0) for p in ps]
    order = sorted(range(m), key=lambda i: (ps[i], i))
    adjusted = [0.0] * m
    if method == "holm":
        running = 0.0
        for rank, i in enumerate(order, start=1):
            running = max(running, (m - rank + 1) * ps[i])
            adjusted[i] = min(running, 1.0)
        return adjusted
    if method == "bhy":
        c = sum(1 / j for j in range(1, m + 1))
        running = 1.0
        for rank in range(m, 0, -1):
            i = order[rank - 1]
            running = min(running, m * c * ps[i] / rank)
            adjusted[i] = running
        return adjusted
    raise StatsError(f"the multiple-testing method is one of {', '.join(METHODS)}")


def p_value(sharpe_per_period: float, n_obs: int) -> float:
    """The two-sided p-value of a per-period Sharpe ratio over ``n_obs``
    sessions (Harvey and Liu, eq. 1 to 3): t = SR sqrt(T) follows a t
    distribution with T - 1 degrees of freedom when the mean return is zero."""
    from scipy.stats import t as student_t

    sr = _finite(sharpe_per_period, "the Sharpe ratio")
    n = _count(n_obs, "the number of sessions", 2)
    return float(min(1.0, 2 * student_t.sf(abs(sr) * math.sqrt(n), n - 1)))


def sharpe_for_p_value(p: float, n_obs: int) -> float:
    """The non-negative per-period Sharpe ratio whose two-sided p-value over
    ``n_obs`` sessions is ``p``: the inverse of ``p_value``."""
    from scipy.stats import t as student_t

    n = _count(n_obs, "the number of sessions", 2)
    (p,) = _p_values([p])
    if p == 0:
        return math.inf
    if p == 1:
        return 0.0
    return float(student_t.isf(p / 2, n - 1)) / math.sqrt(n)


def haircut(sharpe_per_period: float, n_obs: int, n_trials: int, method: str,
            other_p_values: Sequence[float] = ()) -> dict:
    """The haircut Sharpe ratio of a strategy chosen among ``n_trials``
    (Harvey and Liu): the Sharpe ratio whose single-test p-value equals the
    strategy's p-value adjusted for the ``n_trials`` tests by ``method``.

    Holm and BHY need every trial's p-value; ``other_p_values`` holds those
    known besides the candidate's. Trials whose p-value is not known (they
    left no return series to test) are taken to have p = 1, which never lowers
    the candidate's adjusted p-value under either method. Bonferroni needs
    only the count. The haircut is 1 - HSR / SR. The candidate must have a
    positive Sharpe ratio: the haircut of a losing strategy means nothing.
    """
    sr = _positive_sharpe(sharpe_per_period)
    n = _count(n_obs, "the number of sessions", 2)
    m = _count(n_trials, "the number of trials", 1)
    if method not in METHODS:
        raise StatsError(f"the multiple-testing method is one of {', '.join(METHODS)}")
    others = list(other_p_values)  # each one is checked by adjusted_p_values below
    if len(others) > m - 1:
        raise StatsError(f"{len(others)} other p-values for {m} trials")
    unseen = m - 1 - len(others)
    p = p_value(sr, n)
    adjusted = adjusted_p_values([p, *others, *([1.0] * unseen)], method)[0]
    hsr = min(sr, sharpe_for_p_value(adjusted, n))
    return {"method": method, "n_trials": m, "unseen_trials": unseen, "n_obs": n,
            "sharpe": sr, "t_statistic": sr * math.sqrt(n), "p_value": p,
            "adjusted_p_value": adjusted, "haircut_sharpe": hsr, "haircut": 1 - hsr / sr}


# --------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# --------------------------------------------------------------------------

def _rank_key(n: int, total: int, squares: int) -> float:
    """A value that orders the Sharpe ratios of series of ``n`` sessions:
    total / sqrt(dispersion), the Sharpe ratio without its constant factor
    sqrt((n - 1) / n). A constant series ranks by its sign: 0 if flat,
    +/- infinity otherwise (only ever compared, never reported)."""
    dispersion = _dispersion(n, total, squares)
    if dispersion == 0:
        return 0.0 if total == 0 else math.copysign(math.inf, total)
    return total / math.sqrt(dispersion)


def pbo_cscv(columns: Sequence[Sequence[int]], n_splits: int = 16) -> dict:
    """The probability of backtest overfitting of choosing the best of the
    ``columns`` (N return series of equal length, as quanta) by Sharpe ratio,
    estimated by combinatorially symmetric cross-validation (Bailey et al.,
    Algorithm 2.3).

    The sessions are cut into ``n_splits`` (S, even) consecutive blocks of
    equal length; when the length is not a multiple of S, the oldest
    T mod S sessions are left out. For each of the C(S, S/2) ways to choose
    half the blocks, the chosen blocks in their original order are the
    in-sample set and the rest the out-of-sample set. The strategy with the
    best in-sample Sharpe ratio is found, and its out-of-sample rank r
    (1 = worst, N = best) gives the relative rank w = r / (N + 1) and the
    logit lambda = ln(w / (1 - w)).

    PBO is the share of combinations with lambda <= 0, the mass of the
    logits' distribution on (-inf, 0] (Section 3.1). lambda <= 0 is
    w <= 1/2, that is 2r <= N + 1, which is counted in integers. The
    paper's Definition 2.2 writes the event as r < N/2; for N even that
    misses r = N/2, which this counts as overfitting, the stricter reading.

    Ties, which the paper does not treat, are resolved against the strategy:
    among strategies tied for the best in-sample Sharpe ratio the one with
    the worst out-of-sample Sharpe ratio is chosen, and its out-of-sample
    rank is 1 plus the number of strategies strictly worse (the lowest rank
    among those tied with it, not the average).

    Every Sharpe ratio is computed from exact integer sums, so the counts
    are the same on every platform. The paper recommends S = 16 (12,870
    combinations; its text says 12,780, a misprint of C(16, 8)).
    """
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise StatsError("the strategies' return series are a sequence of columns")
    series = [_quanta(column) for column in columns]
    n_strategies = len(series)
    if n_strategies < 2:
        raise StatsError("overfitting a choice needs at least two strategies to choose from")
    splits = _count(n_splits, "the number of splits", 2)
    if splits % 2:
        raise StatsError("the number of splits must be even")
    rows = len(series[0])
    if any(len(column) != rows for column in series):
        raise StatsError("every strategy needs a return for every session")
    block = rows // splits
    half = block * splits // 2
    if half < 2:
        raise StatsError(f"{rows} sessions are too few for {splits} splits")
    dropped = rows - block * splits

    totals = np.empty((splits, n_strategies), dtype=object)
    squares = np.empty((splits, n_strategies), dtype=object)
    for n, column in enumerate(series):
        for s in range(splits):
            values = column[dropped + s * block: dropped + (s + 1) * block]
            totals[s, n] = sum(values)
            squares[s, n] = sum(v * v for v in values)
    chosen = np.array(list(combinations(range(splits), splits // 2)))
    in_totals, in_squares = totals[chosen].sum(axis=1), squares[chosen].sum(axis=1)
    out_totals, out_squares = totals.sum(axis=0) - in_totals, squares.sum(axis=0) - in_squares

    overfit = 0
    for c in range(len(chosen)):
        in_key = [_rank_key(half, in_totals[c, n], in_squares[c, n]) for n in range(n_strategies)]
        out_key = [_rank_key(half, out_totals[c, n], out_squares[c, n]) for n in range(n_strategies)]
        best = max(in_key)
        pick = min((n for n in range(n_strategies) if in_key[n] == best), key=lambda n: (out_key[n], n))
        rank = 1 + sum(1 for n in range(n_strategies) if out_key[n] < out_key[pick])
        if 2 * rank <= n_strategies + 1:
            overfit += 1
    return {"n_splits": splits, "rows": rows, "rows_used": block * splits, "rows_dropped": dropped,
            "strategies": n_strategies, "combinations": len(chosen), "overfit": overfit,
            "pbo": overfit / len(chosen)}
