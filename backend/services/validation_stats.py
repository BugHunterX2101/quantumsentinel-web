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

* ``deflated_sharpe`` (and ``deflated_sharpe_ratio``, ``probabilistic_sharpe``,
  ``moments``, ``implied_trials``): the deflated Sharpe ratio of Bailey and
  López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias,
  Backtest Overfitting and Non-Normality", Journal of Portfolio Management
  40(5), 2014, eq. (1) and (2), with the number of effectively independent
  trials implied by the trials' average correlation (its Appendix A.3,
  eq. 8 and 9).

The columns these functions take are a trial matrix's (``trial_matrix.build``):
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


# --------------------------------------------------------------------------
# Deflated Sharpe ratio
# --------------------------------------------------------------------------

def moments(column) -> dict:
    """The skewness and the kurtosis of a return series given as quanta, as
    the moments of its sessions: m3 / m2^(3/2) and m4 / m2^2, m_k the k-th
    central moment with divisor n (scipy's ``skew`` and
    ``kurtosis(fisher=False)`` with their default bias=True). The kurtosis is
    plain, not excess: 3 for normal returns, as the deflated Sharpe ratio's
    eq. (2) takes it. Both come from exact integer power sums, so they are
    the same on every platform; a constant series has neither."""
    values = _quanta(column)
    n = len(values)
    if n < 2:
        raise StatsError("moments need at least two sessions")
    s1 = sum(values)
    s2 = sum(v * v for v in values)
    s3 = sum(v * v * v for v in values)
    s4 = sum(v * v * v * v for v in values)
    dispersion = _dispersion(n, s1, s2)  # n^2 m2
    if dispersion == 0:
        raise StatsError("a constant series has no skewness or kurtosis")
    third = n * n * s3 - 3 * n * s1 * s2 + 2 * s1 ** 3  # n^3 m3
    fourth = n ** 3 * s4 - 4 * n * n * s1 * s3 + 6 * n * s1 * s1 * s2 - 3 * s1 ** 4  # n^4 m4
    return {"skewness": third / dispersion / math.sqrt(dispersion),
            "kurtosis": fourth / (dispersion * dispersion)}


def _expected_max_real(n_trials: float) -> float:
    """``expected_max`` for a real number of trials N >= 1, as an implied
    number of independent trials is. As N falls towards 1 the large-N
    approximation falls to minus infinity, where the true value is 0; the
    expected maximum of draws with mean 0 is never negative, so the
    approximation is floored at 0 (and N = 1 gives 0, as ``expected_max``).
    For integer N >= 2 it equals ``expected_max``."""
    n = _finite(n_trials, "the number of trials")
    if n < 1:
        raise StatsError("the number of trials must be at least 1")
    if n == 1:
        return 0.0
    inv = NormalDist().inv_cdf
    return max(0.0, (1 - EULER_GAMMA) * inv(1 - 1 / n) + EULER_GAMMA * inv(1 - 1 / (n * math.e)))


def probabilistic_sharpe(sharpe_per_period: float, benchmark: float, n_obs: int,
                         skewness: float, kurtosis: float) -> float:
    """The probability that the true Sharpe ratio exceeds ``benchmark``,
    given an estimate over ``n_obs`` sessions with that skewness and plain
    kurtosis (Deflated Sharpe Ratio, eq. 2):

        Z[(SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2)].

    The square root's argument is positive for the moments of any sample
    (g4 >= g3^2 + 1); moments that make it zero or negative are refused."""
    sr = _finite(sharpe_per_period, "the Sharpe ratio")
    sr0 = _finite(benchmark, "the benchmark Sharpe ratio")
    n = _count(n_obs, "the number of sessions", 2)
    g3, g4 = _finite(skewness, "the skewness"), _finite(kurtosis, "the kurtosis")
    spread = 1 - g3 * sr + (g4 - 1) / 4 * sr * sr
    if not spread > 0:
        raise StatsError("these moments give the Sharpe ratio's estimate no variance")
    return NormalDist().cdf((sr - sr0) * math.sqrt(n - 1) / math.sqrt(spread))


def deflated_sharpe_ratio(sharpe_per_period: float, n_obs: int, n_trials: float, variance: float,
                          skewness: float, kurtosis: float) -> dict:
    """The deflated Sharpe ratio (Deflated Sharpe Ratio, eq. 1 and 2): the
    probabilistic Sharpe ratio against SR0, the Sharpe ratio expected of the
    best of ``n_trials`` skill-less trials whose Sharpe ratios have
    ``variance`` V across trials,

        SR0 = sqrt(V) ((1 - gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - 1/(N e))).

    ``n_trials`` may be fractional (an implied number of trials)."""
    v = _finite(variance, "the variance of the trials' Sharpe ratios")
    if v <= 0:
        raise StatsError("the variance of the trials' Sharpe ratios must be positive")
    sr0 = math.sqrt(v) * _expected_max_real(n_trials)
    dsr = probabilistic_sharpe(sharpe_per_period, sr0, n_obs, skewness, kurtosis)
    return {"sharpe": float(sharpe_per_period), "n_obs": n_obs, "n_trials": float(n_trials),
            "variance": v, "skewness": float(skewness), "kurtosis": float(kurtosis),
            "benchmark_sharpe": sr0, "dsr": dsr}


def _distinct_columns(columns) -> list[list[int]]:
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise StatsError("the trials' return series are a sequence of columns")
    series = [_quanta(column) for column in columns]
    if not series:
        raise StatsError("there are no return series")
    rows = len(series[0])
    if any(len(column) != rows for column in series):
        raise StatsError("every trial needs a return for every session")
    if len({tuple(column) for column in series}) != len(series):
        raise StatsError("the columns must be distinct: a matrix stores each return series once")
    return series


def implied_trials(columns: Sequence[Sequence[int]], n_trials: int) -> dict:
    """The number of effectively independent trials implied by the average
    correlation of the trials' return series (Deflated Sharpe Ratio,
    Appendix A.3): rho is the mean of the correlations between distinct
    columns (eq. 8) and N = rho + (1 - rho) M (eq. 9), M = ``n_trials``.

    The correlations are those of the distinct columns only: versions that
    share a column are counted in M but add no correlation of 1, so they do
    not pull N down. N falls to 1 as rho rises to 1, and exceeds M when rho
    is negative; it is not capped, a larger N being the stricter. The
    correlations come from exact integer sums; each is clamped to [-1, 1],
    which rounding can overstep by an ulp (a column and 13 times it plus a
    constant give 1.0000000000000002). With the clamp rho <= 1 and, rounding
    being monotone, N >= 1 holds in floating point too. A constant column
    has no correlation and is refused, as are fewer than two columns."""
    series = _distinct_columns(columns)
    m = _count(n_trials, "the number of trials", 1)
    k = len(series)
    if k < 2:
        raise StatsError("an average correlation needs at least two distinct return series")
    if m < k:
        raise StatsError(f"{m} trials cannot have {k} distinct return series")
    n = len(series[0])
    if n < 2:
        raise StatsError("a correlation needs at least two sessions")
    totals = [sum(column) for column in series]
    dispersions = [_dispersion(n, totals[i], sum(v * v for v in series[i])) for i in range(k)]
    if any(d == 0 for d in dispersions):
        raise StatsError("a constant return series has no correlation")
    pairs = 0.0
    for i, j in combinations(range(k), 2):
        cross = n * sum(a * b for a, b in zip(series[i], series[j])) - totals[i] * totals[j]
        pairs += max(-1.0, min(1.0, cross / math.sqrt(dispersions[i] * dispersions[j])))
    rho = 2 * pairs / (k * (k - 1))
    return {"distinct_series": k, "n_trials": m, "average_correlation": rho,
            "implied_trials": rho + (1 - rho) * m}


def deflated_sharpe(columns: Sequence[Sequence[int]], candidate: int, n_trials: int,
                    every_trial_evaluated: bool) -> dict:
    """The deflated Sharpe ratio of the trial matrix column ``candidate``
    (its index in ``columns``, the matrix's distinct return series), chosen
    among ``n_trials`` trials: the family's ledger count, in which every
    strategy version is counted, those that share a column included.

    * The number of trials is the one implied by the columns' average
      correlation (``implied_trials``) when every trial in the ledger has a
      column (``every_trial_evaluated``, from the matrix's ``ledger``).
      Otherwise the trials without one are unknown and the raw ledger count
      is used, as it is when there is a single distinct column.
    * The variance of the trials' Sharpe ratios is V = max(V_null, V_emp).
      V_emp is the sample variance (ddof=1) of the distinct columns'
      per-period Sharpe ratios. V_null = 1 / (T - 1) is the variance of a
      Sharpe ratio's estimate over T sessions when the true ratio is 0
      (eq. 2's denominator at SR = 0): the dispersion skill-less trials show
      by chance alone. The floor keeps a family of near-identical variants
      from deflating nothing. With one distinct column V = V_null.
    * The candidate's Sharpe ratio is ``sharpe`` of its column (ddof=1); its
      skewness and plain kurtosis are ``moments``.
    """
    series = _distinct_columns(columns)
    if isinstance(candidate, bool) or not isinstance(candidate, int) or not 0 <= candidate < len(series):
        raise StatsError("the candidate is the index of one of the columns")
    if not isinstance(every_trial_evaluated, bool):
        raise StatsError("every_trial_evaluated must be true or false")
    m = _count(n_trials, "the number of trials", 1)
    k = len(series)
    if m < k:
        raise StatsError(f"{m} trials cannot have {k} distinct return series")
    sessions = len(series[0])
    sharpes = [sharpe(column) for column in series]
    v_null = 1 / (sessions - 1)
    v_emp = float(np.var(sharpes, ddof=1)) if k > 1 else None
    if every_trial_evaluated and k > 1:
        implied = implied_trials(series, m)
        used, source, rho = implied["implied_trials"], "implied", implied["average_correlation"]
    else:
        used, source, rho = float(m), "ledger", None
    shape = moments(series[candidate])
    variance = v_null if v_emp is None else max(v_null, v_emp)
    result = deflated_sharpe_ratio(sharpes[candidate], sessions, used, variance,
                                   shape["skewness"], shape["kurtosis"])
    return {**result, "ledger_trials": m, "distinct_series": k, "n_trials_source": source,
            "average_correlation": rho, "variance_null": v_null, "variance_empirical": v_emp}
