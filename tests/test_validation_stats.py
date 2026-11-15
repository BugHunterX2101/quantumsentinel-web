"""The statistics behind the validation policy's gates
(services/validation_stats.py), checked against the worked examples of the
papers they come from, against independent implementations, and on every
input where they must refuse to give a number."""
import math
import random
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pytest
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

from backend.services import research_trials, stat_tests, trial_matrix, validation_stats as vs
from backend.services.validation_stats import StatsError

Q = 10 ** 12  # quanta per unit of return


def noise(rows, columns, seed, scale=10 ** 10):
    rng = random.Random(seed)
    return [[rng.randint(-scale, scale) for _ in range(rows)] for _ in range(columns)]


# --------------------------------------------------------------------------
# Sharpe ratio
# --------------------------------------------------------------------------

def test_sharpe_is_the_mean_over_the_sample_standard_deviation():
    (column,) = noise(250, 1, seed=3)
    column = [q + 2 * 10 ** 8 for q in column]
    returns = np.array(column, dtype=float) / Q
    assert vs.sharpe(column) == pytest.approx(returns.mean() / returns.std(ddof=1), rel=1e-12)
    assert vs.sharpe(np.array(column, dtype=np.int64)) == vs.sharpe(column)


@pytest.mark.parametrize("column,reason", [
    ([5], "at least two sessions"),
    ([], "at least two sessions"),
    ([7, 7, 7], "constant series"),
    ([0, 0], "constant series"),
    ([1, 2.0, 3], "integer quanta"),
    ([1, True, 3], "integer quanta"),
    ("123", "sequence of integer quanta"),
    (None, "sequence of integer quanta"),
])
def test_sharpe_refuses_a_series_that_has_none(column, reason):
    with pytest.raises(StatsError, match=reason):
        vs.sharpe(column)


# --------------------------------------------------------------------------
# Expected maximum and minimum backtest length (Pseudo-Mathematics)
# --------------------------------------------------------------------------

def test_expected_max_of_ten_trials_is_the_papers_1_57():
    """'if the researcher tries only N = 10 alternative configurations ... he
    or she is expected to find a strategy with a Sharpe ratio IS of 1.57'."""
    assert vs.expected_max(10) == pytest.approx(1.5745983013, rel=1e-9)
    assert round(vs.expected_max(10), 2) == 1.57


def test_expected_max_grows_with_the_trials_and_stays_below_its_bound():
    values = [vs.expected_max(n) for n in range(2, 2001)]
    assert all(a < b for a, b in zip(values, values[1:]))
    assert all(v < math.sqrt(2 * math.log(n)) for n, v in zip(range(10, 2001), values[8:]))


def test_one_trial_has_no_selection_and_an_expected_maximum_of_zero():
    assert vs.expected_max(1) == 0.0
    assert vs.min_backtest_length(1, 0.05) == 0.0


@pytest.mark.parametrize("n", [0, -3, 2.0, True, None, "10"])
def test_expected_max_refuses_a_count_that_is_not_one(n):
    with pytest.raises(StatsError, match="number of trials"):
        vs.expected_max(n)


def test_min_backtest_length_reproduces_five_years_for_45_trials():
    """'if only 5 years of data are available, no more than 45 independent
    model configurations should be tried' (annualised Sharpe ratio 1). In
    sessions, with 252 a year: 45 trials need under 5 x 252 sessions, 46 more."""
    year = 252
    sr = 1 / math.sqrt(year)
    assert vs.min_backtest_length(45, sr) == pytest.approx(vs.expected_max(45) ** 2 * year, rel=1e-12)
    assert vs.expected_max(45) ** 2 == pytest.approx(4.998087, rel=1e-6)
    assert vs.min_backtest_length(45, sr) < 5 * year < vs.min_backtest_length(46, sr)


def test_min_backtest_length_is_the_squared_ratio_of_expected_max_to_sharpe():
    assert vs.min_backtest_length(20, 0.1) == pytest.approx((vs.expected_max(20) / 0.1) ** 2, rel=1e-12)
    assert vs.min_backtest_length(20, 0.05) == pytest.approx(4 * vs.min_backtest_length(20, 0.1), rel=1e-12)


@pytest.mark.parametrize("sr,reason", [
    (0.0, "positive"), (-0.1, "positive"), (math.nan, "finite"), (math.inf, "finite"),
    (True, "finite"), ("0.1", "finite"), (None, "finite"),
])
def test_min_backtest_length_needs_a_positive_sharpe_ratio(sr, reason):
    with pytest.raises(StatsError, match=reason):
        vs.min_backtest_length(10, sr)


# --------------------------------------------------------------------------
# Multiple testing (Harvey and Liu, "Backtesting")
# --------------------------------------------------------------------------

HL_P = [0.005, 0.009, 0.0128, 0.0135, 0.045, 0.06]


def test_bonferroni_and_holm_reproduce_harvey_and_liu():
    assert vs.adjusted_p_values(HL_P, "bonferroni") == pytest.approx([0.03, 0.054, 0.0768, 0.081, 0.27, 0.36])
    assert vs.adjusted_p_values(HL_P, "holm") == pytest.approx([0.03, 0.045, 0.0512, 0.0512, 0.09, 0.09])


def test_bhy_is_the_published_procedure_not_the_papers_variant():
    """Harvey and Liu's first four BHY values are 0.0496; c(6) = 49/20. Their
    last two (0.06, 0.06) leave the largest p-value unscaled; the published
    procedure scales it, giving 6 c(6) p / rank."""
    c6 = 49 / 20
    expected = [6 * c6 * 0.0135 / 4] * 4 + [6 * c6 * 0.045 / 5, 6 * c6 * 0.06 / 6]
    assert vs.adjusted_p_values(HL_P, "bhy") == pytest.approx(expected, rel=1e-12)
    assert round(expected[0], 4) == 0.0496 and expected[4:] == pytest.approx([0.1323, 0.147])


@pytest.mark.parametrize("method,statsmodels_method", [
    ("bonferroni", "bonferroni"), ("holm", "holm"), ("bhy", "fdr_by")])
def test_adjustments_agree_with_statsmodels(method, statsmodels_method):
    rng = random.Random(11)
    for size in (1, 2, 7, 40):
        ps = [rng.random() ** 3 for _ in range(size)]
        ours = vs.adjusted_p_values(ps, method)
        theirs = multipletests(ps, method=statsmodels_method)[1]
        assert ours == pytest.approx(list(theirs), rel=1e-12, abs=1e-15)


def test_adjusted_p_values_keep_their_order_and_stay_at_most_one():
    ps = [0.6, 0.01, 0.9, 0.2]
    for method in vs.METHODS:
        adjusted = vs.adjusted_p_values(ps, method)
        assert all(a >= p for a, p in zip(adjusted, ps)) and max(adjusted) <= 1.0
        order = sorted(range(4), key=lambda i: ps[i])
        assert [adjusted[i] for i in order] == sorted(adjusted)


@pytest.mark.parametrize("ps,method,reason", [
    ([], "holm", "no p-values"),
    ([0.1, -0.01], "holm", r"\[0, 1\]"),
    ([0.1, 1.01], "bhy", r"\[0, 1\]"),
    ([0.1, math.nan], "bonferroni", r"\[0, 1\]"),
    ([0.1, True], "holm", r"\[0, 1\]"),
    ([0.1, "0.2"], "holm", r"\[0, 1\]"),
    ([0.1, 0.2], "hochberg", "one of bonferroni, holm, bhy"),
])
def test_adjusted_p_values_refuse_what_is_not_a_set_of_p_values(ps, method, reason):
    with pytest.raises(StatsError, match=reason):
        vs.adjusted_p_values(ps, method)


def test_benjamini_yekutieli_reports_beside_an_unchanged_benjamini_hochberg():
    bh = stat_tests.benjamini_hochberg(HL_P)
    by = stat_tests.benjamini_yekutieli(HL_P)
    assert bh["adjusted_p_values"] == [0.02025, 0.02025, 0.02025, 0.02025, 0.054, 0.06]
    assert by["method"] == "benjamini_yekutieli" and by["n_tests"] == 6
    assert by["adjusted_p_values"] == [round(p, 6) for p in vs.adjusted_p_values(HL_P, "bhy")]
    assert by["significant_indices"] == [0, 1, 2, 3] and by["n_significant"] == 4
    assert stat_tests.benjamini_yekutieli([]) == {"significant": [], "n_tests": 0}
    # Significant means strictly below alpha, as for BH.
    for method in (stat_tests.benjamini_hochberg, stat_tests.benjamini_yekutieli):
        assert method([0.05], alpha=0.05)["significant_indices"] == []
        assert method([0.0499], alpha=0.05)["significant_indices"] == [0]

    rng = np.random.default_rng(0)
    full = stat_tests.run_full_stat_tests(rng.normal(0.001, 0.01, 120), strategy_p_values=HL_P,
                                          n_bootstrap=200, n_permutations=200)
    assert full["benjamini_hochberg"] == bh and full["benjamini_yekutieli"] == by


# --------------------------------------------------------------------------
# The haircut Sharpe ratio
# --------------------------------------------------------------------------

def test_sharpe_to_p_value_follows_harvey_and_liu():
    """Twenty years of monthly returns (T = 240), annual Sharpe ratio 0.75.
    The paper quotes p = 0.0008, the normal approximation (two-sided,
    t = 0.75 sqrt(20)); its stated method, a t distribution with T - 1
    degrees of freedom, gives 0.000926, which is what this uses (stricter).
    An adjusted p-value of 0.15 gives the paper's haircut Sharpe ratio 0.32."""
    sr = 0.75 / math.sqrt(12)
    t = 0.75 * math.sqrt(20)
    normal = 2 * (1 - NormalDist().cdf(t))
    assert round(normal, 4) == 0.0008
    assert vs.p_value(sr, 240) == pytest.approx(0.000926, rel=1e-3)
    assert vs.p_value(sr, 240) > normal
    assert round(vs.sharpe_for_p_value(0.15, 240) * math.sqrt(12), 2) == 0.32


@pytest.mark.parametrize("p", [1.5, -0.1, math.nan, True, "0.05"])
def test_sharpe_for_p_value_refuses_what_is_not_a_p_value(p):
    with pytest.raises(StatsError, match=r"\[0, 1\]"):
        vs.sharpe_for_p_value(p, 300)


@pytest.mark.parametrize("n", [1, 0, 300.0, True])
def test_p_value_and_its_inverse_refuse_too_few_sessions(n):
    with pytest.raises(StatsError, match="number of sessions"):
        vs.p_value(0.1, n)
    with pytest.raises(StatsError, match="number of sessions"):
        vs.sharpe_for_p_value(0.05, n)


def test_the_p_value_is_two_sided():
    for sr in (0.01, 0.1, 0.3):
        assert vs.p_value(-sr, 250) == vs.p_value(sr, 250) < 1


def test_the_haircut_is_never_negative_even_by_rounding():
    """With one trial the adjusted p-value is the p-value, and inverting it
    lands on either side of the Sharpe ratio by rounding (about a quarter of
    the time above). The haircut Sharpe ratio is capped at the Sharpe ratio,
    so the haircut is never below zero."""
    for k in range(1, 400):
        for n in (60, 252, 1000):
            result = vs.haircut(k / 2000, n, 1, "bhy")
            assert result["haircut_sharpe"] <= result["sharpe"] and result["haircut"] >= 0


def test_sharpe_for_p_value_inverts_p_value():
    for p in (1e-9, 0.003, 0.05, 0.5, 0.99):
        assert vs.p_value(vs.sharpe_for_p_value(p, 300), 300) == pytest.approx(p, rel=1e-9)
    assert vs.sharpe_for_p_value(1.0, 300) == 0.0
    assert vs.sharpe_for_p_value(0.0, 300) == math.inf


def test_a_single_trial_takes_no_haircut():
    result = vs.haircut(0.12, 252, 1, "bhy")
    assert result["adjusted_p_value"] == pytest.approx(result["p_value"], rel=1e-12)
    assert result["haircut_sharpe"] == pytest.approx(0.12, rel=1e-9)
    assert result["haircut"] == pytest.approx(0.0, abs=1e-9)
    assert result["t_statistic"] == pytest.approx(0.12 * math.sqrt(252))


def test_bonferroni_haircut_needs_only_the_count():
    result = vs.haircut(0.1, 500, 30, "bonferroni")
    assert result["adjusted_p_value"] == pytest.approx(min(1.0, 30 * vs.p_value(0.1, 500)))
    assert result["haircut_sharpe"] == pytest.approx(vs.sharpe_for_p_value(result["adjusted_p_value"], 500))
    assert 0 < result["haircut"] < 1 and result["unseen_trials"] == 29


def test_unseen_trials_count_as_p_one():
    """With no other p-value known, Holm puts the candidate first (Bonferroni's
    factor M) and BHY leaves it at M c(M) p, since every later term is at
    least c(M) >= 1."""
    p = vs.p_value(0.1, 400)
    holm = vs.haircut(0.1, 400, 8, "holm")
    bhy = vs.haircut(0.1, 400, 8, "bhy")
    assert holm["adjusted_p_value"] == pytest.approx(min(1.0, 8 * p))
    c8 = sum(1 / j for j in range(1, 9))
    assert bhy["adjusted_p_value"] == pytest.approx(min(1.0, 8 * c8 * p))
    assert holm["unseen_trials"] == bhy["unseen_trials"] == 7


@pytest.mark.parametrize("method", vs.METHODS)
def test_taking_unseen_trials_as_p_one_never_lowers_the_adjustment(method):
    rng = random.Random(5)
    unseen = vs.haircut(0.08, 400, 10, method)["adjusted_p_value"]
    for _ in range(50):
        known = [rng.random() ** rng.choice((1, 4)) for _ in range(rng.randint(1, 9))]
        assert vs.haircut(0.08, 400, 10, method, known)["adjusted_p_value"] <= unseen + 1e-15


def test_known_p_values_enter_holm_and_bhy():
    others = [0.0001, 0.0002, 0.0003]
    alone = vs.haircut(0.1, 400, 4, "bhy")
    known = vs.haircut(0.1, 400, 4, "bhy", others)
    assert known["unseen_trials"] == 0
    assert known["adjusted_p_value"] < alone["adjusted_p_value"]
    assert known["adjusted_p_value"] == pytest.approx(
        vs.adjusted_p_values([vs.p_value(0.1, 400), *others], "bhy")[0])


def test_the_haircut_sharpe_ratio_never_exceeds_the_sharpe_ratio():
    for m in (1, 2, 10, 1000):
        for method in vs.METHODS:
            result = vs.haircut(0.3, 60, m, method)
            assert 0 <= result["haircut_sharpe"] <= result["sharpe"] and 0 <= result["haircut"] <= 1


@pytest.mark.parametrize("kwargs,reason", [
    ({"sharpe_per_period": 0.0}, "positive"),
    ({"sharpe_per_period": -0.2}, "positive"),
    ({"sharpe_per_period": math.nan}, "finite"),
    ({"sharpe_per_period": math.inf}, "finite"),
    ({"n_obs": 1}, "number of sessions"),
    ({"n_obs": 252.0}, "number of sessions"),
    ({"n_trials": 0}, "number of trials"),
    ({"n_trials": True}, "number of trials"),
    ({"method": "sidak"}, "one of bonferroni, holm, bhy"),
    ({"other_p_values": [0.1, 0.2, 0.3]}, "3 other p-values for 3 trials"),
    ({"other_p_values": [0.1, 2.0]}, r"\[0, 1\]"),
    ({"other_p_values": [math.nan]}, r"\[0, 1\]"),
    ({"other_p_values": [True]}, r"\[0, 1\]"),
    ({"other_p_values": "ab"}, r"\[0, 1\]"),
])
def test_haircut_refuses_inputs_it_cannot_judge(kwargs, reason):
    args = {"sharpe_per_period": 0.1, "n_obs": 252, "n_trials": 3, "method": "holm", **kwargs}
    with pytest.raises(StatsError, match=reason):
        vs.haircut(**args)


# --------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# --------------------------------------------------------------------------

def reference_pbo(columns, splits):
    """Algorithm 2.3 written out from the paper with numpy and scipy, as an
    independent check (floats, so only for inputs without ties)."""
    m = np.array(columns, dtype=float).T
    rows, n = m.shape
    block = rows // splits
    m = m[rows - block * splits:]
    blocks = np.split(np.arange(block * splits), splits)
    overfit = total = 0
    for chosen in combinations(range(splits), splits // 2):
        train = np.concatenate([blocks[i] for i in chosen])
        test = np.concatenate([blocks[i] for i in range(splits) if i not in chosen])
        r_in = m[train].mean(axis=0) / m[train].std(axis=0, ddof=1)
        r_out = m[test].mean(axis=0) / m[test].std(axis=0, ddof=1)
        rank = rankdata(r_out)[int(np.argmax(r_in))]
        w = rank / (n + 1)
        overfit += math.log(w / (1 - w)) <= 0
        total += 1
    return overfit, total


@pytest.mark.parametrize("splits,strategies,rows", [(4, 2, 40), (4, 5, 61), (6, 3, 90), (8, 10, 203)])
def test_pbo_agrees_with_the_algorithm_written_out(splits, strategies, rows):
    for seed in range(4):
        columns = noise(rows, strategies, seed)
        result = vs.pbo_cscv(columns, splits)
        assert (result["overfit"], result["combinations"]) == reference_pbo(columns, splits)
        assert result["pbo"] == result["overfit"] / result["combinations"]


def test_sixteen_splits_give_12870_combinations():
    result = vs.pbo_cscv(noise(160, 3, seed=1), 16)
    assert result["combinations"] == math.comb(16, 8) == 12_870
    assert result["rows_used"] == 160 and result["rows_dropped"] == 0


def block_column(means, unit=10 ** 9):
    """A column whose block k (three sessions) has Sharpe ratio means[k]:
    sessions m-1, m, m+1 (standard deviation 1, so the Sharpe ratio is m)."""
    return [(m + d) * unit for m in means for d in (-1, 0, 1)]


def test_block_column_has_the_sharpe_ratios_it_says():
    column = block_column([4, -2])
    assert vs.sharpe(column[:3]) == pytest.approx(4.0) and vs.sharpe(column[3:]) == pytest.approx(-2.0)


def test_a_strategy_best_everywhere_is_not_overfit():
    rng = random.Random(2)
    columns = [block_column([rng.randint(1, 5) for _ in range(6)]) for _ in range(4)]
    columns.append(block_column([9] * 6))
    assert vs.pbo_cscv(columns, 6)["pbo"] == 0.0


def test_winners_that_always_lose_out_of_sample_are_overfit():
    columns = [block_column([4, 1]), block_column([1, 4])]
    result = vs.pbo_cscv(columns, 2)
    assert (result["overfit"], result["combinations"], result["pbo"]) == (2, 2, 1.0)


def test_a_median_rank_out_of_sample_counts_as_overfitting_for_odd_n():
    """N = 3, two blocks. Each block's winner ranks 2 = (N + 1)/2 in the
    other: w = 1/2, lambda = 0, inside (-inf, 0]."""
    a, b, c = block_column([3, 2]), block_column([2, 1]), block_column([1, 3])
    assert vs.pbo_cscv([a, b, c], 2)["pbo"] == 1.0


def test_rank_n_over_2_counts_as_overfitting_for_even_n():
    """N = 4: the in-sample winner ranks 2 = N/2 out of sample in both
    combinations. lambda = ln(0.4/0.6) < 0, so it is overfitting, although
    the paper's Definition 2.2 (r < N/2) would not count it. Rank N/2 + 1
    (lambda > 0) does not count."""
    a, b, c, d = block_column([4, 2]), block_column([2, 4]), block_column([3, 1]), block_column([1, 3])
    assert vs.pbo_cscv([a, b, c, d], 2)["pbo"] == 1.0
    a, b, c, d = block_column([4, 3]), block_column([3, 4]), block_column([2, 1]), block_column([1, 2])
    assert vs.pbo_cscv([a, b, c, d], 2)["pbo"] == 0.0


def test_an_in_sample_tie_picks_the_worst_out_of_sample():
    """Block A: a and b tie for best. Block B: a ranks 3, b ranks 1. Choosing
    b (rank 1 of 4) is overfitting; choosing a would not be. Block B's winner
    d ranks 1 in A: overfit either way."""
    a = block_column([4, 3])
    b = block_column([4, 1])
    c = block_column([2, 2])
    d = block_column([1, 4])
    assert vs.pbo_cscv([a, b, c, d], 2)["pbo"] == 1.0


def test_an_out_of_sample_tie_takes_the_lowest_rank():
    """Block B's winner d ties with b and c in block A. The lowest of the tied
    ranks is 2 (overfit for N = 4); the average, 3, would not be. Block A's
    winner a ranks 1 in B: overfit either way."""
    a = block_column([4, 1])
    b = block_column([3, 2])
    c = block_column([3, 3])
    d = block_column([3, 4])
    assert vs.pbo_cscv([a, b, c, d], 2)["pbo"] == 1.0


def test_flat_and_constant_blocks_rank_by_their_sign():
    """A flat block has Sharpe ratio 0; a constant non-zero one ranks above
    (or below) every varying block of its sign, without failing. Block A:
    rising (+inf) beats varying (0.5) beats flat (0); block B: varying (2)
    beats rising (1) beats falling (-inf). A's winner, rising, ranks 2 in B
    (overfit, N = 3); B's winner, varying, ranks 2 in A (overfit)."""
    u = 10 ** 9
    rising = [u] * 3 + [0, u, 2 * u]
    varying = [-u // 2, u // 2, 3 * u // 2] + [u, 2 * u, 3 * u]
    flat = [0] * 3 + [-u] * 3
    assert [vs.sharpe(varying[:3]), vs.sharpe(varying[3:]), vs.sharpe(rising[3:])] == pytest.approx([0.5, 2, 1])
    result = vs.pbo_cscv([rising, varying, flat], 2)
    assert (result["combinations"], result["overfit"]) == (2, 2)


def test_a_strategy_flat_throughout_ranks_at_zero_not_above_the_winners():
    """A strategy that never trades returns 0 in every block: Sharpe ratio
    0, below any winning block. Each block's winner ranks 2 of 3 in the
    other (overfit both times). Ranked above the winners instead, the flat
    strategy would be chosen in-sample and rank first out of sample."""
    flat = [0] * 6
    assert vs.pbo_cscv([block_column([4, 1]), block_column([1, 4]), flat], 2)["pbo"] == 1.0


def test_the_oldest_sessions_are_left_out_when_the_length_does_not_divide():
    columns = noise(83, 4, seed=9)
    result = vs.pbo_cscv(columns, 8)
    assert (result["rows"], result["rows_used"], result["rows_dropped"]) == (83, 80, 3)
    changed = [[10 ** 11] * 3 + column[3:] for column in columns]
    assert vs.pbo_cscv(changed, 8) == result
    assert vs.pbo_cscv([column[3:] for column in columns], 8)["overfit"] == result["overfit"]


def test_pbo_accepts_numpy_integers_and_is_repeatable():
    columns = noise(64, 5, seed=4)
    as_numpy = [np.array(column, dtype=np.int64) for column in columns]
    assert vs.pbo_cscv(as_numpy, 8) == vs.pbo_cscv(columns, 8) == vs.pbo_cscv(columns, 8)


def test_pbo_golden_count_is_the_same_on_every_platform():
    """The counts come from exact integer sums and correctly rounded IEEE
    operations only, so this literal holds on every OS CI runs (a platform's
    libm entering the ranking would show here)."""
    rng = random.Random(20261115)
    columns = [[rng.randint(-10 ** 10, 10 ** 10) + k * 10 ** 7 for _ in range(509)] for k in range(12)]
    assert vs.pbo_cscv(columns, 16) == {
        "n_splits": 16, "rows": 509, "rows_used": 496, "rows_dropped": 13, "strategies": 12,
        "combinations": 12_870, "overfit": 4913, "pbo": 4913 / 12_870}


def test_the_statistics_read_a_trial_matrix_as_trial_matrix_builds_it():
    """The producer: trial_matrix.build quantises each version's float returns
    (oldest session first) into a column of Python ints. Its Sharpe ratio,
    annualised the way the evaluation that recorded the run annualised it, is
    the evaluation's Sharpe ratio, and its columns go to pbo_cscv as they are."""
    rng = np.random.default_rng(7)
    dates = [f"2024-{1 + d // 28:02d}-{1 + d % 28:02d}" for d in range(96)]
    runs = {f"qs-sv-sha256-v1:{k:064x}": rng.normal(0.0004 * k, 0.01, len(dates)) for k in range(5)}
    matrix = trial_matrix.build(dates, runs)
    for column in matrix["columns"]:
        assert all(type(q) is int for q in column["returns"])
        (version,) = column["strategy_version_ids"]
        recorded = research_trials.evaluation_metrics(runs[version])
        assert column["returns_sha256"] == recorded["returns_sha256"]
        assert vs.sharpe(column["returns"]) * math.sqrt(252) == pytest.approx(recorded["sharpe"], rel=1e-9)
    result = vs.pbo_cscv([column["returns"] for column in matrix["columns"]], 8)
    assert (result["strategies"], result["rows"], result["combinations"]) == (5, 96, 70)


@pytest.mark.parametrize("columns,splits,reason", [
    ([[1, 2, 3, 4]], 2, "at least two strategies"),
    ([], 2, "at least two strategies"),
    ([[1, 2, 3, 4], [4, 3, 2, 1]], 3, "even"),
    ([[1, 2, 3, 4], [4, 3, 2, 1]], 0, "number of splits"),
    ([[1, 2, 3, 4], [4, 3, 2, 1]], True, "number of splits"),
    ([[1, 2, 3], [3, 2, 1]], 4, "too few"),
    ([[1, 2], [2, 1]], 2, "too few"),
    ([[1, 2, 3, 4], [4, 3, 2]], 2, "every session"),
    ([[1, 2, 3, 4], [4, 3, 2.5, 1]], 2, "integer quanta"),
    ("abcd", 2, "sequence of columns"),
])
def test_pbo_refuses_what_it_cannot_judge(columns, splits, reason):
    with pytest.raises(StatsError, match=reason):
        vs.pbo_cscv(columns, splits)


# --------------------------------------------------------------------------
# Deflated Sharpe ratio (Bailey and López de Prado 2014)
# --------------------------------------------------------------------------

# The paper's worked example. Its typeset symbols did not survive text
# extraction, so its inputs were reconstructed (an annual Sharpe ratio of 2.5
# over five years of 250 sessions, a variance of the trials' annual Sharpe
# ratios of 0.5, skewness -3, kurtosis 10) and confirmed by reproducing all
# three of its published outcomes: SR0 0.1132 and a DSR of about 90% for 100
# trials; 0.9505 for 46 trials; and, for normal returns, 88 trials as the most
# that keep the DSR at 95%.
PAPER = {"sharpe_per_period": 2.5 / math.sqrt(250), "n_obs": 1250, "variance": 0.5 / 250,
         "skewness": -3.0, "kurtosis": 10.0}


def test_the_papers_example_with_100_trials():
    result = vs.deflated_sharpe_ratio(n_trials=100, **PAPER)
    assert result["benchmark_sharpe"] == pytest.approx(0.113172, abs=1e-6)
    assert result["dsr"] == pytest.approx(0.900397, abs=1e-6)


def test_the_papers_example_with_46_trials():
    """The kurtosis is plain (3 for normal returns): taking the paper's 10
    as excess kurtosis would give 0.9495, not the published 0.9505."""
    assert vs.deflated_sharpe_ratio(n_trials=46, **PAPER)["dsr"] == pytest.approx(0.950502, abs=1e-6)
    misread = vs.deflated_sharpe_ratio(n_trials=46, **{**PAPER, "kurtosis": 13.0})["dsr"]
    assert misread == pytest.approx(0.949469, abs=1e-6)


def test_with_normal_returns_88_trials_are_the_most_that_keep_95_percent():
    normal = {**PAPER, "skewness": 0.0, "kurtosis": 3.0}
    assert vs.deflated_sharpe_ratio(n_trials=88, **normal)["dsr"] >= 0.95
    assert vs.deflated_sharpe_ratio(n_trials=89, **normal)["dsr"] < 0.95


def test_the_dsr_is_eq_2_written_out_with_scipy():
    from scipy.stats import norm

    for n_trials, skew, kurt in [(3, 0.4, 5.0), (17.5, -1.2, 9.0), (250, 0.0, 3.0)]:
        sr, t, v = 0.07, 600, 0.0011
        emax = (1 - vs.EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + vs.EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e))
        sr0 = math.sqrt(v) * emax
        z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr * sr)
        result = vs.deflated_sharpe_ratio(sr, t, n_trials, v, skew, kurt)
        assert result["benchmark_sharpe"] == pytest.approx(sr0, rel=1e-9)
        assert result["dsr"] == pytest.approx(norm.cdf(z), rel=1e-9)


def test_the_probabilistic_sharpe_ratio_is_one_half_at_its_benchmark():
    assert vs.probabilistic_sharpe(0.05, 0.05, 500, -0.5, 6.0) == pytest.approx(0.5, abs=1e-15)
    assert vs.probabilistic_sharpe(0.06, 0.05, 500, -0.5, 6.0) > 0.5 > vs.probabilistic_sharpe(0.04, 0.05, 500, -0.5, 6.0)


def test_negative_skew_and_fat_tails_lower_the_probabilistic_sharpe_ratio():
    base = vs.probabilistic_sharpe(0.08, 0.02, 400, 0.0, 3.0)
    assert vs.probabilistic_sharpe(0.08, 0.02, 400, -1.0, 3.0) < base
    assert vs.probabilistic_sharpe(0.08, 0.02, 400, 0.0, 12.0) < base


def test_the_real_valued_expected_maximum_matches_the_integer_one():
    for n in [2, 3, 10, 46, 100, 1000]:
        assert vs._expected_max_real(n) == vs.expected_max(n)
        assert vs._expected_max_real(float(n)) == vs.expected_max(n)
    assert vs._expected_max_real(1) == vs._expected_max_real(1.0) == 0.0


def test_the_real_valued_expected_maximum_is_floored_at_zero_near_one():
    """The approximation falls to minus infinity as N approaches 1 (it is
    negative below about N = 1.28); the true expected maximum is never
    negative, and grows with N."""
    inv = NormalDist().inv_cdf
    raw = (1 - vs.EULER_GAMMA) * inv(1 - 1 / 1.1) + vs.EULER_GAMMA * inv(1 - 1 / (1.1 * math.e))
    assert raw < 0
    assert vs._expected_max_real(1.1) == 0.0
    assert vs._expected_max_real(1.000001) == 0.0
    grid = [1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 7.25, 40.0]
    values = [vs._expected_max_real(n) for n in grid]
    assert values == sorted(values) and values[2] > 0


@pytest.mark.parametrize("n_trials,reason", [
    (0.99, "at least 1"), (0, "at least 1"), (-3, "at least 1"),
    (math.nan, "finite"), (math.inf, "finite"), (True, "finite"), ("5", "finite"),
])
def test_the_real_valued_expected_maximum_refuses_what_is_not_a_count(n_trials, reason):
    with pytest.raises(StatsError, match=reason):
        vs._expected_max_real(n_trials)


@pytest.mark.parametrize("kwargs,reason", [
    ({"skewness": 2.0, "kurtosis": 1.0}, "no variance"),  # 1 - 2 SR + 0 at SR = 0.5: 0
    ({"skewness": 3.0, "kurtosis": 1.0}, "no variance"),
    ({"skewness": math.nan}, "skewness"),
    ({"kurtosis": math.inf}, "kurtosis"),
    ({"n_obs": 1}, "number of sessions"),
    ({"n_obs": 12.0}, "number of sessions"),
    ({"sharpe_per_period": math.nan}, "Sharpe ratio"),
    ({"benchmark": math.inf}, "benchmark"),
])
def test_the_probabilistic_sharpe_ratio_refuses_what_it_cannot_judge(kwargs, reason):
    args = {"sharpe_per_period": 0.5, "benchmark": 0.1, "n_obs": 100, "skewness": 0.0, "kurtosis": 3.0, **kwargs}
    with pytest.raises(StatsError, match=reason):
        vs.probabilistic_sharpe(**args)


@pytest.mark.parametrize("variance", [0.0, -0.001, math.nan, True])
def test_the_dsr_refuses_a_variance_that_is_not_positive(variance):
    with pytest.raises(StatsError, match="variance"):
        vs.deflated_sharpe_ratio(**{**PAPER, "variance": variance}, n_trials=10)


def test_moments_are_scipys_skew_and_plain_kurtosis():
    from scipy.stats import kurtosis, skew

    rng = np.random.default_rng(11)
    for sample in [rng.normal(0.0005, 0.01, 750), rng.standard_t(3, 500) * 0.01, rng.lognormal(0, 0.6, 300) * 0.01]:
        column = [round(r * Q) for r in sample]
        floats = np.array(column, dtype=float) / Q
        result = vs.moments(column)
        assert result["skewness"] == pytest.approx(float(skew(floats)), rel=1e-9, abs=1e-12)
        assert result["kurtosis"] == pytest.approx(float(kurtosis(floats, fisher=False)), rel=1e-9)


def test_normal_returns_have_kurtosis_three_not_zero():
    rng = np.random.default_rng(5)
    result = vs.moments([round(r * Q) for r in rng.normal(0, 0.01, 20_000)])
    assert result["kurtosis"] == pytest.approx(3.0, abs=0.1)
    assert result["skewness"] == pytest.approx(0.0, abs=0.05)


def test_moments_of_a_two_point_series():
    """Two equally likely values: skewness 0 and kurtosis exactly 1, the least
    any distribution has."""
    assert vs.moments([-5, 5, -5, 5]) == {"skewness": 0.0, "kurtosis": 1.0}
    assert vs.moments([0, 0, 0, 7]) == pytest.approx({"skewness": 2 / math.sqrt(3), "kurtosis": 7 / 3}, rel=1e-15)


@pytest.mark.parametrize("column,reason", [
    ([4], "at least two sessions"), ([3, 3, 3], "constant"), ([1, 2.5, 3], "integer quanta"),
    ("12", "sequence of integer quanta"),
])
def test_moments_refuse_a_series_that_has_none(column, reason):
    with pytest.raises(StatsError, match=reason):
        vs.moments(column)


def average_correlation(columns):
    c = np.corrcoef(np.array(columns, dtype=float))
    k = len(columns)
    return (c.sum() - np.trace(c)) / (k * (k - 1))


def test_implied_trials_are_appendix_a3_written_out_with_numpy():
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.01, 400)
    columns = [[round((0.6 * common[t] + 0.8 * x) * Q) for t, x in enumerate(rng.normal(0, 0.01, 400))]
               for _ in range(6)]
    rho = average_correlation(columns)
    result = vs.implied_trials(columns, 9)
    assert result["average_correlation"] == pytest.approx(rho, rel=1e-9)
    assert result["implied_trials"] == pytest.approx(rho + (1 - rho) * 9, rel=1e-9)
    assert result["distinct_series"] == 6 and result["n_trials"] == 9
    assert 0.2 < rho < 0.5


def test_perfectly_correlated_series_are_one_trial():
    base = [3, -1, 4, 1, -5, 9, 2, -6]
    columns = [base, [2 * q + 7 for q in base], [5 * q - 11 for q in base]]
    result = vs.implied_trials(columns, 3)
    assert result["average_correlation"] == 1.0
    assert result["implied_trials"] == 1.0


def test_a_correlation_that_rounds_past_one_is_clamped():
    """Computed in floating point, the correlation of this column with 13
    times it plus a constant is 1.0000000000000002; unclamped, the implied
    number of trials would fall below 1."""
    x = [-1139023663, 3391358857, 6153210692, -9098250963, -7904671614]
    y = [13 * q - 70639805 for q in x]
    n, tx, ty = len(x), sum(x), sum(y)
    raw = (n * sum(a * b for a, b in zip(x, y)) - tx * ty) / math.sqrt(
        (n * sum(q * q for q in x) - tx * tx) * (n * sum(q * q for q in y) - ty * ty))
    assert raw > 1
    result = vs.implied_trials([x, y], 2)
    assert result["average_correlation"] == 1.0
    assert result["implied_trials"] == 1.0
    # and one that rounds below -1: -3 times a column plus a constant
    x = [-7241366701, 2283376829, 8992804953, -2826092214, -107107555, 6675120324]
    result = vs.implied_trials([x, [-3 * q + 12345 for q in x]], 2)
    assert result["average_correlation"] == -1.0
    assert result["implied_trials"] == 3.0


def test_independent_series_are_about_as_many_trials_as_the_ledger_counts():
    result = vs.implied_trials(noise(2000, 8, seed=19), 8)
    assert abs(result["average_correlation"]) < 0.02
    assert result["implied_trials"] == pytest.approx(8, abs=0.15)


def test_negatively_correlated_series_imply_more_trials_than_counted():
    """N = rho + (1 - rho) M exceeds M when rho < 0; it is not capped at M."""
    base = [3, -1, 4, 1, -5, 9, 2, -6]
    result = vs.implied_trials([base, [-q for q in base]], 2)
    assert result["average_correlation"] == -1.0
    assert result["implied_trials"] == 3.0


def test_versions_sharing_a_column_count_in_m_but_add_no_correlation():
    """Decision (b): every strategy version is a trial, including versions
    that differ only in fields the engine does not read and so share a
    column; the correlation is taken over the distinct columns."""
    columns = noise(500, 3, seed=23)
    three, five = vs.implied_trials(columns, 3), vs.implied_trials(columns, 5)
    rho = three["average_correlation"]
    assert five["average_correlation"] == rho
    assert five["implied_trials"] == pytest.approx(rho + (1 - rho) * 5, rel=1e-12)
    assert five["implied_trials"] > three["implied_trials"]


def test_implied_trials_are_the_same_on_every_platform():
    """Exact integer sums: a golden value, not a tolerance."""
    result = vs.implied_trials(noise(300, 5, seed=29), 7)
    assert result["average_correlation"] == -0.025019496644124083


@pytest.mark.parametrize("columns,n_trials,reason", [
    ([[1, 2, 3]], 1, "at least two distinct"),
    ([], 2, "no return series"),
    ([[1, 2, 3], [1, 2, 3]], 2, "distinct"),
    ([[1, 2, 3], [4, 4, 4]], 2, "constant"),
    ([[1, 2, 3], [3, 1, 2], [2, 3, 1]], 2, "cannot have 3"),
    ([[1, 2, 3], [3, 1]], 2, "every session"),
    ([[1], [2]], 2, "two sessions"),
    ([[1, 2, 3], [3, 1, 2]], 0, "number of trials"),
    ([[1, 2, 3], [3, 1, 2]], 2.0, "number of trials"),
    ([[1, 2, 3], [3, 1.5, 2]], 2, "integer quanta"),
    ("abc", 2, "sequence of columns"),
])
def test_implied_trials_refuse_what_they_cannot_judge(columns, n_trials, reason):
    with pytest.raises(StatsError, match=reason):
        vs.implied_trials(columns, n_trials)


def reference_dsr(columns, candidate, n_trials, every_trial_evaluated):
    """The deflated Sharpe ratio of a matrix written out with numpy/scipy."""
    from scipy.stats import kurtosis, norm, skew

    m = np.array(columns, dtype=float) / Q
    k, t = m.shape
    sharpes = m.mean(axis=1) / m.std(axis=1, ddof=1)
    v = max(1 / (t - 1), float(np.var(sharpes, ddof=1)) if k > 1 else 0.0)
    if every_trial_evaluated and k > 1:
        rho = average_correlation(columns)
        n = rho + (1 - rho) * n_trials
    else:
        n = n_trials
    emax = (1 - vs.EULER_GAMMA) * norm.ppf(1 - 1 / n) + vs.EULER_GAMMA * norm.ppf(1 - 1 / (n * math.e))
    sr, g3, g4 = sharpes[candidate], skew(m[candidate]), kurtosis(m[candidate], fisher=False)
    z = (sr - math.sqrt(v) * emax) * math.sqrt(t - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr * sr)
    return float(norm.cdf(z))


def family(rows=500, k=6, seed=31):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, rows)
    return [[round((0.0006 * j + 0.5 * common[t] + 0.85 * x) * Q) for t, x in enumerate(rng.normal(0, 0.01, rows))]
            for j in range(k)]


@pytest.mark.parametrize("every_trial_evaluated", [True, False])
def test_the_dsr_of_a_matrix_column_is_the_paper_written_out(every_trial_evaluated):
    columns = family()
    for candidate in (0, 5):
        result = vs.deflated_sharpe(columns, candidate, 8, every_trial_evaluated)
        assert result["dsr"] == pytest.approx(reference_dsr(columns, candidate, 8, every_trial_evaluated), rel=1e-9)
        assert result["sharpe"] == vs.sharpe(columns[candidate])
        assert result["n_obs"] == 500 and result["ledger_trials"] == 8 and result["distinct_series"] == 6


def test_the_implied_number_of_trials_is_used_only_when_every_trial_is_evaluated():
    columns = family()
    implied = vs.implied_trials(columns, 8)
    complete = vs.deflated_sharpe(columns, 5, 8, True)
    assert (complete["n_trials_source"], complete["n_trials"]) == ("implied", implied["implied_trials"])
    assert complete["average_correlation"] == implied["average_correlation"]
    incomplete = vs.deflated_sharpe(columns, 5, 8, False)
    assert (incomplete["n_trials_source"], incomplete["n_trials"], incomplete["average_correlation"]) == ("ledger", 8.0, None)
    # Correlated variants are fewer independent trials, so the DSR is higher.
    assert complete["n_trials"] < 8 and complete["dsr"] > incomplete["dsr"]


def test_the_trial_variance_is_the_empirical_one_above_its_floor():
    columns = family()
    result = vs.deflated_sharpe(columns, 5, 6, True)
    sharpes = [vs.sharpe(c) for c in columns]
    assert result["variance_empirical"] == pytest.approx(float(np.var(sharpes, ddof=1)), rel=1e-12)
    assert result["variance_null"] == 1 / 499
    assert result["variance"] == max(result["variance_null"], result["variance_empirical"])
    assert result["variance"] == result["variance_empirical"]


def test_near_identical_variants_deflate_by_at_least_the_null_variance():
    """Variants whose Sharpe ratios hardly differ (V_emp ~ 0) still face the
    dispersion skill-less trials show by chance, V_null = 1 / (T - 1)."""
    (base,) = noise(400, 1, seed=37)
    base = [q + 10 ** 9 for q in base]
    columns = [base, [q + (1 if t == 0 else 0) for t, q in enumerate(base)]]
    result = vs.deflated_sharpe(columns, 0, 2, True)
    assert result["variance_empirical"] < 1e-12
    assert result["variance"] == result["variance_null"] == 1 / 399


def test_a_single_distinct_column_uses_the_ledger_count_and_the_null_variance():
    (column,) = noise(300, 1, seed=41)
    result = vs.deflated_sharpe([[q + 10 ** 9 for q in column]], 0, 4, True)
    assert result["n_trials_source"] == "ledger" and result["n_trials"] == 4.0
    assert result["variance_empirical"] is None and result["variance"] == 1 / 299


def test_more_trials_never_raise_the_dsr():
    columns = family()
    values = [vs.deflated_sharpe(columns, 5, m, False)["dsr"] for m in (6, 7, 10, 50, 500)]
    assert values == sorted(values, reverse=True) and values[0] > values[-1]


def test_a_twin_version_shares_a_column_and_still_counts_as_a_trial():
    """Through the producer: two versions with the same returns share one
    column of trial_matrix.build; the ledger counts both, the correlation
    sees the column once, and the implied N rises with the ledger count."""
    rng = np.random.default_rng(43)
    dates = [f"2024-{1 + d // 28:02d}-{1 + d % 28:02d}" for d in range(200)]
    returns = [rng.normal(0.0005 * k, 0.01, len(dates)) for k in range(3)]
    runs = {f"qs-sv-sha256-v1:{k:064x}": returns[k] for k in range(3)}
    runs[f"qs-sv-sha256-v1:{9:064x}"] = returns[2]  # differs only in risk_limits
    matrix = trial_matrix.build(dates, runs)
    columns = [c["returns"] for c in matrix["columns"]]
    assert len(columns) == 3 and sorted(len(c["strategy_version_ids"]) for c in matrix["columns"]) == [1, 1, 2]
    with_twin, without = vs.deflated_sharpe(columns, 0, 4, True), vs.deflated_sharpe(columns, 0, 3, True)
    assert with_twin["average_correlation"] == without["average_correlation"]
    assert with_twin["n_trials"] > without["n_trials"]
    assert with_twin["dsr"] <= without["dsr"]


@pytest.mark.parametrize("columns,candidate,n_trials,complete,reason", [
    ([[1, 2, 3], [3, 1, 2]], 2, 2, True, "index of one of the columns"),
    ([[1, 2, 3], [3, 1, 2]], -1, 2, True, "index of one of the columns"),
    ([[1, 2, 3], [3, 1, 2]], True, 2, True, "index of one of the columns"),
    ([[1, 2, 3], [3, 1, 2]], 0, 2, 1, "true or false"),
    ([[1, 2, 3], [3, 1, 2]], 0, 1, True, "cannot have 2"),
    ([[1, 2, 3], [3, 1, 2]], 0, 1, False, "cannot have 2"),
    ([[1, 2, 3], [1, 2, 3]], 0, 2, True, "distinct"),
    ([[1, 2, 3], [5, 5, 5]], 0, 2, False, "constant series"),
    ([[1, 2, 3], [5, 5, 5]], 0, 2, True, "constant series"),
    ([[1], [2]], 0, 2, False, "two sessions"),
    ([], 0, 1, True, "no return series"),
])
def test_the_dsr_of_a_matrix_refuses_what_it_cannot_judge(columns, candidate, n_trials, complete, reason):
    with pytest.raises(StatsError, match=reason):
        vs.deflated_sharpe(columns, candidate, n_trials, complete)
