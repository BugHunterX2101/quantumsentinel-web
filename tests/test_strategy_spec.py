"""Strategy specs (DSL v1): ids, validation and the compiler (backend/services/strategy_spec.py)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from backend.services import strategy_spec as ss
from backend.services.dataset_format import canonical_json
from backend.services.event_simulator import run_event_backtest

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "qs_sv_sha256_v1.json").read_text("utf-8"))
_DELETE = object()


def spec(**changes) -> dict:
    """A valid spec; ``changes`` maps dotted paths to new values (or _DELETE)."""
    value = {
        "dsl": "qs-strategy-v1",
        "signal": {"template": "ma_crossover", "fast": 20, "slow": 50},
        "universe": {"family": "EQUITY_EOD", "symbols": ["AAPL", "MSFT", "GOOGL"]},
        "sizing": {"method": "fixed_fractional", "fraction_bps": 200},
        "cost_model": "retail",
        "allow_short": False,
        "leverage_limit_bps": 10_000,
        "execution": "next_bar_close",
        "risk_limits": {"max_drawdown_bps": 2_000, "max_position_bps": 5_000},
    }
    for path, new in changes.items():
        *parents, leaf = path.split(".")
        node = value
        for key in parents:
            node = node[key]
        if new is _DELETE:
            del node[leaf]
        else:
            node[leaf] = new
    return value


def prices(symbols, bars=320, seed=7) -> dict:
    rng = np.random.default_rng(seed)
    return {s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, bars))) for s in symbols}


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

VECTOR_IDS = [v["spec"]["signal"]["template"] for v in VECTORS["vectors"]]


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=VECTOR_IDS)
def test_golden_vectors(vector):
    parsed = ss.parse(vector["spec"])
    assert ss.canonical(parsed).decode("utf-8") == vector["canonical"]
    assert ss.version_id(parsed) == vector["id"]
    # The id is the plain SHA-256 of the canonical bytes, so another
    # implementation needs only the canonical form.
    assert vector["id"] == f"{VECTORS['scheme']}:" + hashlib.sha256(vector["canonical"].encode()).hexdigest()
    assert ss.STRATEGY_VERSION_ID.fullmatch(vector["id"])


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=VECTOR_IDS)
def test_a_canonical_spec_parses_to_itself(vector):
    # No defaults: parsing adds nothing, so the canonical bytes are exactly
    # the JCS form of what was stored.
    stored = json.loads(vector["canonical"])
    assert ss.canonical(ss.parse(stored)) == canonical_json(stored) == vector["canonical"].encode()


def test_symbol_case_spacing_and_order_do_not_change_the_id():
    a = ss.parse(spec(**{"universe.symbols": ["AAPL", "MSFT", "GOOGL"]}))
    b = ss.parse(spec(**{"universe.symbols": [" googl", "msft ", "Aapl"]}))
    assert ss.version_id(a) == ss.version_id(b)
    assert a.universe.symbols == ["AAPL", "GOOGL", "MSFT"]


# Every leaf of the default spec and a different valid value for it.
LEAVES = {
    "signal.fast": 10, "signal.slow": 60,
    "universe.family": "INDEX_EOD", "universe.symbols": ["AAPL", "MSFT", "AMZN"],
    "sizing.fraction_bps": 300, "cost_model": "institutional", "allow_short": True,
    "leverage_limit_bps": 15_000,
    "risk_limits.max_drawdown_bps": 1_000, "risk_limits.max_position_bps": 4_000,
}
# Fields that name or limit the strategy without changing what the engine
# runs: the family selects data (and its holdout); the risk limits are
# checked against a backtest by the validation gate.
NOT_RUN = {"universe.family", "risk_limits.max_drawdown_bps", "risk_limits.max_position_bps"}

SIX = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA"]
VARIANTS = {
    "signal": [{"template": "momentum", "lookback": 60, "n_long": 1, "n_short": 1},
               {"template": "momentum", "lookback": 30, "n_long": 1, "n_short": 1},
               {"template": "momentum", "lookback": 60, "n_long": 2, "n_short": 1},
               {"template": "momentum", "lookback": 60, "n_long": 1, "n_short": 0},
               {"template": "mean_reversion", "window": 20, "n_std_centi": 200},
               {"template": "mean_reversion", "window": 30, "n_std_centi": 200},
               {"template": "mean_reversion", "window": 20, "n_std_centi": 150}],
    "sizing": [{"method": "volatility_target", "target_vol_bps": 1_500},
               {"method": "volatility_target", "target_vol_bps": 1_000}],
}


def _compiled(value):
    c = ss.compile_spec(ss.parse(value))
    return c.tickers, json.dumps(c.engine_kwargs, sort_keys=True)


def test_the_leaf_list_covers_the_spec():
    def leaves(node, prefix=""):
        for key, item in node.items():
            if isinstance(item, dict):
                yield from leaves(item, f"{prefix}{key}.")
            else:
                yield f"{prefix}{key}"
    assert set(leaves(spec())) - {"dsl", "execution", "signal.template", "sizing.method"} == set(LEAVES)


@pytest.mark.parametrize("path", sorted(LEAVES))
def test_every_field_is_part_of_the_id_and_only_declared_ones_skip_the_engine(path):
    base, changed = spec(), spec(**{path: LEAVES[path]})
    assert ss.version_id(ss.parse(base)) != ss.version_id(ss.parse(changed))
    assert (_compiled(base) == _compiled(changed)) == (path in NOT_RUN)


@pytest.mark.parametrize("field", sorted(VARIANTS))
def test_every_template_and_sizing_parameter_reaches_the_engine(field):
    # Six symbols, so momentum may trade two on a side.
    variants = [spec(**{field: v, "universe.symbols": SIX}) for v in VARIANTS[field]]
    compiled = {_compiled(v) for v in variants}
    ids = {ss.version_id(ss.parse(v)) for v in variants}
    assert len(compiled) == len(ids) == len(variants)


def test_momentum_counts_up_to_a_third_and_a_short_only_side_when_shorting():
    momentum = {"template": "momentum", "lookback": 60, "n_long": 2, "n_short": 2}
    assert ss.parse(spec(**{"signal": momentum, "universe.symbols": SIX})).signal.n_long == 2
    short_only = {"template": "momentum", "lookback": 60, "n_long": 0, "n_short": 1}
    assert ss.parse(spec(**{"signal": short_only, "allow_short": True})).signal.n_short == 1


# --------------------------------------------------------------------------
# The compiler runs the engine and nothing else
# --------------------------------------------------------------------------

ENGINE_CASES = [
    ({"template": "ma_crossover", "fast": 10, "slow": 30}, {"fast": 10, "slow": 30}),
    ({"template": "momentum", "lookback": 40, "n_long": 1, "n_short": 1},
     {"lookback": 40, "n_long": 1, "n_short": 1}),
    ({"template": "mean_reversion", "window": 15, "n_std_centi": 150}, {"window": 15, "n_std": 1.5}),
]
SIZING_CASES = [
    ({"method": "fixed_fractional", "fraction_bps": 500}, {"fraction": 0.05}),
    ({"method": "volatility_target", "target_vol_bps": 1_200}, {"target_vol": 0.12}),
]


@pytest.mark.parametrize("allow_short,leverage_bps,leverage", [(False, 10_000, 1.0), (True, 15_000, 1.5)],
                         ids=["long_only", "short"])
@pytest.mark.parametrize("sizing,sizing_params", SIZING_CASES, ids=["fixed_fractional", "volatility_target"])
@pytest.mark.parametrize("signal,signal_params", ENGINE_CASES,
                         ids=["ma_crossover", "momentum", "mean_reversion"])
def test_a_compiled_spec_runs_exactly_what_the_engine_runs(signal, signal_params, sizing, sizing_params,
                                                           allow_short, leverage_bps, leverage):
    symbols = ["AAPL", "GOOGL", "MSFT"]
    value = spec(**{"signal": signal, "sizing": sizing, "allow_short": allow_short,
                    "leverage_limit_bps": leverage_bps, "cost_model": "institutional",
                    "universe.symbols": ["msft", "AAPL", "googl"]})
    data = prices(symbols)
    got = ss.compile_spec(ss.parse(value)).run(data, 250_000.0)
    want = run_event_backtest(
        tickers=symbols, price_data=data, strategy_name=signal["template"],
        strategy_params={**signal_params, **sizing_params, "leverage_limit": leverage},
        initial_capital=250_000.0, cost_model_name="institutional", allow_short=allow_short,
        sizing_method=sizing["method"])
    assert got == want
    assert got["n_trades"] > 0


def test_run_uses_only_the_universe_and_refuses_missing_prices():
    compiled = ss.compile_spec(ss.parse(spec()))
    data = prices(["AAPL", "GOOGL", "MSFT", "TSLA"])
    assert compiled.run(data, 100_000.0)["tickers"] == ["AAPL", "GOOGL", "MSFT"]
    del data["MSFT"]
    with pytest.raises(ss.StrategySpecError, match="no prices for MSFT"):
        compiled.run(data, 100_000.0)


# --------------------------------------------------------------------------
# What a spec may not be
# --------------------------------------------------------------------------

@pytest.mark.parametrize("changes", [
    {"signal.fast": 20.0},                      # float
    {"signal.fast": "20"},                      # string integer
    {"signal.fast": True},                      # boolean integer
    {"allow_short": 0},                         # integer boolean
    {"leverage_limit_bps": 1e4},                # float bps
    {"signal.slow": 20},                        # slow not above fast
    {"signal.fast": 1},                         # fast below range
    {"signal.slow": 501},                       # slow above range
    {"signal": {"template": "momentum", "lookback": 60, "n_long": 0, "n_short": 0},
     "allow_short": True},                      # trades nothing even when shorting
    {"signal": {"template": "momentum", "lookback": 60, "n_long": 1, "n_short": 1},
     "universe.symbols": ["AAPL", "MSFT"]},     # momentum on 2 symbols
    {"signal": {"template": "momentum", "lookback": 60, "n_long": 2, "n_short": 1}},
    {"signal": {"template": "momentum", "lookback": 60, "n_long": 1, "n_short": 2},
     "allow_short": True},                      # more than a third on either side
    {"signal": {"template": "momentum", "lookback": 60, "n_long": 0, "n_short": 1}},  # long-only, no long side
    {"signal": {"template": "mean_reversion", "window": 20, "n_std": 2}},
    {"signal": {"template": "obi_momentum", "threshold": 1}},
    {"signal.extra": 1},
    {"notes": "x"},
    {"risk_limits": {"max_drawdown_bps": 2_000}},
    {"execution": _DELETE},
    {"execution": "next_bar_open"},             # not what the engine does
    {"dsl": "qs-strategy-v2"},
    {"universe.family": "NOT_A_FAMILY"},
    {"universe.symbols": ["AAPL", "aapl"]},     # duplicate after normalising
    {"universe.symbols": ["AA PL"]},
    {"universe.symbols": []},
    {"universe.symbols": [f"S{i}" for i in range(11)]},
    {"sizing.fraction_bps": 0},
    {"sizing.fraction_bps": 10_001},
    {"leverage_limit_bps": 0},
    {"cost_model": "free"},
])
def test_what_a_spec_may_not_be(changes):
    with pytest.raises(ValidationError):
        ss.parse(spec(**changes))


def test_a_parsed_spec_cannot_be_changed():
    parsed = ss.parse(spec())
    with pytest.raises(ValidationError):
        parsed.allow_short = True
    with pytest.raises(ValidationError):
        parsed.signal.fast = 5
