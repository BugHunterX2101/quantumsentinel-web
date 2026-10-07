"""Research-core correctness: date alignment, one shared strategy definition
with a one-bar decision delay, multi-asset walk-forward, a calibrated
Deflated Sharpe test with server-counted trials, a race-free audit chain, and
no execution against stale quotes. All offline (yfinance is faked)."""
import math
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from backend import main, models, schemas
from backend.crypto import pqc
from backend.services import (backtest_service, event_simulator, historical_data, paper_broker,
                              research_trials, security_service, stat_tests, strategy_signals,
                              trading_service, walk_forward)
from backend.services.execution_model import zero_cost_config


# ── helpers ──────────────────────────────────────────────────────────────────

def yf_frame(series: dict[str, pd.Series]) -> pd.DataFrame:
    """A yfinance-shaped multi-ticker frame: (field, ticker) columns."""
    cols = {}
    for ticker, s in series.items():
        for field in ("Open", "High", "Low", "Close"):
            cols[(field, ticker)] = s
        cols[("Volume", ticker)] = s * 0 + 2e6
    frame = pd.DataFrame(cols)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def random_walk(seed: int, n: int, drift: float = 0.0) -> np.ndarray:
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(drift, 0.015, n)))


@pytest.fixture
def db(make_engine):
    session = sessionmaker(bind=make_engine())()
    yield session
    session.close()


def make_user(db, email="researcher@example.com", role="user"):
    user = models.User(email=email, password_hash="x", role=role)
    db.add(user)
    db.commit()
    return user


# ── 1. Date alignment ────────────────────────────────────────────────────────

def mixed_calendar():
    """A 7-day asset and a weekday asset whose prices encode their date."""
    days = pd.date_range("2025-01-01", periods=400, freq="D")
    ordinal = np.arange(len(days), dtype=float)
    coin = pd.Series(1000 + ordinal, days)
    equity = pd.Series(np.where(days.dayofweek < 5, 2000 + ordinal, np.nan), days)
    return days, coin, equity


class TestAlignment:
    def test_panel_rows_are_the_same_date_for_every_ticker(self):
        days, coin, equity = mixed_calendar()
        panel = historical_data.aligned_panel(yf_frame({"COIN": coin, "EQ": equity}), ["COIN", "EQ"])
        assert len(panel) == int((days.dayofweek < 5).sum())
        coin_dates = days[(panel.close["COIN"].to_numpy() - 1000).astype(int)]
        eq_dates = days[(panel.close["EQ"].to_numpy() - 2000).astype(int)]
        assert (coin_dates == eq_dates).all() and (coin_dates == panel.dates).all()

    def test_short_histories_are_dropped_not_allowed_to_shrink_the_calendar(self):
        days, coin, equity = mixed_calendar()
        stub = pd.Series(np.nan, days)
        stub.iloc[-5:] = 50.0
        panel = historical_data.aligned_panel(yf_frame({"COIN": coin, "EQ": equity, "NEW": stub}),
                                              ["COIN", "EQ", "NEW"], min_rows=60)
        assert panel.tickers == ["COIN", "EQ"]

    def test_benchmark_on_a_closed_day_carries_its_last_close(self):
        days, coin, equity = mixed_calendar()
        data = yf_frame({"COIN": coin, "SPY": equity})
        bench = historical_data.series_on_calendar(data, "SPY", pd.DatetimeIndex(days[7:14]))
        expected = [equity[:d].dropna().iloc[-1] for d in days[7:14]]
        assert list(bench) == expected

    def test_flat_single_symbol_frames_are_supported(self):
        idx = pd.bdate_range("2024-01-01", periods=30)
        flat = pd.DataFrame({"Close": np.linspace(10, 20, 30), "High": 21.0, "Low": 9.0, "Volume": 5.0}, idx)
        panel = historical_data.aligned_panel(flat, ["ABC"])
        assert panel.tickers == ["ABC"] and len(panel) == 30

    def test_backtest_prices_both_calendars_on_the_same_date(self, monkeypatch):
        days, coin, equity = mixed_calendar()
        monkeypatch.setattr(backtest_service.yf, "download",
                            lambda *a, **k: yf_frame({"COIN": coin, "EQ": equity, "SPY": equity}))
        seen = {}
        original = backtest_service.BacktestEngine._run_strategy

        def spy(self, asset_data, start_bar, n_bars, dates=None):
            seen.update(asset_data=asset_data, n_bars=n_bars, dates=dates)
            return original(self, asset_data, start_bar, n_bars, dates)

        monkeypatch.setattr(backtest_service.BacktestEngine, "_run_strategy", spy)
        result = backtest_service.BacktestEngine(backtest_service.BacktestConfig(
            assets=["COIN", "EQ"], strategy=backtest_service.StrategyConfig(strategy_type="momentum"),
            execution=zero_cost_config())).run()
        last = seen["n_bars"] - 1
        coin_day = days[int(seen["asset_data"]["COIN"]["close"][last] - 1000)]
        eq_day = days[int(seen["asset_data"]["EQ"]["close"][last] - 2000)]
        assert coin_day == eq_day == days[days.dayofweek < 5][-1]
        assert result["data_end"] == eq_day.strftime("%Y-%m-%d")
        assert result["assets_traded"] == ["COIN", "EQ"]


# ── 2. One strategy definition, one timing rule ──────────────────────────────

STRATEGIES = [
    strategy_signals.StrategyConfig(strategy_type="ma_crossover", fast_window=5, slow_window=20),
    strategy_signals.StrategyConfig(strategy_type="momentum"),
    strategy_signals.StrategyConfig(strategy_type="mean_reversion", mr_lookback=15),
    strategy_signals.StrategyConfig(strategy_type="sba_signal"),
]


class TestStrategySignals:
    @pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.strategy_type)
    def test_signal_at_t_never_depends_on_later_closes(self, strategy):
        close = random_walk(4, 300)
        base = strategy_signals.signal_series(close, strategy)
        for cut in (60, 150, 240):
            changed = close.copy()
            changed[cut + 1:] *= np.random.default_rng(cut).uniform(0.5, 1.5, len(close) - cut - 1)
            assert np.array_equal(strategy_signals.signal_series(changed, strategy)[:cut + 1], base[:cut + 1])

    @pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.strategy_type)
    def test_latest_signal_is_the_series_last_element(self, strategy):
        close = random_walk(9, 200)
        series = strategy_signals.signal_series(close, strategy)
        for t in (80, 150, 199):
            assert strategy_signals.latest_signal(close[:t + 1], strategy) == series[t]

    def _step_engine(self, delay=1):
        cfg = backtest_service.BacktestConfig(
            assets=["X"], strategy=backtest_service.StrategyConfig(fast_window=5, slow_window=20),
            execution=zero_cost_config())
        cfg.execution.execution_delay_bars = delay
        return backtest_service.BacktestEngine(cfg)

    def test_backtester_fills_one_bar_after_the_signal_bar(self):
        px = np.r_[np.full(80, 100.0), np.linspace(100.5, 130, 60)]
        signal_bar = int(np.flatnonzero(strategy_signals.ma_crossover_series(px, 5, 20) > 0)[0])
        data = {"X": {"close": px, "volume": np.full(len(px), 1e6), "high": px, "low": px}}
        for delay in (1, 2):
            first_buy = self._step_engine(delay)._run_strategy(data, 25, len(px))["trade_log"][0]
            assert first_buy.date == f"bar_{signal_bar + delay}"
            assert first_buy.fill_price == px[signal_bar + delay]

    def test_same_bar_execution_is_refused(self):
        px = np.linspace(100, 120, 100)
        data = {"X": {"close": px, "volume": np.full(100, 1e6), "high": px, "low": px}}
        with pytest.raises(ValueError, match="at least 1"):
            self._step_engine(delay=0)._run_strategy(data, 30, 100)

    def test_event_simulator_uses_the_shared_crossover(self):
        px = {"A": random_walk(1, 300), "B": random_walk(2, 300)}
        strat = event_simulator.MACrossoverStrategy(fast=5, slow=20)
        strat.prepare(px)
        for t, close in px.items():
            expected = strategy_signals.ma_crossover_series(close, 5, 20)
            emitted = np.zeros(len(close))
            for bar in range(len(close)):
                for sig in strat.on_market(event_simulator.MarketEvent(ticker=t, timestamp=bar),
                                           event_simulator.Portfolio(), {t: list(close[:bar + 1])}):
                    emitted[bar] = sig.suggested_direction
            assert np.array_equal(emitted, expected)

    def test_event_simulator_refuses_unaligned_arrays(self):
        with pytest.raises(ValueError, match="date-aligned"):
            event_simulator.run_event_backtest(["A", "B"], {"A": np.ones(100), "B": np.ones(90)})


# ── 3. Walk-forward covers every asset through the same engine ──────────────

@pytest.fixture
def wf_market(monkeypatch):
    idx = pd.bdate_range("2019-01-01", periods=5 * 252 + 20)
    frame = yf_frame({t: pd.Series(random_walk(i, len(idx)), idx) for i, t in enumerate(["AAA", "BBB"])})
    monkeypatch.setattr(walk_forward.yf, "download", lambda *a, **k: frame)


def run_wf(assets, optimize=True):
    return walk_forward.WalkForwardEngine(walk_forward.WalkForwardConfig(
        assets=assets, execution=zero_cost_config(), optimize_parameters=optimize)).run()


class TestWalkForward:
    def test_every_asset_is_evaluated_out_of_sample(self, wf_market):
        both, a, b = run_wf(["AAA", "BBB"], False), run_wf(["AAA"], False), run_wf(["BBB"], False)
        combined = both["aggregated_oos"]["total_return"]
        assert combined not in (a["aggregated_oos"]["total_return"], b["aggregated_oos"]["total_return"])
        assert both["assets"] == ["AAA", "BBB"]

    def test_result_does_not_depend_on_asset_order(self, wf_market):
        assert run_wf(["AAA", "BBB"])["aggregated_oos"] == run_wf(["BBB", "AAA"])["aggregated_oos"]

    def test_walk_forward_is_long_only(self, wf_market):
        engine = walk_forward.WalkForwardEngine(walk_forward.WalkForwardConfig(assets=["AAA"]))
        assert engine._execution.allow_short_selling is False
        close = random_walk(5, 800)
        bt = backtest_service.BacktestEngine(backtest_service.BacktestConfig(
            assets=["X"], strategy=backtest_service.StrategyConfig(fast_window=10, slow_window=40),
            execution=engine._execution))
        trades = bt._run_strategy(walk_forward._asset_data_from_closes({"X": close}), 60, 800)["trade_log"]
        assert trades and all(t.position_after >= 0 for t in trades)

    def test_reports_trials_and_full_out_of_sample_coverage(self, wf_market):
        result = run_wf(["AAA"])
        assert result["n_trials"] == 25 and len(result["parameter_grid"]) == 25
        assert run_wf(["AAA"], optimize=False)["n_trials"] == 1
        assert result["aggregated_oos"]["n_oos_days"] == result["n_folds"] * 252
        assert result["folds"][0]["test_dates"].count("..") == 1

    def test_fold_evaluation_is_the_backtester(self):
        close = random_walk(3, 600)
        engine = walk_forward.WalkForwardEngine(walk_forward.WalkForwardConfig(
            assets=["X"], execution=zero_cost_config()))
        metrics = engine._evaluate_period({"X": close}, ["X"], 300, 600, 10, 40)
        bt = backtest_service.BacktestEngine(backtest_service.BacktestConfig(
            assets=["X"], strategy=backtest_service.StrategyConfig(fast_window=10, slow_window=40),
            execution=engine._execution, initial_capital=100_000.0))
        direct = bt._run_strategy(walk_forward._asset_data_from_closes({"X": close}), 300, 600)
        assert metrics["equity"][1:] == [float(v) for v in direct["equity_curve_net"]]


# ── 4. Deflated Sharpe ratio ────────────────────────────────────────────────

def _annual_sharpe(r):
    return float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(252))


class TestDeflatedSharpe:
    @pytest.mark.parametrize("n_obs", [60, 756])
    def test_false_positive_rate_is_nominal_at_any_sample_length(self, n_obs):
        rng = np.random.default_rng(n_obs)
        sims = 2000
        flagged = sum(
            stat_tests.deflated_sharpe_ratio(_annual_sharpe(r), 1, n_obs, stat_tests._skewness(r),
                                             stat_tests._kurtosis(r))["significant_5pct"]
            for r in (rng.normal(0, 0.01, n_obs) for _ in range(sims)))
        assert 0.03 <= flagged / sims <= 0.07

    def test_short_sample_is_not_significant(self):
        # Annualised SR 1.5 over 60 days has a standard error of ~2.
        r = np.random.default_rng(3).normal(0, 1, 60)
        r = (r - r.mean()) / r.std(ddof=1) * 0.01 + 1.5 / math.sqrt(252) * 0.01
        dsr = stat_tests.run_full_stat_tests(r, n_strategies_tested=1)["deflated_sharpe"]
        assert dsr["sharpe_standard_error"] > 1.5 and not dsr["significant_5pct"]

    def test_short_samples_accepted_by_the_endpoint_are_handled(self):
        # The stat-test endpoint accepts 5+ observations; below 10 the
        # permutation test cannot randomize but must still answer.
        result = stat_tests.run_full_stat_tests(np.array([0.01, -0.02, 0.015, 0.003, -0.004, 0.02]))
        assert result["permutation_test"]["significant_5pct"] is False
        assert result["summary"]["permutation_significant"] is False

    def test_benchmark_grows_with_trials(self):
        assert stat_tests.expected_max_sharpe(1, 0.1) == 0.0
        values = [stat_tests.expected_max_sharpe(n, 0.1) for n in (2, 10, 100, 1000)]
        assert values == sorted(values) and values[0] > 0


class TestResearchReport:
    def test_report_pipeline_runs_on_aligned_prices(self, monkeypatch):
        import yfinance
        from backend.services import report_generator

        idx = pd.bdate_range("2019-01-01", periods=5 * 252 + 20)
        names = ["AAA", "BBB", "CCC", "DDD"]
        series = {t: pd.Series(random_walk(i + 10, len(idx)), idx) for i, t in enumerate(names)}
        series["DDD"].iloc[:300] = np.nan                     # listed later than the others
        monkeypatch.setattr(yfinance, "download", lambda *a, **k: yf_frame(series))
        report = report_generator.run_full_report_pipeline(names, period="5y")
        assert "error" not in report
        assert report["metadata"]["tickers_used"] == names
        assert report["metadata"]["n_bars"] == len(idx) - 300 - 1   # common days, no back-filled history


# ── 5. Server-counted trials ────────────────────────────────────────────────

class TestResearchTrials:
    def test_distinct_configs_are_counted_once_per_user_and_family(self, db):
        alice, bob = make_user(db, "alice@example.com"), make_user(db, "bob@example.com")
        fam = research_trials.family_hash("ma_crossover", ["aapl", "MSFT"])
        assert fam == research_trials.family_hash("ma_crossover", ["MSFT", "AAPL"])
        configs = [{"fast": f, "slow": 50} for f in (10, 20, 30)]
        assert research_trials.record(db, alice.id, fam, configs, "walk_forward") == 3
        assert research_trials.record(db, alice.id, fam, configs[:1] + [{"fast": 40, "slow": 50}], "backtest") == 4
        assert research_trials.count(db, bob.id, fam) == 0

    def test_stat_test_uses_at_least_the_server_count(self, db):
        user = make_user(db)
        fam = research_trials.family_hash("ma_crossover", ["AAPL"])
        research_trials.record(db, user.id, fam, [{"fast": f} for f in range(25)], "walk_forward")
        returns = list(np.random.default_rng(1).normal(0.001, 0.01, 300))
        declared_only = main.statistical_tests(
            schemas.StatTestRequest(returns=returns, n_strategies_tested=1), user, db)
        counted = main.statistical_tests(
            schemas.StatTestRequest(returns=returns, n_strategies_tested=1, trial_family=fam), user, db)
        assert declared_only["trial_count"]["used"] == 1
        assert counted["trial_count"] == {"used": 25, "declared": 1, "server_counted": 25, "family": fam}
        assert counted["deflated_sharpe"]["n_trials"] == 25
        assert (counted["deflated_sharpe"]["expected_max_sharpe"]
                > declared_only["deflated_sharpe"]["expected_max_sharpe"])


# ── 6. Audit hash chain ─────────────────────────────────────────────────────

@pytest.fixture
def file_db(make_engine):
    return sessionmaker(bind=make_engine())


@pytest.fixture
def frequent_thread_switches():
    """Switch threads every microsecond so interleavings that a busy server
    hits only occasionally happen on every run."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(previous)


class TestPqcThreadSafety:
    def test_concurrent_ml_dsa_signatures_all_verify(self, frequent_thread_switches):
        pk, sk, _ = pqc.dsa_keygen()
        results = []

        def signer(n):
            for i in range(4):
                message = f"signer-{n}-{i}".encode()
                signature, _ = pqc.dsa_sign(sk, message)
                results.append(pqc.dsa_verify(pk, message, signature))

        threads = [threading.Thread(target=signer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == [True] * 16


class TestAuditChain:
    def test_concurrent_writers_keep_one_contiguous_valid_chain(self, file_db, frequent_thread_switches):
        security_service.write_audit_log(file_db(), None, "GENESIS")
        errors = []

        def writer(n):
            session = file_db()
            try:
                for i in range(8):
                    security_service.write_audit_log(session, None, "CONCURRENT", metadata={"w": n, "i": i})
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)
            finally:
                session.close()

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        db = file_db()
        sequences = [s for (s,) in db.execute(text("SELECT sequence FROM audit_chain_links ORDER BY sequence"))]
        assert sequences == list(range(1, 50))
        status = security_service.audit_chain_status(db)
        assert status.pop("last_full_verification_at") is not None
        assert status == {"valid": True, "links": 49, "unchained_events": 0,
                          "first_invalid_sequence": None, "reason": None,
                          "mode": "full", "verified_from_sequence": 1}

    def test_postgres_lock_is_taken_before_the_chain_head_is_read(self, db, monkeypatch):
        # Recorded at the driver, which sees every statement whichever API
        # sent it (the lock and the head read go out as one query string).
        import psycopg

        executed = []
        real_execute = psycopg.Cursor.execute

        def recording_execute(cursor, query, *args, **kwargs):
            executed.append(query if isinstance(query, str) else query.as_string(cursor))
            return real_execute(cursor, query, *args, **kwargs)

        monkeypatch.setattr(psycopg.Cursor, "execute", recording_execute)
        security_service.write_audit_log(db, None, "LOCK_ORDER")
        sent = "\n".join(executed)
        assert "pg_advisory_xact_lock" in sent and "FROM audit_chain_links" in sent
        assert sent.index("pg_advisory_xact_lock") < sent.index("FROM audit_chain_links")

    def test_tampering_is_located(self, db):
        for i in range(3):
            security_service.write_audit_log(db, None, "EVENT", metadata={"i": i})
        second = db.execute(text("SELECT audit_log_id FROM audit_chain_links WHERE sequence = 2")).scalar()
        db.execute(text("UPDATE audit_logs SET action = 'FORGED' WHERE id = :id"), {"id": second})
        db.commit()
        db.expire_all()
        status = security_service.audit_chain_status(db)
        assert not status["valid"] and status["first_invalid_sequence"] == 2

    def test_chain_endpoint_is_operator_only(self, db):
        user, admin = make_user(db), make_user(db, "ops@example.com", role="risk_admin")
        security_service.write_audit_log(db, user.id, "EVENT")
        with pytest.raises(HTTPException) as exc:
            main.audit_chain(user, db)
        assert exc.value.status_code == 403
        assert main.audit_chain(admin, db)["valid"] is True


# ── 7. No execution against stale quotes ────────────────────────────────────

class _QuoteTicker:
    price, market_time = 341.07, 0.0

    def __init__(self, _symbol):
        self.history_metadata = {}

    def history(self, **_kw):
        self.history_metadata = {"regularMarketPrice": self.price, "regularMarketTime": self.market_time}
        return pd.DataFrame()


class TestQuoteFreshness:
    def _quote(self, monkeypatch, age_seconds):
        _QuoteTicker.market_time = time.time() - age_seconds
        monkeypatch.setattr(trading_service.yf, "Ticker", _QuoteTicker)
        trading_service._price_cache.clear()

    def test_fresh_trade_is_executable(self, monkeypatch):
        self._quote(monkeypatch, 60)
        assert trading_service.get_last_price("AAPL") == 341.07
        assert trading_service.get_mark_price("AAPL") == (341.07, False)

    def test_closed_market_close_is_a_mark_not_an_execution_price(self, monkeypatch):
        self._quote(monkeypatch, 62 * 3600)
        with pytest.raises(trading_service.StaleMarketData) as exc:
            trading_service.get_last_price("AAPL")
        assert exc.value.price == 341.07 and exc.value.age_seconds > 61 * 3600
        assert "2.6 days ago" in str(exc.value)
        assert trading_service.get_mark_price("AAPL") == (341.07, True)

    @pytest.fixture
    def market(self, monkeypatch):
        """asset -> (price, age_seconds)."""
        quotes: dict[str, tuple[float, float]] = {}

        def last_price(asset):
            price, age = quotes[asset]
            if age > trading_service.MAX_QUOTE_AGE_SECONDS:
                raise trading_service.StaleMarketData(asset, price, time.time() - age,
                                                      trading_service.MAX_QUOTE_AGE_SECONDS)
            return price

        monkeypatch.setattr(trading_service, "get_last_price", last_price)
        return quotes

    def _order(self, db, user, **fields):
        req = schemas.OrderRequest(**{"asset": "AAPL", "side": "buy", "quantity": 10,
                                      "order_type": "market", **fields})
        return main.place_order(req, user, db, None)

    def test_market_order_on_a_stale_quote_is_refused(self, db, market):
        user = make_user(db)
        market["AAPL"] = (341.07, 62 * 3600)
        with pytest.raises(HTTPException) as exc:
            self._order(db, user)
        assert exc.value.status_code == 409 and "not trading" in exc.value.detail
        assert db.query(models.Trade).count() == 0

    def test_limit_order_rests_and_fills_only_against_a_live_quote(self, db, market):
        user = make_user(db)
        market["AAPL"] = (90.0, 62 * 3600)               # closed, and "marketable" at 90
        placed = self._order(db, user, order_type="limit", limit_price=100.0)
        assert placed["status"] == "ACCEPTED" and placed["filled_price"] is None
        assert main.portfolio_account(user, db)["reserved_cash"] == pytest.approx(1000.0)
        assert paper_broker.process_open_orders(db) == []          # still stale: no fill
        market["AAPL"] = (95.0, 30)                                # market opens
        outcomes = paper_broker.process_open_orders(db)
        assert [(t.id, o.status) for t, o in outcomes] == [(placed["order_id"], "FILLED")]
        assert float(db.get(models.Trade, placed["order_id"]).filled_price) == 95.0

    def test_ioc_order_on_a_stale_quote_expires(self, db, market):
        user = make_user(db)
        market["AAPL"] = (90.0, 62 * 3600)
        placed = self._order(db, user, order_type="limit", limit_price=100.0, time_in_force="ioc")
        assert placed["status"] == "EXPIRED"
        assert main.portfolio_account(user, db)["reserved_cash"] == 0.0
