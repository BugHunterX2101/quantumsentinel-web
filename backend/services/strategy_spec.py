"""QuantumSentinel — strategy specs (DSL v1), their ids and their compiler.

A strategy that research produces, by hand now and by a model later, is data,
never code. A spec names one of the engine's tested strategy templates with
its parameters, and the compiler turns it into the arguments of
``event_simulator.run_event_backtest``. No generated code ever runs.

* ``StrategySpec`` is DSL v1: a flat object of required fields. A spec has no
  defaults, so its canonical form is exactly what was given (after symbols
  are upper-cased and sorted). Values the engine would clamp or ignore are
  refused rather than stored, so every field in the bytes is what runs.
* Numbers are integers: fractions are basis points (``*_bps``, 1/10,000) and
  the mean-reversion band width is in hundredths (``n_std_centi``). Floats
  are refused, so the id never depends on how a language prints a float.
* A spec's id is ``qs-sv-sha256-v1:<hex>``, the SHA-256 of its canonical
  JSON (RFC 8785 on the restricted domain of
  ``dataset_format.canonical_json``). tests/fixtures/qs_sv_sha256_v1.json
  holds golden vectors for other implementations.
* Every field changes what the engine runs, except ``universe.family``, which
  selects the data (and its holdout), and ``risk_limits``. Those are
  declared limits that the validation gate checks a backtest never breached
  (R3); nothing here enforces them. ``execution`` has one value,
  ``next_bar_close``: the engine fills an order at the close of the bar after
  its signal. ``momentum`` rebalances every fifth bar; the other templates
  act on every bar.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Mapping, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .dataset_format import canonical_json
from .dataset_providers import FAMILIES
from .event_simulator import run_event_backtest

SPEC_SCHEME = "qs-strategy-v1"
VERSION_SCHEME = "qs-sv-sha256-v1"
STRATEGY_VERSION_ID = re.compile(rf"{VERSION_SCHEME}:[0-9a-f]{{64}}")
MAX_SYMBOLS = 10

_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.^=-]{0,19}")
_BPS = 10_000


class StrategySpecError(ValueError):
    """A spec that cannot run on the data it was given."""


class _Model(BaseModel):
    # strict: an integer field refuses 2.0, "2" and True alike.
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


# --------------------------------------------------------------------------
# Signal templates: the engine's strategies
# --------------------------------------------------------------------------

class MACrossover(_Model):
    """Long while the fast moving average is above the slow one."""
    template: Literal["ma_crossover"]
    fast: int = Field(ge=2, le=200)
    slow: int = Field(ge=5, le=500)

    @model_validator(mode="after")
    def _slow_exceeds_fast(self):
        if self.slow <= self.fast:
            raise ValueError("slow must be larger than fast")
        return self


class Momentum(_Model):
    """Cross-sectional momentum over ``lookback`` bars, every fifth bar: long
    the top ``n_long``, sell the bottom ``n_short``, each at most a third of
    the universe."""
    template: Literal["momentum"]
    lookback: int = Field(ge=2, le=500)
    n_long: int = Field(ge=0, le=MAX_SYMBOLS)
    n_short: int = Field(ge=0, le=MAX_SYMBOLS)

    @model_validator(mode="after")
    def _trades_something(self):
        if self.n_long + self.n_short == 0:
            raise ValueError("n_long and n_short cannot both be 0")
        return self


class MeanReversion(_Model):
    """Bollinger-band mean reversion: buy below and sell above the mean by
    ``n_std_centi / 100`` population standard deviations of ``window`` bars."""
    template: Literal["mean_reversion"]
    window: int = Field(ge=2, le=500)
    n_std_centi: int = Field(ge=1, le=1_000)


Signal = Annotated[Union[MACrossover, Momentum, MeanReversion], Field(discriminator="template")]


# --------------------------------------------------------------------------
# Sizing, universe, limits
# --------------------------------------------------------------------------

class FixedFractional(_Model):
    """Each order is ``fraction_bps`` of equity, scaled by signal strength."""
    method: Literal["fixed_fractional"]
    fraction_bps: int = Field(ge=1, le=_BPS)


class VolatilityTarget(_Model):
    """Each order targets ``target_vol_bps`` annualised volatility from the
    last 21 returns, at most 5x leverage and a quarter of equity."""
    method: Literal["volatility_target"]
    target_vol_bps: int = Field(ge=1, le=2 * _BPS)


Sizing = Annotated[Union[FixedFractional, VolatilityTarget], Field(discriminator="method")]


class Universe(_Model):
    """The instruments the strategy trades, from one dataset family."""
    family: str
    symbols: list[str] = Field(min_length=1, max_length=MAX_SYMBOLS)

    @field_validator("family")
    @classmethod
    def _known_family(cls, value: str) -> str:
        if value not in FAMILIES:
            raise ValueError(f"unknown dataset family {value!r}")
        return value

    @field_validator("symbols")
    @classmethod
    def _canonical_symbols(cls, value: list[str]) -> list[str]:
        symbols = [s.strip().upper() for s in value]
        for symbol in symbols:
            if not _SYMBOL.fullmatch(symbol):
                raise ValueError(f"invalid symbol {symbol!r}")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be distinct")
        return sorted(symbols)


class RiskLimits(_Model):
    """Declared limits: the most equity the strategy may lose from a peak, and
    the largest position as a share of equity. Checked by the validation gate
    against a backtest, never enforced while it runs."""
    max_drawdown_bps: int = Field(ge=1, le=_BPS)
    max_position_bps: int = Field(ge=1, le=_BPS)


class StrategySpec(_Model):
    dsl: Literal["qs-strategy-v1"]
    signal: Signal
    universe: Universe
    sizing: Sizing
    cost_model: Literal["zero_cost", "retail", "institutional"]
    allow_short: bool
    # Largest gross exposure as a share of equity; an order that would take
    # the portfolio past it is skipped.
    leverage_limit_bps: int = Field(ge=1, le=5 * _BPS)
    execution: Literal["next_bar_close"]
    risk_limits: RiskLimits

    @model_validator(mode="after")
    def _momentum_fits_the_universe(self):
        # The engine trades at most a third of the ranked universe on each
        # side and silently lowers larger counts, so a spec asking for more
        # would claim positions it never takes. Without shorting, a sell can
        # only flatten, so a long-only momentum spec needs a long side.
        if not isinstance(self.signal, Momentum):
            return self
        third = len(self.universe.symbols) // 3
        if third == 0:
            raise ValueError("momentum needs at least 3 symbols")
        if self.signal.n_long > third or self.signal.n_short > third:
            raise ValueError(f"n_long and n_short can be at most a third of the universe ({third})")
        if self.signal.n_long == 0 and not self.allow_short:
            raise ValueError("long-only momentum needs n_long of at least 1")
        return self


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def parse(value: Any) -> StrategySpec:
    """Validate a spec given as parsed JSON (a dict)."""
    return StrategySpec.model_validate(value)


def canonical(spec: StrategySpec) -> bytes:
    return canonical_json(spec.model_dump(mode="python"))


def version_id(spec: StrategySpec) -> str:
    return f"{VERSION_SCHEME}:{hashlib.sha256(canonical(spec)).hexdigest()}"


# --------------------------------------------------------------------------
# Compiler
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledStrategy:
    """A spec as the event engine's arguments."""
    version_id: str
    tickers: tuple[str, ...]
    engine_kwargs: Mapping[str, Any]

    def run(self, price_data: Mapping[str, np.ndarray], initial_capital: float) -> dict:
        """Backtest on date-aligned close arrays, one per universe symbol."""
        missing = sorted(set(self.tickers) - set(price_data))
        if missing:
            raise StrategySpecError(f"no prices for {', '.join(missing)}")
        return run_event_backtest(tickers=list(self.tickers),
                                  price_data={t: price_data[t] for t in self.tickers},
                                  initial_capital=initial_capital, **self.engine_kwargs)


def _signal_params(signal) -> dict:
    if isinstance(signal, MACrossover):
        return {"fast": signal.fast, "slow": signal.slow}
    if isinstance(signal, Momentum):
        return {"lookback": signal.lookback, "n_long": signal.n_long, "n_short": signal.n_short}
    return {"window": signal.window, "n_std": signal.n_std_centi / 100}


def _sizing_params(sizing) -> dict:
    if isinstance(sizing, FixedFractional):
        return {"fraction": sizing.fraction_bps / _BPS}
    return {"target_vol": sizing.target_vol_bps / _BPS}


def compile_spec(spec: StrategySpec) -> CompiledStrategy:
    params = {**_signal_params(spec.signal), **_sizing_params(spec.sizing),
              "leverage_limit": spec.leverage_limit_bps / _BPS}
    return CompiledStrategy(
        version_id=version_id(spec),
        tickers=tuple(spec.universe.symbols),
        engine_kwargs={
            "strategy_name": spec.signal.template,
            "strategy_params": params,
            "cost_model_name": spec.cost_model,
            "allow_short": spec.allow_short,
            "sizing_method": spec.sizing.method,
        },
    )
