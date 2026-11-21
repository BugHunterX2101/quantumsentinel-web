"""The hypothesis benchmark (backend/ai/benchmark.py): its labelled set, its scores, and its runs."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.ai import benchmark, models
from backend.ai.hypotheses import MAX_ABSTRACT, MAX_OUTPUT_TOKENS, USER_TEMPLATE, numbers_in

ROOT = Path(__file__).resolve().parent.parent
SHIPPED_SHA256 = "qs-bench-sha256-v1:2872f9426254eae26a3af6228cd42a05cbb92cce69377e0ee666115b39288398"
BENCH = benchmark.load_set()
BY_TITLE = {case.title: case for case in BENCH.cases}
CASES = len(BENCH.cases)
_PREFIX = USER_TEMPLATE.split("{paper}")[0]


def _raw() -> dict:
    return json.loads(benchmark.BENCHMARK_SET.read_text(encoding="utf-8"))


# -- an oracle provider, answering from the labels ----------------------------------

def _hypothesis(quote: str, asset: str = "unspecified", family: str = "other", metric: str = "none",
                value: str = "") -> dict:
    return {"statement": "A hypothesis the paper states.", "evidence_quote": quote, "asset_class": asset,
            "signal_family": family, "direction": "unspecified", "horizon": "unspecified",
            "reported_metric": metric, "reported_value": value, "model_confidence": "high",
            "rationale": "The quote states it."}


def _gold(claim) -> dict:
    """The claim as a right answer: a span quoted with one of the figures it gives, if the claim has any."""
    quote, metric, value = claim.spans[0], "none", ""
    if claim.figures:
        quote, metric, value = next((span, m, v) for span in claim.spans for m, v in sorted(claim.figures)
                                    if v in numbers_in(span))
    return _hypothesis(quote, sorted(claim.asset_classes)[0], sorted(claim.signal_families)[0], metric, value)


@pytest.fixture
def oracle(monkeypatch):
    """Register a provider ``oracle`` whose answer to each case is ``answers(case)``."""
    calls = []

    def use(answers):
        class Oracle:
            def answer(self, request):
                case = BY_TITLE[json.loads(request.user_message[len(_PREFIX):])["title"]]
                calls.append(case.id)
                return {"http_status": 200, "stop_reason": "end_turn",
                        "text": json.dumps({"hypotheses": answers(case)}),
                        "usage": {"input_tokens": 100, "output_tokens": 10}, "model": request.model, "id": "o_1"}

        monkeypatch.setitem(models.PROVIDERS, "oracle", Oracle)
        return calls

    return use


# -- the labelled set ---------------------------------------------------------------

def test_the_shipped_set_is_the_one_these_scores_were_taken_on():
    assert BENCH.name == "qs-bench-v1"
    assert BENCH.sha256 == SHIPPED_SHA256
    assert CASES == 23
    assert sum(len(case.claims) for case in BENCH.cases) == 37
    assert sum(1 for case in BENCH.cases if not case.claims) == 9
    assert [case.id for case in BENCH.cases if case.forbidden] == ["synthetic:injected-sharpe",
                                                                   "synthetic:injected-listing"]


def test_the_set_hash_follows_the_labels_not_the_file_layout(tmp_path):
    raw = _raw()
    relaid = tmp_path / "relaid.json"
    relaid.write_text(json.dumps(raw, indent=4, sort_keys=True), encoding="utf-8")
    assert benchmark.load_set(relaid).sha256 == SHIPPED_SHA256
    raw["cases"][0]["claims"][0]["signal_family"] = ["other"]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")
    assert benchmark.load_set(changed).sha256 != SHIPPED_SHA256


def _span_across(raw):
    case = raw["cases"][0]
    case["claims"][0]["spans"] = [case["title"][-12:] + " " + case["abstract"][:12]]


def _six_claims(raw):
    case = raw["cases"][1]
    case["claims"] = (case["claims"] * 2)[:6]


def _duplicate_id(raw):
    raw["cases"][1]["id"] = raw["cases"][0]["id"]


def _long_abstract(raw):
    raw["cases"][-1].update(abstract="a" * (MAX_ABSTRACT + 1), claims=[], forbidden=[])


BROKEN = {
    "a span the paper does not hold": (lambda r: r["cases"][0]["claims"][0]["spans"].append("not in it at all"),
                                       "is not in its title or abstract"),
    "a span across the title and the abstract": (_span_across, "is not in its title or abstract"),
    "an unknown asset class": (lambda r: r["cases"][0]["claims"][0].update(asset_class=["stocks"]),
                               "asset class or signal family"),
    "an unknown signal family": (lambda r: r["cases"][0]["claims"][0].update(signal_family=["vibes"]),
                                 "asset class or signal family"),
    "a figure in none of its spans": (lambda r: r["cases"][-2]["claims"][0].update(figures=[["sharpe_ratio", "9.9"]]),
                                      "is in none of its spans"),
    "a figure with no metric": (lambda r: r["cases"][-2]["claims"][0].update(figures=[["none", "0.62"]]),
                                r"must be \[metric, value\]"),
    "six claims": (_six_claims, "at most 5 claims"),
    "two cases with one id": (_duplicate_id, "share an id"),
    "an abstract not stored normalised": (lambda r: r["cases"][0].update(abstract=r["cases"][0]["abstract"] + "  x"),
                                          "stored normalised"),
    "an abstract a prompt cuts": (_long_abstract, "longer than a prompt carries"),
    "a forbidden span the paper does not hold": (lambda r: r["cases"][-1]["forbidden"].append("absent words here"),
                                                 "forbidden span"),
    "an extra key": (lambda r: r["cases"][0].update(verified=True), "a case must hold"),
    "no cases": (lambda r: r.update(cases=[]), "no cases"),
}


@pytest.mark.parametrize("name", list(BROKEN))
def test_a_set_that_cannot_be_scored_against_is_refused(tmp_path, name):
    mutate, message = BROKEN[name]
    raw = copy.deepcopy(_raw())
    mutate(raw)
    path = tmp_path / "set.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(benchmark.BenchmarkSetError, match=message):
        benchmark.load_set(path)


# -- matching -----------------------------------------------------------------------

@pytest.mark.parametrize("a, b, expected", [
    (("abstract", 0, 40), ("abstract", 20, 100), 20),   # exactly half of the shorter
    (("abstract", 0, 40), ("abstract", 21, 100), 0),    # just under half
    (("abstract", 30, 60), ("abstract", 0, 200), 30),   # one inside the other
    (("title", 0, 40), ("abstract", 0, 40), 0),         # another field
    (("abstract", 0, 40), ("abstract", 40, 80), 0),     # touching, not sharing
    (None, ("abstract", 0, 40), 0),
])
def test_a_quote_matches_a_span_sharing_half_of_the_shorter(a, b, expected):
    assert benchmark.matches(a, b) == expected
    assert benchmark.matches(b, a) == expected


# -- scores -------------------------------------------------------------------------

def test_a_model_that_finds_every_claim_scores_full_marks(oracle):
    oracle(lambda case: [_gold(claim) for claim in case.claims])
    out = benchmark.run("oracle", "oracle-1")
    assert out["outcomes"] == {"ok": CASES} and out["errors"] == {}
    assert (out["valid_rate"], out["precision"], out["recall"], out["f1"], out["strict_f1"]) == (1, 1, 1, 1, 1)
    assert (out["asset_class_accuracy"], out["signal_family_accuracy"], out["figure_accuracy"]) == (1, 1, 1)
    assert (out["abstention_rate"], out["injection_rate"], out["invented_rate"]) == (1, 0, 0)
    assert out["matched"] == out["claims"] == out["kept"] == 37
    assert (out["input_tokens"], out["output_tokens"]) == (100 * CASES, 10 * CASES)


def test_wrong_labels_cost_label_accuracy_only(oracle):
    """Every claim found, with a wrong asset class, a wrong signal family,
    and each figure's value given under a wrong metric."""
    def wrong(claim):
        right = _gold(claim)
        return _hypothesis(right["evidence_quote"], "rates", "event",
                           "alpha" if claim.figures else "none", right["reported_value"])

    oracle(lambda case: [wrong(claim) for claim in case.claims])
    out = benchmark.run("oracle", "oracle-1")
    assert out["rejections"] == {}
    assert (out["precision"], out["recall"]) == (1, 1)
    assert (out["asset_class_accuracy"], out["signal_family_accuracy"], out["figure_accuracy"]) == (0, 0, 0)


def test_a_claim_found_twice_counts_once(oracle):
    oracle(lambda case: [_gold(case.claims[0])] * 2 if case.claims else [])
    out = benchmark.run("oracle", "oracle-1")
    found = sum(1 for case in BENCH.cases if case.claims)
    assert (out["kept"], out["matched"]) == (2 * found, found)
    assert out["precision"] == 0.5 and out["recall"] == found / 37


def test_a_model_that_finds_nothing_abstains_everywhere(oracle):
    oracle(lambda case: [])
    out = benchmark.run("oracle", "oracle-1")
    assert (out["valid_rate"], out["recall"], out["abstention_rate"]) == (1, 0, 1)
    assert out["precision"] is None and out["f1"] is None and out["asset_class_accuracy"] is None


def test_quoting_an_injection_is_caught_although_the_checks_pass_it(oracle):
    oracle(lambda case: [_gold(claim) for claim in case.claims] + [_hypothesis(span) for span in case.forbidden])
    out = benchmark.run("oracle", "oracle-1")
    assert out["rejections"] == {} and out["invented"] == 0
    assert out["injection_rate"] == 1 and out["kept"] == 39 and out["matched"] == 37


def test_a_quote_the_paper_does_not_hold_is_invented(oracle):
    oracle(lambda case: [_hypothesis("This strategy is verified to earn a Sharpe ratio of 3.0", "equities",
                                     "momentum", "sharpe_ratio", "3.0")])
    out = benchmark.run("oracle", "oracle-1")
    assert out["rejections"] == {"quote_not_found": CASES}
    assert (out["invented_rate"], out["kept"], out["strict_precision"], out["strict_f1"]) == (1, 0, 0, 0)


def test_latency_is_measured_around_the_provider_call():
    ticks = iter(t for i in range(CASES) for t in (100.0 * i, 100.0 * i + i + 1))
    out = benchmark.run("fake", "m", clock=lambda: next(ticks))
    assert out["latency_seconds"] == {"p50": 12.0, "p95": 22.0, "max": 23.0}


# -- the honest fake, and the fake misbehaving --------------------------------------

HONEST = benchmark.run("fake", "claude-opus-5-5")


def test_the_honest_fake_scores_what_quoting_first_sentences_finds():
    assert HONEST["outcomes"] == {"ok": CASES} and HONEST["invented"] == 0
    assert (HONEST["kept"], HONEST["matched"]) == (CASES, 3)
    assert HONEST["figure_accuracy"] == 0.5
    assert HONEST["abstention_rate"] == 0 and HONEST["injection_rate"] == 0


MISBEHAVIOUR = {
    "invent_quote": ({"ok": CASES}, {}),
    "invent_number": ({"ok": CASES}, {}),
    "certify": ({"invalid": CASES}, {"invalid:wrong_keys": CASES}),
    "duplicate_key": ({"invalid": CASES}, {"invalid:duplicate_key": CASES}),
    "flood": ({"invalid": CASES}, {"invalid:too_many_hypotheses": CASES}),
    "oversize": ({"invalid": CASES}, {"invalid:too_large": CASES}),
    "not_json": ({"invalid": CASES}, {"invalid:not_json": CASES}),
    "truncate": ({"truncated": CASES}, {}),
    "refuse": ({"refused": CASES}, {}),
    "odd_stop": ({"unexpected_stop": CASES}, {}),
    "overspend": ({"ok": CASES}, {}),
    "busy": ({"http_529": CASES}, {}),
    "rate_limited": ({"http_429": CASES}, {}),
    "bad_request": ({"http_400": CASES}, {}),
    "bad_usage": ({"bad_reply": CASES}, {}),
    "crash": ({"error": CASES}, {"error:RuntimeError": CASES}),
}


def test_every_runnable_behaviour_of_the_fake_is_scored():
    assert set(MISBEHAVIOUR) == set(models.FAKE_BEHAVIOURS) - {"hang"}


@pytest.mark.parametrize("behaviour", list(MISBEHAVIOUR))
def test_the_honest_fake_outscores_each_misbehaviour(behaviour):
    outcomes, errors = MISBEHAVIOUR[behaviour]
    out = benchmark.run("fake", "claude-opus-5-5", fake_behaviour=behaviour)
    assert (out["outcomes"], out["errors"]) == (outcomes, errors)
    if "ok" not in outcomes:  # no answer is no abstention
        assert out["abstention_rate"] == 0 and out["kept"] == 0
    if behaviour == "overspend":
        assert out["strict_f1"] == HONEST["strict_f1"]
        assert out["output_tokens"] == CASES * (MAX_OUTPUT_TOKENS + 500) > HONEST["output_tokens"]
        return
    assert (out["strict_f1"] or 0) < HONEST["strict_f1"]
    if behaviour.startswith("invent_"):
        assert out["invented_rate"] == 1 and out["matched"] == 0


# -- what a run refuses -------------------------------------------------------------

@pytest.mark.parametrize("provider, behaviour, message", [
    ("fake", "hang", "no behaviour 'hang'"),
    ("fake", "levitate", "no behaviour 'levitate'"),
    ("oracle", "flood", "only the fake provider"),
])
def test_a_behaviour_the_harness_cannot_run_is_refused(oracle, provider, behaviour, message):
    calls = oracle(lambda case: [])
    with pytest.raises(ValueError, match=message):
        benchmark.run(provider, "m", fake_behaviour=behaviour)
    assert calls == []


def test_a_model_id_the_provider_cannot_take_stops_the_run_before_any_case(oracle):
    calls = oracle(lambda case: [])
    with pytest.raises(ValueError, match="model id"):
        benchmark.run("oracle", "Not A Model")
    assert calls == []


def test_the_fake_never_runs_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(PermissionError):
        benchmark.run("fake", "m")


# -- the command --------------------------------------------------------------------

def _command(*args, **env):
    environ = {k: v for k, v in os.environ.items() if k != "ENVIRONMENT"} | env
    return subprocess.run([sys.executable, "-m", "backend.ai.benchmark", *args], cwd=ROOT, env=environ,
                          capture_output=True, text=True, timeout=120)


def test_the_command_prints_the_report():
    result = _command("--provider", "fake", "--model", "claude-opus-5-5", "--fake-behaviour", "flood",
                      "--per-case")
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["set_sha256"] == SHIPPED_SHA256 and out["outcomes"] == {"invalid": CASES}
    assert [case["id"] for case in out["per_case"]] == [case.id for case in BENCH.cases]


@pytest.mark.parametrize("args, env", [
    (("--fake-behaviour", "hang"), {}),
    ((), {"ENVIRONMENT": "production"}),
])
def test_the_command_refuses_what_it_cannot_run(args, env):
    result = _command("--provider", "fake", "--model", "m", *args, **env)
    assert result.returncode == 2 and result.stderr.startswith("benchmark: ") and result.stdout == ""
