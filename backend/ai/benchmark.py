"""How well a research model answers the hypothesis prompt, on a labelled set of papers.

The set (benchmark_cases.json, BENCHMARK_SET) is a few dozen titles and
abstracts, each labelled with the hypotheses it states (its *claims*). A
claim is the spans of the paper that state it, the asset classes and signal
families that are right for it, and the figures (metric and value) it
reports, if any. A paper may state none; a paper may also carry a prompt
injection, whose text is a *forbidden* span.

``run`` sends every paper to a provider exactly as the extract driver does
(hypotheses.prompt_for, then models.answer, extract.read_reply and
hypotheses.check_output), so what is scored is what extraction would have
recorded, and the same prompt version. A hypothesis that survives the checks
*matches* a claim when its evidence_quote and one of the claim's spans
overlap by at least half of the shorter of the two; hypotheses and claims are
paired one to one, the largest overlaps first. The report gives:

* the outcomes (ok, refused, truncated, invalid, unexpected_stop, http_<status>,
  bad_reply, error), why (``errors``: an invalid answer's code, a failed
  provider's exception) and ``valid_rate``, the share of papers answered ok;
* ``invented_rate``: the hypotheses dropped for a quote the paper does not
  hold or a figure its quote does not give, over all hypotheses proposed;
* ``precision`` (matched over kept), ``recall`` (claims matched over claims)
  and ``f1``; ``strict_precision`` and ``strict_f1`` count every proposed
  hypothesis, so one the checks dropped is a false positive there too;
* on matched pairs, the accuracy of asset_class and signal_family, and of the
  figure where the claim reports one;
* ``abstention_rate``: papers without claims that got no hypothesis;
* ``injection_rate``: papers with a forbidden span where a hypothesis quotes it;
* tokens as the provider reported them, and latency around the provider call.

The report names the set by the hash of its content and the prompt version,
so two reports compare only when both agree. Prices are not applied: they
change, and tokens are what the ledger counts.

The fake provider can be asked to misbehave on every paper
(``fake_behaviour``, one of models.FAKE_BEHAVIOURS except ``hang``: this
harness calls the provider in its own process, with no job process to stop).

    python -m backend.ai.benchmark --provider fake --model claude-opus-5-5 [--fake-behaviour flood]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import extract, models
from .documents import canonical
from .hypotheses import (ASSET_CLASSES, MAX_HYPOTHESES, METRICS, PROMPT_VERSION, SIGNAL_FAMILIES, Checked,
                         check_output, normalise, numbers_in, prompt_for)

BENCHMARK_SET = Path(__file__).with_name("benchmark_cases.json")
SET_SCHEME = "qs-bench-sha256-v1"
# Rejections that mean the model wrote something the paper does not say.
INVENTED = frozenset({"quote_not_found", "value_not_in_quote", "value_not_a_number"})
DEFAULT_TIMEOUT_SECONDS = 120


class BenchmarkSetError(ValueError):
    """The labelled set is not one this harness can score against."""


@dataclass(frozen=True)
class Claim:
    spans: tuple[str, ...]
    asset_classes: frozenset[str]
    signal_families: frozenset[str]
    figures: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    abstract: str
    claims: tuple[Claim, ...]
    forbidden: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkSet:
    name: str
    sha256: str
    cases: tuple[Case, ...]


# -- the labelled set ----------------------------------------------------------------

def _locate(case_title: str, case_abstract: str, text: str) -> tuple[str, int, int] | None:
    """Where ``text`` first occurs: in the title, else in the abstract."""
    for name, source in (("title", case_title), ("abstract", case_abstract)):
        start = source.find(text)
        if start >= 0:
            return name, start, start + len(text)
    return None


def _strings(value, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise BenchmarkSetError(f"{where} must be a list of text")
    return value


def _claim(raw, case_id: str, title: str, abstract: str) -> Claim:
    where = f"a claim of {case_id}"
    if not isinstance(raw, dict) or set(raw) != {"spans", "asset_class", "signal_family", "figures"}:
        raise BenchmarkSetError(f"{where} must hold spans, asset_class, signal_family and figures")
    spans = _strings(raw["spans"], f"the spans of {where}")
    for span in spans:
        if _locate(title, abstract, span) is None:
            raise BenchmarkSetError(f"a span of {where} is not in its title or abstract: {span[:60]!r}")
    assets = _strings(raw["asset_class"], f"the asset classes of {where}")
    families = _strings(raw["signal_family"], f"the signal families of {where}")
    if not set(assets) <= set(ASSET_CLASSES) or not set(families) <= set(SIGNAL_FAMILIES):
        raise BenchmarkSetError(f"{where} names an asset class or signal family the schema does not have")
    figures = raw["figures"]
    if not isinstance(figures, list):
        raise BenchmarkSetError(f"the figures of {where} must be a list")
    for figure in figures:
        if (not isinstance(figure, list) or len(figure) != 2 or figure[0] not in METRICS or figure[0] == "none"
                or not isinstance(figure[1], str)):
            raise BenchmarkSetError(f"a figure of {where} must be [metric, value]")
        if not any(figure[1] in numbers_in(span) for span in spans):
            raise BenchmarkSetError(f"a figure of {where} is in none of its spans: {figure[1]!r}")
    return Claim(tuple(spans), frozenset(assets), frozenset(families), frozenset(tuple(f) for f in figures))


def _case(raw) -> Case:
    if not isinstance(raw, dict) or set(raw) != {"id", "source", "title", "abstract", "claims", "forbidden"}:
        raise BenchmarkSetError("a case must hold id, source, title, abstract, claims and forbidden")
    case_id, title, abstract = raw["id"], raw["title"], raw["abstract"]
    if not all(isinstance(value, str) and value for value in (case_id, raw["source"], title, abstract)):
        raise BenchmarkSetError("a case's id, source, title and abstract must be text")
    if normalise(title) != title or normalise(abstract) != abstract:
        raise BenchmarkSetError(f"the title and abstract of {case_id} must be stored normalised")
    if prompt_for(title, abstract).abstract != abstract:
        raise BenchmarkSetError(f"the abstract of {case_id} is longer than a prompt carries")
    if not isinstance(raw["claims"], list) or len(raw["claims"]) > MAX_HYPOTHESES:
        raise BenchmarkSetError(f"{case_id} must have at most {MAX_HYPOTHESES} claims, so all can be found")
    claims = tuple(_claim(item, case_id, title, abstract) for item in raw["claims"])
    forbidden = _strings(raw["forbidden"], f"the forbidden spans of {case_id}") if raw["forbidden"] else []
    for span in forbidden:
        if _locate(title, abstract, span) is None:
            raise BenchmarkSetError(f"a forbidden span of {case_id} is not in its title or abstract")
    return Case(case_id, title, abstract, claims, tuple(forbidden))


def load_set(path: Path = BENCHMARK_SET) -> BenchmarkSet:
    """The labelled set at ``path``, checked; raises BenchmarkSetError."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchmarkSetError(f"the set cannot be read: {exc}") from None
    if not isinstance(raw, dict) or set(raw) != {"set", "about", "cases"} or not isinstance(raw["cases"], list):
        raise BenchmarkSetError("the set must hold set, about and cases")
    if not isinstance(raw["set"], str) or not raw["set"]:
        raise BenchmarkSetError("the set must be named")
    cases = tuple(_case(item) for item in raw["cases"])
    if not cases:
        raise BenchmarkSetError("the set has no cases")
    if len({case.id for case in cases}) != len(cases):
        raise BenchmarkSetError("two cases share an id")
    digest = hashlib.sha256(canonical({"scheme": SET_SCHEME, "set": raw["set"], "cases": raw["cases"]})).hexdigest()
    return BenchmarkSet(raw["set"], f"{SET_SCHEME}:{digest}", cases)


# -- scoring one paper --------------------------------------------------------------

def overlap(a: tuple[str, int, int] | None, b: tuple[str, int, int] | None) -> int:
    """Characters two located texts share (0 unless both are in the same field)."""
    if a is None or b is None or a[0] != b[0]:
        return 0
    return max(0, min(a[2], b[2]) - max(a[1], b[1]))


def matches(a: tuple[str, int, int] | None, b: tuple[str, int, int] | None) -> int:
    """The overlap of ``a`` and ``b`` if it is at least half of the shorter one, else 0."""
    shared = overlap(a, b)
    if shared == 0 or 2 * shared < min(a[2] - a[1], b[2] - b[1]):
        return 0
    return shared


@dataclass
class CaseResult:
    id: str
    outcome: str
    error: str | None = None
    latency_seconds: float | None = None
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    claims: int = 0
    proposed: int = 0
    kept: int = 0
    rejections: dict[str, int] = field(default_factory=dict)
    invented: int = 0
    matched: int = 0
    asset_class_correct: int = 0
    signal_family_correct: int = 0
    figure_claims_matched: int = 0
    figures_correct: int = 0
    injected: int = 0


def score(case: Case, checked: Checked, result: CaseResult) -> CaseResult:
    """``result`` with what ``checked`` (an answer for ``case``) scores."""
    result.outcome, result.error = checked.status, checked.error
    result.kept = len(checked.hypotheses)
    result.proposed = result.kept + len(checked.rejections)
    for _, reason in checked.rejections:
        result.rejections[reason] = result.rejections.get(reason, 0) + 1
    result.invented = sum(1 for _, reason in checked.rejections if reason in INVENTED)
    quotes = [_locate(case.title, case.abstract, h.evidence_quote) for h in checked.hypotheses]
    pairs = []
    for h_index, quote in enumerate(quotes):
        for c_index, claim in enumerate(case.claims):
            best = max(matches(quote, _locate(case.title, case.abstract, span)) for span in claim.spans)
            if best:
                pairs.append((best, h_index, c_index))
    used_h, used_c = set(), set()
    for _, h_index, c_index in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if h_index in used_h or c_index in used_c:
            continue
        used_h.add(h_index)
        used_c.add(c_index)
        h, claim = checked.hypotheses[h_index], case.claims[c_index]
        result.matched += 1
        result.asset_class_correct += h.asset_class in claim.asset_classes
        result.signal_family_correct += h.signal_family in claim.signal_families
        if claim.figures:
            result.figure_claims_matched += 1
            result.figures_correct += (h.reported_metric, h.reported_value) in claim.figures
    forbidden = [_locate(case.title, case.abstract, span) for span in case.forbidden]
    result.injected = sum(1 for quote in quotes if any(overlap(quote, span) for span in forbidden))
    return result


def run_case(case: Case, provider: str, model: str, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
             api_key: str | None = None, fake_behaviour: str | None = None, clock=time.perf_counter) -> CaseResult:
    """Send ``case`` to the provider as the extract driver would, and score the answer."""
    abstract = case.abstract if fake_behaviour is None else f"{case.abstract} [[fake:{fake_behaviour}]]"
    prompt = prompt_for(case.title, abstract)
    payload = {"provider": provider, "model": model, "user_message": prompt.user_message,
               "timeout_seconds": timeout_seconds}
    if api_key is not None:
        payload["api_key"] = api_key
    result = CaseResult(case.id, "error", claims=len(case.claims))
    started = clock()
    try:
        out = models.answer(payload)
    except Exception as exc:  # the provider failed: scored as such, the run goes on
        result.latency_seconds = clock() - started
        result.error = type(exc).__name__
        return result
    result.latency_seconds = clock() - started
    try:
        reply = extract.read_reply(out)
    except extract.ReplyError:
        result.outcome = "bad_reply"
        return result
    if reply.http_status != 200:
        result.outcome = f"http_{reply.http_status}"
        return result
    result.input_tokens = reply.usage.input_tokens
    result.cache_creation_input_tokens = reply.usage.cache_creation_input_tokens
    result.cache_read_input_tokens = reply.usage.cache_read_input_tokens
    result.output_tokens = reply.usage.output_tokens
    return score(case, check_output(reply.stop_reason, reply.text, prompt), result)


# -- the report ---------------------------------------------------------------------

def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _percentile(values: list[float], fraction: float) -> float | None:
    """The nearest-rank percentile of ``values``."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def report(bench: BenchmarkSet, results: list[CaseResult], *, provider: str, model: str,
           fake_behaviour: str | None = None, per_case: bool = False) -> dict:
    """The scores of ``results`` (one per case of ``bench``, in order)."""
    def total(name: str) -> int:
        return sum(getattr(r, name) for r in results)

    outcomes: dict[str, int] = {}
    errors: dict[str, int] = {}
    rejections: dict[str, int] = {}
    for r in results:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        if r.error is not None:
            errors[f"{r.outcome}:{r.error}"] = errors.get(f"{r.outcome}:{r.error}", 0) + 1
        for reason, count in r.rejections.items():
            rejections[reason] = rejections.get(reason, 0) + count
    cases = {case.id: case for case in bench.cases}
    no_claims = [r for r in results if not cases[r.id].claims]
    with_forbidden = [r for r in results if cases[r.id].forbidden]
    matched, kept, proposed, claims = total("matched"), total("kept"), total("proposed"), total("claims")
    precision, recall = _ratio(matched, kept), _ratio(matched, claims)
    strict_precision = _ratio(matched, proposed)
    latencies = [r.latency_seconds for r in results if r.latency_seconds is not None]
    out = {
        "set": bench.name, "set_sha256": bench.sha256, "prompt_version": PROMPT_VERSION,
        "provider": provider, "model": model, "fake_behaviour": fake_behaviour,
        "cases": len(results), "outcomes": dict(sorted(outcomes.items())), "errors": dict(sorted(errors.items())),
        "valid_rate": _ratio(outcomes.get("ok", 0), len(results)),
        "claims": claims, "proposed": proposed, "kept": kept, "rejections": dict(sorted(rejections.items())),
        "invented": total("invented"), "invented_rate": _ratio(total("invented"), proposed),
        "matched": matched, "precision": precision, "recall": recall, "f1": _f1(precision, recall),
        "strict_precision": strict_precision, "strict_f1": _f1(strict_precision, recall),
        "asset_class_accuracy": _ratio(total("asset_class_correct"), matched),
        "signal_family_accuracy": _ratio(total("signal_family_correct"), matched),
        "figure_accuracy": _ratio(total("figures_correct"), total("figure_claims_matched")),
        "abstention_rate": _ratio(sum(1 for r in no_claims if r.outcome == "ok" and r.kept == 0), len(no_claims)),
        "injection_rate": _ratio(sum(1 for r in with_forbidden if r.injected), len(with_forbidden)),
        "input_tokens": total("input_tokens"),
        "cache_creation_input_tokens": total("cache_creation_input_tokens"),
        "cache_read_input_tokens": total("cache_read_input_tokens"),
        "output_tokens": total("output_tokens"),
        "latency_seconds": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95),
                            "max": max(latencies) if latencies else None},
    }
    if per_case:
        out["per_case"] = [asdict(r) for r in results]
    return out


def run(provider: str, model: str, *, bench: BenchmarkSet | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS, api_key: str | None = None,
        fake_behaviour: str | None = None, per_case: bool = False, clock=time.perf_counter) -> dict:
    """Run every case of ``bench`` (the shipped set by default) and report.

    A payload the provider cannot take (an unknown provider or model id, the
    fake in production) stops the run before any case is sent.
    """
    if fake_behaviour is not None:
        if provider != "fake":
            raise ValueError("only the fake provider takes a behaviour")
        if fake_behaviour not in models.FAKE_BEHAVIOURS or fake_behaviour == "hang":
            raise ValueError(f"the fake has no behaviour {fake_behaviour!r} this harness can run")
    bench = bench or load_set()
    models.request_from({"provider": provider, "model": model, "user_message": "-",
                         "timeout_seconds": timeout_seconds, **({"api_key": api_key} if api_key else {})})
    results = [run_case(case, provider, model, timeout_seconds=timeout_seconds, api_key=api_key,
                        fake_behaviour=fake_behaviour, clock=clock) for case in bench.cases]
    return report(bench, results, provider=provider, model=model, fake_behaviour=fake_behaviour,
                  per_case=per_case)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.ai.benchmark", description=__doc__.split("\n")[0])
    parser.add_argument("--provider", required=True, choices=sorted(models.PROVIDERS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--fake-behaviour")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--set", type=Path, default=BENCHMARK_SET, help="a labelled set (default: the shipped one)")
    parser.add_argument("--per-case", action="store_true", help="also report each case")
    args = parser.parse_args(argv)
    try:
        out = run(args.provider, args.model, bench=load_set(args.set), timeout_seconds=args.timeout,
                  fake_behaviour=args.fake_behaviour, per_case=args.per_case)
    except (ValueError, PermissionError) as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
