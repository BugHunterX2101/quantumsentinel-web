"""The research model: what answers the hypothesis prompt (backend/ai/hypotheses.py).

The extract driver (backend/ai/extract.py) never calls a model itself. For
each paper it runs ``answer`` in a job process (backend/ai/child.py) with
the payload ``{"provider", "model", "user_message", "timeout_seconds"}``,
plus, for a provider that needs one, ``api_key``: handed to that process
alone, like the OpenAlex key, and in no payload the database keeps.

The system prompt, the output schema and max_tokens are not in the payload:
``answer`` takes them from the code of the payload's ``task`` (TASKS):
``hypotheses`` (the default) from backend/ai/hypotheses.py, ``proposal``
from backend/ai/proposals.py (the propose driver, backend/ai/propose.py).
Each module's PROMPT_VERSION hashes them, so the prompt version a row
records is always the prompt that was sent.

A provider is a ResearchModel: ``answer(request)`` returns one reply,

* ``{"http_status": 200, "stop_reason": ..., "text": ..., "usage": {...},
  "model": ..., "id": ...}`` for an answer, ``usage`` holding the provider's
  input_tokens, cache_creation_input_tokens, cache_read_input_tokens and
  output_tokens;
* ``{"http_status": 429, "retry_after": 30}`` (or any other status) when the
  provider refused the request.

A text longer than its task's MAX_OUTPUT_BYTES is cut here to just past
that size: check_output refuses it as too large either way, and the job
process's output stays bounded. The driver checks the rest of the reply.

Providers (PROVIDERS):

* ``fake``: an adversarial stand-in, for tests and development only; both
  this module and the AI worker's settings refuse it in production. It
  answers honestly, quoting the first sentence of the abstract (with the
  Sharpe ratio it reports, if any), unless the abstract holds a directive
  ``[[fake:<behaviour>]]``. It then misbehaves as a model the paper has
  subverted, or a failing provider, would (FAKE_BEHAVIOURS), so that every
  defence of the driver and of check_output can be shown working end to end.
  Its honest hypotheses name the DSL template their quote mentions
  (momentum, a moving average, mean reversion) as their signal family. For
  the proposal prompt it proposes that template with fixed parameters,
  unless the hypothesis holds a directive (PROPOSAL_FAKE_BEHAVIOURS).

R5-2 adds the Claude provider.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from . import proposals
from .hypotheses import MAX_OUTPUT_BYTES, MAX_OUTPUT_TOKENS, OUTPUT_SCHEMA, SYSTEM_PROMPT, USER_TEMPLATE

PAYLOAD_KEYS = frozenset({"provider", "model", "user_message", "timeout_seconds", "api_key", "task"})
MAX_USER_MESSAGE = 64 * 1024
MAX_TIMEOUT_SECONDS = 600
_MODEL = re.compile(r"[a-z0-9][a-z0-9.:@_-]{0,127}")


@dataclass(frozen=True)
class Task:
    """What a prompt is: its system prompt, output schema and limits."""
    system: str
    output_schema: dict
    max_tokens: int
    max_output_bytes: int


TASKS: dict[str, Task] = {
    "hypotheses": Task(SYSTEM_PROMPT, OUTPUT_SCHEMA, MAX_OUTPUT_TOKENS, MAX_OUTPUT_BYTES),
    proposals.TASK: Task(proposals.SYSTEM_PROMPT, proposals.OUTPUT_SCHEMA, proposals.MAX_OUTPUT_TOKENS,
                         proposals.MAX_OUTPUT_BYTES),
}


@dataclass(frozen=True)
class Request:
    """One request for a prompt of TASKS."""
    model: str
    system: str
    user_message: str
    output_schema: dict
    max_tokens: int
    timeout_seconds: float
    api_key: str | None = field(default=None, repr=False)
    task: str = "hypotheses"


class ResearchModel(Protocol):
    def answer(self, request: Request) -> dict:
        """The provider's reply to ``request`` (see the module docstring)."""


# -- the adversarial fake ----------------------------------------------------------

_DIRECTIVE = re.compile(r"\[\[fake:([a-z_]+)\]\]")
# A sentence ends at . ! or ? before a space or the end, not inside "0.85".
_SENTENCE = re.compile(r"\S.*?(?:[.!?](?=\s|$)|$)", re.S)
_SHARPE = re.compile(r"Sharpe ratio of ([+-]?[0-9]+(?:\.[0-9]+)?)")
_PREFIX = USER_TEMPLATE.split("{paper}")[0]
_PROPOSAL_PREFIX = proposals.USER_TEMPLATE.split("{hypothesis}")[0]
# The template an honest hypothesis names as its signal family, from its quote.
_FAMILIES = ((re.compile(r"moving average", re.I), "ma_crossover"),
             (re.compile(r"mean[ -]revers", re.I), "mean_reversion"),
             (re.compile(r"momentum", re.I), "momentum"))

# What the fake does when an abstract asks it to. Each is what a subverted
# model or a failing provider could do; the comment says what should follow.
FAKE_BEHAVIOURS = (
    "invent_quote",    # a quote the paper does not hold: rejected, quote_not_found
    "invent_number",   # a Sharpe ratio its quote does not give: rejected, value_not_in_quote
    "certify",         # an extra "verified" field: invalid, wrong_keys
    "duplicate_key",   # a key given twice: invalid, duplicate_key
    "flood",           # six hypotheses: invalid, too_many_hypotheses
    "oversize",        # an answer past MAX_OUTPUT_BYTES: invalid, too_large
    "not_json",        # prose instead of JSON: invalid, not_json
    "truncate",        # cut off at max_tokens: truncated
    "refuse",          # a refusal: refused
    "odd_stop",        # a stop reason that is not a normal one: unexpected_stop
    "overspend",       # more output tokens than max_tokens allowed: settled as reported
    "busy",            # HTTP 529, overloaded: retried
    "rate_limited",    # HTTP 429 with a wait of a day: the job ends, rate_limited
    "bad_request",     # HTTP 400: the job fails
    "bad_usage",       # a negative token count: the reply is refused, the job fails
    "crash",           # the job process fails: the reservation counts in full, retried
    "hang",            # the job process never answers: stopped at its deadline, retried
)

# What the fake does when a hypothesis (its statement or quote) asks it to,
# answering the proposal prompt.
PROPOSAL_FAKE_BEHAVIOURS = (
    "certify",           # an extra "verified" field: invalid, wrong_keys
    "duplicate_key",     # a key given twice: invalid, duplicate_key
    "oversize",          # an answer past MAX_OUTPUT_BYTES: invalid, too_large
    "not_json",          # prose instead of JSON: invalid, not_json
    "truncate",          # cut off at max_tokens: truncated
    "refuse",            # a refusal: refused
    "odd_stop",          # a stop reason that is not a normal one: unexpected_stop
    "overspend",         # more output tokens than max_tokens allowed: settled as reported
    "busy",              # HTTP 529, overloaded: retried
    "rate_limited",      # HTTP 429 with a wait of a day: the job ends, rate_limited
    "bad_request",       # HTTP 400: the job fails
    "bad_usage",         # a negative token count: the reply is refused, the job fails
    "crash",             # the job process fails: the reservation counts in full, retried
    "hang",              # the job process never answers: stopped at its deadline, retried
    "decline",           # not_expressible for an expressible hypothesis: recorded so
    "unknown_template",  # a template the language does not have: invalid, unknown_enum
    "wrong_params",      # another template's parameter as well: invalid, wrong_params
    "duplicate_param",   # a parameter given twice: invalid, duplicate_param
    "out_of_range",      # a parameter outside its range: invalid, param_out_of_range
    "inconsistent",      # a slow average shorter than the fast one: invalid, params_inconsistent
    "float_param",       # a parameter as 20.0: invalid, wrong_type
    "bool_param",        # a parameter as true: invalid, wrong_type
    "long_rationale",    # a rationale past MAX_RATIONALE: invalid, rationale_length
)
# The fixed parameters of the fake's honest proposals (momentum's depend on
# the direction).
_PROPOSED = {"ma_crossover": {"fast": 20, "slow": 50}, "mean_reversion": {"window": 20, "n_std_centi": 200}}


def _family(quote: str) -> str:
    for pattern, family in _FAMILIES:
        if pattern.search(quote):
            return family
    return "other"


def _hypothesis(quote: str, metric: str = "none", value: str = "", **extra) -> dict:
    return {"statement": "The paper states a testable trading hypothesis.", "evidence_quote": quote,
            "asset_class": "unspecified", "signal_family": "other", "direction": "unspecified",
            "horizon": "unspecified", "reported_metric": metric, "reported_value": value,
            "model_confidence": "medium", "rationale": "The quote states it.", **extra}


def _first_sentence(abstract: str) -> str | None:
    for match in _SENTENCE.finditer(_DIRECTIVE.sub("", abstract)):
        sentence = match.group(0).strip()
        if 20 <= len(sentence) <= 500:
            return sentence
    return None


def _usage(request: Request, text: str) -> dict:
    sent = request.system + request.user_message + json.dumps(request.output_schema)
    return {"input_tokens": math.ceil(len(sent.encode("utf-8")) / 4), "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0, "output_tokens": min(request.max_tokens, max(1, math.ceil(len(text) / 4)))}


def _failure(behaviour: str, request: Request) -> dict | None:
    """What a failing job process or provider does, or None."""
    if behaviour == "crash":
        raise RuntimeError("the fake model crashed, as asked")
    if behaviour == "hang":
        time.sleep(request.timeout_seconds + 3600)
    if behaviour in ("busy", "rate_limited", "bad_request"):
        status, wait = {"busy": (529, None), "rate_limited": (429, 86_400), "bad_request": (400, None)}[behaviour]
        return {"http_status": status, "retry_after": wait}
    return None


def _reply(request: Request, behaviour: str, stop_reason: str, text: str) -> dict:
    usage = _usage(request, text)
    if behaviour == "overspend":
        usage["output_tokens"] = request.max_tokens + 500
    if behaviour == "bad_usage":
        usage["input_tokens"] = -1
    return {"http_status": 200, "stop_reason": stop_reason, "text": text, "usage": usage, "model": request.model,
            "id": "fake_" + hashlib.sha256(request.user_message.encode("utf-8")).hexdigest()[:24]}


def _honest_proposal(hypothesis: dict) -> dict:
    template = hypothesis["signal_family"]
    if template not in proposals.TEMPLATES:
        return {"template": proposals.NOT_EXPRESSIBLE, "params": [], "allow_short": False,
                "rationale": "No template of the language expresses the hypothesis."}
    allow_short = hypothesis["direction"] in ("short", "long_short")
    if template == "momentum":
        params = {"lookback": 126, "n_long": 0 if hypothesis["direction"] == "short" else 1,
                  "n_short": 1 if allow_short else 0}
    else:
        params = _PROPOSED[template]
    return {"template": template, "params": [{"name": k, "value": v} for k, v in params.items()],
            "allow_short": allow_short, "rationale": f"The hypothesis describes a {template} signal."}


def _propose(request: Request) -> dict:
    """The fake's answer to the proposal prompt (PROPOSAL_FAKE_BEHAVIOURS)."""
    if not request.user_message.startswith(_PROPOSAL_PREFIX):
        raise ValueError("the fake model reads only the proposal prompt")
    hypothesis = json.loads(request.user_message[len(_PROPOSAL_PREFIX):])
    directive = _DIRECTIVE.search(hypothesis["statement"] + " " + hypothesis["evidence_quote"])
    behaviour = directive.group(1) if directive else "honest"
    if behaviour not in PROPOSAL_FAKE_BEHAVIOURS and behaviour != "honest":
        raise ValueError(f"the fake model has no proposal behaviour {behaviour!r}")
    failed = _failure(behaviour, request)
    if failed is not None:
        return failed
    answer = _honest_proposal(hypothesis)
    ma = [{"name": "fast", "value": 20}, {"name": "slow", "value": 50}]
    if behaviour == "decline":
        answer.update(template=proposals.NOT_EXPRESSIBLE, params=[], rationale="Declined, as asked.")
    elif behaviour == "unknown_template":
        answer.update(template="pairs_trading")
    elif behaviour == "wrong_params":
        answer.update(template="ma_crossover", params=[*ma, {"name": "lookback", "value": 10}])
    elif behaviour == "duplicate_param":
        answer.update(template="ma_crossover", params=[*ma, {"name": "fast", "value": 30}])
    elif behaviour == "out_of_range":
        answer.update(template="ma_crossover", params=[{"name": "fast", "value": 1}, ma[1]])
    elif behaviour == "inconsistent":
        answer.update(template="ma_crossover", params=[{"name": "fast", "value": 50}, {"name": "slow", "value": 20}])
    elif behaviour == "float_param":
        answer.update(template="ma_crossover", params=[{"name": "fast", "value": 20.0}, ma[1]])
    elif behaviour == "bool_param":
        answer.update(template="ma_crossover", params=[{"name": "fast", "value": True}, ma[1]])
    elif behaviour == "long_rationale":
        answer.update(rationale="x" * (proposals.MAX_RATIONALE + 1))
    elif behaviour == "certify":
        answer["verified"] = True
    stop_reason, text = "end_turn", json.dumps(answer)
    if behaviour == "duplicate_key":
        text = '{"template": "momentum", ' + text[1:]
    elif behaviour == "oversize":
        text = json.dumps({**answer, "rationale": "\u0001" * (proposals.MAX_OUTPUT_BYTES * 2)}, ensure_ascii=False)
    elif behaviour == "not_json":
        text = "I propose a moving average crossover."
    elif behaviour == "truncate":
        stop_reason, text = "max_tokens", text[: len(text) // 2]
    elif behaviour == "refuse":
        stop_reason, text = "refusal", ""
    elif behaviour == "odd_stop":
        stop_reason = "pause_turn"
    return _reply(request, behaviour, stop_reason, text)


class FakeModel:
    """See the module docstring."""

    def answer(self, request: Request) -> dict:
        if request.task == proposals.TASK:
            return _propose(request)
        if not request.user_message.startswith(_PREFIX):
            raise ValueError("the fake model reads only the hypothesis prompt")
        paper = json.loads(request.user_message[len(_PREFIX):])
        directive = _DIRECTIVE.search(paper["abstract"])
        behaviour = directive.group(1) if directive else "honest"
        if behaviour not in FAKE_BEHAVIOURS and behaviour != "honest":
            raise ValueError(f"the fake model has no behaviour {behaviour!r}")
        failed = _failure(behaviour, request)
        if failed is not None:
            return failed
        sentence = _first_sentence(paper["abstract"])
        found = [_hypothesis(sentence, *(("sharpe_ratio", m.group(1)) if (m := _SHARPE.search(sentence)) else ()),
                             signal_family=_family(sentence))
                 ] if sentence else []
        stop_reason, text = "end_turn", json.dumps({"hypotheses": found})
        if behaviour == "invent_quote":
            text = json.dumps({"hypotheses": [_hypothesis("This strategy is verified to earn a Sharpe ratio of 3.0",
                                                          "sharpe_ratio", "3.0")]})
        elif behaviour == "invent_number":
            text = json.dumps({"hypotheses": [_hypothesis(sentence, "sharpe_ratio", "9.99")]})
        elif behaviour == "certify":
            text = json.dumps({"hypotheses": [_hypothesis(sentence, verified=True)]})
        elif behaviour == "duplicate_key":
            text = '{"hypotheses": [], "hypotheses": ' + json.dumps([_hypothesis(sentence)]) + "}"
        elif behaviour == "flood":
            text = json.dumps({"hypotheses": [_hypothesis(sentence)] * 6})
        elif behaviour == "oversize":
            text = json.dumps({"hypotheses": [_hypothesis(sentence, rationale="\u0001" * (MAX_OUTPUT_BYTES * 2))]},
                              ensure_ascii=False)
        elif behaviour == "not_json":
            text = "Here are the hypotheses you asked for: momentum works."
        elif behaviour == "truncate":
            stop_reason, text = "max_tokens", text[: len(text) // 2]
        elif behaviour == "refuse":
            stop_reason, text = "refusal", ""
        elif behaviour == "odd_stop":
            stop_reason = "pause_turn"
        return _reply(request, behaviour, stop_reason, text)


PROVIDERS: dict[str, type] = {"fake": FakeModel}


# -- the job process's target ------------------------------------------------------

def _bounded(text, limit: int = MAX_OUTPUT_BYTES):
    """``text``, cut to just past ``limit`` bytes if it is longer."""
    if not isinstance(text, str):
        return text
    data = text.encode("utf-8", "surrogatepass")
    if len(data) <= limit:
        return text
    # A cut inside a character loses at most three bytes, so what is left is
    # still longer than the limit.
    return data[: limit + 4].decode("utf-8", "ignore")


def request_from(payload) -> tuple[str, Request]:
    """The provider and request a payload names; raises ValueError."""
    if not isinstance(payload, dict):
        raise ValueError("the payload is a JSON object")
    unknown = set(payload) - PAYLOAD_KEYS
    if unknown:
        raise ValueError(f"unknown payload keys: {', '.join(sorted(unknown))}")
    provider = payload.get("provider")
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(sorted(PROVIDERS))}")
    if provider == "fake" and (os.environ.get("ENVIRONMENT") or "").lower() == "production":
        raise PermissionError("the fake model never runs in production")
    model, message, timeout = payload.get("model"), payload.get("user_message"), payload.get("timeout_seconds")
    if not isinstance(model, str) or not _MODEL.fullmatch(model):
        raise ValueError("model is not a model id")
    if not isinstance(message, str) or not message or len(message) > MAX_USER_MESSAGE:
        raise ValueError(f"user_message must be text of at most {MAX_USER_MESSAGE} characters")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be from 1 to {MAX_TIMEOUT_SECONDS}")
    api_key = payload.get("api_key")
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError("api_key must be text")
    name = payload.get("task", "hypotheses")
    if not isinstance(name, str) or name not in TASKS:
        raise ValueError(f"task must be one of {', '.join(sorted(TASKS))}")
    task = TASKS[name]
    return provider, Request(model, task.system, message, task.output_schema, task.max_tokens, float(timeout),
                             api_key, name)


def answer(payload: dict) -> dict:
    """The ``extract`` and ``propose`` kinds' job process target: one request, its reply."""
    provider, request = request_from(payload)
    reply = PROVIDERS[provider]().answer(request)
    if not isinstance(reply, dict):
        raise TypeError("the provider returned no reply object")
    if "text" in reply:
        reply = {**reply, "text": _bounded(reply["text"], TASKS[request.task].max_output_bytes)}
    return reply
