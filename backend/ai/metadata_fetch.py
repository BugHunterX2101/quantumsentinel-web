"""Look up one work on OpenAlex or Crossref: an AI job process's target.

The AI worker's metadata driver (backend/ai/enrich.py) runs this in a job
process (backend/ai/child.py), once per document. The payload names the
source, its base URL (from the worker's settings) and the DOI to look up;
this builds the one URL itself, makes one GET, and reads the few fields the
registry keeps out of the JSON answer. It decides nothing: the worker checks
everything it returns before any of it reaches the database.

* The DOI must match DOI_PATTERN (Crossref's own pattern for modern DOIs)
  and is percent-encoded whole, ``/`` included, so it cannot add a path
  segment, a query or a fragment to the URL. Both services accept it so.
* An OpenAlex API key, when the payload has one (``api_key``, from the
  worker's AI_OPENALEX_API_KEY), goes in an ``Authorization: Bearer``
  header, never in the URL, and only to OpenAlex. It must match API_KEY, so
  it cannot add a header; no message here repeats it.
* Redirects are not followed, the proxy comes only from
  HTTP_PROXY/HTTPS_PROXY, and a body over MAX_BYTES is refused.
* Only an HTTP 200 is read. Any other status is returned with the longer of
  its Retry-After and X-RateLimit-Reset (OpenAlex's seconds until its daily
  budget resets), when they are a number of seconds.

It returns ``{"http_status": 200, "work": {...}}``, ``{"http_status": 200,
"unreadable": "<why>"}`` for a 200 whose body is not a work it can read, or
``{"http_status": <status>, "retry_after": <int or None>}``.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "quantumsentinel-ai-harvester/1"
SOURCES = ("openalex", "crossref")
MAX_BYTES = 4 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 60
MAX_DOI = 255
# Crossref's pattern for the DOIs it issues today (case-insensitive).
DOI_PATTERN = re.compile(r"10\.[0-9]{4,9}/[-._;()/:A-Za-z0-9]+")
# What OpenAlex sends of a work; the rest (abstract, references, ...) is
# large and unused.
OPENALEX_SELECT = ("id,doi,type,publication_date,cited_by_count,is_retracted,"
                   "primary_location,primary_topic")
API_KEY = re.compile(r"[A-Za-z0-9_-]{8,128}")
_BASE_URL = re.compile(r"https?://[\x21-\x7e]+")
_SECONDS = re.compile(r"[0-9]{1,9}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib then raises HTTPError with the 3xx status


def valid_doi(doi) -> bool:
    return isinstance(doi, str) and len(doi) <= MAX_DOI and DOI_PATTERN.fullmatch(doi) is not None


def request_url(payload: dict) -> str:
    """The URL of the one GET a payload asks for; raises ValueError."""
    if not isinstance(payload, dict):
        raise ValueError("the payload is a JSON object")
    source, base, doi = payload.get("source"), payload.get("base_url"), payload.get("doi")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}")
    if not isinstance(base, str) or not _BASE_URL.fullmatch(base) or "?" in base or "#" in base \
            or base.endswith("/"):
        raise ValueError("base_url must be an http(s) URL without a query or a final /")
    if not valid_doi(doi):
        raise ValueError("doi must be a DOI such as 10.1234/abc")
    quoted = urllib.parse.quote(doi, safe="")
    if source == "openalex":
        return f"{base}/works/doi:{quoted}?select={OPENALEX_SELECT}"
    return f"{base}/works/{quoted}"


def request_headers(payload: dict) -> dict:
    """The headers of the GET; raises ValueError."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", "Accept-Encoding": "identity"}
    key = payload.get("api_key")
    if key is not None:
        if payload.get("source") != "openalex":
            raise ValueError("only an OpenAlex request takes an api_key")
        if not isinstance(key, str) or not API_KEY.fullmatch(key):
            raise ValueError("api_key must be 8 to 128 letters, digits, - or _")
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _opener() -> urllib.request.OpenerDirector:
    proxies = {scheme: os.environ[name] for scheme, name in (("http", "HTTP_PROXY"), ("https", "HTTPS_PROXY"))
               if os.environ.get(name)}
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies), _NoRedirect())


def _seconds(headers, name: str) -> int | None:
    value = (headers.get(name) or "").strip() if headers is not None else ""
    return int(value) if _SECONDS.fullmatch(value) else None


def _wait(headers) -> int | None:
    """The longer of Retry-After and X-RateLimit-Reset, in seconds."""
    waits = [w for w in (_seconds(headers, "Retry-After"), _seconds(headers, "X-RateLimit-Reset")) if w is not None]
    return max(waits) if waits else None


def _string(value, limit: int = 1000) -> str | None:
    return value[:limit] if isinstance(value, str) else None


def _number(value) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _object(value) -> dict:
    return value if isinstance(value, dict) else {}


def _openalex(data: dict) -> dict:
    location = _object(data.get("primary_location"))
    venue = _object(location.get("source"))
    topic = _object(data.get("primary_topic"))
    field = _object(topic.get("field"))
    retracted = data.get("is_retracted")
    return {"id": _string(data.get("id")), "doi": _string(data.get("doi")), "type": _string(data.get("type")),
            "publication_date": _string(data.get("publication_date")),
            "cited_by_count": _number(data.get("cited_by_count")),
            "is_retracted": retracted if isinstance(retracted, bool) else None,
            "venue": _string(venue.get("display_name")), "topic": _string(topic.get("display_name")),
            "field": _string(field.get("display_name"))}


def _date_parts(value) -> list | None:
    parts = _object(value).get("date-parts")
    first = parts[0] if isinstance(parts, list) and parts and isinstance(parts[0], list) else None
    if first and all(isinstance(p, int) and not isinstance(p, bool) for p in first[:3]):
        return first[:3]
    return None


def _crossref(data: dict) -> dict:
    message = _object(data.get("message"))
    titles = message.get("container-title")
    licences = message.get("license") if isinstance(message.get("license"), list) else []
    updates = message.get("updated-by") if isinstance(message.get("updated-by"), list) else []
    published = None
    for key in ("published", "published-print", "published-online", "issued"):
        published = _date_parts(message.get(key))
        if published:
            break
    return {"doi": _string(message.get("DOI")), "type": _string(message.get("type")),
            "venue": _string(titles[0]) if isinstance(titles, list) and titles else None,
            "publisher": _string(message.get("publisher")),
            "cited_by_count": _number(message.get("is-referenced-by-count")),
            "published": published,
            "licence_urls": [u for u in (_string(_object(item).get("URL"), 300) for item in licences[:10]) if u],
            "update_types": [t for t in (_string(_object(item).get("type"), 64) for item in updates[:50]) if t]}


def read_work(source: str, body: bytes) -> dict:
    """The fields of a work in a 200 body; raises ValueError if it is not one."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("the answer is not JSON") from None
    if not isinstance(data, dict):
        raise ValueError("the answer is not a JSON object")
    return _openalex(data) if source == "openalex" else _crossref(data)


def fetch_work(payload: dict) -> dict:
    url = request_url(payload)
    timeout = payload.get("timeout_seconds", 30)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be a number from 1 to {MAX_TIMEOUT_SECONDS}")
    request = urllib.request.Request(url, headers=request_headers(payload))
    try:
        # request_url allows only http(s) URLs.
        with _opener().open(request, timeout=timeout) as response:  # nosec B310
            status = response.status
            body = response.read(MAX_BYTES + 1) if status == 200 else b""
            headers = response.headers
    except urllib.error.HTTPError as exc:
        with exc:
            return {"http_status": exc.code, "retry_after": _wait(exc.headers)}
    if status != 200:
        return {"http_status": status, "retry_after": _wait(headers)}
    if len(body) > MAX_BYTES:
        return {"http_status": 200, "unreadable": f"the answer is larger than {MAX_BYTES} bytes"}
    try:
        return {"http_status": 200, "work": read_work(payload["source"], body)}
    except ValueError as exc:
        return {"http_status": 200, "unreadable": str(exc)}
