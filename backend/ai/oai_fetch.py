"""Fetch and parse one OAI-PMH ListRecords page: an AI job process's target.

The AI worker's arXiv driver (backend/ai/harvest.py) runs this in a job
process (backend/ai/child.py), once per page. The worker builds the request
from its own settings and its harvest cursor; this checks it again, makes
one GET and parses the answer (backend/ai/oai.py). It decides nothing: the
worker checks everything it returns before any of it reaches the database.

* Redirects are not followed: any 3xx is reported as it is.
* The proxy comes only from HTTP_PROXY/HTTPS_PROXY (the egress proxy in
  production, R4-2), never from the system's settings.
* A body larger than oai.MAX_BYTES is refused, and only an HTTP 200 is
  parsed. Any other status is returned with its Retry-After, if it is a
  number of seconds; the worker decides what to do with it.

It returns ``{"http_status": 200, "page": <parse_list_records result>}`` or
``{"http_status": <status>, "retry_after": <int or None>}``.
"""
from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse
import urllib.request

from . import oai

USER_AGENT = "quantumsentinel-ai-harvester/1"
PARAMETERS = frozenset({"verb", "metadataPrefix", "set", "from", "until", "resumptionToken"})
MAX_TIMEOUT_SECONDS = 120
_BASE_URL = re.compile(r"https?://[\x21-\x7e]+")
_VALUE = re.compile(r"[\x21-\x7e]{1,1024}")
_RETRY_AFTER = re.compile(r"[0-9]{1,7}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib then raises HTTPError with the 3xx status


def request_url(payload: dict) -> str:
    """The URL of the one GET a payload asks for; raises ValueError."""
    if not isinstance(payload, dict):
        raise ValueError("the payload is a JSON object")
    base, params = payload.get("base_url"), payload.get("params")
    if not isinstance(base, str) or not _BASE_URL.fullmatch(base) or "?" in base or "#" in base:
        raise ValueError("base_url must be an http(s) URL without a query")
    if not isinstance(params, dict) or not params or not set(params) <= PARAMETERS:
        raise ValueError(f"params must name only {', '.join(sorted(PARAMETERS))}")
    if params.get("verb") != "ListRecords":
        raise ValueError("the only verb is ListRecords")
    if not all(isinstance(value, str) and _VALUE.fullmatch(value) for value in params.values()):
        raise ValueError("every parameter is printable ASCII")
    return base + "?" + urllib.parse.urlencode(sorted(params.items()))


def _opener() -> urllib.request.OpenerDirector:
    proxies = {scheme: os.environ[name] for scheme, name in (("http", "HTTP_PROXY"), ("https", "HTTPS_PROXY"))
               if os.environ.get(name)}
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies), _NoRedirect())


def _retry_after(headers) -> int | None:
    value = (headers.get("Retry-After") or "").strip() if headers is not None else ""
    return int(value) if _RETRY_AFTER.fullmatch(value) else None


def fetch_page(payload: dict) -> dict:
    url = request_url(payload)
    timeout = payload.get("timeout_seconds", 60)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be a number from 1 to {MAX_TIMEOUT_SECONDS}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    try:
        # request_url allows only http(s) URLs.
        with _opener().open(request, timeout=timeout) as response:  # nosec B310
            status = response.status
            body = response.read(oai.MAX_BYTES + 1) if status == 200 else b""
            headers = response.headers
    except urllib.error.HTTPError as exc:
        with exc:
            return {"http_status": exc.code, "retry_after": _retry_after(exc.headers)}
    if status != 200:
        return {"http_status": status, "retry_after": _retry_after(headers)}
    if len(body) > oai.MAX_BYTES:
        raise oai.OaiError(f"the response is larger than {oai.MAX_BYTES} bytes")
    return {"http_status": 200, "page": oai.parse_list_records(body)}
