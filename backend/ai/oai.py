"""Parse an OAI-PMH ListRecords response in arXiv's ``arXiv`` metadata format.

This runs in an AI job process (backend/ai/child.py), never in the AI worker
itself, because its input comes from the internet. It extracts what the
response holds and decides nothing: the worker re-checks every field of the
result before any of it reaches the database (backend/ai/documents.py).

The XML is parsed only once it is known to declare no DTD. A response must be
UTF-8 (OAI-PMH requires it) and must contain no ``<!DOCTYPE`` or
``<!ENTITY``. Without a DTD no entity can be declared, so neither external
entities nor entity expansion can occur. The check runs on the strictly
decoded text, and any other declared encoding is refused, so a UTF-16 body
cannot hide a DTD from it.

What it returns (all strings as found, or None when absent)::

    {"response_date": "2026-11-20T06:47:36Z",
     "error": None or {"code": "noRecordsMatch", "message": "..."},
     "records": [{"identifier", "datestamp", "sets": [...], "deleted": bool,
                  "id", "created", "updated", "title", "abstract",
                  "categories", "license", "doi", "journal_ref",
                  "authors": [{"keyname", "forenames", "suffix"}, ...],
                  "author_count": int}],
     "resumption_token": None or "...", "token_expires": None or "..."}

An empty or missing resumptionToken means the list is complete.
"""
from __future__ import annotations

import re
# Only UTF-8 input without a DTD reaches the parser (see _check).
import xml.etree.ElementTree as ET  # nosec B405

OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"
MAX_BYTES = 16 * 1024 * 1024
# arXiv serves 1300 records a page; anything far beyond that is not a page.
MAX_RECORDS = 5000
# Authors kept per record; author_count still says how many there were.
MAX_AUTHORS = 200

_DECLARATION = re.compile(r"""\A\s*<\?xml\b[^>]*?\bencoding\s*=\s*["']([^"']*)["']""")
_DTD = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)", re.IGNORECASE)


class OaiError(ValueError):
    """A response that is not an OAI-PMH response this parser accepts."""


def _check(data: bytes) -> None:
    if not isinstance(data, (bytes, bytearray)):
        raise OaiError("an OAI-PMH response is bytes")
    if len(data) > MAX_BYTES:
        raise OaiError(f"the response is larger than {MAX_BYTES} bytes")
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError:
        raise OaiError("the response is not UTF-8") from None
    declared = _DECLARATION.match(text.lstrip("﻿"))
    if declared and declared.group(1).strip().lower() not in ("utf-8", "utf8"):
        raise OaiError(f"the response declares encoding {declared.group(1)[:20]!r}, not UTF-8")
    if _DTD.search(text):
        raise OaiError("the response has a DTD")


def _text(element) -> str | None:
    return None if element is None else "".join(element.itertext())


def _child(parent, tag: str) -> str | None:
    return None if parent is None else _text(parent.find(tag))


def _authors(meta) -> tuple[list[dict], int]:
    found = [] if meta is None else meta.findall(f"{ARXIV}authors/{ARXIV}author")
    kept = [{"keyname": _child(a, f"{ARXIV}keyname"), "forenames": _child(a, f"{ARXIV}forenames"),
             "suffix": _child(a, f"{ARXIV}suffix")} for a in found[:MAX_AUTHORS]]
    return kept, len(found)


def _record(element) -> dict:
    header = element.find(f"{OAI}header")
    meta = element.find(f"{OAI}metadata/{ARXIV}arXiv")
    authors, count = _authors(meta)
    return {
        "identifier": _child(header, f"{OAI}identifier"),
        "datestamp": _child(header, f"{OAI}datestamp"),
        "sets": [] if header is None else [_text(s) for s in header.findall(f"{OAI}setSpec")],
        "deleted": header is not None and header.get("status") == "deleted",
        "id": _child(meta, f"{ARXIV}id"),
        "created": _child(meta, f"{ARXIV}created"),
        "updated": _child(meta, f"{ARXIV}updated"),
        "title": _child(meta, f"{ARXIV}title"),
        "abstract": _child(meta, f"{ARXIV}abstract"),
        "categories": _child(meta, f"{ARXIV}categories"),
        "license": _child(meta, f"{ARXIV}license"),
        "doi": _child(meta, f"{ARXIV}doi"),
        "journal_ref": _child(meta, f"{ARXIV}journal-ref"),
        "authors": authors,
        "author_count": count,
    }


def parse_list_records(data: bytes) -> dict:
    """One ListRecords response, as the module docstring describes; raises
    OaiError for anything that is not a well-formed OAI-PMH response."""
    _check(data)
    try:
        # _check has refused any DTD and any input that is not UTF-8.
        root = ET.fromstring(bytes(data))  # nosec B314
    except ET.ParseError as exc:
        raise OaiError(f"the response is not well-formed XML: {exc}") from None
    if root.tag != f"{OAI}OAI-PMH":
        raise OaiError("the response is not an OAI-PMH response")
    out = {"response_date": _child(root, f"{OAI}responseDate"), "error": None, "records": [],
           "resumption_token": None, "token_expires": None}
    error = root.find(f"{OAI}error")
    if error is not None:
        out["error"] = {"code": error.get("code"), "message": (_text(error) or "").strip()[:500]}
        return out
    listing = root.find(f"{OAI}ListRecords")
    if listing is None:
        raise OaiError("the response has neither ListRecords nor an error")
    records = listing.findall(f"{OAI}record")
    if len(records) > MAX_RECORDS:
        raise OaiError(f"the response has more than {MAX_RECORDS} records")
    out["records"] = [_record(record) for record in records]
    token = listing.find(f"{OAI}resumptionToken")
    if token is not None and (token.text or "").strip():
        out["resumption_token"] = token.text.strip()
        out["token_expires"] = token.get("expirationDate")
    return out
