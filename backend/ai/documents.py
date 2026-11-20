"""The document registry (table ``documents`` in the AI schema).

The AI worker records here the papers it harvests. A job process fetches and
parses each page (backend/ai/oai.py). Its result is untrusted: it came from
the internet through code that holds no secret, so ``clean_page`` checks
every field again in the worker before ``upsert_page`` writes anything.

* A result that is not a page at all (wrong types, a bad responseDate or
  resumptionToken, too many records) raises PageError, and nothing from it
  is used.
* A single record that fails a check is dropped and counted; the rest of the
  page is kept. Otherwise one malformed record would fail its page on every
  retry.
* Text loses control and format characters (including bidirectional
  overrides and zero-width characters), runs of whitespace become one space,
  and every field has a length cap. Titles reach the UI, and later prompts.

A document is keyed by (source, source_id). arXiv's ``arXiv`` metadata
format carries no version number, so a document holds the latest metadata of
its paper. Re-harvesting is idempotent: a record replaces the stored one only
when its datestamp is later, or the same day with different content
(``content_hash``) or a different deleted state. OAI-PMH datestamps are
days, so two changes on one day share one. A deleted record keeps its row,
with ``deleted_at`` set.

``reuse_class`` says what the paper's licence lets this product do with its
full text. arXiv's metadata (title, authors, abstract) is CC0 whatever the
licence.

* ``open``: CC BY, CC BY-SA or CC0. The text may be shown and quoted, also
  commercially, with attribution; adaptations of CC BY-SA work share alike.
* ``noncommercial``: CC BY-NC, BY-NC-SA or BY-NC-ND. No commercial reuse.
* ``restricted``: anything else, including arXiv's non-exclusive licence (the
  usual one), the licence assumed for 1991-2003 papers, no licence at all
  and any URL this code does not recognise. Link to the paper only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass

from psycopg import Connection, sql
from psycopg.types.json import Jsonb

SOURCE = "arxiv"
OAI_PREFIX = "oai:arXiv.org:"
HASH_SCHEME = "qs-doc-sha256-v1"
REUSE_CLASSES = ("open", "noncommercial", "restricted")

MAX_RECORDS = 5000
MAX_TITLE = 1000
MAX_ABSTRACT = 8000
MAX_AUTHOR_NAME = 300
MAX_AUTHORS = 100
MAX_AUTHOR_COUNT = 100_000
MAX_CATEGORIES = 20
MAX_SETS = 20
MAX_SET_SPEC = 128
MAX_JOURNAL_REF = 500
MAX_URL = 255
MAX_DOI = 255
MAX_TOKEN = 1024
MAX_ERROR_MESSAGE = 500
# Rejected records listed by identifier in a page's outcome; all are counted.
MAX_REJECTED_LISTED = 20

_NEW_ID = re.compile(r"[0-9]{4}\.[0-9]{4,5}")
_OLD_ID = re.compile(r"[a-z]+(?:-[a-z]+)*(?:\.[A-Z]{2})?/[0-9]{7}")
_CATEGORY = re.compile(r"[a-z]+(?:-[a-z]+)*(?:\.[A-Za-z]+(?:-[A-Za-z]+)*)?")
_SET_SPEC = re.compile(r"[A-Za-z][A-Za-z0-9.-]*(?::[A-Za-z0-9.-]+)*")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_DOI = re.compile(r"10\.[0-9]{4,9}/[\x21-\x7e]+")
_URL = re.compile(r"https?://[\x21-\x7e]+")
_TOKEN = re.compile(r"[\x21-\x7e]+")
_ERROR_CODE = re.compile(r"[A-Za-z]{1,64}")
_CC = re.compile(r"creativecommons\.org/licenses/(by|by-sa|by-nc|by-nc-sa|by-nc-nd)/[0-9]\.[0-9]/")
_CC0 = "creativecommons.org/publicdomain/zero/1.0/"
# Control characters become spaces (they include newlines and tabs);
# format characters, surrogates and private-use characters are dropped.
_AS_SPACE = frozenset({"Cc"})
_DROPPED = frozenset({"Cf", "Cs", "Co"})


class PageError(ValueError):
    """A job's result that is not a page of records."""


class RecordError(ValueError):
    """A record that fails a check; it is dropped from its page."""


@dataclass(frozen=True)
class Document:
    source_id: str
    datestamp: dt.date
    set_specs: tuple[str, ...]
    title: str
    authors: tuple[str, ...]
    author_count: int
    abstract: str
    categories: tuple[str, ...]
    license_url: str | None
    reuse_class: str
    doi: str | None
    journal_ref: str | None
    created: dt.date
    updated: dt.date | None

    def content(self) -> dict:
        """The fields ``content_hash`` covers: the paper's metadata, not the
        harvest's bookkeeping (datestamp, sets) or what is derived from it."""
        return {
            "source": SOURCE, "source_id": self.source_id, "title": self.title,
            "authors": list(self.authors), "author_count": self.author_count, "abstract": self.abstract,
            "categories": list(self.categories), "license_url": self.license_url, "doi": self.doi,
            "journal_ref": self.journal_ref, "created": self.created.isoformat(),
            "updated": self.updated.isoformat() if self.updated else None,
        }

    @property
    def content_hash(self) -> str:
        return content_hash(self.content())


@dataclass(frozen=True)
class Deletion:
    source_id: str
    datestamp: dt.date


@dataclass(frozen=True)
class Page:
    response_date: dt.datetime
    error_code: str | None
    error_message: str | None
    documents: tuple[Document, ...]
    deletions: tuple[Deletion, ...]
    resumption_token: str | None
    token_expires: dt.datetime | None
    rejected_count: int
    rejected: tuple[dict, ...]


# -- pure checks ---------------------------------------------------------------

def canonical(value) -> bytes:
    """Canonical JSON of strings, integers, booleans, None, lists and
    objects with ASCII keys: RFC 8785 for these values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def content_hash(content: dict) -> str:
    return f"{HASH_SCHEME}:{hashlib.sha256(canonical(content)).hexdigest()}"


def reuse_class(license_url) -> str:
    """The reuse class of a licence URL (see the module docstring)."""
    if not isinstance(license_url, str):
        return "restricted"
    url = license_url.strip().lower()
    for scheme in ("https://", "http://"):
        if url.startswith(scheme):
            url = url[len(scheme):]
            break
    else:
        return "restricted"
    url = url.removeprefix("www.")
    if not url.endswith("/"):
        url += "/"
    if url == _CC0:
        return "open"
    licence = _CC.fullmatch(url)
    if licence is None:
        return "restricted"
    return "open" if licence.group(1) in ("by", "by-sa") else "noncommercial"


def clean_text(value, limit: int) -> str:
    """``value`` without control or format characters, its whitespace
    collapsed, cut to ``limit`` characters; "" for None."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise RecordError("a text field is not a string")
    kept = []
    for ch in value:
        category = unicodedata.category(ch)
        if category in _AS_SPACE:
            kept.append(" ")
        elif category not in _DROPPED:
            kept.append(ch)
    return " ".join("".join(kept).split())[:limit].rstrip()


def _source_id(value) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise RecordError("not an arXiv identifier")
    value = value.strip()
    if not (_NEW_ID.fullmatch(value) or _OLD_ID.fullmatch(value)):
        raise RecordError("not an arXiv identifier")
    return value


def _date(value, what: str) -> dt.date:
    if not isinstance(value, str) or not _DATE.fullmatch(value.strip()):
        raise RecordError(f"the {what} is not a date")
    try:
        return dt.date.fromisoformat(value.strip())
    except ValueError:
        raise RecordError(f"the {what} is not a date") from None


def _timestamp(value, what: str) -> dt.datetime:
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value.strip()):
        raise PageError(f"the {what} is not a UTC timestamp")
    try:
        return dt.datetime.strptime(value.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        raise PageError(f"the {what} is not a UTC timestamp") from None


def _set_specs(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RecordError("the record's sets are not a list")
    specs = {s.strip() for s in value if isinstance(s, str) and len(s.strip()) <= MAX_SET_SPEC
             and _SET_SPEC.fullmatch(s.strip())}
    return tuple(sorted(specs)[:MAX_SETS])


def _categories(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        raise RecordError("the record's categories are not a string")
    seen: list[str] = []
    for category in value.split():
        if len(category) <= 64 and _CATEGORY.fullmatch(category) and category not in seen:
            seen.append(category)
    return tuple(seen[:MAX_CATEGORIES])


def _authors(value, count) -> tuple[tuple[str, ...], int]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise RecordError("the record's authors are not a list")
    names = []
    for author in value:
        if not isinstance(author, dict):
            raise RecordError("an author is not an object")
        parts = (clean_text(author.get(key), MAX_AUTHOR_NAME) for key in ("forenames", "keyname", "suffix"))
        name = " ".join(part for part in parts if part)[:MAX_AUTHOR_NAME].rstrip()
        if name:
            names.append(name)
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= MAX_AUTHOR_COUNT:
        raise RecordError("the record's author count is not a count")
    return tuple(names[:MAX_AUTHORS]), max(count, len(names))


def _license(value) -> str | None:
    url = clean_text(value, MAX_URL + 1)
    return url if url and len(url) <= MAX_URL and _URL.fullmatch(url) else None


def _doi(value) -> str | None:
    """The first well-formed DOI; arXiv's field may list several."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise RecordError("the record's DOI is not a string")
    for candidate in clean_text(value, 4 * MAX_DOI).split():
        if len(candidate) <= MAX_DOI and _DOI.fullmatch(candidate):
            return candidate
    return None


def clean_record(raw) -> Document | Deletion:
    """One record of a job's result, checked; raises RecordError."""
    if not isinstance(raw, dict):
        raise RecordError("a record is not an object")
    identifier = raw.get("identifier")
    if not isinstance(identifier, str) or not identifier.startswith(OAI_PREFIX):
        raise RecordError("the record's identifier is not an arXiv OAI identifier")
    source_id = _source_id(identifier[len(OAI_PREFIX):])
    datestamp = _date(raw.get("datestamp"), "datestamp")
    deleted = raw.get("deleted", False)
    if not isinstance(deleted, bool):
        raise RecordError("the record's deleted flag is not a boolean")
    if deleted:
        return Deletion(source_id, datestamp)
    if _source_id(raw.get("id")) != source_id:
        raise RecordError("the record's header and metadata name different papers")
    title = clean_text(raw.get("title"), MAX_TITLE)
    if not title:
        raise RecordError("the record has no title")
    authors, author_count = _authors(raw.get("authors"), raw.get("author_count", 0))
    license_url = _license(raw.get("license"))
    updated = raw.get("updated")
    return Document(
        source_id=source_id, datestamp=datestamp, set_specs=_set_specs(raw.get("sets")), title=title,
        authors=authors, author_count=author_count, abstract=clean_text(raw.get("abstract"), MAX_ABSTRACT),
        categories=_categories(raw.get("categories")), license_url=license_url,
        reuse_class=reuse_class(license_url), doi=_doi(raw.get("doi")),
        journal_ref=clean_text(raw.get("journal_ref"), MAX_JOURNAL_REF) or None,
        created=_date(raw.get("created"), "creation date"),
        updated=None if updated is None else _date(updated, "update date"))


def _listed(raw) -> str:
    identifier = raw.get("identifier") if isinstance(raw, dict) else None
    return clean_text(identifier, 80) if isinstance(identifier, str) else ""


def clean_page(result) -> Page:
    """A job's result, checked: raises PageError if it is not a page, and
    drops (and counts) each record that fails a check."""
    if not isinstance(result, dict):
        raise PageError("the result is not an object")
    response_date = _timestamp(result.get("response_date"), "responseDate")
    error = result.get("error")
    error_code = error_message = None
    if error is not None:
        if not isinstance(error, dict) or not isinstance(error.get("code"), str) \
                or not _ERROR_CODE.fullmatch(error["code"]):
            raise PageError("the result's error has no OAI-PMH error code")
        error_code = error["code"]
        message = error.get("message")
        error_message = clean_text(message if isinstance(message, str) else None, MAX_ERROR_MESSAGE)
    records = result.get("records")
    if not isinstance(records, list) or len(records) > MAX_RECORDS:
        raise PageError(f"the result's records are not a list of at most {MAX_RECORDS}")
    if error_code and records:
        raise PageError("the result has both an error and records")
    token = result.get("resumption_token")
    if token is not None and (not isinstance(token, str) or len(token) > MAX_TOKEN
                              or not _TOKEN.fullmatch(token)):
        raise PageError("the result's resumptionToken is malformed")
    expires = result.get("token_expires")
    token_expires = None if expires is None else _timestamp(expires, "token expiry")

    kept: dict[str, Document | Deletion] = {}
    rejected: list[dict] = []
    rejected_count = 0
    for raw in records:
        try:
            record = clean_record(raw)
        except RecordError as exc:
            reason = str(exc)
        else:
            held = kept.get(record.source_id)
            if held is None or record.datestamp > held.datestamp:
                kept[record.source_id] = record
                if held is None:
                    continue
            reason = "the page lists this paper more than once"
        rejected_count += 1
        if len(rejected) < MAX_REJECTED_LISTED:
            rejected.append({"identifier": _listed(raw), "reason": reason})
    return Page(
        response_date=response_date, error_code=error_code, error_message=error_message,
        documents=tuple(r for r in kept.values() if isinstance(r, Document)),
        deletions=tuple(r for r in kept.values() if isinstance(r, Deletion)),
        resumption_token=token, token_expires=token_expires,
        rejected_count=rejected_count, rejected=tuple(rejected))


# -- the database ----------------------------------------------------------------

def _t(schema: str) -> sql.Composable:
    return sql.Identifier(schema, "documents")


def _row(doc: Document) -> dict:
    return {
        "source_id": doc.source_id, "datestamp": doc.datestamp.isoformat(), "set_specs": list(doc.set_specs),
        "title": doc.title, "authors": list(doc.authors), "author_count": doc.author_count,
        "abstract": doc.abstract, "categories": list(doc.categories), "license_url": doc.license_url,
        "reuse_class": doc.reuse_class, "doi": doc.doi, "journal_ref": doc.journal_ref,
        "created": doc.created.isoformat(), "updated": doc.updated.isoformat() if doc.updated else None,
        "content_hash": doc.content_hash,
    }


_UPSERT = """
    WITH r AS (
        SELECT * FROM jsonb_to_recordset(%(rows)s) AS r(
            source_id text, datestamp date, set_specs text[], title text, authors jsonb, author_count integer,
            abstract text, categories text[], license_url text, reuse_class text, doi text, journal_ref text,
            created date, updated date, content_hash text)),
    before AS (
        SELECT d.source_id, d.content_hash, d.deleted_at FROM {docs} d
        JOIN r ON d.source = %(source)s AND d.source_id = r.source_id),
    written AS (
        INSERT INTO {docs} AS d (source, source_id, datestamp, set_specs, title, authors, author_count, abstract,
                                 categories, primary_category, license_url, reuse_class, doi, journal_ref,
                                 created, updated, content_hash)
        SELECT %(source)s, r.source_id, r.datestamp, r.set_specs, r.title, r.authors, r.author_count, r.abstract,
               r.categories, r.categories[1], r.license_url, r.reuse_class, r.doi, r.journal_ref,
               r.created, r.updated, r.content_hash
        FROM r
        ON CONFLICT (source, source_id) DO UPDATE SET
            datestamp = EXCLUDED.datestamp, set_specs = EXCLUDED.set_specs, title = EXCLUDED.title,
            authors = EXCLUDED.authors, author_count = EXCLUDED.author_count, abstract = EXCLUDED.abstract,
            categories = EXCLUDED.categories, primary_category = EXCLUDED.primary_category,
            license_url = EXCLUDED.license_url, reuse_class = EXCLUDED.reuse_class, doi = EXCLUDED.doi,
            journal_ref = EXCLUDED.journal_ref, created = EXCLUDED.created, updated = EXCLUDED.updated,
            content_hash = EXCLUDED.content_hash,
            changed_at = CASE WHEN d.content_hash <> EXCLUDED.content_hash OR d.deleted_at IS NOT NULL
                              THEN now() ELSE d.changed_at END,
            deleted_at = NULL
        WHERE EXCLUDED.datestamp > d.datestamp
           OR (EXCLUDED.datestamp = d.datestamp
               AND (EXCLUDED.content_hash <> d.content_hash OR d.deleted_at IS NOT NULL))
        RETURNING d.source_id, d.content_hash)
    SELECT count(*) FILTER (WHERE b.source_id IS NULL),
           count(*) FILTER (WHERE b.source_id IS NOT NULL
                            AND (b.content_hash <> w.content_hash OR b.deleted_at IS NOT NULL))
    FROM written w LEFT JOIN before b USING (source_id)
"""

_DELETE = """
    WITH r AS (SELECT * FROM jsonb_to_recordset(%(rows)s) AS r(source_id text, datestamp date)),
    before AS (
        SELECT d.source_id, d.deleted_at FROM {docs} d
        JOIN r ON d.source = %(source)s AND d.source_id = r.source_id),
    written AS (
        UPDATE {docs} d SET datestamp = r.datestamp, deleted_at = coalesce(d.deleted_at, now()),
                            changed_at = CASE WHEN d.deleted_at IS NULL THEN now() ELSE d.changed_at END
        FROM r
        WHERE d.source = %(source)s AND d.source_id = r.source_id
          AND (r.datestamp > d.datestamp OR (r.datestamp = d.datestamp AND d.deleted_at IS NULL))
        RETURNING d.source_id)
    SELECT count(*) FILTER (WHERE b.deleted_at IS NULL) FROM written w JOIN before b USING (source_id)
"""


def upsert_page(conn: Connection, schema: str, page: Page) -> dict:
    """Record a checked page in the caller's transaction; returns counts.

    ``inserted`` and ``updated`` count papers new to the registry and papers
    whose metadata or deleted state changed; ``deleted`` counts papers that
    this page marked deleted. A deletion of a paper the registry has never
    held is ignored.
    """
    inserted = updated = deleted = 0
    if page.documents:
        inserted, updated = conn.execute(sql.SQL(_UPSERT).format(docs=_t(schema)), {
            "rows": Jsonb([_row(doc) for doc in page.documents]), "source": SOURCE}).fetchone()
    if page.deletions:
        deleted = conn.execute(sql.SQL(_DELETE).format(docs=_t(schema)), {
            "rows": Jsonb([{"source_id": d.source_id, "datestamp": d.datestamp.isoformat()}
                           for d in page.deletions]), "source": SOURCE}).fetchone()[0]
    return {"records": len(page.documents), "inserted": inserted, "updated": updated,
            "unchanged": len(page.documents) - inserted - updated,
            "deletions": len(page.deletions), "deleted": deleted, "rejected": page.rejected_count}
