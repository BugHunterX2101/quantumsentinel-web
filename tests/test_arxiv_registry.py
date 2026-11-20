"""The arXiv OAI-PMH parser (backend/ai/oai.py) and the document registry
(backend/ai/documents.py, migration 0013).

The fixtures under tests/fixtures/arxiv_oai are real responses of
https://oaipmh.arxiv.org/oai (set q-fin, metadataPrefix arXiv, captured
2026-11-20), trimmed to a few records each. The parser runs in a job process
on input from the internet, so it must refuse anything with a DTD; the
worker trusts nothing it returns, so every field is checked again before a
page reaches the database. Re-harvesting is idempotent.
"""
import copy
import datetime as dt
import json
from pathlib import Path

import psycopg
import pytest

from backend.ai import documents, oai
from backend.ai.documents import PageError, RecordError, clean_page, clean_record, reuse_class
from test_ai_tier import Ai

FIXTURES = Path(__file__).parent / "fixtures"
OAI_FIXTURES = FIXTURES / "arxiv_oai"
VECTORS = json.loads((FIXTURES / "qs_doc_sha256_v1.json").read_text("utf-8"))


def fixture(name: str) -> bytes:
    return (OAI_FIXTURES / name).read_bytes()


def parsed(name: str) -> dict:
    return oai.parse_list_records(fixture(name))


def envelope(records: str = "", token: str = "") -> bytes:
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            f"<responseDate>2026-11-20T06:47:36Z</responseDate><request>x</request>"
            f"<ListRecords>{records}{token}</ListRecords></OAI-PMH>").encode()


def xml_record(source_id="2611.00001", datestamp="2026-11-02", authors=None, status="", metadata=True) -> str:
    authors = authors if authors is not None else "<author><keyname>Doe</keyname><forenames>Jane</forenames></author>"
    meta = (f'<metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>{source_id}</id>'
            f"<created>2026-11-01</created><authors>{authors}</authors><title>A title</title>"
            f"<categories>q-fin.TR</categories><abstract>An abstract.</abstract></arXiv></metadata>") if metadata else ""
    return (f"<record><header{status}><identifier>oai:arXiv.org:{source_id}</identifier>"
            f"<datestamp>{datestamp}</datestamp><setSpec>q-fin:q-fin:TR</setSpec></header>{meta}</record>")


def raw(**overrides) -> dict:
    """One record as a job process returns it."""
    record = {
        "identifier": "oai:arXiv.org:2611.00001", "datestamp": "2026-11-02", "sets": ["q-fin:q-fin:TR"],
        "deleted": False, "id": "2611.00001", "created": "2026-11-01", "updated": None,
        "title": "A title", "abstract": "An abstract.", "categories": "q-fin.TR math.OC",
        "license": "http://creativecommons.org/licenses/by/4.0/", "doi": None, "journal_ref": None,
        "authors": [{"keyname": "Doe", "forenames": "Jane", "suffix": None}], "author_count": 1,
    }
    record.update(overrides)
    if "identifier" not in overrides and "id" in overrides and isinstance(overrides["id"], str):
        record["identifier"] = "oai:arXiv.org:" + overrides["id"]
    return record


def page(*records, **overrides) -> dict:
    result = {"response_date": "2026-11-20T06:47:36Z", "error": None, "records": list(records),
              "resumption_token": None, "token_expires": None}
    result.update(overrides)
    return result


# -- the parser ------------------------------------------------------------------

def test_a_page_parses_with_its_records_and_resumption_token():
    result = parsed("arxiv_page1.xml")
    assert result["response_date"] == "2026-11-20T06:47:13Z"
    assert result["error"] is None
    assert result["resumption_token"] == ("verb%3DListRecords%26metadataPrefix%3DarXiv%26from%3D2026-06-17"
                                          "%26until%3D2026-10-31%26set%3Dq-fin%26skip%3D14")
    assert result["token_expires"] == "2026-11-21T00:00:00Z"
    assert len(result["records"]) == 6
    first = result["records"][0]
    assert first["abstract"].strip().startswith("A continuous-time financial portfolio selection model")
    assert {k: v for k, v in first.items() if k != "abstract"} == {
        "identifier": "oai:arXiv.org:0709.4467", "datestamp": "2026-06-03",
        "sets": ["q-fin:q-fin:PM", "cs:cs:NA", "math:math:NA", "math:math:OC", "math:math:PR"],
        "deleted": False, "id": "0709.4467", "created": "2007-09-27", "updated": "2026-06-03",
        "title": "A Convex Stochastic Optimization Problem Arising from Portfolio Selection",
        "categories": "q-fin.PM cs.NA math.NA math.OC math.PR", "license": None,
        "doi": "10.1111/j.1467-9965.2007.00327.x",
        "journal_ref": "Mathematical Finance, Vol. 18, No. 1 (January 2008), 171-183",
        "authors": [{"keyname": "Jin", "forenames": "Hanqing", "suffix": None},
                    {"keyname": "Xu", "forenames": "Zuo Quan", "suffix": None},
                    {"keyname": "Zhou", "forenames": "Xun Yu", "suffix": None}],
        "author_count": 3}


def test_the_last_page_has_no_resumption_token():
    result = parsed("arxiv_last_page.xml")
    assert len(result["records"]) == 3
    assert result["resumption_token"] is None and result["token_expires"] is None


def test_an_empty_token_element_also_ends_the_list():
    result = oai.parse_list_records(envelope(xml_record(), "<resumptionToken completeListSize='1'/>"))
    assert result["resumption_token"] is None


@pytest.mark.parametrize("name, code", [("arxiv_no_records.xml", "noRecordsMatch"),
                                        ("arxiv_bad_token.xml", "badResumptionToken")])
def test_an_oai_error_is_returned_not_raised(name, code):
    result = parsed(name)
    assert result["error"]["code"] == code and result["error"]["message"]
    assert result["records"] == [] and result["resumption_token"] is None


def test_a_deleted_record_has_no_metadata():
    result = oai.parse_list_records(envelope(xml_record(status=' status="deleted"', metadata=False)))
    record = result["records"][0]
    assert record["deleted"] is True and record["identifier"] == "oai:arXiv.org:2611.00001"
    assert record["id"] is None and record["title"] is None and record["authors"] == []


def test_a_long_author_list_is_cut_but_counted():
    authors = "".join(f"<author><keyname>K{i}</keyname></author>" for i in range(250))
    record = oai.parse_list_records(envelope(xml_record(authors=authors)))["records"][0]
    assert len(record["authors"]) == oai.MAX_AUTHORS and record["author_count"] == 250
    assert record["authors"][0] == {"keyname": "K0", "forenames": None, "suffix": None}
    document = clean_record(record)
    assert len(document.authors) == documents.MAX_AUTHORS and document.author_count == 250


BILLION_LAUGHS = ('<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
                  '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]>'
                  '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><x>&lol2;</x></OAI-PMH>')
EXTERNAL_ENTITY = ('<?xml version="1.0"?><!DOCTYPE r [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
                   '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><x>&xxe;</x></OAI-PMH>')


@pytest.mark.parametrize("data, reason", [
    (BILLION_LAUGHS.encode(), "DTD"),
    (EXTERNAL_ENTITY.encode(), "DTD"),
    (EXTERNAL_ENTITY.replace("DOCTYPE", "doctype").encode(), "DTD"),
    (b"<!ENTITY x 'y'>" + envelope(), "DTD"),
    # UTF-16 could hide a DTD from a scan of the raw bytes: refused undecoded.
    (BILLION_LAUGHS.replace('version="1.0"', 'version="1.0" encoding="UTF-16"').encode("utf-16"), "not UTF-8"),
    (envelope().replace(b'encoding="UTF-8"', b'encoding="ISO-8859-1"'), "declares encoding"),
    (b"\xff\xfe<x/>", "not UTF-8"),
    (b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><x>&undeclared;</x></OAI-PMH>', "not well-formed"),
    (b"<OAI-PMH", "not well-formed"),
    (b"<html><body>Service Unavailable</body></html>", "not an OAI-PMH response"),
    (b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><responseDate>x</responseDate></OAI-PMH>',
     "neither ListRecords nor an error"),
    ("text, not bytes", "bytes"),
])
def test_what_is_not_a_safe_oai_pmh_response_is_refused(data, reason):
    with pytest.raises(oai.OaiError, match=reason):
        oai.parse_list_records(data)


def test_size_and_record_limits(monkeypatch):
    monkeypatch.setattr(oai, "MAX_BYTES", 100)
    with pytest.raises(oai.OaiError, match="larger than 100 bytes"):
        oai.parse_list_records(envelope())
    monkeypatch.setattr(oai, "MAX_BYTES", 1 << 20)
    monkeypatch.setattr(oai, "MAX_RECORDS", 2)
    with pytest.raises(oai.OaiError, match="more than 2 records"):
        oai.parse_list_records(envelope(xml_record() * 3))


# -- checking a job's result -------------------------------------------------------

@pytest.mark.parametrize("name", sorted(p.name for p in OAI_FIXTURES.glob("*.xml")))
def test_every_record_of_a_real_page_passes_the_checks(name):
    result = parsed(name)
    checked = clean_page(json.loads(json.dumps(result)))  # as it arrives from the job process
    assert checked.rejected_count == 0
    assert len(checked.documents) == len(result["records"])
    assert checked.resumption_token == result["resumption_token"]


def test_a_checked_document_from_a_real_page():
    checked = clean_page(parsed("arxiv_page1.xml"))
    assert checked.response_date == dt.datetime(2026, 11, 20, 6, 47, 13, tzinfo=dt.timezone.utc)
    assert checked.token_expires == dt.datetime(2026, 11, 21, tzinfo=dt.timezone.utc)
    doc = checked.documents[0]
    assert doc.source_id == "0709.4467" and doc.datestamp == dt.date(2026, 6, 3)
    assert doc.authors == ("Hanqing Jin", "Zuo Quan Xu", "Xun Yu Zhou") and doc.author_count == 3
    assert doc.categories == ("q-fin.PM", "cs.NA", "math.NA", "math.OC", "math.PR")
    assert doc.set_specs == ("cs:cs:NA", "math:math:NA", "math:math:OC", "math:math:PR", "q-fin:q-fin:PM")
    assert doc.license_url is None and doc.reuse_class == "restricted"
    assert doc.abstract.startswith("A continuous-time financial portfolio selection model")
    assert "  " not in doc.abstract and "\n" not in doc.abstract
    last = clean_page(parsed("arxiv_last_page.xml"))
    assert [(d.source_id, d.reuse_class) for d in last.documents] == [
        ("2610.33400", "restricted"), ("2606.04574", "noncommercial"), ("2606.25986", "open")]


@pytest.mark.parametrize("url, expected", [
    ("http://creativecommons.org/licenses/by/4.0/", "open"),
    ("http://creativecommons.org/licenses/by/3.0/", "open"),
    ("http://creativecommons.org/licenses/by-sa/4.0/", "open"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "open"),
    ("HTTPS://www.CreativeCommons.org/licenses/by/4.0", "open"),
    ("http://creativecommons.org/licenses/by-nc/4.0/", "noncommercial"),
    ("http://creativecommons.org/licenses/by-nc-sa/3.0/", "noncommercial"),
    ("http://creativecommons.org/licenses/by-nc-sa/4.0/", "noncommercial"),
    ("http://creativecommons.org/licenses/by-nc-nd/4.0/", "noncommercial"),
    ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "restricted"),
    ("http://arxiv.org/licenses/assumed-1991-2003/", "restricted"),
    ("http://creativecommons.org/licenses/by-nd/4.0/", "restricted"),
    ("http://creativecommons.org/licenses/by/4.0/legalcode", "restricted"),
    ("http://creativecommons.org.evil.example/licenses/by/4.0/", "restricted"),
    ("ftp://creativecommons.org/licenses/by/4.0/", "restricted"),
    ("creativecommons.org/licenses/by/4.0/", "restricted"),
    ("", "restricted"),
    (None, "restricted"),
    (42, "restricted"),
])
def test_reuse_classes(url, expected):
    assert reuse_class(url) == expected
    assert expected in documents.REUSE_CLASSES


def test_text_loses_control_and_format_characters():
    hostile = "A\x00B‮C⁦D​E﻿F­G\tH\r\nI\x1b[31mJK   L"
    assert documents.clean_text(hostile, 100) == "A BCDEFG H I [31mJK L"
    assert documents.clean_text("  many   spaces  ", 100) == "many spaces"
    assert documents.clean_text("x" * 50, 10) == "x" * 10
    assert documents.clean_text(None, 10) == ""
    with pytest.raises(RecordError):
        documents.clean_text(5, 10)


def test_identifiers_categories_and_sets():
    for source_id in ("2611.00001", "0709.4467", "hep-th/9901001", "math/0309136", "math.GT/0309136",
                      "cond-mat/0102536"):
        assert clean_record(raw(id=source_id)).source_id == source_id
    for bad in ("2611.001", "2611.000001", "HEP-TH/9901001", "hep-th/990100", "../etc/passwd",
                "2611.00001v2", "٢٦١١.00001"):
        with pytest.raises(RecordError, match="not an arXiv identifier"):
            clean_record(raw(id=bad))
    doc = clean_record(raw(categories="cond-mat.stat-mech q-fin.TR astro-ph BAD! q-fin.TR physics.soc-ph",
                           sets=["q-fin:q-fin:TR", "physics:cond-mat", "bad set", 7, "q-fin:q-fin:TR"]))
    assert doc.categories == ("cond-mat.stat-mech", "q-fin.TR", "astro-ph", "physics.soc-ph")
    assert doc.set_specs == ("physics:cond-mat", "q-fin:q-fin:TR")


def test_doi_licence_and_dates():
    assert clean_record(raw(doi="10.1000/a 10.1000/b")).doi == "10.1000/a"
    assert clean_record(raw(doi="not a doi")).doi is None
    assert clean_record(raw(license="javascript:alert(1)")).license_url is None
    assert clean_record(raw(license="javascript:alert(1)")).reuse_class == "restricted"
    assert clean_record(raw(license="http://x/" + "a" * 300)).license_url is None
    assert clean_record(raw(updated="2026-11-02")).updated == dt.date(2026, 11, 2)
    assert clean_record(raw(journal_ref="  J. Fin.\n 1 ")).journal_ref == "J. Fin. 1"
    for field, value in (("datestamp", "2026-13-01"), ("datestamp", "02/11/2026"), ("created", None),
                         ("updated", "yesterday"), ("datestamp", "٢٠٢٦-11-02")):
        with pytest.raises(RecordError, match="date"):
            clean_record(raw(**{field: value}))


@pytest.mark.parametrize("record, reason", [
    ("not an object", "not an object"),
    (raw(identifier="oai:example.org:2611.00001"), "not an arXiv OAI identifier"),
    (raw(identifier=None), "not an arXiv OAI identifier"),
    (raw(identifier="oai:arXiv.org:2611.00002"), "different papers"),
    (raw(title=" ​\n "), "no title"),
    (raw(title=["a"]), "not a string"),
    (raw(author_count=True), "author count"),
    (raw(author_count=-1), "author count"),
    (raw(authors="Jane Doe"), "authors are not a list"),
    (raw(authors=["Jane Doe"]), "author is not an object"),
    (raw(deleted="yes"), "deleted flag"),
    (raw(categories=["q-fin.TR"]), "categories are not a string"),
    (raw(sets="q-fin"), "sets are not a list"),
    (raw(doi=10), "DOI is not a string"),
])
def test_a_record_that_fails_a_check_is_dropped_and_the_rest_kept(record, reason):
    checked = clean_page(page(raw(id="2611.00009"), record))
    assert [d.source_id for d in checked.documents] == ["2611.00009"]
    assert checked.rejected_count == 1 and reason in checked.rejected[0]["reason"]


def test_rejections_are_all_counted_but_only_some_listed():
    checked = clean_page(page(*[raw(title="")] * 30))
    assert checked.rejected_count == 30 and len(checked.rejected) == documents.MAX_REJECTED_LISTED
    assert checked.rejected[0] == {"identifier": "oai:arXiv.org:2611.00001", "reason": "the record has no title"}


def test_a_paper_listed_twice_keeps_its_latest_record():
    checked = clean_page(page(raw(datestamp="2026-11-02", title="old"), raw(datestamp="2026-11-05", title="new"),
                              raw(datestamp="2026-11-03", title="middle")))
    assert [(d.title, d.datestamp.day) for d in checked.documents] == [("new", 5)]
    assert checked.rejected_count == 2


def test_a_deletion_is_kept_apart():
    checked = clean_page(page(raw(id="2611.00002"), {"identifier": "oai:arXiv.org:2611.00001",
                                                     "datestamp": "2026-11-03", "deleted": True}))
    assert [d.source_id for d in checked.documents] == ["2611.00002"]
    assert checked.deletions == (documents.Deletion("2611.00001", dt.date(2026, 11, 3)),)


@pytest.mark.parametrize("result, reason", [
    ([], "not an object"),
    (page(response_date=None), "responseDate"),
    (page(response_date="2026-11-20 06:47:36"), "responseDate"),
    (page(response_date="2026-02-30T00:00:00Z"), "responseDate"),
    (page(records=None), "records"),
    (page(records="x"), "records"),
    (page(records=[raw()] * (documents.MAX_RECORDS + 1)), "records"),
    (page(error="noRecordsMatch"), "error code"),
    (page(error={"code": "no records"}), "error code"),
    (page(raw(), error={"code": "noRecordsMatch"}), "both an error and records"),
    (page(resumption_token="has space"), "resumptionToken"),
    (page(resumption_token="x" * (documents.MAX_TOKEN + 1)), "resumptionToken"),
    (page(resumption_token=5), "resumptionToken"),
    (page(token_expires="tomorrow"), "token expiry"),
])
def test_a_result_that_is_not_a_page_is_refused(result, reason):
    with pytest.raises(PageError, match=reason):
        clean_page(result)


def test_an_error_page():
    checked = clean_page(page(error={"code": "noRecordsMatch", "message": "No\x00 records\n"}))
    assert (checked.error_code, checked.error_message) == ("noRecordsMatch", "No records")
    assert checked.documents == () and checked.deletions == ()


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda v: v["content"]["source_id"])
def test_content_hash_golden_vectors(vector):
    assert VECTORS["scheme"] == documents.HASH_SCHEME
    assert documents.canonical(vector["content"]).decode("utf-8") == vector["canonical"]
    assert documents.content_hash(vector["content"]) == vector["content_hash"]


def test_the_golden_vectors_are_what_the_checks_produce():
    real = next(d for d in clean_page(parsed("arxiv_page2.xml")).documents if d.source_id == "2602.16401")
    assert real.content() == VECTORS["vectors"][0]["content"]
    assert real.content_hash == VECTORS["vectors"][0]["content_hash"]
    content = VECTORS["vectors"][1]["content"]
    assert content["title"] == "Knötchen in São Paulo and Łódź"
    assert content["authors"] == ["José Müller Jr", "Collaboration"]


def test_the_hash_covers_the_metadata_not_the_harvest():
    doc = clean_record(raw())
    assert clean_record(raw(datestamp="2027-01-01", sets=["q-fin"])).content_hash == doc.content_hash
    for change in ({"title": "Another"}, {"abstract": "x"}, {"license": None}, {"doi": "10.1000/x"},
                   {"categories": "q-fin.ST"}, {"updated": "2026-11-02"}, {"author_count": 2}):
        assert clean_record(raw(**change)).content_hash != doc.content_hash, change


# -- the registry -----------------------------------------------------------------

class Registry(Ai):
    def upsert(self, result) -> dict:
        with self.conn.transaction():
            return documents.upsert_page(self.conn, self.schema, clean_page(copy.deepcopy(result)))

    def row(self, source_id: str) -> dict:
        cursor = self.conn.execute(
            f'SELECT * FROM "{self.schema}".documents WHERE source = %s AND source_id = %s', ("arxiv", source_id))
        values = cursor.fetchone()
        return None if values is None else dict(zip([c.name for c in cursor.description], values))

    def count(self) -> int:
        return self.conn.execute(f'SELECT count(*) FROM "{self.schema}".documents').fetchone()[0]


@pytest.fixture
def registry(make_engine):
    db = Registry(make_engine(migrated=True))
    yield db
    db.conn.close()


def counts(records=0, inserted=0, updated=0, deletions=0, deleted=0, rejected=0) -> dict:
    return {"records": records, "inserted": inserted, "updated": updated,
            "unchanged": records - inserted - updated, "deletions": deletions, "deleted": deleted,
            "rejected": rejected}


def test_a_real_page_is_recorded_once_and_harvesting_it_again_changes_nothing(registry):
    result = parsed("arxiv_page1.xml")
    assert registry.upsert(result) == counts(records=6, inserted=6)
    row = registry.row("0709.4467")
    assert row["title"] == "A Convex Stochastic Optimization Problem Arising from Portfolio Selection"
    assert row["authors"] == ["Hanqing Jin", "Zuo Quan Xu", "Xun Yu Zhou"] and row["author_count"] == 3
    assert row["categories"] == ["q-fin.PM", "cs.NA", "math.NA", "math.OC", "math.PR"]
    assert row["primary_category"] == "q-fin.PM"
    assert row["datestamp"] == dt.date(2026, 6, 3) and row["created"] == dt.date(2007, 9, 27)
    assert row["reuse_class"] == "restricted" and row["license_url"] is None
    assert row["doi"] == "10.1111/j.1467-9965.2007.00327.x" and row["deleted_at"] is None
    assert row["content_hash"] == clean_page(result).documents[0].content_hash
    changed_at = row["changed_at"]
    assert registry.upsert(result) == counts(records=6)
    assert registry.count() == 6 and registry.row("0709.4467")["changed_at"] == changed_at
    assert registry.upsert(parsed("arxiv_page2.xml")) == counts(records=3, inserted=3)
    assert registry.upsert(parsed("arxiv_no_records.xml")) == counts()


def test_a_later_record_replaces_an_earlier_one_never_the_reverse(registry):
    registry.upsert(page(raw(datestamp="2026-11-05", title="Fifth")))
    assert registry.upsert(page(raw(datestamp="2026-11-04", title="Fourth"))) == counts(records=1)
    assert registry.row("2611.00001")["title"] == "Fifth"
    assert registry.upsert(page(raw(datestamp="2026-11-06", title="Sixth"))) == counts(records=1, updated=1)
    row = registry.row("2611.00001")
    assert (row["title"], row["datestamp"]) == ("Sixth", dt.date(2026, 11, 6))


def test_a_later_datestamp_with_the_same_content_is_not_a_change(registry):
    registry.upsert(page(raw(datestamp="2026-11-05")))
    before = registry.row("2611.00001")
    assert registry.upsert(page(raw(datestamp="2026-11-07", sets=["q-fin:q-fin:ST"]))) == counts(records=1)
    after = registry.row("2611.00001")
    assert after["datestamp"] == dt.date(2026, 11, 7) and after["set_specs"] == ["q-fin:q-fin:ST"]
    assert after["changed_at"] == before["changed_at"]


def test_a_change_on_the_same_day_is_applied(registry):
    """OAI-PMH datestamps are days: two changes on one day share one."""
    registry.upsert(page(raw(title="Morning")))
    assert registry.upsert(page(raw(title="Evening"))) == counts(records=1, updated=1)
    assert registry.row("2611.00001")["title"] == "Evening"


def deletion(source_id="2611.00001", datestamp="2026-11-03") -> dict:
    return {"identifier": f"oai:arXiv.org:{source_id}", "datestamp": datestamp, "deleted": True}


def test_deletion_and_reinstatement(registry):
    registry.upsert(page(raw(title="Kept", datestamp="2026-11-02")))
    assert registry.upsert(page(deletion(datestamp="2026-11-01"))) == counts(deletions=1)
    assert registry.row("2611.00001")["deleted_at"] is None
    assert registry.upsert(page(deletion(datestamp="2026-11-03"))) == counts(deletions=1, deleted=1)
    row = registry.row("2611.00001")
    assert row["deleted_at"] is not None and row["title"] == "Kept" and row["datestamp"] == dt.date(2026, 11, 3)
    assert registry.upsert(page(deletion(datestamp="2026-11-04"))) == counts(deletions=1)
    assert registry.row("2611.00001")["deleted_at"] == row["deleted_at"]
    assert registry.upsert(page(raw(title="Back", datestamp="2026-11-03"))) == counts(records=1)
    assert registry.upsert(page(raw(title="Back", datestamp="2026-11-05"))) == counts(records=1, updated=1)
    row = registry.row("2611.00001")
    assert row["deleted_at"] is None and row["title"] == "Back"


def test_a_deletion_and_a_reinstatement_on_the_same_day_are_applied(registry):
    registry.upsert(page(raw(datestamp="2026-11-03")))
    assert registry.upsert(page(deletion(datestamp="2026-11-03"))) == counts(deletions=1, deleted=1)
    assert registry.upsert(page(raw(datestamp="2026-11-03"))) == counts(records=1, updated=1)
    assert registry.row("2611.00001")["deleted_at"] is None


def test_a_deletion_of_a_paper_never_recorded_is_ignored(registry):
    assert registry.upsert(page(deletion("2611.00042"))) == counts(deletions=1)
    assert registry.count() == 0


def test_a_page_with_rejected_records_records_the_rest(registry):
    assert registry.upsert(page(raw(id="2611.00003"), raw(title=""), "junk")) == counts(
        records=1, inserted=1, rejected=2)
    assert registry.count() == 1


def test_the_table_refuses_what_the_checks_never_produce(registry):
    good = clean_record(raw())
    columns = ("source, source_id, datestamp, title, author_count, abstract, reuse_class, created, content_hash")
    values = {"source": "arxiv", "source_id": good.source_id, "datestamp": good.datestamp, "title": good.title,
              "author_count": 1, "abstract": "", "reuse_class": "open", "created": good.created,
              "content_hash": good.content_hash}
    statement = (f'INSERT INTO "{registry.schema}".documents ({columns}) VALUES (%(source)s, %(source_id)s,'
                 " %(datestamp)s, %(title)s, %(author_count)s, %(abstract)s, %(reuse_class)s, %(created)s,"
                 " %(content_hash)s)")
    for bad in ({"source": "ssrn"}, {"source_id": "../x"}, {"title": ""}, {"title": "t" * 1001},
                {"abstract": "a" * 8001}, {"reuse_class": "public"}, {"author_count": -1},
                {"content_hash": "sha256:" + "0" * 64}):
        with pytest.raises(psycopg.errors.CheckViolation):
            with registry.conn.transaction():
                registry.conn.execute(statement, {**values, **bad})
    with registry.conn.transaction():
        registry.conn.execute(statement, values)
    with pytest.raises(psycopg.errors.UniqueViolation):
        with registry.conn.transaction():
            registry.conn.execute(statement, values)
