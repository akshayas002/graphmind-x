"""
Sanity test for entity_extractor.py using a small hand-built fake EPO OPS
page — no network or API credentials needed. Run this first, before
touching the real API, to confirm the parsing logic matches the field
names you're actually seeing (OPS's JSON shape has moved before).

Usage:
    python -m pytest tests/test_entity_extractor.py -v
"""
from src.extraction.entity_extractor import EntityExtractor, as_list, text_of

# Shaped like a real EPO OPS /published-data/search/biblio response
# (single-result case — deliberately using bare dicts, not lists, to
# exercise the as_list() normalization the real API's quirk requires).
FAKE_PAGE = {
    "ops:world-patent-data": {
        "ops:biblio-search": {
            "@total-result-count": "1",
            "ops:search-result": {
                "exchange-documents": {
                    "exchange-document": {
                        "@country": "IN",
                        "@doc-number": "202541012345",
                        "@kind": "A",
                        "@family-id": "12345678",
                        "bibliographic-data": {
                            "invention-title": [
                                {"@lang": "en", "$": "Solid-state battery electrolyte with lithium coating"},
                                {"@lang": "hi", "$": "ठोस अवस्था बैटरी"},
                            ],
                            "parties": {
                                "inventors": {
                                    "inventor": [
                                        {"inventor-name": {"name": {"$": "Jane Doe"}}},
                                        {"inventor-name": {"name": {"$": "John Smith"}}},
                                    ]
                                },
                                "applicants": {
                                    "applicant": {"applicant-name": {"name": {"$": "Acme Battery Pvt Ltd"}}}
                                },
                            },
                            "classifications-ipcr": {
                                "classification-ipcr": [
                                    {"text": {"$": "H01M 10/0562"}},
                                    {"text": {"$": "H01M 4/13"}},
                                ]
                            },
                        },
                    }
                }
            },
        }
    }
}


def test_as_list_normalizes_single_and_multi():
    assert as_list(None) == []
    assert as_list({"a": 1}) == [{"a": 1}]
    assert as_list([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]


def test_text_of_handles_dollar_and_plain_string():
    assert text_of({"$": "hello"}) == "hello"
    assert text_of("hello") == "hello"
    assert text_of({}) == ""


def test_extracts_patent_node_with_english_title():
    ex = EntityExtractor()
    ex.process_page(FAKE_PAGE)
    assert ("Patent", "IN202541012345A") in ex.nodes
    props = ex.nodes[("Patent", "IN202541012345A")]["properties"]
    assert props["title"] == "Solid-state battery electrolyte with lithium coating"
    assert props["country"] == "IN"


def test_extracts_inventors_and_edges():
    ex = EntityExtractor()
    ex.process_page(FAKE_PAGE)
    assert ("Inventor", "Jane Doe") in ex.nodes
    assert ("Inventor", "John Smith") in ex.nodes
    invented_edges = [e for e in ex.edges if e["type"] == "INVENTED"]
    assert len(invented_edges) == 2


def test_extracts_applicant_from_single_dict_not_list():
    """Applicant is a bare dict in the fixture (single result) — must still work."""
    ex = EntityExtractor()
    ex.process_page(FAKE_PAGE)
    assert ("Applicant", "Acme Battery Pvt Ltd") in ex.nodes


def test_extracts_ipc_codes():
    ex = EntityExtractor()
    ex.process_page(FAKE_PAGE)
    assert ("ClassificationCode", "H01M 10/0562") in ex.nodes
    assert ("ClassificationCode", "H01M 4/13") in ex.nodes


def test_skips_document_without_country_or_doc_number():
    ex = EntityExtractor()
    ex.process_exchange_document({"bibliographic-data": {"invention-title": {"$": "No id here"}}})
    assert len(ex.nodes) == 0
    assert ex._skipped == 1
