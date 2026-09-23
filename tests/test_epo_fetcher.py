from src.ingestion.epo_fetcher import EPOFetcher


def test_build_cql_query_exact_phrase_default():
    q = EPOFetcher.build_cql_query("solid state battery", "IN", loose=False)
    assert q == 'pn=IN AND ta="solid state battery"'


def test_build_cql_query_loose_ands_individual_words():
    q = EPOFetcher.build_cql_query("solid state battery", "IN", loose=True)
    assert q == "pn=IN AND ta=solid AND ta=state AND ta=battery"


def test_build_cql_query_loose_single_word():
    q = EPOFetcher.build_cql_query("battery", "IN", loose=True)
    assert q == "pn=IN AND ta=battery"


def test_count_documents_handles_single_dict():
    page = {
        "ops:world-patent-data": {
            "ops:biblio-search": {
                "ops:search-result": {"exchange-documents": {"exchange-document": {"@doc-number": "1"}}}
            }
        }
    }
    assert EPOFetcher._count_documents(page) == 1


def test_count_documents_handles_list():
    page = {
        "ops:world-patent-data": {
            "ops:biblio-search": {
                "ops:search-result": {
                    "exchange-documents": {"exchange-document": [{"@doc-number": "1"}, {"@doc-number": "2"}]}
                }
            }
        }
    }
    assert EPOFetcher._count_documents(page) == 2


def test_count_documents_handles_missing_shape():
    assert EPOFetcher._count_documents({}) == 0


def test_extract_total_count():
    page = {"ops:world-patent-data": {"ops:biblio-search": {"@total-result-count": "42"}}}
    assert EPOFetcher._extract_total_count(page) == 42


def test_extract_total_count_missing_shape_returns_none():
    assert EPOFetcher._extract_total_count({}) is None


def test_probe_aggregates_counts_per_keyword():
    fetcher = EPOFetcher.__new__(EPOFetcher)
    responses_in_order = [
        {"ops:world-patent-data": {"ops:biblio-search": {"@total-result-count": "340"}}},
        {"ops:world-patent-data": {"ops:biblio-search": {"@total-result-count": "52"}}},
        {"ops:world-patent-data": {"ops:biblio-search": {"@total-result-count": "1"}}},
    ]
    call_log = []

    def fake_get_page(cql_query, start, end):
        call_log.append(cql_query)
        return responses_in_order[len(call_log) - 1]

    fetcher._get_page = fake_get_page

    counts = fetcher.probe(["battery", "lithium ion", "solid state battery"], "IN", loose=False)

    assert counts == {"battery": 340, "lithium ion": 52, "solid state battery": 1}
    assert call_log[0] == 'pn=IN AND ta="battery"'
    assert call_log[2] == 'pn=IN AND ta="solid state battery"'