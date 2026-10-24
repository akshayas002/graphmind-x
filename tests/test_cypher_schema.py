"""Tests for the deterministic schema gate and its integration into the Explorer."""
from src.agents.graph_explorer_agent import GraphExplorerAgent
from src.agents.models import SubQuestion
from src.graph.cypher_schema import check_and_fix

SCHEMA = {
    "labels": ["Patent", "Inventor", "Applicant", "ClassificationCode", "Concept"],
    "relationship_types": ["INVENTED", "OWNS", "CLASSIFIED_AS", "MENTIONS"],
    "relationship_patterns": [
        {"from": "Inventor", "type": "INVENTED", "to": "Patent"},
        {"from": "Applicant", "type": "OWNS", "to": "Patent"},
        {"from": "Patent", "type": "CLASSIFIED_AS", "to": "ClassificationCode"},
        {"from": "Patent", "type": "MENTIONS", "to": "Concept"},
    ],
    "label_properties": {},
}
SQ = SubQuestion(id="sq1", text="Who invented the sulphide patent?")


def test_flips_reversed_right_arrow():
    q = "MATCH (p:Patent)-[:INVENTED]->(i:Inventor) RETURN i.name"
    r = check_and_fix(q, SCHEMA)
    assert r.error is None
    assert "(p:Patent)<-[:INVENTED]-(i:Inventor)" in r.cypher
    assert r.fixes


def test_flips_reversed_left_arrow():
    r = check_and_fix("MATCH (a:Applicant)<-[:OWNS]-(p:Patent) RETURN a.name", SCHEMA)
    assert r.error is None and "-[:OWNS]->" in r.cypher


def test_correct_query_is_untouched():
    q = "MATCH (i:Inventor)-[:INVENTED]->(p:Patent) RETURN i.name"
    r = check_and_fix(q, SCHEMA)
    assert r.cypher == q and not r.fixes and r.error is None


def test_rejects_unknown_label_and_relationship():
    q = "MATCH (p:Patent)-[:kind]->(k:Kind) RETURN p.title"
    r = check_and_fix(q, SCHEMA)
    assert r.error is not None and "Kind" in r.error


def test_rejects_unknown_relationship_with_valid_labels():
    r = check_and_fix("MATCH (p:Patent)-[:WROTE]->(i:Inventor) RETURN p", SCHEMA)
    assert r.error is not None and "WROTE" in r.error


def test_rejects_relationship_between_wrong_labels():
    r = check_and_fix("MATCH (a:Applicant)-[:INVENTED]->(p:Patent) RETURN a", SCHEMA)
    assert r.error is not None and "never connects" in r.error


def test_resolves_unlabeled_endpoint_from_earlier_variable():
    q = "MATCH (p:Patent) MATCH (p)-[:INVENTED]->(i:Inventor) RETURN i.name"
    r = check_and_fix(q, SCHEMA)
    assert "(p)<-[:INVENTED]-(i:Inventor)" in r.cypher


def test_fixes_only_the_bad_hop_in_a_chain():
    q = "MATCH (a:Applicant)-[:OWNS]->(p:Patent)-[:INVENTED]->(i:Inventor) RETURN a.name, i.name"
    r = check_and_fix(q, SCHEMA)
    assert "(a:Applicant)-[:OWNS]->(p:Patent)<-[:INVENTED]-(i:Inventor)" in r.cypher


def test_string_literals_are_ignored():
    q = "MATCH (p:Patent) WHERE p.title CONTAINS '(x:Fake)-[:nope]->(y:Fake)' RETURN p"
    r = check_and_fix(q, SCHEMA)
    assert r.error is None and r.cypher == q


def test_undirected_and_empty_schema_are_not_blocked():
    assert check_and_fix("MATCH (p:Patent)-[:INVENTED]-(i:Inventor) RETURN p", SCHEMA).error is None
    assert check_and_fix("MATCH (p:Kind) RETURN p", {"labels": [], "relationship_patterns": []}).error is None


# ---- Explorer integration -------------------------------------------------

class _LLM:
    def __init__(self, responses):
        self.responses, self.prompts = list(responses), []

    def generate(self, prompt, temperature=0.1):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class _DB:
    """Returns a row only when the arrow matches the real graph direction."""

    def __init__(self):
        self.log = []

    def run_read_query(self, cypher, parameters=None):
        self.log.append(cypher)
        ok = "(p:Patent)<-[:INVENTED]-(i:Inventor)" in cypher or "(i:Inventor)-[:INVENTED]->(p:Patent)" in cypher
        return [{"i.name": "Jane"}] if ok else []


_REVERSED = "```cypher\nMATCH (p:Patent)-[:INVENTED]->(i:Inventor) RETURN i.name\n```"
_GOOD = "```cypher\nMATCH (i:Inventor)-[:INVENTED]->(p:Patent) RETURN i.name\n```"
_BAD_LABEL = "```cypher\nMATCH (p:Patent)-[:kind]->(k:Kind) RETURN p.title\n```"


def test_explorer_answers_reversed_arrow_on_first_attempt():
    llm, db = _LLM([_REVERSED]), _DB()
    r = GraphExplorerAgent(llm, db).explore(SQ, SCHEMA)
    assert r.error is None and r.row_count == 1
    assert len(llm.prompts) == 1  # auto-flip made the retry unnecessary


def test_explorer_rejects_invented_label_before_db_and_tells_model_why():
    llm, db = _LLM([_BAD_LABEL, _GOOD]), _DB()
    r = GraphExplorerAgent(llm, db).explore(SQ, SCHEMA)
    assert len(llm.prompts) == 2
    assert "Kind" in llm.prompts[1] and "Valid labels" in llm.prompts[1]
    assert len(db.log) == 1  # the invalid query never reached the database
    assert r.error is None and r.row_count == 1


def test_explore_once_never_retries_by_itself():
    llm, db = _LLM([_BAD_LABEL]), _DB()
    r = GraphExplorerAgent(llm, db).explore_once(SQ, SCHEMA)
    assert len(llm.prompts) == 1 and r.error and db.log == []

def test_reports_missing_data_when_relationship_has_no_edges():
    """Pattern list is complete (<50 triples) and INVENTED isn't in it: the type
    exists but no data was loaded, so say so instead of silently returning 0 rows."""
    schema = {
        "labels": ["Patent", "Inventor", "Concept"],
        "relationship_types": ["INVENTED", "MENTIONS"],
        "relationship_patterns": [{"from": "Patent", "type": "MENTIONS", "to": "Concept"}],
    }
    q = "MATCH (i:Inventor)-[:INVENTED]->(p:Patent) RETURN i.name"
    r = check_and_fix(q, schema)
    assert r.error is not None and "no [:INVENTED]" in r.error
    ok = "MATCH (p:Patent)-[:MENTIONS]->(c:Concept) RETURN c.name"
    assert check_and_fix(ok, schema).error is None


def test_does_not_claim_missing_data_when_pattern_list_may_be_truncated():
    patterns = [{"from": f"A{i}", "type": f"T{i}", "to": "B"} for i in range(50)]  # at the query limit
    schema = {
        "labels": ["Patent", "Inventor"],
        "relationship_types": ["INVENTED"],
        "relationship_patterns": patterns,
    }
    q = "MATCH (i:Inventor)-[:INVENTED]->(p:Patent) RETURN i.name"
    assert check_and_fix(q, schema).error is None