from src.graph.cypher_validator import extract_cypher, validate, enforce_limit, sanitize_and_validate


def test_extract_cypher_from_fenced_block():
    assert extract_cypher("```cypher\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"


def test_extract_cypher_with_surrounding_prose():
    text = "Sure! Here is the query:\n```\nMATCH (p:Patent) RETURN p\n```\nHope that helps!"
    assert extract_cypher(text) == "MATCH (p:Patent) RETURN p"


def test_extract_cypher_no_fence():
    assert extract_cypher("MATCH (n) RETURN n") == "MATCH (n) RETURN n"


def test_validate_accepts_simple_read_query():
    ok, reason = validate("MATCH (p:Patent) RETURN p LIMIT 10")
    assert ok, reason


def test_validate_rejects_destructive_keywords():
    destructive = [
        "MATCH (n) DETACH DELETE n",
        "CREATE (n:X) RETURN n",
        "MATCH (n) SET n.hacked = true RETURN n",
        "MATCH (n) REMOVE n.title RETURN n",
        "DROP CONSTRAINT foo",
        "MERGE (n:X {id: 1}) RETURN n",
        'CALL apoc.periodic.iterate("...", "...", {}) YIELD batches RETURN batches',
        'LOAD CSV FROM "file:///x.csv" AS row RETURN row',
    ]
    for query in destructive:
        ok, reason = validate(query)
        assert not ok, f"should have rejected: {query}"


def test_validate_rejects_statement_chaining():
    ok, reason = validate("MATCH (n) RETURN n; MATCH (m) DETACH DELETE m")
    assert not ok
    assert "multiple statements" in reason


def test_validate_allows_single_trailing_semicolon():
    ok, reason = validate("MATCH (n) RETURN n;")
    assert ok, reason


def test_validate_rejects_missing_return_or_match():
    assert not validate("MATCH (n)")[0]
    assert not validate("RETURN 1")[0]
    assert not validate("")[0]
    assert not validate("   ")[0]


def test_enforce_limit_appends_when_missing():
    assert enforce_limit("MATCH (n) RETURN n") == "MATCH (n) RETURN n\nLIMIT 100"


def test_enforce_limit_leaves_existing_limit():
    assert enforce_limit("MATCH (n) RETURN n LIMIT 5") == "MATCH (n) RETURN n LIMIT 5"


def test_sanitize_and_validate_rejects_sneaky_fenced_chained_attack():
    sneaky = '```cypher\nMATCH (n:Patent) RETURN n; MATCH (m) DETACH DELETE m\n```'
    safe, reason = sanitize_and_validate(sneaky)
    assert safe is None


def test_sanitize_and_validate_happy_path():
    good = "```cypher\nMATCH (p:Patent)-[:INVENTED]-(i:Inventor) RETURN p.title, i.name\n```"
    safe, reason = sanitize_and_validate(good)
    assert safe == "MATCH (p:Patent)-[:INVENTED]-(i:Inventor) RETURN p.title, i.name\nLIMIT 100"
