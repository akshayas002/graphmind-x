from tests.fakes import FakeLLM, FakeNeo4jClient
from src.agents.graph_explorer_agent import GraphExplorerAgent
from src.agents.models import SubQuestion

SCHEMA = {"labels": ["Patent", "Inventor"], "relationship_types": ["INVENTED"], "label_properties": {}}
SQ = SubQuestion(id="sq1", text="who invented X", requires_graph=True)


def test_explorer_happy_path():
    llm = FakeLLM(response="```cypher\nMATCH (p:Patent)-[:INVENTED]-(i:Inventor) RETURN p, i\n```")
    db = FakeNeo4jClient(records=[{"p": {"title": "X"}, "i": {"name": "Jane"}}])
    explorer = GraphExplorerAgent(llm, db)
    result = explorer.explore(SQ, SCHEMA)
    assert result.error is None
    assert result.row_count == 1
    assert "LIMIT" in result.cypher


def test_explorer_rejects_destructive_query_before_hitting_db():
    llm = FakeLLM(response="```cypher\nMATCH (n) DETACH DELETE n\n```")
    db = FakeNeo4jClient()
    explorer = GraphExplorerAgent(llm, db)
    result = explorer.explore(SQ, SCHEMA)
    assert result.error is not None
    assert "rejected" in result.error
    assert db.last_query is None, "destructive query must never reach the database"


def test_explorer_handles_llm_unreachable():
    llm = FakeLLM(raise_error=True)
    explorer = GraphExplorerAgent(llm, FakeNeo4jClient())
    result = explorer.explore(SQ, SCHEMA)
    assert result.error is not None
    assert result.cypher is None


def test_explorer_handles_neo4j_execution_error():
    llm = FakeLLM(response="```cypher\nMATCH (p:Patent) RETURN p\n```")
    db = FakeNeo4jClient(raise_error=True)
    explorer = GraphExplorerAgent(llm, db)
    result = explorer.explore(SQ, SCHEMA)
    assert result.error is not None
    assert "Neo4j execution failed" in result.error


def test_explorer_works_without_code_fence():
    llm = FakeLLM(response="MATCH (p:Patent) RETURN p.title")
    db = FakeNeo4jClient(records=[{"p.title": "Some Patent"}])
    explorer = GraphExplorerAgent(llm, db)
    result = explorer.explore(SQ, SCHEMA)
    assert result.error is None
    assert result.row_count == 1
