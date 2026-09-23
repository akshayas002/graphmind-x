from tests.fakes import FakeNeo4jClient


class RoutingFakeLLM:
    """Returns a different canned response depending on which agent's prompt
    it receives, simulating one local model serving all four agent roles."""

    def generate(self, prompt, temperature=0.1):
        if "query planner" in prompt:
            return (
                '{"reasoning": "single lookup", "sub_questions": '
                '[{"id": "sq1", "text": "who invented the battery patent", "requires_graph": true}]}'
            )
        if "Cypher query generator" in prompt:
            return "```cypher\nMATCH (i:Inventor)-[:INVENTED]->(p:Patent) RETURN i.name, p.title\n```"
        if "checking whether graph query results" in prompt:
            return '{"passed": true, "reason": "result directly names an inventor"}'
        if "answering a user" in prompt:
            return "Jane Doe invented the sulphide-rich composite battery patent, per the graph."
        return None


def test_orchestrator_full_happy_path():
    from src.orchestrator import Orchestrator

    fake_neo4j = FakeNeo4jClient(
        records=[{"i.name": "Jane Doe", "p.title": "Sulphide rich composite battery"}],
        schema={"labels": ["Patent", "Inventor"], "relationship_types": ["INVENTED"], "label_properties": {}},
    )
    orch = Orchestrator(llm_client=RoutingFakeLLM(), neo4j_client=fake_neo4j)
    result = orch.answer("Who invented the sulphide-rich composite battery patent?")

    assert not result.used_fallback
    assert "Jane Doe" in result.answer
    agents_seen = {step.agent for step in result.trace}
    assert agents_seen == {"Planner", "Explorer", "Verifier", "Synthesizer"}


def test_orchestrator_degrades_gracefully_when_neo4j_is_down():
    from src.orchestrator import Orchestrator

    class BrokenNeo4jClient:
        def get_schema_summary(self):
            raise ConnectionError("Neo4j is not running")

        def close(self):
            pass

    class MinimalLLM:
        def generate(self, prompt, temperature=0.1):
            return '{"reasoning": "x", "sub_questions": [{"id": "sq1", "text": "anything", "requires_graph": true}]}'

    orch = Orchestrator(llm_client=MinimalLLM(), neo4j_client=BrokenNeo4jClient())
    result = orch.answer("any question")
    assert result.used_fallback
    assert "connect" in result.answer.lower() or "neo4j" in result.answer.lower()
