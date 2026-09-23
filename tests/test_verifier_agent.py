from tests.fakes import FakeLLM
from src.agents.verifier_agent import VerifierAgent
from src.agents.models import SubQuestion, ExplorerOutput

SQ = SubQuestion(id="sq1", text="who invented X")


def test_verifier_auto_fails_on_explorer_error():
    v = VerifierAgent(FakeLLM(raise_error=True))
    eo = ExplorerOutput(sub_question_id="sq1", cypher=None, error="Neo4j down")
    result = v.verify(SQ, eo)
    assert result.passed is False
    assert "Explorer failed" in result.reason


def test_verifier_auto_fails_on_zero_rows():
    v = VerifierAgent(FakeLLM(raise_error=True))
    eo = ExplorerOutput(sub_question_id="sq1", cypher="MATCH (n) RETURN n", records=[], row_count=0)
    result = v.verify(SQ, eo)
    assert result.passed is False
    assert "zero rows" in result.reason


def test_verifier_llm_confirms_relevance():
    v = VerifierAgent(FakeLLM(response='{"passed": true, "reason": "matches"}'))
    eo = ExplorerOutput(sub_question_id="sq1", cypher="...", records=[{"name": "Jane"}], row_count=1)
    result = v.verify(SQ, eo)
    assert result.passed is True


def test_verifier_llm_flags_irrelevant_despite_nonempty_results():
    v = VerifierAgent(FakeLLM(response='{"passed": false, "reason": "irrelevant"}'))
    eo = ExplorerOutput(sub_question_id="sq1", cypher="...", records=[{"name": "Jane"}], row_count=1)
    result = v.verify(SQ, eo)
    assert result.passed is False


def test_verifier_fails_open_when_llm_unreachable_but_structural_ok():
    v = VerifierAgent(FakeLLM(raise_error=True))
    eo = ExplorerOutput(sub_question_id="sq1", cypher="...", records=[{"name": "Jane"}], row_count=1)
    result = v.verify(SQ, eo)
    assert result.passed is True
