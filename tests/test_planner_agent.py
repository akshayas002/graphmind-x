from tests.fakes import FakeLLM
from src.agents.planner_agent import PlannerAgent


def test_planner_parses_clean_json():
    response = (
        '```json\n{"reasoning": "needs one lookup", '
        '"sub_questions": [{"id": "sq1", "text": "who invented battery X", '
        '"requires_graph": true}]}\n```'
    )
    planner = PlannerAgent(FakeLLM(response=response))
    result = planner.plan("who invented battery X?")
    assert not result.used_fallback
    assert len(result.sub_questions) == 1
    assert result.sub_questions[0].text == "who invented battery X"


def test_planner_falls_back_on_llm_unreachable():
    planner = PlannerAgent(FakeLLM(raise_error=True))
    result = planner.plan("what is the meaning of life")
    assert result.used_fallback
    assert len(result.sub_questions) == 1
    assert result.sub_questions[0].text == "what is the meaning of life"
    assert result.sub_questions[0].requires_graph is True


def test_planner_falls_back_on_non_json_response():
    planner = PlannerAgent(FakeLLM(response="I think you should ask someone else."))
    result = planner.plan("who owns patent Y")
    assert result.used_fallback


def test_planner_falls_back_on_malformed_shape():
    planner = PlannerAgent(FakeLLM(response='{"sub_questions": [{"id": "sq1"}]}'))
    result = planner.plan("some question")
    assert result.used_fallback  # missing required "text" key — must not crash


def test_planner_falls_back_on_empty_sub_questions():
    planner = PlannerAgent(FakeLLM(response='{"sub_questions": []}'))
    result = planner.plan("some question")
    assert result.used_fallback
