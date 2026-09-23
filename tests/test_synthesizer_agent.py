from tests.fakes import FakeLLM
from src.agents.synthesizer_agent import SynthesizerAgent
from src.agents.models import SubQuestion, ExplorerOutput, VerificationResult

SQS = [SubQuestion(id="sq1", text="who invented X")]
EOS = [ExplorerOutput(sub_question_id="sq1", cypher="...", records=[{"name": "Jane Doe"}], row_count=1)]


def test_synthesizer_clean_path():
    s = SynthesizerAgent(FakeLLM(response="Jane Doe invented X, according to the patent record."))
    verifs = [VerificationResult(sub_question_id="sq1", passed=True, reason="ok")]
    result = s.synthesize("who invented X", SQS, EOS, verifs)
    assert not result.used_fallback
    assert "Jane Doe" in result.answer


def test_synthesizer_honest_when_nothing_verified():
    s = SynthesizerAgent(FakeLLM(response="should not be called"))
    verifs_failed = [VerificationResult(sub_question_id="sq1", passed=False, reason="no data")]
    result = s.synthesize("who invented X", SQS, EOS, verifs_failed)
    assert result.used_fallback
    assert "could" in result.answer.lower()  # "couldn't find" / "could not"


def test_synthesizer_templated_fallback_when_llm_down():
    s = SynthesizerAgent(FakeLLM(raise_error=True))
    verifs = [VerificationResult(sub_question_id="sq1", passed=True, reason="ok")]
    result = s.synthesize("who invented X", SQS, EOS, verifs)
    assert result.used_fallback
    assert "Jane Doe" in result.answer or "1 result" in result.answer
