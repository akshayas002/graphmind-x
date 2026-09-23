"""
Week 3: Synthesizer agent.

Takes everything the pipeline gathered — sub-questions, Explorer results,
Verifier verdicts — and produces one natural-language answer. If the LLM
is unavailable, falls back to a templated answer built directly from the
raw records rather than failing the whole pipeline at the last step.
"""
from src.agents.base_agent import BaseAgent
from src.agents.models import ExplorerOutput, FinalAnswer, SubQuestion, VerificationResult
from src.agents.prompts import SYNTHESIZER_PROMPT


class SynthesizerAgent(BaseAgent):
    def synthesize(
        self,
        original_question: str,
        sub_questions: list[SubQuestion],
        explorer_outputs: list[ExplorerOutput],
        verifications: list[VerificationResult],
    ) -> FinalAnswer:
        verified_by_id = {v.sub_question_id: v for v in verifications}
        passed_outputs = [
            eo for eo in explorer_outputs
            if verified_by_id.get(eo.sub_question_id) and verified_by_id[eo.sub_question_id].passed
        ]

        if not passed_outputs:
            return FinalAnswer(
                answer="I couldn't find verified information in the graph to answer this question. "
                       "The query may need rephrasing, or this data may not be in the graph yet.",
                used_fallback=True,
            )

        facts_text = self._format_facts(sub_questions, passed_outputs)
        prompt = SYNTHESIZER_PROMPT.format(question=original_question, facts=facts_text)

        answer_text = self._call_llm(prompt, temperature=0.2)
        if answer_text is None:
            return FinalAnswer(
                answer=self._templated_fallback(passed_outputs),
                used_fallback=True,
            )

        return FinalAnswer(answer=answer_text.strip(), used_fallback=False)

    @staticmethod
    def _format_facts(sub_questions: list[SubQuestion], outputs: list[ExplorerOutput]) -> str:
        text_by_id = {sq.id: sq.text for sq in sub_questions}
        lines = []
        for eo in outputs:
            question_text = text_by_id.get(eo.sub_question_id, eo.sub_question_id)
            lines.append(f"Sub-question: {question_text}")
            for record in eo.records[:10]:  # cap context size
                lines.append(f"  - {record}")
        return "\n".join(lines)

    @staticmethod
    def _templated_fallback(outputs: list[ExplorerOutput]) -> str:
        parts = ["Here's what the graph returned (LLM summarization was unavailable):"]
        for eo in outputs:
            parts.append(f"- {eo.row_count} result(s): {eo.records[:5]}")
        return "\n".join(parts)
