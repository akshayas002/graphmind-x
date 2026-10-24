from src.agents.base_agent import BaseAgent
from src.agents.models import FinalAnswer
from src.agents.prompts import SYNTHESIZER_PROMPT

NOT_FOUND = ("I couldn't find verified information in the graph to answer this question. "
             "The query may need rephrasing, or this data may not be in the graph yet.")


class SynthesizerAgent(BaseAgent):
    def synthesize(self, original_question, sub_questions, explorer_outputs, verifications):
        by_id = {v.sub_question_id: v for v in verifications}
        passed = [eo for eo in explorer_outputs
                  if by_id.get(eo.sub_question_id) and by_id[eo.sub_question_id].passed]

        if not passed:
            # Say why, so the user can tell a query problem from a data problem.
            reasons = [eo.error for eo in explorer_outputs if eo.error]
            reasons += [v.reason for v in verifications if not v.passed]
            msg = NOT_FOUND + (f" Details: {reasons[-1]}" if reasons else "")
            return FinalAnswer(msg, used_fallback=True)

        text_by_id = {sq.id: sq.text for sq in sub_questions}
        lines = []
        for eo in passed:
            lines.append(f"Sub-question: {text_by_id.get(eo.sub_question_id, eo.sub_question_id)}")
            lines += [f"  - {r}" for r in eo.records[:10]]
        ans = self._call_llm(SYNTHESIZER_PROMPT.format(question=original_question, facts="\n".join(lines)),
                             temperature=0.2)
        if ans is None:
            parts = ["Here's what the graph returned (LLM summarization was unavailable):"]
            parts += [f"- {eo.row_count} result(s): {eo.records[:5]}" for eo in passed]
            return FinalAnswer("\n".join(parts), used_fallback=True)
        return FinalAnswer(ans.strip(), used_fallback=False)