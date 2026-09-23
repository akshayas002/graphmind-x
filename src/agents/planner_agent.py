"""
Week 2: Planner agent.

Decomposes a user's natural-language question into 1-3 sub-questions the
Graph Explorer can turn into Cypher. If the local LLM fails or returns
something unparseable, falls back to treating the whole question as a
single sub-question — always produces a usable PlannerOutput, never raises.
"""
from src.agents.base_agent import BaseAgent
from src.agents.models import PlannerOutput, SubQuestion
from src.agents.prompts import PLANNER_PROMPT


class PlannerAgent(BaseAgent):
    def plan(self, question: str) -> PlannerOutput:
        prompt = PLANNER_PROMPT.format(question=question)
        parsed = self._call_llm_for_json(prompt)

        if parsed is None:
            return self._fallback(question)

        try:
            raw_sub_questions = parsed["sub_questions"]
            if not raw_sub_questions:
                return self._fallback(question)

            sub_questions = [
                SubQuestion(
                    id=str(sq.get("id", f"sq{i+1}")),
                    text=str(sq["text"]),
                    requires_graph=bool(sq.get("requires_graph", True)),
                )
                for i, sq in enumerate(raw_sub_questions)
            ]
        except (KeyError, TypeError):
            # Shape matched enough to parse as JSON but not enough to build
            # SubQuestions from — treat identically to an outright LLM failure.
            return self._fallback(question)

        return PlannerOutput(
            sub_questions=sub_questions,
            reasoning=str(parsed.get("reasoning", "")),
            used_fallback=False,
        )

    @staticmethod
    def _fallback(question: str) -> PlannerOutput:
        return PlannerOutput(
            sub_questions=[SubQuestion(id="sq1", text=question, requires_graph=True)],
            reasoning="Planner LLM output was unavailable or unparseable; treating the "
                      "original question as a single sub-question.",
            used_fallback=True,
        )
