"""
Week 3: Verifier agent.

Sanity-checks an ExplorerOutput before it's trusted by the Synthesizer.
Deliberately layered: cheap structural checks run first and can reject
outright (an execution error, or zero rows for a question that presumably
expects data) without needing the LLM at all. Only when structural checks
pass does it optionally ask the LLM whether the results actually look
relevant — and if that call fails, it fails OPEN (passes) rather than
blocking the whole pipeline on a flaky local model. This is a heuristic,
not a rigorous verifier — the RL-trained routing/verification planned for
later weeks is what makes this actually reliable; for now it's a
best-effort net to catch the obvious failures (errors, empty results).
"""
from src.agents.base_agent import BaseAgent
from src.agents.models import ExplorerOutput, SubQuestion, VerificationResult
from src.agents.prompts import VERIFIER_PROMPT


class VerifierAgent(BaseAgent):
    def verify(self, sub_question: SubQuestion, explorer_output: ExplorerOutput) -> VerificationResult:
        # --- Structural checks first, no LLM needed ---
        if explorer_output.error is not None:
            return VerificationResult(
                sub_question_id=sub_question.id,
                passed=False,
                reason=f"Explorer failed: {explorer_output.error}",
            )

        if explorer_output.row_count == 0:
            return VerificationResult(
                sub_question_id=sub_question.id,
                passed=False,
                reason="Query executed successfully but returned zero rows.",
            )

        # --- Optional LLM relevance check ---
        sample = explorer_output.records[:3]
        prompt = VERIFIER_PROMPT.format(
            question=sub_question.text,
            row_count=explorer_output.row_count,
            sample_records=sample,
        )
        parsed = self._call_llm_for_json(prompt)

        if parsed is None:
            # Fail OPEN: structural checks already passed, and we don't want
            # a flaky local model to block an otherwise-good result.
            return VerificationResult(
                sub_question_id=sub_question.id,
                passed=True,
                reason="Structural checks passed; LLM relevance check unavailable, defaulting to pass.",
            )

        passed = bool(parsed.get("passed", True))
        reason = str(parsed.get("reason", "")) or "LLM relevance check completed."
        return VerificationResult(sub_question_id=sub_question.id, passed=passed, reason=reason)
