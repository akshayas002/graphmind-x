"""
Deterministic stand-ins for GraphExplorerAgent/VerifierAgent, used to train
and smoke-test the RL router without Ollama or Neo4j.

Scenarios:
  easy                 first attempt finds plenty of rows (many), verify passes
  few_results_pass      first attempt finds a SMALL number of rows, verify passes
                        -- this is the common real case (a specific, narrow
                        question) and was missing before: without it, the
                        router has no learned values for "few rows, first
                        try" and picks arbitrarily, including rejecting a
                        correct small answer without ever verifying it.
  few_results_fail      first attempt finds a small number of rows, but they
                        don't actually answer the question (verify fails) --
                        so "few rows" alone must not be learned as "always good"
  needs_one_retry       first attempt finds 0 rows, any retry finds rows
  unrecoverable          always 0 rows
  error_then_fixed      first attempt is a schema/validation error (no DB call),
                        any retry succeeds
  error_unrecoverable    every attempt errors
"""
from src.agents.models import ExplorerOutput, VerificationResult

SCENARIOS = [
    "easy",
    "few_results_pass",
    "few_results_fail",
    "needs_one_retry",
    "unrecoverable",
    "error_then_fixed",
    "error_unrecoverable",
]

_SMALL_ROW_COUNT = 2  # within the "few" bucket (1-5), matches a narrow real question


class SimulatedExplorer:
    def __init__(self, scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario
        self.call_count = 0

    def explore_once(self, sub_question, schema, previous_cypher=None, previous_error=None):
        self.call_count += 1
        is_retry = previous_cypher is not None
        cypher = f"MATCH (n) RETURN n /* call {self.call_count} */"

        if self.scenario in ("error_then_fixed", "error_unrecoverable"):
            fixed = self.scenario == "error_then_fixed" and is_retry
            if not fixed:
                return ExplorerOutput(
                    sub_question_id=sub_question.id, cypher=cypher,
                    error="Generated Cypher doesn't match the graph schema: simulated",
                )
            row_count = 8
        elif self.scenario == "easy":
            row_count = 10
        elif self.scenario in ("few_results_pass", "few_results_fail"):
            row_count = _SMALL_ROW_COUNT
        elif self.scenario == "needs_one_retry":
            row_count = 8 if is_retry else 0
        else:  # unrecoverable
            row_count = 0

        eo = ExplorerOutput(
            sub_question_id=sub_question.id, cypher=cypher,
            records=[{"dummy": i} for i in range(row_count)], row_count=row_count,
        )
        if self.scenario == "few_results_fail":
            eo._force_fail = True  # tells SimulatedVerifier these rows don't actually answer the question
        return eo


class SimulatedVerifier:
    def verify(self, sub_question, explorer_output):
        if explorer_output.row_count == 0:
            passed = False
        elif getattr(explorer_output, "_force_fail", False):
            passed = False
        else:
            passed = True
        return VerificationResult(
            sub_question_id=sub_question.id, passed=passed,
            reason="simulated: row_count > 0" if passed else "simulated: zero rows or flagged irrelevant",
        )