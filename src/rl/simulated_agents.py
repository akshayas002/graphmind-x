"""
Deterministic stand-ins for GraphExplorerAgent/VerifierAgent, used to train
and smoke-test the RL router without needing Ollama or Neo4j running.
"""
from src.agents.models import ExplorerOutput, VerificationResult


class SimulatedExplorer:
    def __init__(self, scenario: str):
        self.scenario = scenario
        self.call_count = 0

    def explore_once(self, sub_question, schema, previous_cypher=None):
        self.call_count += 1
        is_retry = previous_cypher is not None

        if self.scenario == "easy":
            row_count = 10
        elif self.scenario == "needs_one_retry":
            row_count = 0 if not is_retry else 8
        elif self.scenario == "unrecoverable":
            row_count = 0
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

        return ExplorerOutput(
            sub_question_id=sub_question.id,
            cypher=f"MATCH (n) RETURN n /* call {self.call_count} */",
            records=[{"dummy": i} for i in range(row_count)],
            row_count=row_count,
        )


class SimulatedVerifier:
    def verify(self, sub_question, explorer_output):
        passed = explorer_output.row_count > 0
        return VerificationResult(
            sub_question_id=sub_question.id, passed=passed,
            reason="simulated: row_count > 0" if passed else "simulated: zero rows",
        )