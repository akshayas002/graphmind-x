"""
Inference-time RL router: loads a trained Q-table and uses it (greedy) to
drive one sub-question through Explorer/Verifier. Falls back to a sensible
default policy if no trained Q-table exists yet, rather than crashing.
"""
from pathlib import Path

from src.agents.models import ExplorerOutput, SubQuestion, VerificationResult
from src.rl.actions import ACCEPT, EXPLORE, REJECT, VERIFY, legal_actions
from src.rl.environment import RouterEnvironment
from src.rl.q_learning import QLearningAgent


class RLRouter:
    def __init__(self, explorer, verifier, q_table_path: str | Path | None = None):
        self.explorer = explorer
        self.verifier = verifier
        self.agent = QLearningAgent(epsilon=0.0)
        self.has_trained_policy = False

        if q_table_path and Path(q_table_path).exists():
            self.agent.load(q_table_path)
            self.has_trained_policy = True

    def route(self, sub_question: SubQuestion, schema: dict):
        env = RouterEnvironment(self.explorer, self.verifier, sub_question, schema)
        state = env.reset()
        action_trace: list[str] = []
        done = False
        last_verification: VerificationResult | None = None

        while not done:
            actions = legal_actions(state)
            if self.has_trained_policy:
                action = self.agent.choose_action(state.as_tuple(), actions, greedy=True)
            else:
                action = self._default_policy(state, actions)

            action_trace.append(action)
            state, reward, done, info = env.step(action)

            if action == VERIFY:
                last_verification = VerificationResult(
                    sub_question_id=sub_question.id,
                    passed=(state.verified == "pass"),
                    reason=f"RL router: verified as {state.verified}",
                )

        return env.explorer_output, last_verification, action_trace

    @staticmethod
    def _default_policy(state, legal: list[str]) -> str:
        if EXPLORE in legal and state.stage == "start":
            return EXPLORE
        if VERIFY in legal:
            return VERIFY
        if state.verified == "pass" and ACCEPT in legal:
            return ACCEPT
        if REJECT in legal:
            return REJECT
        return legal[0]