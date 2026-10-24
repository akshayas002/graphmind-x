from pathlib import Path

from src.rl.actions import ACCEPT, EXPLORE, REJECT, VERIFY, legal_actions
from src.rl.environment import RouterEnvironment
from src.rl.q_learning import QLearningAgent


class RLRouter:
    def __init__(self, explorer, verifier, q_table_path=None):
        self.explorer = explorer
        self.verifier = verifier
        self.agent = QLearningAgent(epsilon=0.0)
        self.has_trained_policy = False
        if q_table_path and Path(q_table_path).exists():
            self.agent.load(q_table_path)
            self.has_trained_policy = True

    def route(self, sub_question, schema):
        env = RouterEnvironment(self.explorer, self.verifier, sub_question, schema)
        state = env.reset()
        trace, done = [], False
        while not done:
            actions = legal_actions(state)
            if self.has_trained_policy:
                action = self.agent.choose_action(state.as_tuple(), actions, greedy=True)
            else:
                action = self._default_policy(state, actions)
            trace.append(action)
            state, reward, done, info = env.step(action)
        # Return the verifier's own result, including its real reason.
        return env.explorer_output, env.last_verification, trace

    @staticmethod
    def _default_policy(state, legal):
        if EXPLORE in legal and state.stage == "start":
            return EXPLORE
        if VERIFY in legal:
            return VERIFY
        if ACCEPT in legal:
            return ACCEPT
        if REJECT in legal:
            return REJECT
        return legal[0]