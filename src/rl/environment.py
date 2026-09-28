"""
Wraps the Graph Explorer + Verifier agents as an MDP for one sub-question,
so the Q-learning router can learn when to retry, verify, accept, or reject.
"""
from src.agents.models import SubQuestion
from src.rl.actions import ACCEPT, EXPLORE, REJECT, VERIFY, legal_actions
from src.rl.state import START_STATE, RouterState, bucket_row_count

MAX_STEPS = 6  # must comfortably exceed the worst-case legal path: initial EXPLORE +
               # MAX_RETRIES(=2) retries + VERIFY + ACCEPT/REJECT = 5 steps minimum.
               # Too low would force-terminate a fully-legal episode right before
               # it could reach its terminal action, corrupting the reward signal.
STEP_COST = -0.05


class RouterEnvironment:
    def __init__(self, explorer, verifier, sub_question: SubQuestion, schema: dict):
        self.explorer = explorer
        self.verifier = verifier
        self.sub_question = sub_question
        self.schema = schema
        self.state: RouterState = START_STATE
        self.explorer_output = None
        self.steps_taken = 0

    def reset(self) -> RouterState:
        self.state = START_STATE
        self.explorer_output = None
        self.steps_taken = 0
        return self.state

    def step(self, action: str) -> tuple[RouterState, float, bool, dict]:
        if action not in legal_actions(self.state):
            raise ValueError(f"Action {action} is not legal from state {self.state}")

        self.steps_taken += 1
        reward = STEP_COST
        done = False

        if action == EXPLORE:
            is_retry = self.state.stage == "after_explore"
            previous_cypher = self.explorer_output.cypher if is_retry and self.explorer_output else None
            self.explorer_output = self.explorer.explore_once(
                self.sub_question, self.schema, previous_cypher=previous_cypher
            )
            had_error = self.explorer_output.error is not None
            row_bucket = "zero" if had_error else bucket_row_count(self.explorer_output.row_count)
            retries = self.state.retries_used + (1 if is_retry else 0)
            self.state = RouterState(
                stage="after_explore", row_bucket=row_bucket, had_error=had_error,
                retries_used=retries, verified="unknown",
            )

        elif action == VERIFY:
            v = self.verifier.verify(self.sub_question, self.explorer_output)
            self.state = RouterState(
                stage="after_verify", row_bucket=self.state.row_bucket, had_error=self.state.had_error,
                retries_used=self.state.retries_used, verified="pass" if v.passed else "fail",
            )

        elif action == ACCEPT:
            reward += 1.0 if self.state.verified == "pass" else -1.0
            done = True

        elif action == REJECT:
            reward += -1.0 if self.state.verified == "pass" else -0.3
            done = True

        if not done and self.steps_taken >= MAX_STEPS:
            reward += -1.0
            done = True

        return self.state, reward, done, {"explorer_output": self.explorer_output}