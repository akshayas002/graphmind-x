from src.agents.models import SubQuestion
from src.rl.actions import ACCEPT, EXPLORE, REJECT, VERIFY, legal_actions
from src.rl.state import START_STATE, RouterState, bucket_row_count

MAX_STEPS = 6
STEP_COST = -0.05


class RouterEnvironment:
    def __init__(self, explorer, verifier, sub_question: SubQuestion, schema: dict):
        self.explorer = explorer
        self.verifier = verifier
        self.sub_question = sub_question
        self.schema = schema
        self.state = START_STATE
        self.explorer_output = None
        self.last_verification = None  # the verifier's full result, including its reason
        self.steps_taken = 0

    def reset(self):
        self.state = START_STATE
        self.explorer_output = None
        self.last_verification = None
        self.steps_taken = 0
        return self.state

    def step(self, action: str):
        if action not in legal_actions(self.state):
            raise ValueError(f"Action {action} is not legal from state {self.state}")

        self.steps_taken += 1
        reward = STEP_COST
        done = False

        if action == EXPLORE:
            is_retry = self.state.stage == "after_explore"
            previous_cypher = self.explorer_output.cypher if is_retry and self.explorer_output else None
            kwargs = {}
            if is_retry and self.explorer_output is not None and self.explorer_output.error:
                kwargs["previous_error"] = self.explorer_output.error
            self.explorer_output = self.explorer.explore_once(
                self.sub_question, self.schema, previous_cypher=previous_cypher, **kwargs
            )
            had_error = self.explorer_output.error is not None
            row_bucket = "zero" if had_error else bucket_row_count(self.explorer_output.row_count)
            retries = self.state.retries_used + (1 if is_retry else 0)
            self.state = RouterState("after_explore", row_bucket, had_error, retries, "unknown")

        elif action == VERIFY:
            v = self.verifier.verify(self.sub_question, self.explorer_output)
            self.last_verification = v
            self.state = RouterState(
                "after_verify", self.state.row_bucket, self.state.had_error,
                self.state.retries_used, "pass" if v.passed else "fail",
            )

        elif action == ACCEPT:
            reward += 1.0
            done = True

        elif action == REJECT:
            reward += -1.0 if self.state.verified == "pass" else -0.3
            done = True

        if not done and self.steps_taken >= MAX_STEPS:
            reward += -1.0
            done = True

        return self.state, reward, done, {"explorer_output": self.explorer_output}