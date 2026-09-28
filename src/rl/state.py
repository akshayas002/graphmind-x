"""
State representation for the RL router.

Deliberately small and hand-crafted rather than raw text/embeddings: the
router's job is a sequential control decision (retry? verify? accept?
reject?) about ONE sub-question's progress through the pipeline, not
natural-language understanding — that's already the Planner/Explorer/
Verifier's job. A small discrete state space is exactly what makes tabular
Q-learning appropriate here instead of needing a neural policy network.
"""
from dataclasses import dataclass

MAX_RETRIES = 2


@dataclass(frozen=True)
class RouterState:
    stage: str        # "start" | "after_explore" | "after_verify"
    row_bucket: str    # "none" | "zero" | "few" | "many"
    had_error: bool
    retries_used: int  # 0..MAX_RETRIES
    verified: str      # "unknown" | "pass" | "fail"

    def as_tuple(self) -> tuple:
        """Hashable key for the Q-table."""
        return (self.stage, self.row_bucket, self.had_error, self.retries_used, self.verified)


def bucket_row_count(n: int) -> str:
    if n == 0:
        return "zero"
    if n <= 5:
        return "few"
    return "many"


START_STATE = RouterState(stage="start", row_bucket="none", had_error=False, retries_used=0, verified="unknown")