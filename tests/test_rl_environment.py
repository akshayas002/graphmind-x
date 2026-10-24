from src.agents.models import ExplorerOutput, SubQuestion, VerificationResult
from src.rl.actions import ACCEPT, EXPLORE, REJECT, VERIFY
from src.rl.environment import RouterEnvironment

SQ = SubQuestion(id="sq1", text="q")


class _Exp:
    def __init__(self, outs):
        self.outs, self.calls, self.prev = outs, 0, []

    def explore_once(self, sq, schema, previous_cypher=None, previous_error=None):
        self.prev.append((previous_cypher, previous_error))
        out = self.outs[min(self.calls, len(self.outs) - 1)]
        self.calls += 1
        return out


class _Ver:
    def __init__(self, passed):
        self.passed = passed

    def verify(self, sq, eo):
        return VerificationResult("sq1", self.passed, "fake")


def test_explore_success_transitions():
    env = RouterEnvironment(_Exp([ExplorerOutput("sq1", "c", row_count=3)]), _Ver(True), SQ, {})
    env.reset()
    s, r, d, _ = env.step(EXPLORE)
    assert s.stage == "after_explore" and s.row_bucket == "few" and not d


def test_retry_after_error_passes_failure_reason():
    exp = _Exp([ExplorerOutput("sq1", "MATCH (k:Kind) RETURN k", error="Label 'Kind' missing"),
                ExplorerOutput("sq1", "MATCH (p) RETURN p", row_count=4)])
    env = RouterEnvironment(exp, _Ver(True), SQ, {})
    env.reset()
    env.step(EXPLORE)
    env.step(EXPLORE)
    assert exp.prev[0] == (None, None)
    assert exp.prev[1][1] == "Label 'Kind' missing"


def test_verify_then_accept_on_pass_is_positive():
    env = RouterEnvironment(_Exp([ExplorerOutput("sq1", "c", row_count=2)]), _Ver(True), SQ, {})
    env.reset()
    env.step(EXPLORE)
    env.step(VERIFY)
    _, r, d, _ = env.step(ACCEPT)
    assert d and r > 0.9


def test_reject_after_failed_verify_is_mild():
    env = RouterEnvironment(_Exp([ExplorerOutput("sq1", "c", row_count=2)]), _Ver(False), SQ, {})
    env.reset()
    env.step(EXPLORE)
    env.step(VERIFY)
    _, r, d, _ = env.step(REJECT)
    assert d and -0.5 < r < 0