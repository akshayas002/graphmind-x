"""
Action space for the RL router, and the legal-actions function that
constrains which actions are valid from a given state.
"""
from src.rl.state import RouterState, MAX_RETRIES

EXPLORE = "EXPLORE"
VERIFY = "VERIFY"
ACCEPT = "ACCEPT"
REJECT = "REJECT"

ALL_ACTIONS = [EXPLORE, VERIFY, ACCEPT, REJECT]


def legal_actions(state: RouterState) -> list[str]:
    if state.stage == "start":
        return [EXPLORE]

    if state.stage == "after_explore":
        actions = [VERIFY, REJECT]
        if state.retries_used < MAX_RETRIES:
            actions.append(EXPLORE)
        return actions

    if state.stage == "after_verify":
        return [ACCEPT, REJECT]

    return []