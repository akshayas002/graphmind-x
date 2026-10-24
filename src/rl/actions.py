from src.rl.state import RouterState, MAX_RETRIES

EXPLORE, VERIFY, ACCEPT, REJECT = "EXPLORE", "VERIFY", "ACCEPT", "REJECT"


def legal_actions(state: RouterState) -> list[str]:
    if state.stage == "start":
        return [EXPLORE]
    if state.stage == "after_explore":
        actions = [VERIFY, REJECT]
        if state.retries_used < MAX_RETRIES:
            actions.append(EXPLORE)
        return actions
    if state.stage == "after_verify":
        # ACCEPT is only legal when verification passed. Accepting a failed result
        # is a safety rule, not something to learn, so it is removed from the set.
        return [ACCEPT, REJECT] if state.verified == "pass" else [REJECT]
    return []