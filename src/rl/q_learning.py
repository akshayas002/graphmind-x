"""
Tabular Q-learning agent. Tabular (not a neural Q-network) deliberately:
the state space is small and fully enumerable, so a table converges faster,
is exactly reproducible, needs no GPU, and is trivially inspectable.
"""
import ast
import json
import random
from collections import defaultdict
from pathlib import Path


class QLearningAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2, seed: int | None = None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict[tuple, dict[str, float]] = defaultdict(dict)
        self._rng = random.Random(seed)

    def get_q(self, state_key: tuple, action: str) -> float:
        return self.q_table[state_key].get(action, 0.0)

    def choose_action(self, state_key: tuple, legal_actions: list[str], greedy: bool = False) -> str:
        if not legal_actions:
            raise ValueError(f"No legal actions from state {state_key}")

        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.choice(legal_actions)

        q_values = [(a, self.get_q(state_key, a)) for a in legal_actions]
        max_q = max(q for _, q in q_values)
        best = [a for a, q in q_values if q == max_q]
        return self._rng.choice(best)

    def update(self, state_key, action, reward, next_state_key, next_legal_actions, done):
        current_q = self.get_q(state_key, action)
        if done or not next_legal_actions:
            target = reward
        else:
            next_max = max(self.get_q(next_state_key, a) for a in next_legal_actions)
            target = reward + self.gamma * next_max
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state_key][action] = new_q

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {repr(k): v for k, v in self.q_table.items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load(self, path: str | Path):
        with open(path) as f:
            data = json.load(f)
        self.q_table = defaultdict(dict)
        for key_repr, action_values in data.items():
            key = ast.literal_eval(key_repr)
            self.q_table[key] = action_values

    def policy_summary(self) -> dict[str, str]:
        summary = {}
        for state_key, action_values in self.q_table.items():
            if not action_values:
                continue
            best_action = max(action_values, key=action_values.get)
            summary[str(state_key)] = f"{best_action} (Q={action_values[best_action]:.3f})"
        return summary