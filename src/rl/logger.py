"""Logs every transition to JSONL — this IS the "logged pipeline runs" the Q-table trains from."""
import json
from datetime import datetime, timezone
from pathlib import Path

from src.rl.state import RouterState


class TransitionLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, episode: int, step: int, state: RouterState, action: str, reward: float,
             next_state: RouterState, done: bool):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode": episode, "step": step,
            "state": list(state.as_tuple()), "action": action, "reward": reward,
            "next_state": list(next_state.as_tuple()), "done": done,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records