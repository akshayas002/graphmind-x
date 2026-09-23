"""
Thin wrapper around a locally-running Ollama instance.

Kept deliberately minimal and injectable: every agent takes an
OllamaClient-shaped object in its constructor, so tests substitute a fake
one and never need a real Ollama server running. See tests/test_agents/
for examples.
"""
import json
import re

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src import config


class OllamaError(Exception):
    """Raised when Ollama is unreachable or returns something unusable."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.OLLAMA_MODEL,
        timeout: int = config.OLLAMA_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Single-turn generation. Low default temperature — we want
        consistent Cypher/JSON, not creative variation, for these agents.
        """
        try:
            resp = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError as e:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}. Is it running? "
                f"Try: ollama serve (and `ollama pull {self.model}` if you haven't)."
            ) from e
        except requests.HTTPError as e:
            raise OllamaError(f"Ollama returned an error: {e}") from e

        data = resp.json()
        text = data.get("response", "")
        if not text:
            raise OllamaError(f"Ollama returned an empty response. Raw payload: {data}")
        return text


def extract_json(raw_text: str) -> dict | None:
    """
    Best-effort extraction of a JSON object from an LLM response that may
    include markdown fences, leading/trailing prose, or minor formatting
    slop — local models are notably less reliable than hosted ones at
    emitting *only* JSON when asked to. Returns None (never raises) so
    callers can fall back to a deterministic default instead of crashing.
    """
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    candidates = []
    if fence_match:
        candidates.append(fence_match.group(1))

    # Fallback: grab the first {...} block by brace matching, in case there's
    # no fence at all (common with smaller local models).
    brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
