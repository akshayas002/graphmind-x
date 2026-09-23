"""
Base class every agent inherits from. Centralizes the "call the LLM, try to
parse JSON out of it, fall back to something safe if that fails" pattern so
each agent doesn't reimplement (and potentially get wrong) its own version.
"""
from src.llm.ollama_client import OllamaClient, OllamaError, extract_json


class BaseAgent:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str | None:
        """Returns None (never raises) on any LLM failure, so callers always
        have a clean path to a deterministic fallback."""
        try:
            return self.llm_client.generate(prompt, temperature=temperature)
        except OllamaError:
            return None

    def _call_llm_for_json(self, prompt: str, temperature: float = 0.1) -> dict | None:
        """Returns None if the LLM call failed OR the response wasn't
        parseable JSON — callers treat both cases identically (fall back)."""
        raw = self._call_llm(prompt, temperature=temperature)
        if raw is None:
            return None
        return extract_json(raw)
