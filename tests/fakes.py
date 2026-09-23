"""Shared fake/mock objects for agent tests — no real Ollama or Neo4j needed."""
from src.llm.ollama_client import OllamaError


class FakeLLM:
    """Generic fake LLM client. Pass a fixed response, or a router function
    for tests that need different responses per agent/prompt."""
    def __init__(self, response=None, raise_error=False, router=None):
        self.response = response
        self.raise_error = raise_error
        self.router = router  # optional: callable(prompt) -> str

    def generate(self, prompt, temperature=0.1):
        if self.raise_error:
            raise OllamaError("fake failure")
        if self.router:
            return self.router(prompt)
        return self.response


class FakeNeo4jClient:
    def __init__(self, records=None, raise_error=False, schema=None):
        self.records = records if records is not None else []
        self.raise_error = raise_error
        self.schema = schema or {"labels": [], "relationship_types": [], "label_properties": {}}
        self.last_query = None

    def run_read_query(self, cypher, parameters=None):
        self.last_query = cypher
        if self.raise_error:
            raise RuntimeError("fake db error")
        return self.records

    def get_schema_summary(self):
        return self.schema

    def close(self):
        pass
