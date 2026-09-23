"""
Week 2: Graph Explorer agent.

Turns a sub-question into a Cypher query (grounded in the live graph schema,
not a hardcoded assumption), runs it through the safety validator, and
executes it against Neo4j. Every failure mode — LLM unreachable, unparseable
output, rejected Cypher, Neo4j execution error — produces an ExplorerOutput
with `error` set rather than raising, so the orchestrator never crashes
partway through a multi-sub-question plan.
"""
from src.agents.base_agent import BaseAgent
from src.agents.models import ExplorerOutput, SubQuestion
from src.agents.prompts import GRAPH_EXPLORER_PROMPT
from src.graph.cypher_validator import sanitize_and_validate
from src.graph.neo4j_client import Neo4jClient
from src import config


class GraphExplorerAgent(BaseAgent):
    def __init__(self, llm_client, neo4j_client: Neo4jClient):
        super().__init__(llm_client)
        self.neo4j_client = neo4j_client

    def explore(self, sub_question: SubQuestion, schema: dict) -> ExplorerOutput:
        prompt = GRAPH_EXPLORER_PROMPT.format(
            labels=", ".join(schema.get("labels", [])),
            relationship_types=", ".join(schema.get("relationship_types", [])),
            label_properties=schema.get("label_properties", {}),
            question=sub_question.text,
        )

        raw = self._call_llm(prompt)
        if raw is None:
            return ExplorerOutput(
                sub_question_id=sub_question.id,
                cypher=None,
                error="LLM was unreachable — could not generate a Cypher query.",
            )

        safe_cypher, reject_reason = sanitize_and_validate(raw, default_limit=config.CYPHER_RESULT_LIMIT)
        if safe_cypher is None:
            return ExplorerOutput(
                sub_question_id=sub_question.id,
                cypher=raw.strip(),
                error=f"Generated Cypher was rejected: {reject_reason}",
            )

        try:
            records = self.neo4j_client.run_read_query(safe_cypher)
        except Exception as e:
            return ExplorerOutput(
                sub_question_id=sub_question.id,
                cypher=safe_cypher,
                error=f"Neo4j execution failed: {e}",
            )

        return ExplorerOutput(
            sub_question_id=sub_question.id,
            cypher=safe_cypher,
            records=records,
            row_count=len(records),
        )
