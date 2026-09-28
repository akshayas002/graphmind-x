"""
Week 2/RL: Graph Explorer agent.

explore_once() is a SINGLE attempt — no internal retry — so that whichever
caller is in charge of retry decisions (the static orchestrator's fixed
one-retry policy, or the RL router's learned policy) is the only place that
decides. Nesting an automatic retry inside explore_once() itself would let
one RL "EXPLORE" action silently trigger two LLM calls behind the router's
back, corrupting its retries_used state and reward accounting — exactly the
kind of bug that's invisible until you look at training data. explore()
below is a thin, backward-compatible wrapper providing the original
fixed one-retry behavior for the static (non-RL) orchestrator.
"""
from src.agents.base_agent import BaseAgent
from src.agents.models import ExplorerOutput, SubQuestion
from src.agents.prompts import GRAPH_EXPLORER_PROMPT, GRAPH_EXPLORER_RETRY_PROMPT
from src.graph.cypher_validator import sanitize_and_validate
from src.graph.neo4j_client import Neo4jClient
from src import config


def _format_relationship_patterns(patterns: list[dict]) -> str:
    if not patterns:
        return "(none found — graph may be empty)"
    return "\n".join(f"  ({p['from']})-[:{p['type']}]->({p['to']})" for p in patterns)


class GraphExplorerAgent(BaseAgent):
    def __init__(self, llm_client, neo4j_client: Neo4jClient):
        super().__init__(llm_client)
        self.neo4j_client = neo4j_client

    def explore_once(
        self, sub_question: SubQuestion, schema: dict, previous_cypher: str | None = None
    ) -> ExplorerOutput:
        """
        A single Cypher-generation-and-execution attempt. Pass previous_cypher
        to get the "broaden your query" retry prompt instead of the initial
        prompt — the caller (static orchestrator or RL router) decides when
        that's appropriate, this method just executes one attempt either way.
        """
        labels_str = ", ".join(schema.get("labels", []))
        rel_patterns_str = _format_relationship_patterns(schema.get("relationship_patterns", []))
        label_properties = schema.get("label_properties", {})

        if previous_cypher is None:
            prompt = GRAPH_EXPLORER_PROMPT.format(
                labels=labels_str,
                relationship_patterns=rel_patterns_str,
                label_properties=label_properties,
                question=sub_question.text,
            )
        else:
            prompt = GRAPH_EXPLORER_RETRY_PROMPT.format(
                previous_cypher=previous_cypher,
                labels=labels_str,
                relationship_patterns=rel_patterns_str,
                label_properties=label_properties,
                question=sub_question.text,
            )

        return self._generate_and_run(sub_question, prompt)

    def explore(self, sub_question: SubQuestion, schema: dict) -> ExplorerOutput:
        """
        Backward-compatible wrapper for the static (non-RL) orchestrator:
        one attempt, and if it executes cleanly but finds zero rows, exactly
        one retry with the broadened prompt.
        """
        result = self.explore_once(sub_question, schema)

        if result.error is None and result.row_count == 0:
            retry_result = self.explore_once(sub_question, schema, previous_cypher=result.cypher)
            if retry_result.error is None and retry_result.row_count > 0:
                return retry_result

        return result

    def _generate_and_run(self, sub_question: SubQuestion, prompt: str) -> ExplorerOutput:
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