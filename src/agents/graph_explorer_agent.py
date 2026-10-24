"""
Week 2/RL: Graph Explorer agent.

explore_once() is a SINGLE attempt — no internal retry — so that whichever
caller owns retry decisions (the static orchestrator's fixed one-retry policy,
or the RL router's learned policy) is the only place that decides.

Every generated query goes through two deterministic gates before it can reach
Neo4j: the read-only safety validator, then a schema check that auto-flips
reversed relationship arrows and rejects labels/relationships that don't
exist (see src/graph/cypher_schema.py). A schema rejection costs no database
call and its message is fed back to the model on retry.
"""
from src.agents.base_agent import BaseAgent
from src.agents.models import ExplorerOutput, SubQuestion
from src.agents.prompts import GRAPH_EXPLORER_PROMPT, GRAPH_EXPLORER_RETRY_PROMPT
from src.graph.cypher_schema import check_and_fix
from src.graph.cypher_validator import sanitize_and_validate
from src.graph.neo4j_client import Neo4jClient
from src import config

ZERO_ROWS_REASON = (
    "It ran successfully but returned ZERO results, which likely means it was too strict "
    "(e.g. exact match instead of partial match, an assumed id/property value that doesn't "
    "exist, or a relationship written in the wrong direction)."
)


def _format_relationship_patterns(patterns: list[dict]) -> str:
    if not patterns:
        return "(none found — graph may be empty)"
    return "\n".join(f"  ({p['from']})-[:{p['type']}]->({p['to']})" for p in patterns)


class GraphExplorerAgent(BaseAgent):
    def __init__(self, llm_client, neo4j_client: Neo4jClient):
        super().__init__(llm_client)
        self.neo4j_client = neo4j_client

    def explore_once(
        self,
        sub_question: SubQuestion,
        schema: dict,
        previous_cypher: str | None = None,
        previous_error: str | None = None,
    ) -> ExplorerOutput:
        """
        One Cypher-generation-and-execution attempt. Pass previous_cypher (and
        optionally previous_error) to use the corrective retry prompt; the
        caller decides when that's appropriate.
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
                failure_reason=previous_error or ZERO_ROWS_REASON,
                labels=labels_str,
                relationship_patterns=rel_patterns_str,
                label_properties=label_properties,
                question=sub_question.text,
            )

        return self._generate_and_run(sub_question, prompt, schema)

    def explore(self, sub_question: SubQuestion, schema: dict) -> ExplorerOutput:
        """
        Backward-compatible wrapper for the static (non-RL) orchestrator: one
        attempt, and if it produced no rows (zero results, schema rejection, or
        a Neo4j error — anything except the LLM being unreachable), exactly one
        corrective retry.
        """
        result = self.explore_once(sub_question, schema)

        if result.row_count == 0 and result.cypher is not None:
            retry_result = self.explore_once(
                sub_question, schema, previous_cypher=result.cypher, previous_error=result.error
            )
            if retry_result.error is None and retry_result.row_count > 0:
                return retry_result

        return result

    def _generate_and_run(self, sub_question: SubQuestion, prompt: str, schema: dict) -> ExplorerOutput:
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

        schema_result = check_and_fix(safe_cypher, schema)
        if schema_result.error is not None:
            return ExplorerOutput(
                sub_question_id=sub_question.id,
                cypher=safe_cypher,
                error=f"Generated Cypher doesn't match the graph schema: {schema_result.error}",
            )
        safe_cypher = schema_result.cypher  # may have had reversed arrows flipped

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