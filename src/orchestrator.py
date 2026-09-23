"""
Week 3: Orchestrator.

Sequential pipeline for now (Planner -> Explorer -> Verifier -> Synthesizer)
— static, not learned. This deliberately mirrors the shape the RL router
will eventually replace: the RL agent's job (later weeks) will be choosing
which agent to call next and in what order; that decision here is just
"always all four, in order." Keeping the interface stable now means the
RL router slots in later without reshaping everything downstream of it.

Every step is wrapped so a single sub-question's failure doesn't take down
the others — the trace records what happened at each step either way.
"""
from src.agents.graph_explorer_agent import GraphExplorerAgent
from src.agents.models import FinalAnswer, TraceStep
from src.agents.planner_agent import PlannerAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.agents.verifier_agent import VerifierAgent
from src.graph.neo4j_client import Neo4jClient
from src.llm.ollama_client import OllamaClient


class Orchestrator:
    def __init__(self, llm_client: OllamaClient | None = None, neo4j_client: Neo4jClient | None = None):
        llm_client = llm_client or OllamaClient()
        neo4j_client = neo4j_client or Neo4jClient()

        self.neo4j_client = neo4j_client
        self.planner = PlannerAgent(llm_client)
        self.explorer = GraphExplorerAgent(llm_client, neo4j_client)
        self.verifier = VerifierAgent(llm_client)
        self.synthesizer = SynthesizerAgent(llm_client)

    def answer(self, question: str) -> FinalAnswer:
        trace: list[TraceStep] = []

        # --- Plan ---
        planner_output = self.planner.plan(question)
        trace.append(
            TraceStep(
                agent="Planner",
                description=f"Decomposed into {len(planner_output.sub_questions)} sub-question(s)."
                             + (" (fallback used)" if planner_output.used_fallback else ""),
                detail=planner_output.reasoning,
            )
        )

        # --- Fetch schema once, reuse across all sub-questions this turn ---
        try:
            schema = self.neo4j_client.get_schema_summary()
        except Exception as e:
            trace.append(TraceStep(agent="Orchestrator", description="Could not fetch graph schema.", detail=str(e)))
            return FinalAnswer(
                answer="I couldn't connect to the knowledge graph to answer this. "
                       "Is Neo4j running? (docker compose up -d)",
                trace=trace,
                used_fallback=True,
            )

        # --- Explore + Verify each sub-question ---
        explorer_outputs = []
        verifications = []
        for sq in planner_output.sub_questions:
            if not sq.requires_graph:
                trace.append(TraceStep(agent="Explorer", description=f"Skipped '{sq.text}' (no graph lookup needed)."))
                continue

            eo = self.explorer.explore(sq, schema)
            explorer_outputs.append(eo)
            trace.append(
                TraceStep(
                    agent="Explorer",
                    description=f"Sub-question '{sq.text}': "
                                 + (f"error — {eo.error}" if eo.error else f"{eo.row_count} row(s) returned."),
                    detail=eo.cypher or "",
                )
            )

            v = self.verifier.verify(sq, eo)
            verifications.append(v)
            trace.append(
                TraceStep(
                    agent="Verifier",
                    description=f"Sub-question '{sq.text}': {'PASSED' if v.passed else 'FAILED'} — {v.reason}",
                )
            )

        # --- Synthesize ---
        final = self.synthesizer.synthesize(
            original_question=question,
            sub_questions=planner_output.sub_questions,
            explorer_outputs=explorer_outputs,
            verifications=verifications,
        )
        trace.append(
            TraceStep(
                agent="Synthesizer",
                description="Composed final answer." + (" (fallback used)" if final.used_fallback else ""),
            )
        )

        return FinalAnswer(answer=final.answer, trace=trace, used_fallback=final.used_fallback)

    def close(self):
        self.neo4j_client.close()
