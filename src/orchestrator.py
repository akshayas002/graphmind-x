"""
Orchestrator — Planner -> Explorer -> Verifier -> Synthesizer.

Two routing modes, selected at construction time:
  use_rl_router=False (default): the original static sequence.
  use_rl_router=True: each sub-question is driven by the trained RL router.

If every sub-question the Planner produced is out of the graph's domain
(requires_graph=False for all of them), the orchestrator returns a direct,
honest "this isn't something I can look up in the patent graph" answer
without ever calling the Explorer, Verifier, or Synthesizer LLM — there is
nothing for any of them to usefully do with an out-of-domain question, so
skipping them saves time and avoids the generic "couldn't find verified
information" message, which wrongly implies a data gap rather than a
question that was never answerable from this graph in the first place.
"""
from src.agents.graph_explorer_agent import GraphExplorerAgent
from src.agents.models import FinalAnswer, TraceStep
from src.agents.planner_agent import PlannerAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.agents.verifier_agent import VerifierAgent
from src.graph.neo4j_client import Neo4jClient
from src.llm.ollama_client import OllamaClient
from src.rl.router import RLRouter
from src import config

OUT_OF_SCOPE_MESSAGE = (
    "This doesn't look like a question about the patent data in the graph. "
    "I can answer questions about patents, their inventors, applicants, "
    "classification codes, and concepts mentioned in their titles — try "
    "rephrasing around one of those."
)


class Orchestrator:
    def __init__(self, llm_client=None, neo4j_client=None, use_rl_router=False, q_table_path=None):
        llm_client = llm_client or OllamaClient()
        neo4j_client = neo4j_client or Neo4jClient()
        self.neo4j_client = neo4j_client
        self.planner = PlannerAgent(llm_client)
        self.explorer = GraphExplorerAgent(llm_client, neo4j_client)
        self.verifier = VerifierAgent(llm_client)
        self.synthesizer = SynthesizerAgent(llm_client)
        self.use_rl_router = use_rl_router
        self.rl_router = None
        if use_rl_router:
            default_q_path = str(config.ROOT_DIR / "data" / "rl" / "q_table.json")
            self.rl_router = RLRouter(self.explorer, self.verifier, q_table_path=q_table_path or default_q_path)

    def answer(self, question: str) -> FinalAnswer:
        trace: list[TraceStep] = []

        planner_output = self.planner.plan(question)
        trace.append(
            TraceStep(
                agent="Planner",
                description=f"Decomposed into {len(planner_output.sub_questions)} sub-question(s)."
                             + (" (fallback used)" if planner_output.used_fallback else ""),
                detail=planner_output.reasoning,
            )
        )

        graph_sub_questions = [sq for sq in planner_output.sub_questions if sq.requires_graph]
        if not graph_sub_questions:
            trace.append(
                TraceStep(
                    agent="Orchestrator",
                    description="No sub-question needs the graph — answering directly without querying it.",
                )
            )
            return FinalAnswer(answer=OUT_OF_SCOPE_MESSAGE, trace=trace, used_fallback=False)

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

        explorer_outputs = []
        verifications = []
        for sq in planner_output.sub_questions:
            if not sq.requires_graph:
                trace.append(TraceStep(agent="Explorer", description=f"Skipped '{sq.text}' (no graph lookup needed)."))
                continue

            if self.use_rl_router:
                eo, v, action_trace = self.rl_router.route(sq, schema)
                label = "learned" if self.rl_router.has_trained_policy else "default (untrained)"
                trace.append(TraceStep(agent="RLRouter", description=f"Sub-question '{sq.text}': {label} policy took actions {action_trace}."))
            else:
                eo = self.explorer.explore(sq, schema)
                v = self.verifier.verify(sq, eo)

            explorer_outputs.append(eo)
            trace.append(
                TraceStep(
                    agent="Explorer",
                    description=f"Sub-question '{sq.text}': "
                                 + (f"error — {eo.error}" if eo.error else f"{eo.row_count} row(s) returned."),
                    detail=eo.cypher or "",
                )
            )

            if v is not None:
                verifications.append(v)
                trace.append(TraceStep(agent="Verifier", description=f"Sub-question '{sq.text}': {'PASSED' if v.passed else 'FAILED'} — {v.reason}"))

        final = self.synthesizer.synthesize(question, planner_output.sub_questions, explorer_outputs, verifications)
        trace.append(TraceStep(agent="Synthesizer", description="Composed final answer." + (" (fallback used)" if final.used_fallback else "")))
        return FinalAnswer(answer=final.answer, trace=trace, used_fallback=final.used_fallback)

    def close(self):
        self.neo4j_client.close()