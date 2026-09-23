"""
Shared data contracts between agents. Using dataclasses instead of raw
dicts here specifically because this pipeline has already been bitten once
(Week 1's field-name mismatch against real EPO data) by assuming a shape
that turned out to be wrong — typed models make that class of bug loud
and immediate (AttributeError at the call site) instead of a silent
`.get()` returning None three layers downstream.
"""
from dataclasses import dataclass, field


@dataclass
class SubQuestion:
    id: str
    text: str
    requires_graph: bool = True


@dataclass
class PlannerOutput:
    sub_questions: list[SubQuestion]
    reasoning: str = ""
    used_fallback: bool = False  # True if the LLM output was unusable and we fell back


@dataclass
class ExplorerOutput:
    sub_question_id: str
    cypher: str | None
    records: list[dict] = field(default_factory=list)
    row_count: int = 0
    error: str | None = None  # set if Cypher was rejected or execution failed


@dataclass
class VerificationResult:
    sub_question_id: str
    passed: bool
    reason: str


@dataclass
class TraceStep:
    agent: str
    description: str
    detail: str = ""


@dataclass
class FinalAnswer:
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    used_fallback: bool = False
