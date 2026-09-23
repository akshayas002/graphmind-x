"""
All LLM prompt templates in one place — easier to tune against your actual
local model's quirks without hunting through agent logic.
"""

PLANNER_PROMPT = """You are a query planner for a patent knowledge graph system.
Break the user's question into 1-3 sub-questions that can each be answered by
querying a graph database. If the question is already simple, return it as a
single sub-question.

Respond with ONLY a JSON object in this exact shape, nothing else:
{{
  "reasoning": "brief explanation of your breakdown",
  "sub_questions": [
    {{"id": "sq1", "text": "...", "requires_graph": true}}
  ]
}}

User question: {question}
"""

GRAPH_EXPLORER_PROMPT = """You are a Cypher query generator for a Neo4j graph database.

The graph has these node labels: {labels}
Relationship types: {relationship_types}
Example properties per label: {label_properties}

Write ONE read-only Cypher query (MATCH ... RETURN ...) that answers this question:
"{question}"

Rules:
- Only use MATCH and RETURN. Never use CREATE, MERGE, DELETE, SET, REMOVE, DROP, or CALL.
- Only reference the labels, relationship types, and properties listed above.
- Return only the Cypher query, wrapped in a ```cypher code block. No explanation.
"""

VERIFIER_PROMPT = """You are checking whether graph query results plausibly answer a question.

Question: {question}
Number of results returned: {row_count}
Sample results: {sample_records}

Respond with ONLY a JSON object:
{{"passed": true or false, "reason": "one short sentence"}}
"""

SYNTHESIZER_PROMPT = """You are answering a user's question using facts retrieved from a patent knowledge graph.
Use ONLY the facts provided below — do not invent patents, names, or numbers not present here.

Original question: {question}

Retrieved facts:
{facts}

Write a clear, concise answer (2-4 sentences) grounded only in the facts above.
If the facts are insufficient to answer confidently, say so plainly.
"""
