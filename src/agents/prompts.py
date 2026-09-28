"""
All LLM prompt templates in one place.
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

Relationships that actually exist in the data (direction matters — use these exactly,
do not guess or invert direction/type):
{relationship_patterns}

Example properties per label: {label_properties}

Write ONE read-only Cypher query (MATCH ... RETURN ...) that answers this question:
"{question}"

Rules:
- Only use MATCH and RETURN. Never use CREATE, MERGE, DELETE, SET, REMOVE, DROP, or CALL.
- Only reference the labels and relationships listed above, with the exact direction shown.
- Never guess or invent an `id` value (e.g. from words in the question). IDs are not
  derived from titles. To find a specific patent by name, match on title text instead:
  WHERE toLower(p.title) CONTAINS toLower('some keyword')
- Prefer partial/case-insensitive text matching (toLower(...) CONTAINS toLower(...))
  over exact equality whenever matching against a name or title.
- Return only the Cypher query, wrapped in a ```cypher code block. No explanation.
"""

GRAPH_EXPLORER_RETRY_PROMPT = """Your previous Cypher query ran successfully but returned
ZERO results, which likely means it was too strict (e.g. exact match instead of partial
match, or an assumed id/property value that doesn't actually exist).

Previous query:
{previous_cypher}

Same graph schema as before:
Node labels: {labels}
Relationships (direction matters): {relationship_patterns}
Example properties per label: {label_properties}

Question: "{question}"

Write a BROADER read-only Cypher query that's more likely to find a match — use
toLower(...) CONTAINS toLower(...) for any text matching, and double-check relationship
direction against the list above. Same rules as before (read-only, only listed
labels/relationships, no invented id values).

Return only the Cypher query, wrapped in a ```cypher code block. No explanation.
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