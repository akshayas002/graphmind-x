"""
All LLM prompt templates in one place.
"""

PLANNER_PROMPT = """You are a query planner for a patent knowledge graph system.
The graph ONLY contains data about patents: their titles, inventors, applicants
(owning companies), classification codes, and concepts mentioned in their titles.
It contains NOTHING about anything else — not about this software system itself,
not general knowledge, not conversation or "trace" history, nothing outside patents.

Break the user's question into 1-3 sub-questions that can each be answered by
querying the patent graph. If the question is already simple, return it as a
single sub-question.

If the question is NOT about patents, inventors, applicants, or patent
classifications/concepts — for example a question about this system itself,
about "traces" or "optimal paths", about general knowledge, or small talk —
set "requires_graph": false for that sub-question instead of inventing a
patent-related interpretation of it. Do not guess what the user "really meant"
if the question has no patent-domain content at all.

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
- Do NOT invent a relationship or an extra node just to filter or describe something.
  If the question is only about one kind of thing (e.g. listing or filtering patents by a
  word in the title), query that single label directly with a WHERE clause. No
  relationship is needed at all for a question like that. For example, "what patents
  mention battery" is simply:
  MATCH (p:Patent) WHERE toLower(p.title) CONTAINS toLower('battery') RETURN p.title
  Only add a MATCH for a relationship when the question explicitly connects two
  different kinds of things (e.g. an inventor AND a patent).
- If the question asks "how many" or for a total COUNT, RETURN ONLY the count —
  nothing else. Do not also return a non-aggregated field like p.title alongside
  count(*): Cypher silently groups by any non-aggregated field you include, which
  turns a total count into a per-group count and gives a misleading small number.
  Correct: MATCH (p:Patent)-[:MENTIONS]->(c:Concept)
           WHERE toLower(c.name) CONTAINS toLower('lithium ion battery')
           RETURN count(DISTINCT p) AS patentCount
  Wrong (silently groups by title, not a real total):
           RETURN p.title AS title, count(*) AS patentCount
- Never guess or invent an `id` value (e.g. from words in the question). IDs are not
  derived from titles. To find a specific patent by name, match on title text instead:
  WHERE toLower(p.title) CONTAINS toLower('some keyword')
- Prefer partial/case-insensitive text matching (toLower(...) CONTAINS toLower(...))
  over exact equality whenever matching against a name or title.
- Return only the Cypher query, wrapped in a ```cypher code block. No explanation.
"""

GRAPH_EXPLORER_RETRY_PROMPT = """Your previous Cypher query did not work.

Previous query:
{previous_cypher}

What went wrong: {failure_reason}

Same graph schema as before:
Node labels: {labels}
Relationships (direction matters): {relationship_patterns}
Example properties per label: {label_properties}

Question: "{question}"

Write a corrected read-only Cypher query. Use ONLY the labels and relationships listed
above, with the exact direction shown. If the failure mentions a relationship or label
that doesn't exist, the simplest fix is usually to remove it: query a single label
directly with a WHERE clause instead of inventing a connection. If this is a "how many"
question, RETURN only the count (e.g. count(DISTINCT p)) and nothing else — mixing in a
non-aggregated field like a title silently changes a total into a per-group count. Use
toLower(...) CONTAINS toLower(...) for any text matching, and never invent id values.
Same rules as before (read-only).

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