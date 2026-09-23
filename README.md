# GraphMind-X — Phase 1 (Week 1): Data Foundation (India-scoped)

Goal of this phase: **real Indian patent data sitting in a real Neo4j graph,
with zero ML, zero agents, zero complexity.** Everything after this builds
on top of it, so don't skip ahead until you can run the full pipeline below
end-to-end and see nodes in Neo4j Browser.

## ⚠️ Two things that changed since earlier planning

1. **IP India has no public API.** The Indian Patent Office's only public
   system, InPASS, is a JS-heavy, CAPTCHA-protected web portal built for
   humans — no REST/JSON endpoint, no key you can register for. Scraping it
   directly is fragile and sits in a legal/ToS gray zone, so it's not a
   sound foundation for a 6-month data pipeline.

2. **The free, structured route to Indian patent data is the EPO's Open
   Patent Services (OPS) API.** It's official, free (4 GB/month), and
   includes Indian bibliographic + legal-status data via INPADOC (the EPO's
   worldwide patent-family database, which IP India reports into). The
   trade-off: it's strongest on bibliographic data (title, applicants,
   inventors, IPC codes, family links) and generally thinner on full
   specification text for India-origin filings — fine for building the
   graph's structure now; deeper text work is what the Explorer/Verifier
   agents (later weeks) are for.

## Setup (do this once)

1. **Get a free EPO OPS Consumer Key + Secret**
   - Register at https://developers.epo.org (free, "Non-paying" access tier)
   - Under "My Apps," create an app to get your Consumer Key + Consumer Secret

2. **Install Docker** (for Neo4j) if you don't have it — https://docs.docker.com/get-docker/

3. **Clone/copy this repo, then:**
   ```bash
   cd graphmind-x
   cp .env.example .env
   # edit .env: paste your EPO_OPS_CONSUMER_KEY and EPO_OPS_CONSUMER_SECRET
   ```

4. **Start Neo4j:**
   ```bash
   docker compose up -d
   # Neo4j Browser: http://localhost:7474  (login: neo4j / graphmindpass)
   ```

5. **Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Run the pipeline

```bash
# 1. Sanity-check the parsing logic with a fake record (no network needed)
python -m pytest tests/ -v

# 2. Pull real Indian patent data — set INGESTION_SAMPLE_SIZE=10 in .env
#    for this first run, so you can eyeball the raw JSON before a big pull
python -m src.ingestion.epo_fetcher

# 3. Extract entities → graph-ready nodes/edges
python -m src.extraction.entity_extractor data/raw/epo_IN_<whatever_was_printed>.json

# 4. Load into Neo4j
python -m src.graph.neo4j_client data/processed/epo_IN_<whatever>_graph.json
```

Then open http://localhost:7474 and run:
```cypher
MATCH (p:Patent)-[:CLASSIFIED_AS]->(c:IPCCode) RETURN p, c LIMIT 50
```
You should see real Indian patent applications connected to IPC
classification codes, inventors, and applicants.

**Before your first big pull:** the country filter uses `pn={country_code}`
(publication-number country prefix) in the CQL query, which is the commonly
documented pattern for restricting by country of publication. Cross-check a
handful of results against Espacenet's Advanced Search
(worldwide.espacenet.com) to confirm it's actually returning Indian
publications before you spend your monthly data quota on a big batch.

## What each file does

| File | Role |
|---|---|
| `src/config.py` | All env-driven config — no hardcoded secrets/paths anywhere else |
| `src/ingestion/epo_fetcher.py` | OAuth2 + CQL search against EPO OPS, saves raw JSON to `data/raw/` |
| `src/extraction/entity_extractor.py` | Parses raw JSON → `Patent`, `Inventor`, `Applicant`, `IPCCode`, `Concept` nodes + edges |
| `src/graph/neo4j_client.py` | Loads nodes/edges into Neo4j via idempotent `MERGE` |
| `tests/test_entity_extractor.py` | Fast offline sanity check — run this *before* burning API quota |

## Known limitations (intentional, for Week 1)

- **Concept extraction is naive** — noun chunks from the title only. OPS's
  search/biblio endpoint doesn't return full abstract/claims text for most
  filings; deeper full-text retrieval is a later-week concern once the
  Explorer/Verifier agents exist to make use of it.
- **No RL, no agents yet.** This phase is purely: fetch → extract → load.
- **IPC codes aren't linked to a formal IPC hierarchy** — stored as flat
  string nodes for now. A good Week 2 stretch goal if Week 1 goes smoothly.
- **OPS's JSON has a known single-item/multi-item quirk** (a field is a bare
  dict when there's one, a list when there's more) — `as_list()` in
  `entity_extractor.py` normalizes this, and the test suite specifically
  exercises the single-item case so it doesn't silently break later.

## Definition of done for Phase 1

- [ ] `docker compose up -d` runs Neo4j locally without errors
- [ ] `.env` has working `EPO_OPS_CONSUMER_KEY` / `EPO_OPS_CONSUMER_SECRET`
- [ ] `pytest tests/` passes
- [ ] A small test pull (~10 records) visibly returns Indian publications
- [ ] `epo_fetcher.py` pulls ≥100 real records for your chosen keyword
- [ ] `entity_extractor.py` produces a non-trivial node/edge count from them
- [ ] `neo4j_client.py` loads without errors and the summary counts look sane
- [ ] You can open Neo4j Browser and visually see the graph

Once all of these are checked, come back and we'll move to Week 2:
the Planner and Explorer agents that actually query this graph.

---

# Phase 2 & 3: Reasoning Pipeline (Planner → Explorer → Verifier → Synthesizer)

Goal: given a natural-language question, decompose it, generate and safely
execute Cypher against the graph, sanity-check the results, and produce a
grounded natural-language answer — all running on a **free local LLM via
Ollama**, zero API cost.

## Additional setup for Phase 2/3

1. **Install Ollama:** https://ollama.com/download
2. **Pull a model:**
   ```bash
   ollama pull llama3.1
   ```
   If your machine is CPU-only and llama3.1:8b feels slow, try a smaller
   model instead (edit `OLLAMA_MODEL` in `.env` to match):
   ```bash
   ollama pull qwen2.5:3b-instruct
   ```
3. **Start Ollama** (if it isn't already running as a service):
   ```bash
   ollama serve
   ```
4. Your `.env` already has `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and
   `OLLAMA_TIMEOUT_SECONDS` — defaults work for a standard local install.

## Run it

```bash
# Run the full pytest suite first — 43 tests, all offline/mocked, no
# Ollama or Neo4j required to run these:
python -m pytest tests/ -v

# Then, with Neo4j running and loaded (Phase 1) and Ollama running:
python -m src.main "Who invented the sulphide rich composite battery patent?"
```

You'll see the final answer plus a full step-by-step trace: what the
Planner decided, the exact Cypher the Explorer generated and ran, whether
the Verifier passed or failed each result, and how the Synthesizer used
that to compose the answer.

## What each new file does

| File | Role |
|---|---|
| `src/llm/ollama_client.py` | Thin wrapper around local Ollama; `extract_json()` defensively parses LLM output that may not be clean JSON |
| `src/graph/cypher_validator.py` | Safety gate — rejects any non-read-only or multi-statement Cypher before it can reach Neo4j |
| `src/agents/models.py` | Typed dataclasses (`SubQuestion`, `ExplorerOutput`, etc.) shared between agents |
| `src/agents/prompts.py` | All prompt templates in one place |
| `src/agents/base_agent.py` | Shared "call LLM, parse JSON, fall back safely" logic |
| `src/agents/planner_agent.py` | Decomposes a question into 1-3 graph-answerable sub-questions |
| `src/agents/graph_explorer_agent.py` | NL sub-question → validated Cypher → Neo4j results |
| `src/agents/verifier_agent.py` | Structural + LLM sanity-check on Explorer results before they're trusted |
| `src/agents/synthesizer_agent.py` | Composes the final natural-language answer from verified results |
| `src/orchestrator.py` | Wires all four agents into one sequential pipeline with a full trace |
| `src/main.py` | CLI: `python -m src.main "your question"` |
| `src/graph/neo4j_client.py` *(extended)* | Added `run_read_query()` and `get_schema_summary()` |
| `tests/fakes.py` | Shared `FakeLLM` / `FakeNeo4jClient` test doubles |
| `tests/test_cypher_validator.py` | Adversarial safety tests (destructive queries, injection attempts) |
| `tests/test_ollama_client.py` | JSON-extraction robustness tests |
| `tests/test_planner_agent.py` | Planner happy-path + every fallback path |
| `tests/test_graph_explorer_agent.py` | Explorer happy-path + destructive-query rejection + failure modes |
| `tests/test_verifier_agent.py` | Structural auto-fail cases + LLM relevance check + fail-open behavior |
| `tests/test_synthesizer_agent.py` | Clean synthesis + honest "not found" + templated fallback |
| `tests/test_orchestrator.py` | Full end-to-end happy path + graceful degradation when Neo4j is down |

## Design decisions worth knowing

- **Everything fails soft, never crashes.** LLM unreachable, malformed JSON,
  rejected Cypher, Neo4j down, empty results — every one of these produces
  a typed result with an error/fallback flag set, not an exception that
  kills the pipeline. This matters especially with a local model, which
  will be flakier about following instructions than a hosted one.
- **The Cypher validator is defense-in-depth, not the only safeguard.**
  For any real deployment, also run Neo4j with a read-only DB user. The
  validator protects against a model generating a destructive query; a
  read-only DB user protects against a validator bug.
- **The orchestrator is intentionally static/sequential** (always Planner →
  Explorer → Verifier → Synthesizer, in that order). This is the exact
  interface the RL router (later weeks) will replace — instead of always
  calling all four in fixed order, the RL policy will learn which agent to
  call next. Keeping this interface stable now avoids a rewrite later.
- **The Verifier fails open on LLM failure, closed on structural failure.**
  A local-model hiccup shouldn't block an otherwise-good result, but a
  genuine Neo4j error or empty result set should always be caught.

## Definition of done for Phase 2/3

- [ ] `ollama pull llama3.1` (or your chosen model) completed
- [ ] `ollama serve` running (or Ollama's background service is up)
- [ ] `python -m pytest tests/ -v` — all 43 tests pass
- [ ] `python -m src.main "<a real question about your loaded data>"` returns
      a sensible answer with a trace showing all four agents ran
- [ ] Try a question you know the graph *can't* answer — confirm you get an
      honest "couldn't find" answer, not a hallucinated one
- [ ] Try to get the Explorer to generate something destructive by asking a
      leading question (e.g. "delete all patents by X") — confirm the
      validator rejects it and nothing in Neo4j changes

Once these pass, come back and we'll talk about Week 4+: the RL router
that learns agent orchestration instead of the current fixed sequence.
