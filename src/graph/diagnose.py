"""
Prints exactly what the graph contains and what the agents are told about it,
so "why does it say it can't find anything?" has a definite answer.

Usage:
    python -m src.graph.diagnose
"""
THIN_DATA_THRESHOLD = 20


def _rows(client, cypher):
    try:
        return client.run_read_query(cypher), None
    except Exception as e:  # keep going: a partial report beats a crash
        return [], str(e)


def run_diagnostics(client) -> list[str]:
    out: list[str] = []

    # ---- what the agents are told ----
    try:
        schema = client.get_schema_summary()
    except Exception as e:
        return [f"Could not read the graph schema: {e}", "Is Neo4j running? (docker compose up -d)"]

    patterns = schema.get("relationship_patterns")
    out.append("== Schema the agents receive ==")
    out.append(f"labels: {schema.get('labels')}")
    if patterns is None:
        out.append("relationship_patterns: MISSING -> your src/graph/neo4j_client.py get_schema_summary() "
                   "is the OLD version. Replace it; direction fixing cannot work without it.")
    elif not patterns:
        out.append("relationship_patterns: EMPTY -> the graph has no relationships between nodes.")
    else:
        for p in patterns:
            out.append(f"  ({p['from']})-[:{p['type']}]->({p['to']})")

    # ---- what is actually stored ----
    node_rows, err1 = _rows(client, "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC")
    rel_rows, err2 = _rows(client, "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC")
    title_rows, err3 = _rows(client, "MATCH (p:Patent) RETURN p.id AS id, p.title AS title LIMIT 10")

    nodes = {r["label"]: r["n"] for r in node_rows}
    rels = {r["type"]: r["n"] for r in rel_rows}

    out.append("")
    out.append("== Nodes ==")
    out.extend(f"  {label}: {n}" for label, n in nodes.items()) if nodes else out.append(f"  (none) {err1 or ''}".rstrip())
    out.append("== Relationships ==")
    out.extend(f"  {t}: {n}" for t, n in rels.items()) if rels else out.append(f"  (none) {err2 or ''}".rstrip())
    out.append("== Sample patents ==")
    out.extend(f"  {r['id']}: {r['title']}" for r in title_rows) if title_rows else out.append(f"  (none) {err3 or ''}".rstrip())

    # ---- verdict ----
    out.append("")
    out.append("== Verdict ==")
    verdict: list[str] = []
    patents = nodes.get("Patent", 0)
    if patents == 0:
        verdict.append("No Patent nodes: nothing has been loaded. Run the fetch -> extract -> load steps.")
    elif patents < THIN_DATA_THRESHOLD:
        verdict.append(f"Only {patents} patent(s) loaded. Too thin for meaningful questions. Try: "
                       "python -m src.ingestion.epo_fetcher --probe \"battery\" \"lithium ion\" \"energy storage\"")
    if rels.get("INVENTED", 0) == 0:
        verdict.append("No INVENTED relationships: inventor questions CANNOT be answered from this graph. "
                       "EPO often returns empty applicant/inventor fields for Indian filings; that is a "
                       "data limitation, not a code bug.")
    if rels.get("OWNS", 0) == 0:
        verdict.append("No OWNS relationships: applicant/owner questions cannot be answered either.")
    if rels.get("CLASSIFIED_AS", 0) == 0:
        verdict.append("No CLASSIFIED_AS relationships: classification questions cannot be answered.")
    if rels.get("MENTIONS", 0) > 0:
        verdict.append("MENTIONS data exists, so questions about patent titles/topics (e.g. "
                       "\"What are the battery patents?\") can be answered.")
    out.extend(f"  - {v}" for v in verdict) if verdict else out.append("  Graph looks healthy.")
    return out


def main():
    from src.graph.neo4j_client import Neo4jClient

    client = Neo4jClient()
    try:
        print("\n".join(run_diagnostics(client)))
    finally:
        client.close()


if __name__ == "__main__":
    main()