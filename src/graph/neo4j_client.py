"""
Week 1, Step: load extracted nodes/edges into Neo4j.
Everything uses MERGE (not CREATE), so re-running this is safe and idempotent.

Usage:
    python -m src.graph.neo4j_client data/processed/..._graph.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm

from src import config

logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


class Neo4jClient:
    def __init__(self, uri=config.NEO4J_URI, user=config.NEO4J_USER, password=config.NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def verify_connectivity(self):
        self.driver.verify_connectivity()

    def ensure_constraints(self):
        labels = ["Patent", "Inventor", "Applicant", "ClassificationCode", "Concept"]
        with self.driver.session() as session:
            for label in labels:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

    def load_nodes(self, nodes: list[dict]):
        by_label: dict[str, list[dict]] = {}
        for n in nodes:
            by_label.setdefault(n["label"], []).append(n)
        with self.driver.session() as session:
            for label, group in by_label.items():
                session.run(
                    f"""
                    UNWIND $batch AS node
                    MERGE (n:{label} {{id: node.id}})
                    SET n += node.properties
                    """,
                    batch=group,
                )

    def load_edges(self, edges: list[dict]):
        by_shape: dict[tuple[str, str, str], list[dict]] = {}
        for e in edges:
            key = (e["start_label"], e["type"], e["end_label"])
            by_shape.setdefault(key, []).append(e)
        with self.driver.session() as session:
            for (start_label, rel_type, end_label), group in tqdm(by_shape.items(), desc="Loading edges"):
                session.run(
                    f"""
                    UNWIND $batch AS edge
                    MATCH (a:{start_label} {{id: edge.start_id}})
                    MATCH (b:{end_label} {{id: edge.end_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    """,
                    batch=group,
                )

    def summary(self):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC"
            )
            return [(r["label"], r["count"]) for r in result]

    def run_read_query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """
        Execute an already-validated read-only Cypher query and return records
        as plain dicts. Callers (Graph Explorer agent) are responsible for
        running the query through src.graph.cypher_validator first.
        """
        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    def get_schema_summary(self) -> dict:
        """
        Introspect the live graph so agents can ground Cypher generation in
        what's actually in the database, rather than a hardcoded assumption
        that can drift out of sync. This is the piece that must be present
        for src/graph/cypher_schema.py to flip reversed relationship arrows
        and catch invented/nonexistent relationships — without
        relationship_patterns here, that whole safety layer is blind.
        """
        with self.driver.session() as session:
            labels = [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label")]
            rel_types = [
                r["relationshipType"]
                for r in session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            ]

            label_properties: dict[str, list[str]] = {}
            for label in labels:
                try:
                    rows = session.run(f"MATCH (n:{label}) RETURN n LIMIT 5")
                    keys: set[str] = set()
                    for row in rows:
                        keys.update(row["n"].keys())
                    label_properties[label] = sorted(keys)
                except Exception:
                    label_properties[label] = []

            relationship_patterns: list[dict] = []
            try:
                rows = session.run(
                    """
                    MATCH (a)-[r]->(b)
                    RETURN DISTINCT labels(a)[0] AS from_label, type(r) AS rel_type, labels(b)[0] AS to_label
                    LIMIT 50
                    """
                )
                relationship_patterns = [
                    {"from": r["from_label"], "type": r["rel_type"], "to": r["to_label"]} for r in rows
                ]
            except Exception:
                pass

        return {
            "labels": labels,
            "relationship_types": rel_types,
            "relationship_patterns": relationship_patterns,
            "label_properties": label_properties,
        }


def main():
    parser = argparse.ArgumentParser(description="Load extracted graph JSON into Neo4j.")
    parser.add_argument("input_file", help="Path to a *_graph.json file from entity_extractor.py")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    if not in_path.exists():
        print(f"File not found: {in_path}")
        sys.exit(1)

    with open(in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    client = Neo4jClient()
    try:
        print("Connecting to Neo4j...")
        client.verify_connectivity()
    except Exception as e:
        print(f"Could not connect to Neo4j at {config.NEO4J_URI}: {e}")
        sys.exit(1)

    print("Ensuring uniqueness constraints...")
    client.ensure_constraints()
    print(f"Loading {len(payload['nodes'])} nodes...")
    client.load_nodes(payload["nodes"])
    print(f"Loading {len(payload['edges'])} edges...")
    client.load_edges(payload["edges"])

    print("\nGraph summary:")
    for label, count in client.summary():
        print(f"  {label}: {count}")

    client.close()
    print("\nDone. Open http://localhost:7474 (Neo4j Browser) to explore the graph visually.")


if __name__ == "__main__":
    main()