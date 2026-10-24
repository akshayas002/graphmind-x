from src.graph.diagnose import run_diagnostics


class FakeClient:
    def __init__(self, schema, nodes, rels, titles, fail_schema=False):
        self.schema, self.nodes, self.rels, self.titles, self.fail_schema = schema, nodes, rels, titles, fail_schema

    def get_schema_summary(self):
        if self.fail_schema:
            raise ConnectionError("down")
        return self.schema

    def run_read_query(self, cypher, parameters=None):
        if "labels(n)[0]" in cypher:
            return [{"label": k, "n": v} for k, v in self.nodes.items()]
        if "type(r)" in cypher:
            return [{"type": k, "n": v} for k, v in self.rels.items()]
        if "MATCH (p:Patent)" in cypher:
            return [{"id": i, "title": t} for i, t in self.titles]
        return []


SCHEMA = {"labels": ["Patent", "Concept"], "relationship_patterns": [{"from": "Patent", "type": "MENTIONS", "to": "Concept"}]}


def test_reports_missing_inventor_data_and_thin_graph():
    c = FakeClient(SCHEMA, {"Patent": 1, "Concept": 3}, {"MENTIONS": 3}, [("IN1A", "SULPHIDE RICH COMPOSITE")])
    text = "\n".join(run_diagnostics(c))
    assert "No INVENTED relationships" in text
    assert "Only 1 patent" in text
    assert "MENTIONS data exists" in text
    assert "IN1A" in text


def test_flags_old_schema_without_relationship_patterns():
    c = FakeClient({"labels": ["Patent"]}, {"Patent": 50}, {"INVENTED": 5, "OWNS": 5, "CLASSIFIED_AS": 5}, [])
    text = "\n".join(run_diagnostics(c))
    assert "OLD version" in text


def test_healthy_graph_has_no_warnings():
    c = FakeClient(SCHEMA, {"Patent": 100}, {"INVENTED": 9, "OWNS": 9, "CLASSIFIED_AS": 9, "MENTIONS": 9}, [("a", "b")])
    text = "\n".join(run_diagnostics(c))
    assert "No INVENTED" not in text and "Only" not in text


def test_neo4j_down_does_not_crash():
    c = FakeClient({}, {}, {}, [], fail_schema=True)
    assert "Could not read the graph schema" in run_diagnostics(c)[0]