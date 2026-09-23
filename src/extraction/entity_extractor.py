"""
Week 1, Step: turn raw EPO OPS JSON pages into graph-ready nodes and edges.

EPO OPS's JSON is a fairly direct conversion of its XML schema, which means
one well-known quirk to defend against: a field that could appear once or
many times is returned as a bare dict when there's exactly one, and as a
list when there's more than one. `as_list()` below normalizes that
everywhere so the rest of the code never has to think about it.

Two sources of entities per patent:
  1. STRUCTURED bibliographic fields OPS gives us directly — applicants,
     inventors, IPC classification codes. No NLP needed, these are exact.
  2. UNSTRUCTURED text (title) — run through spaCy NER/noun-chunking to pull
     out candidate technical concepts. Intentionally naive for Week 1 — the
     Explorer/Verifier agents (later weeks) replace this with something
     more reliable once full text is in the pipeline.

Output: a single JSON file with "nodes" and "edges" lists, shaped so
src/graph/neo4j_client.py can load them directly with MERGE (idempotent).

Usage:
    python -m src.extraction.entity_extractor data/raw/epo_IN_..._....json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import spacy
from tqdm import tqdm

from src import config


def load_spacy_model():
    try:
        return spacy.load(config.SPACY_MODEL)
    except OSError:
        print(f"spaCy model '{config.SPACY_MODEL}' not found.")
        print(f"Run: python -m spacy download {config.SPACY_MODEL}")
        sys.exit(1)


def as_list(value: Any) -> list:
    """Normalize EPO OPS's single-item-is-a-dict / multi-item-is-a-list quirk."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def text_of(node: Any) -> str:
    """OPS text nodes are usually {'$': 'the text'} or {'@lang': .., '$': ..}; sometimes a bare str."""
    if isinstance(node, dict):
        return str(node.get("$", "")).strip()
    if isinstance(node, str):
        return node.strip()
    return ""


def safe_get(d: dict, *keys, default=None):
    """Walk nested dict keys defensively — OPS's JSON shape has moved before."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


class EntityExtractor:
    def __init__(self):
        self.nlp = load_spacy_model()
        self.nodes: dict[tuple[str, str], dict] = {}  # (label, id) -> node dict, dedups automatically
        self.edges: list[dict] = []
        self._skipped = 0

    def _add_node(self, label: str, node_id: str, properties: dict):
        key = (label, node_id)
        if key not in self.nodes:
            self.nodes[key] = {"label": label, "id": node_id, "properties": properties}

    def _add_edge(self, start_label: str, start_id: str, rel_type: str, end_label: str, end_id: str):
        self.edges.append(
            {
                "start_label": start_label,
                "start_id": start_id,
                "type": rel_type,
                "end_label": end_label,
                "end_id": end_id,
            }
        )

    def process_exchange_document(self, doc: dict):
        country = doc.get("@country", "")
        doc_number = doc.get("@doc-number", "")
        kind = doc.get("@kind", "")
        family_id = doc.get("@family-id", "")

        if not (country and doc_number):
            self._skipped += 1
            return  # can't build a stable node id without these

        patent_id = f"{country}{doc_number}{kind}"
        biblio = safe_get(doc, "bibliographic-data", default={}) or {}

        # --- Title (prefer English if multiple language variants exist) ---
        titles = as_list(safe_get(biblio, "invention-title"))
        title = ""
        for t in titles:
            if isinstance(t, dict) and t.get("@lang") == "en":
                title = text_of(t)
                break
        if not title and titles:
            title = text_of(titles[0])

        self._add_node(
            "Patent",
            patent_id,
            {"title": title, "country": country, "kind": kind, "family_id": family_id},
        )

        # --- Structured: inventors ---
        inventors = as_list(safe_get(biblio, "parties", "inventors", "inventor"))
        for inv in inventors:
            name = text_of(safe_get(inv, "inventor-name", "name"))
            if not name:
                continue
            self._add_node("Inventor", name, {"name": name})
            self._add_edge("Inventor", name, "INVENTED", "Patent", patent_id)

        # --- Structured: applicants ---
        applicants = as_list(safe_get(biblio, "parties", "applicants", "applicant"))
        for app in applicants:
            name = text_of(safe_get(app, "applicant-name", "name"))
            if not name:
                continue
            self._add_node("Applicant", name, {"name": name})
            self._add_edge("Applicant", name, "OWNS", "Patent", patent_id)

        # --- Structured: classification (CPC/IPC) ---
        # Real OPS responses use "patent-classifications" (this record's is empty —
        # genuinely no classification synced for this filing, not a parsing failure).
        # Some older/other-shaped responses use the legacy "classifications-ipcr" —
        # kept as a fallback. Field names inside a *populated* patent-classification
        # entry are a best-effort guess (section/class/subclass/main-group/subgroup)
        # until we see a real populated example — flagging this rather than pretending
        # certainty. If codes come out looking wrong once you have real data, share a
        # populated record and this gets tightened up.
        classification_entries = as_list(safe_get(biblio, "patent-classifications", "patent-classification"))
        if not classification_entries:
            classification_entries = as_list(safe_get(biblio, "classifications-ipcr", "classification-ipcr"))

        for entry in classification_entries:
            code = text_of(safe_get(entry, "text"))
            if not code:
                # Try assembling from structured parts (CPC/IPC symbol pieces)
                parts = [
                    text_of(safe_get(entry, "section")),
                    text_of(safe_get(entry, "class")),
                    text_of(safe_get(entry, "subclass")),
                    text_of(safe_get(entry, "main-group")),
                    text_of(safe_get(entry, "subgroup")),
                ]
                if any(parts):
                    code = "".join(p for p in parts[:3]) + " " + "/".join(p for p in parts[3:] if p)
                    code = code.strip()
            code = code.strip() if code else ""
            if not code:
                continue
            scheme = safe_get(entry, "classification-scheme", "@scheme") or "unknown"
            self._add_node("ClassificationCode", code, {"code": code, "scheme": scheme})
            self._add_edge("Patent", patent_id, "CLASSIFIED_AS", "ClassificationCode", code)

        # --- Unstructured: naive concept extraction from the title ---
        if title:
            doc_nlp = self.nlp(title)
            for chunk in doc_nlp.noun_chunks:
                concept = chunk.text.strip().lower()
                if len(concept) < 3 or concept in {"a", "the", "an"}:
                    continue
                self._add_node("Concept", concept, {"name": concept, "source": "title_noun_chunk"})
                self._add_edge("Patent", patent_id, "MENTIONS", "Concept", concept)

    def process_page(self, page: dict):
        exchange_docs = as_list(
            safe_get(
                page,
                "ops:world-patent-data",
                "ops:biblio-search",
                "ops:search-result",
                "exchange-documents",
            )
        )
        for wrapper in exchange_docs:
            doc = safe_get(wrapper, "exchange-document")
            if doc is None:
                self._skipped += 1
                continue
            self.process_exchange_document(doc)

    def process_file(self, in_path: Path):
        with open(in_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        for page in tqdm(pages, desc="Extracting entities"):
            self.process_page(page)

        if self._skipped:
            print(f"Warning: skipped {self._skipped} exchange-document(s) with missing/unexpected shape.")

    def save(self, out_path: Path):
        payload = {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract graph nodes/edges from raw EPO OPS JSON.")
    parser.add_argument("input_file", help="Path to a raw JSON file from epo_fetcher.py")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    if not in_path.exists():
        print(f"File not found: {in_path}")
        sys.exit(1)

    extractor = EntityExtractor()
    extractor.process_file(in_path)

    out_path = config.DATA_PROCESSED_DIR / (in_path.stem + "_graph.json")
    extractor.save(out_path)

    print(f"\nExtracted {len(extractor.nodes)} nodes and {len(extractor.edges)} edges.")
    print(f"Saved to {out_path}")
    print("Next: run `python -m src.graph.neo4j_client` to load this into Neo4j.")


if __name__ == "__main__":
    main()
