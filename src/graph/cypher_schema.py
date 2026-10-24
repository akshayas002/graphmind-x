"""
Deterministic schema check for LLM-generated Cypher.

Small local models routinely (a) write a relationship arrow backwards, e.g.
(p:Patent)-[:INVENTED]->(i:Inventor) when the graph only has
(Inventor)-[:INVENTED]->(Patent), and (b) invent labels or relationship types
that don't exist (e.g. :Kind). Both produce valid-looking Cypher that
silently returns zero rows. Telling the model "use the right direction" in
the prompt isn't reliable, so this module enforces it in code:

  - reversed arrow whose reverse exists in the schema  -> auto-flipped
  - unknown label / relationship type                  -> rejected with a
    specific message that gets fed back to the model on the retry
  - relationship that exists but never between those two labels -> rejected

If the schema has no labels at all (empty graph) the check is skipped.
"""
import re
from dataclasses import dataclass, field

PATTERN_QUERY_LIMIT = 50

_STRING_PATTERN = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")

# lookahead + capture groups so chains like (a)-[:R]->(b)-[:S]->(c) all match
_TRIPLE = re.compile(
    r"(?=(?P<n1>\(\s*(?P<v1>\w*)\s*(?::\s*(?P<l1>\w+))?[^()]*\))"
    r"\s*(?P<rel>(?P<left><)?-\s*\[\s*\w*\s*(?::\s*(?P<types>[\w|:]+))?[^\]]*\]\s*-(?P<right>>)?)"
    r"\s*(?P<n2>\(\s*(?P<v2>\w*)\s*(?::\s*(?P<l2>\w+))?[^()]*\)))"
)
_LABELED_NODE = re.compile(r"\(\s*(?P<var>\w*)\s*:\s*(?P<label>\w+)[^()]*\)")


@dataclass
class SchemaCheckResult:
    cypher: str
    error: str | None = None
    fixes: list[str] = field(default_factory=list)


def _mask_strings(cypher: str) -> str:
    """Blank out string literal contents (same length, so offsets still line up)."""
    return _STRING_PATTERN.sub(lambda m: m.group(0)[0] + "_" * (len(m.group(0)) - 2) + m.group(0)[-1], cypher)


def _split_types(types: str | None) -> list[str]:
    if not types:
        return []
    return [t for t in re.split(r"[|:]", types) if t]


def check_and_fix(cypher: str, schema: dict) -> SchemaCheckResult:
    labels = set(schema.get("labels", []))
    patterns = schema.get("relationship_patterns", [])
    rel_types = set(schema.get("relationship_types", [])) | {p["type"] for p in patterns}

    if not labels:
        return SchemaCheckResult(cypher=cypher)  # empty graph: nothing to check against

    masked = _mask_strings(cypher)

    # --- unknown labels ---
    var_label: dict[str, str] = {}
    unknown_labels: list[str] = []
    for m in _LABELED_NODE.finditer(masked):
        label = m.group("label")
        if label not in labels:
            if label not in unknown_labels:
                unknown_labels.append(label)
        elif m.group("var") and m.group("var") not in var_label:
            var_label[m.group("var")] = label
    if unknown_labels:
        return SchemaCheckResult(
            cypher=cypher,
            error=f"Label(s) {unknown_labels} do not exist in the graph. Valid labels: {sorted(labels)}.",
        )

    # --- relationships: existence + direction ---
    def matches(src, rel, dst) -> bool:
        return any(
            p["type"] == rel and (src is None or p["from"] == src) and (dst is None or p["to"] == dst)
            for p in patterns
        )
    pattern_types = {p["type"] for p in patterns}
    patterns_complete = bool(patterns) and len(patterns) < PATTERN_QUERY_LIMIT
    edits: list[tuple[int, int, str]] = []
    fixes: list[str] = []

    for m in _TRIPLE.finditer(masked):
        types = _split_types(m.group("types"))
        bad = [t for t in types if t not in rel_types]
        if bad:
            return SchemaCheckResult(
                cypher=cypher,
                error=f"Relationship type(s) {bad} do not exist in the graph. "
                      f"Valid relationships: {sorted(rel_types)}.",
            )
        if patterns_complete and types and all(t not in pattern_types for t in types):
            shown = ", ".join(f"[:{t}]" for t in types)
            return SchemaCheckResult(
                cypher=cypher,
                error=f"The graph contains no {shown} relationships at all (the type exists but no data "
                      f"was loaded for it), so this cannot be answered from the graph. "
                      f"Relationships that do have data: {sorted(pattern_types)}.",
            )
        left, right = bool(m.group("left")), bool(m.group("right"))
        if len(types) != 1 or left == right:
            continue  # undirected, bidirectional, or alternation: existence check only

        rel = types[0]
        l1 = m.group("l1") or var_label.get(m.group("v1"))
        l2 = m.group("l2") or var_label.get(m.group("v2"))
        if l1 is None and l2 is None:
            continue

        src, dst = (l1, l2) if right else (l2, l1)  # actual direction of the arrow as written
        if matches(src, rel, dst):
            continue
        if matches(dst, rel, src):
            rel_start, rel_end = m.start("rel"), m.end("rel")
            inner = re.search(r"\[.*\]", cypher[rel_start:rel_end], re.DOTALL).group(0)
            flipped = f"<-{inner}-" if right else f"-{inner}->"
            edits.append((rel_start, rel_end, flipped))
            fixes.append(f"flipped reversed [:{rel}] between {l1 or '?'} and {l2 or '?'}")
            continue

        valid = [f"({p['from']})-[:{p['type']}]->({p['to']})" for p in patterns if p["type"] == rel]
        if not valid:
            # The type exists but we hold no endpoint information for it (pattern
            # introspection failed or was truncated). We can't disprove the query,
            # so let it through rather than reject something that may be correct.
            continue
        return SchemaCheckResult(
            cypher=cypher,
            error=f"[:{rel}] never connects {src or 'any'} to {dst or 'any'} in this graph. "
                  f"Where it exists: {', '.join(valid)}.",
        )

    fixed = cypher
    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        fixed = fixed[:start] + replacement + fixed[end:]
    return SchemaCheckResult(cypher=fixed, fixes=fixes)