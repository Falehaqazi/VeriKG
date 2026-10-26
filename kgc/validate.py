"""Structural validation of the KGX output against the schema in biolink.py.

Errors break the graph (dangling edges, wrong domain or range, bad provenance).
Warnings are allowed but reported (non-standard ID prefixes).
"""

import re
from collections import Counter

import pandas as pd

from . import biolink as B

CURIE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]*:[^\s|]+$")
STANDARD_PREFIXES = {
    "UniProtKB", "NCBIGene", "HGNC", "CHEMBL.COMPOUND", "PUBCHEM.COMPOUND",
    "DRUGCENTRAL", "GTOPDB", "UNII", "MONDO", "DOID", "UMLS", "OMIM", "MESH",
    "HP", "REACT", "GO",
}
CATEGORIES = {B.PROTEIN, B.GENE, B.CHEMICAL, B.DRUG, B.DISEASE, B.PATHWAY}


def validate(nodes: pd.DataFrame, edges: pd.DataFrame):
    errors, warnings = Counter(), Counter()

    if nodes["id"].duplicated().any():
        errors["duplicate_node_id"] += int(nodes["id"].duplicated().sum())
    if edges["id"].duplicated().any():
        errors["duplicate_edge_id"] += int(edges["id"].duplicated().sum())

    for nid in nodes["id"]:
        if not CURIE.match(str(nid)):
            errors["malformed_curie"] += 1
        elif nid.split(":", 1)[0] not in STANDARD_PREFIXES:
            warnings[f"nonstandard_prefix:{nid.split(':', 1)[0]}"] += 1
    bad_cat = ~nodes["category"].isin(CATEGORIES)
    if bad_cat.any():
        errors["unknown_category"] += int(bad_cat.sum())

    cat = dict(zip(nodes["id"], nodes["category"]))
    for e in edges.itertuples(index=False):
        if e.subject not in cat or e.object not in cat:
            errors["dangling_edge"] += 1
            continue
        rule = B.DOMAIN_RANGE.get(e.predicate)
        if rule is None:
            errors["unknown_predicate"] += 1
            continue
        dom, rng = rule
        if cat[e.subject] not in dom:
            errors[f"domain_violation:{e.predicate}"] += 1
        if cat[e.object] not in rng:
            errors[f"range_violation:{e.predicate}"] += 1
        if e.subject == e.object:
            errors["self_loop"] += 1
        if not str(e.primary_knowledge_source).startswith("infores:"):
            errors["primary_source_not_infores"] += 1
        if e.knowledge_level not in B.KNOWLEDGE_LEVELS:
            errors["bad_knowledge_level"] += 1
        if e.agent_type not in B.AGENT_TYPES:
            errors["bad_agent_type"] += 1

    return {"ok": not errors, "errors": dict(errors), "warnings": dict(warnings),
            "nodes": len(nodes), "edges": len(edges)}
