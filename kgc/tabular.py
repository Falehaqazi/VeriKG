"""Bring your own table: any CSV/TSV of relationships becomes a Biolink KGX graph.

This is how the agent generalizes beyond Pharos. A researcher drops in a
spreadsheet (drug-target screens, gene-disease lists, metabolite-pathway maps,
a lab's own results); the agent proposes which columns are the two entities,
what kind of entities they are, which Biolink predicate links them and which
columns are evidence. The user confirms or edits the mapping, and the same
validate -> prune -> ask loop runs on the result.

The mapping suggestion is deterministic (identifier patterns and header
words), so it works offline; a language model can refine it, but never writes
edges itself.
"""

import hashlib
import re

import numpy as np
import pandas as pd

from . import biolink as B
from .kgx import edge_id

# Category -> (identifier regex, CURIE prefix to add if the value is bare)
ID_PATTERNS = [
    ("biolink:Protein", r"^(UniProtKB:)?([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$", "UniProtKB"),
    ("biolink:SmallMolecule", r"^(CHEMBL\.COMPOUND:)?CHEMBL\d+$", "CHEMBL.COMPOUND"),
    ("biolink:SmallMolecule", r"^(CHEBI:)\d+$", None),
    ("biolink:SmallMolecule", r"^PUBCHEM\.COMPOUND:\d+$", None),
    ("biolink:Disease", r"^(MONDO|DOID|OMIM|ORPHANET|EFO)[:_]\d+$", None),
    ("biolink:Gene", r"^(HGNC|NCBIGene|ENSEMBL):[A-Z0-9]+$", None),
    ("biolink:Gene", r"^ENSG\d{11}$", "ENSEMBL"),
    ("biolink:Pathway", r"^(REACT:)?R-[A-Z]{3}-\d+$", "REACT"),
    ("biolink:BiologicalProcess", r"^GO:\d{7}$", None),
    ("biolink:PhenotypicFeature", r"^HP:\d{7}$", None),
]
HEADER_HINTS = [
    ("biolink:SmallMolecule", r"compound|chem|ligand|molecule|smiles|inchi|metabolite"),
    ("biolink:Drug", r"\bdrug|medication|medicine"),
    ("biolink:Protein", r"protein|uniprot|target|receptor|enzyme|kinase"),
    ("biolink:Gene", r"gene|symbol|hgnc|entrez|ensembl"),
    ("biolink:Disease", r"disease|condition|disorder|indication|cancer|mondo"),
    ("biolink:PhenotypicFeature", r"phenotype|symptom|\bhpo?\b"),
    ("biolink:Pathway", r"pathway|reactome|kegg"),
    ("biolink:BiologicalProcess", r"process|\bgo\b|function"),
    ("biolink:AnatomicalEntity", r"tissue|organ|anatom"),
    ("biolink:Cell", r"\bcell"),
]
CATEGORIES = sorted({c for c, _, _ in ID_PATTERNS} | {c for c, _ in HEADER_HINTS})

DEFAULT_PREDICATE = {
    ("biolink:SmallMolecule", "biolink:Protein"): B.INTERACTS_PHYS,
    ("biolink:Drug", "biolink:Protein"): B.INTERACTS_PHYS,
    ("biolink:SmallMolecule", "biolink:Gene"): "biolink:affects",
    ("biolink:Drug", "biolink:Gene"): "biolink:affects",
    ("biolink:Protein", "biolink:Protein"): B.PPI,
    ("biolink:Gene", "biolink:Gene"): "biolink:interacts_with",
    ("biolink:Gene", "biolink:Disease"): B.ASSOC_CONDITION,
    ("biolink:Protein", "biolink:Disease"): "biolink:associated_with",
    ("biolink:Gene", "biolink:PhenotypicFeature"): "biolink:has_phenotype",
    ("biolink:Disease", "biolink:PhenotypicFeature"): "biolink:has_phenotype",
    ("biolink:Drug", "biolink:Disease"): B.TREATS,
    ("biolink:SmallMolecule", "biolink:Disease"): B.TREATS,
    ("biolink:Gene", "biolink:Pathway"): B.PARTICIPATES_IN,
    ("biolink:Protein", "biolink:Pathway"): B.PARTICIPATES_IN,
    ("biolink:Gene", "biolink:BiologicalProcess"): B.PARTICIPATES_IN,
    ("biolink:Protein", "biolink:BiologicalProcess"): B.PARTICIPATES_IN,
    ("biolink:SmallMolecule", "biolink:Pathway"): B.PARTICIPATES_IN,
    ("biolink:Gene", "biolink:AnatomicalEntity"): "biolink:expressed_in",
    ("biolink:Gene", "biolink:Cell"): "biolink:expressed_in",
}
PREDICATES = sorted(set(DEFAULT_PREDICATE.values()) | {"biolink:related_to"})


def _id_category(values):
    vals = [str(v).strip() for v in values if isinstance(v, str) or not pd.isna(v)][:200]
    if not vals:
        return None, 0.0, None
    best = (None, 0.0, None)
    for cat, pat, prefix in ID_PATTERNS:
        share = np.mean([bool(re.match(pat, v)) for v in vals])
        if share > best[1]:
            best = (cat, float(share), prefix)
    return best


def _header_category(name):
    for cat, pat in HEADER_HINTS:
        if re.search(pat, name.lower()):
            return cat
    return None


def suggest_mapping(df):
    """Propose subject/object columns, categories, predicate and evidence columns."""
    cols = list(df.columns)
    info = {}
    for c in cols:
        cat, share, prefix = _id_category(df[c].dropna().astype(str))
        num = pd.to_numeric(df[c], errors="coerce").notna().mean()
        info[c] = {"id_category": cat if share >= 0.6 else None, "id_share": share, "prefix": prefix,
                   "header_category": _header_category(c), "numeric": float(num)}
    ent = [c for c in cols if info[c]["numeric"] < 0.5 and (info[c]["id_category"] or info[c]["header_category"])]
    if len(ent) < 2:
        ent = [c for c in cols if info[c]["numeric"] < 0.5][:2]
    subj, obj = (ent + [None, None])[:2]
    cat = lambda c: (info[c]["id_category"] or info[c]["header_category"] or "biolink:NamedThing") if c else None
    sc, oc = cat(subj), cat(obj)
    pred = DEFAULT_PREDICATE.get((sc, oc)) or DEFAULT_PREDICATE.get((oc, sc))
    if pred and (sc, oc) not in DEFAULT_PREDICATE and (oc, sc) in DEFAULT_PREDICATE:
        subj, obj, sc, oc = obj, subj, oc, sc
    score_cols = [c for c in cols if c not in (subj, obj) and info[c]["numeric"] >= 0.8]
    pub_cols = [c for c in cols if re.search(r"pmid|pubmed|reference|citation|doi", c.lower())]
    src_cols = [c for c in cols if re.search(r"source|database|provenance", c.lower())]
    return {
        "subject_column": subj, "object_column": obj,
        "subject_category": sc, "object_category": oc,
        "subject_prefix": info[subj]["prefix"] if subj else None,
        "object_prefix": info[obj]["prefix"] if obj else None,
        "predicate": pred or "biolink:related_to",
        "score_column": score_cols[0] if score_cols else None,
        "publication_column": pub_cols[0] if pub_cols else None,
        "source_column": src_cols[0] if src_cols else None,
        "source_name": "user_table",
        "knowledge_level": "knowledge_assertion", "agent_type": "manual_agent",
        "columns": info,
    }


def _curie(v, prefix, fallback_prefix):
    v = str(v).strip()
    if not v:
        return None
    if re.match(r"^[A-Za-z][A-Za-z0-9_.]*:[^\s]+$", v):
        return v
    if prefix:
        return f"{prefix}:{v}"
    # names without an identifier: a stable local CURIE, flagged as non-standard
    slug = re.sub(r"[^A-Za-z0-9]+", "_", v).strip("_")[:60] or hashlib.sha1(v.encode()).hexdigest()[:10]
    return f"{fallback_prefix}:{slug}"


def build_from_table(df, m, run_id=None):
    """Return (nodes, edges, issues) in KGX form with provenance on every edge."""
    run_id = run_id or hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]
    src = "infores:" + re.sub(r"[^a-z0-9_]+", "_", str(m.get("source_name") or "user_table").lower())
    nodes, edges, issues = {}, {}, {"rows": len(df), "skipped_rows": 0}
    for row in df.to_dict("records"):
        s = _curie(row.get(m["subject_column"]), m.get("subject_prefix"), "LOCAL")
        o = _curie(row.get(m["object_column"]), m.get("object_prefix"), "LOCAL")
        if not s or not o or s == o:
            issues["skipped_rows"] += 1
            continue
        for nid, cat, raw in ((s, m["subject_category"], row.get(m["subject_column"])),
                              (o, m["object_category"], row.get(m["object_column"]))):
            nodes.setdefault(nid, {"id": nid, "category": cat, "name": str(raw),
                                   "provided_by": src, "type_iri": B.NODE_TYPE_IRI.get(cat)})
        p = m["predicate"]
        a, b = (min(s, o), max(s, o)) if p in B.SYMMETRIC else (s, o)
        eid = edge_id(a, p, b)
        e = edges.setdefault(eid, {
            "id": eid, "subject": a, "predicate": p, "object": b, "relation": B.PREDICATE_IRI.get(p, ""),
            "primary_knowledge_source": src, "aggregator_knowledge_source": "",
            "knowledge_level": m.get("knowledge_level", "not_provided"),
            "agent_type": m.get("agent_type", "not_provided"),
            "_pubs": set(), "_src": set(), "evidence_count": 0, "score_max": None,
            "source_version": "user_upload", "ingest_run": run_id})
        e["evidence_count"] += 1
        if m.get("score_column"):
            v = pd.to_numeric(pd.Series([row.get(m["score_column"])]), errors="coerce").iloc[0]
            if not pd.isna(v):
                e["score_max"] = v if e["score_max"] is None else max(e["score_max"], v)
        if m.get("publication_column") and not pd.isna(row.get(m["publication_column"])):
            for tok in re.findall(r"\d{5,9}", str(row[m["publication_column"]])):
                e["_pubs"].add(f"PMID:{tok}")
        if m.get("source_column") and not pd.isna(row.get(m["source_column"])):
            e["_src"].add("infores:" + re.sub(r"[^a-z0-9_]+", "_", str(row[m["source_column"]]).lower()))
    rows = []
    for e in edges.values():
        r = {k: v for k, v in e.items() if not k.startswith("_")}
        r["publications"] = "|".join(sorted(e["_pubs"]))
        r["supporting_sources"] = "|".join(sorted(e["_src"] | {src}))
        rows.append(r)
    n = pd.DataFrame(list(nodes.values()))
    issues["local_ids"] = int(n["id"].str.startswith("LOCAL:").sum()) if len(n) else 0
    return n, pd.DataFrame(rows), issues


def validate_generic(nodes, edges):
    """Structural checks for an uploaded graph: CURIEs, dangling edges, provenance."""
    errors, warnings = {}, {}
    ids = set(nodes["id"])
    bad = (~edges["subject"].isin(ids) | ~edges["object"].isin(ids)).sum()
    if bad:
        errors["dangling_edge"] = int(bad)
    local = int(nodes["id"].str.startswith("LOCAL:").sum())
    if local:
        warnings["names_without_identifier (LOCAL: prefix)"] = local
    nocat = int((nodes["category"] == "biolink:NamedThing").sum())
    if nocat:
        warnings["unspecific_category"] = nocat
    rule = B.DOMAIN_RANGE.get(edges["predicate"].iloc[0]) if len(edges) else None
    if rule:
        cat = dict(zip(nodes["id"], nodes["category"]))
        dom = (~edges["subject"].map(cat).isin(rule[0])).sum()
        rng = (~edges["object"].map(cat).isin(rule[1])).sum()
        if dom:
            warnings["subject_outside_predicate_domain"] = int(dom)
        if rng:
            warnings["object_outside_predicate_range"] = int(rng)
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "nodes": len(nodes), "edges": len(edges)}


def topology_plausibility(edges, seed=0):
    """Generic context score for any predicate: shared neighbours of the two
    ends (excluding the edge), learned against random swaps, out of fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    rng = np.random.default_rng(seed)
    nb = {}
    for s, o in zip(edges["subject"], edges["object"]):
        nb.setdefault(s, set()).add(o)
        nb.setdefault(o, set()).add(s)

    def feats(s, o):
        a, b = nb.get(s, set()) - {o}, nb.get(o, set()) - {s}
        two = sum(len(nb.get(x, set()) & b) for x in a) if a and b else 0
        return [np.log1p(two), len(a & b) / (1 + len(a | b))]
    pairs = list(zip(edges["subject"], edges["object"]))
    known = set(pairs)
    objs = sorted(set(edges["object"]))
    neg = []
    for s, _ in pairs:
        o2 = objs[rng.integers(len(objs))]
        if (s, o2) not in known and s != o2:
            neg.append((s, o2))
    X = np.array([feats(*p) for p in pairs + neg])
    y = np.r_[np.ones(len(pairs)), np.zeros(len(neg))]
    if len(pairs) < 20 or len(set(y)) < 2:
        return pd.Series(np.nan, index=edges.index)
    p = cross_val_predict(LogisticRegression(class_weight="balanced"), X, y, cv=5, method="predict_proba")[:, 1]
    return pd.Series(p[:len(pairs)].round(4), index=edges.index)
