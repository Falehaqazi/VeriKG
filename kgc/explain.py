"""Edge lineage and link explanations: why does A connect to B?

explain_edge       an edge that exists: its sources, evidence and verdict, in plain words.
explain_prediction a pair with no edge: the paths through the training graph
                   that make the link plausible (shared ligands, PPIs, pathways),
                   each path listed with the IDs of the edges it uses.
"""

import numpy as np
import pandas as pd

from . import biolink as B
from .kgx import edge_id

LIST_SEP = "|"


def _text(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)


def _short(predicate):
    return predicate.split(":")[-1].replace("_", " ")


def _label(nodes_by_id, curie):
    name = _text((nodes_by_id.get(curie) or {}).get("name"))
    return f"{name} ({curie})" if name else curie


def _split(v):
    return [x for x in _text(v).split(LIST_SEP) if x]


def explain_edge(nodes, edges, subject, obj, predicate=None):
    """Lineage of every edge between subject and obj (either direction)."""
    nodes_by_id = nodes.set_index("id").to_dict("index")
    m = (((edges["subject"] == subject) & (edges["object"] == obj))
         | ((edges["subject"] == obj) & (edges["object"] == subject)))
    if predicate:
        m &= edges["predicate"] == predicate
    hits = edges[m]
    if hits.empty:
        return {"found": False,
                "text": f"No edge between {_label(nodes_by_id, subject)} and {_label(nodes_by_id, obj)}."}

    records, lines = [], []
    for e in hits.to_dict("records"):
        pmids = [p.split(":")[-1] for p in _split(e.get("publications"))]
        rec = {
            "edge_id": e["id"], "subject": e["subject"], "predicate": e["predicate"],
            "object": e["object"],
            "primary_knowledge_source": _text(e.get("primary_knowledge_source")),
            "aggregator_knowledge_source": _text(e.get("aggregator_knowledge_source")),
            "supporting_sources": _split(e.get("supporting_sources")),
            "knowledge_level": _text(e.get("knowledge_level")),
            "agent_type": _text(e.get("agent_type")),
            "evidence_count": e.get("evidence_count"),
            "publications": [f"https://pubmed.ncbi.nlm.nih.gov/{p}/" for p in pmids],
            "source_version": _text(e.get("source_version")),
            "confidence": e.get("confidence"),
            "verdict": _text(e.get("verdict")),
            "confidence_notes": _text(e.get("confidence_notes")).split(";") if _text(e.get("confidence_notes")) else [],
        }
        records.append(rec)
        line = (f"{_label(nodes_by_id, e['subject'])} {_short(e['predicate'])} "
                f"{_label(nodes_by_id, e['object'])}. "
                f"Reported by {rec['primary_knowledge_source']}, aggregated by "
                f"{rec['aggregator_knowledge_source']} ({rec['source_version']}). "
                f"Knowledge level: {rec['knowledge_level']}; produced by: {rec['agent_type']}.")
        if len(rec["supporting_sources"]) > 1:
            line += f" Supporting sources: {', '.join(rec['supporting_sources'])}."
        if pmids:
            line += f" Literature: {', '.join('PMID:' + p for p in pmids[:5])}" \
                    f"{' and more' if len(pmids) > 5 else ''}."
        if rec["verdict"]:
            line += f" Confidence {rec['confidence']} ({rec['verdict']}): {'; '.join(rec['confidence_notes'])}."
        lines.append(line)
    return {"found": True, "edges": records, "text": "\n".join(lines)}


def explain_prediction(edges, chemical, protein, nodes=None, top=3):
    """Why might chemical bind protein? Paths through the (training) graph.

    Pass the training graph, not the full graph, when explaining a held-out
    pair, so the pair itself is not part of its own evidence.
    """
    nodes_by_id = nodes.set_index("id").to_dict("index") if nodes is not None else {}
    task = edges[edges["predicate"] == B.INTERACTS_PHYS]
    known = bool(((task["subject"] == chemical) & (task["object"] == protein)).any())
    targets = sorted(set(task.loc[task["subject"] == chemical, "object"]) - {protein})

    ligands_of = {}
    for s, o in zip(task["subject"], task["object"]):
        ligands_of.setdefault(o, set()).add(s)
    prot_ligs = ligands_of.get(protein, set()) - {chemical}

    paths = {"similar_compound": [], "shared_ligand": [], "ppi": [], "pathway": []}

    # a structurally similar compound binds the protein
    sim = edges[edges["predicate"] == B.SIMILAR] if B.SIMILAR in set(edges["predicate"]) else None
    if sim is not None and len(prot_ligs):
        m1 = sim[(sim["subject"] == chemical) & sim["object"].isin(prot_ligs)]
        m2 = sim[(sim["object"] == chemical) & sim["subject"].isin(prot_ligs)]
        cand = [(o, sc) for o, sc in zip(m1["object"], m1.get("similarity_score", [None] * len(m1)))] + \
               [(s_, sc) for s_, sc in zip(m2["subject"], m2.get("similarity_score", [None] * len(m2)))]
        for c2, sc in sorted(cand, key=lambda x: -(x[1] or 0)):
            a, b = sorted((chemical, c2))
            paths["similar_compound"].append({
                "via_chemical": c2, "similarity": sc,
                "edges": [edge_id(a, B.SIMILAR, b), edge_id(c2, B.INTERACTS_PHYS, protein)],
                "text": f"{_label(nodes_by_id, chemical)} is structurally similar "
                        f"(Tanimoto {sc:.2f}) to {_label(nodes_by_id, c2)}, which binds "
                        f"{_label(nodes_by_id, protein)}."})

    # chemical binds t, another ligand c2 binds both t and protein
    for t in targets:
        for c2 in sorted(ligands_of.get(t, set()) & prot_ligs):
            paths["shared_ligand"].append({
                "via_protein": t, "via_chemical": c2,
                "edges": [edge_id(chemical, B.INTERACTS_PHYS, t),
                          edge_id(c2, B.INTERACTS_PHYS, t),
                          edge_id(c2, B.INTERACTS_PHYS, protein)],
                "text": f"{_label(nodes_by_id, chemical)} binds {_label(nodes_by_id, t)}, which "
                        f"{_label(nodes_by_id, c2)} also binds, and {_label(nodes_by_id, c2)} "
                        f"binds {_label(nodes_by_id, protein)}."})

    # chemical binds t, t interacts with protein
    ppi = edges[edges["predicate"] == B.PPI]
    ppi_pairs = {(s, o) for s, o in zip(ppi["subject"], ppi["object"])}
    for t in targets:
        pair = (t, protein) if (t, protein) in ppi_pairs else (protein, t) if (protein, t) in ppi_pairs else None
        if pair:
            paths["ppi"].append({
                "via_protein": t,
                "edges": [edge_id(chemical, B.INTERACTS_PHYS, t), edge_id(pair[0], B.PPI, pair[1])],
                "text": f"{_label(nodes_by_id, chemical)} binds {_label(nodes_by_id, t)}, which "
                        f"physically interacts with {_label(nodes_by_id, protein)}."})

    # chemical binds t, t and protein share a pathway
    pw = edges[edges["predicate"] == B.PARTICIPATES_IN]
    pw_of = {}
    for s, o in zip(pw["subject"], pw["object"]):
        pw_of.setdefault(s, set()).add(o)
    for t in targets:
        for p in sorted(pw_of.get(t, set()) & pw_of.get(protein, set())):
            paths["pathway"].append({
                "via_protein": t, "via_pathway": p,
                "edges": [edge_id(chemical, B.INTERACTS_PHYS, t),
                          edge_id(t, B.PARTICIPATES_IN, p),
                          edge_id(protein, B.PARTICIPATES_IN, p)],
                "text": f"{_label(nodes_by_id, chemical)} binds {_label(nodes_by_id, t)}, which shares "
                        f"{_label(nodes_by_id, p)} with {_label(nodes_by_id, protein)}."})

    counts = {k: len(v) for k, v in paths.items()}
    shown = {k: v[:top] for k, v in paths.items()}
    head = (f"{_label(nodes_by_id, chemical)} already has a recorded interaction with "
            f"{_label(nodes_by_id, protein)}; showing the other evidence around it."
            if known else
            f"No recorded interaction between {_label(nodes_by_id, chemical)} and "
            f"{_label(nodes_by_id, protein)}.")
    if sum(counts.values()) == 0:
        body = " No supporting path was found in the graph."
    else:
        body = (f" Supporting paths: {counts['similar_compound']} through structurally similar compounds, "
                f"{counts['shared_ligand']} through shared ligands, "
                f"{counts['ppi']} through protein interactions, {counts['pathway']} through shared pathways.")
    lines = [head + body] + [p["text"] for k in ("similar_compound", "shared_ligand", "ppi", "pathway")
                             for p in shown[k]]
    return {"chemical": chemical, "protein": protein, "already_known": known,
            "path_counts": counts, "paths": shown, "text": "\n".join(lines)}
