"""Edge confidence and keep/prune verdicts.

Deterministic and explainable: every edge gets a confidence in [0, 1] built
from named signals, so a verdict can always be traced back to why it was made.

Signals:
  - reliability prior of the primary source
  - corroboration by further, independent sources
  - literature support (PMIDs)
  - how the knowledge was produced (curated, text mined, computed)
  - predicate-specific strength: ligand potency, PPI score
  - context plausibility (kgc/plausibility.py): does the rest of the graph,
    without this edge, still point to it? Only counts against an edge when
    there was enough surrounding graph to judge it by.
  - independent corroboration from an outside source (Open Targets
    association score for gene-disease edges), when it has been fetched
  - context veto: where the context model is strong (out-of-fold AUROC >= 0.9
    against corrupted copies; in practice the ligand-protein edges), an edge
    with enough surrounding graph that is less plausible than three quarters
    of random mix-ups is capped below the threshold (at most 2% of the edges).
    This is what catches a good record filed under the wrong entity, which
    evidence alone cannot.

Structural edges (gene translates to protein) are kept unconditionally;
computed similarity edges carry their Tanimoto score as confidence.

Two graphs come out of this: the whole graph with a verdict on every edge, and
the pruned graph with only the kept edges, so the two can be compared.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import biolink as B

LIST_SEP = "|"
NOTE_SEP = ";"
DEFAULT_THRESHOLD = 0.45

# Reliability prior by primary source (infores). Curated resources high,
# text mining and aggregated predictions low.
SOURCE_PRIOR = {
    "infores:uniprot": 0.90,
    "infores:omim": 0.90,
    "infores:chembl": 0.90,
    "infores:gtopdb": 0.90,
    "infores:drugcentral": 0.85,
    "infores:reactome": 0.85,
    "infores:ctd": 0.75,
    "infores:bioplex": 0.70,
    "infores:monarchinitiative": 0.60,
    "infores:string": 0.60,
    "infores:disgenet": 0.50,
    "infores:tiga": 0.50,
    "infores:gxa": 0.40,
    "infores:eram": 0.40,
    "infores:pharos": 0.40,
    "infores:diseases": 0.35,
}
DEFAULT_PRIOR = 0.40

LEVEL_ADJ = {
    "knowledge_assertion": 0.0,
    "observation": 0.0,
    "statistical_association": -0.05,
    "prediction": -0.15,
    "not_provided": -0.10,
}
AGENT_ADJ = {
    "text_mining_agent": -0.10,
    "computational_model": -0.05,
}

# Structural edges that are definitional, not claims about biology.
STRUCTURAL = {B.TRANSLATES_TO}


def _text(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def score_edge(predicate, primary, supporting, level, agent, publications,
               activity=None, ppi_score=None, similarity=None):
    """Return (confidence, notes). Notes are signed: +gain, -loss, =prior."""
    if predicate in STRUCTURAL:
        return 1.0, ["+structural"]
    if predicate == B.SIMILAR:
        sim = similarity if similarity is not None else 0.0
        return round(sim, 3), [f"=tanimoto:{sim:.2f}"]

    # Merged edges can carry several sources (an activity from ChEMBL and an
    # unattributed one, a disease link from OMIM and from text mining): the
    # prior is the most reliable source's, the others count as corroboration.
    sources = {s for s in _text(supporting).split(LIST_SEP) if s} | {primary}
    named = sources - {B.AGGREGATOR} or {primary}
    best = max(sorted(named), key=lambda s: SOURCE_PRIOR.get(s, DEFAULT_PRIOR))
    conf = SOURCE_PRIOR.get(best, DEFAULT_PRIOR)
    notes = [f"=source:{best}:{conf:.2f}"]
    if best not in SOURCE_PRIOR:
        notes.append("-unknown_source")

    def add(delta, label):
        nonlocal conf
        conf += delta
        notes.append(f"{'+' if delta >= 0 else '-'}{label}")

    n_extra = len(sources - {best, B.AGGREGATOR})
    if n_extra:
        add(min(0.10 * n_extra, 0.20), f"corroborated_by:{n_extra}")

    n_pub = len([p for p in _text(publications).split(LIST_SEP) if p])
    if n_pub >= 3:
        add(0.15, f"literature:{n_pub}")
    elif n_pub >= 1:
        add(0.10, f"literature:{n_pub}")

    d = LEVEL_ADJ.get(level, 0.0)
    if d:
        add(d, f"level:{level}")
    d = AGENT_ADJ.get(agent, 0.0)
    if d:
        add(d, f"agent:{agent}")

    if predicate == B.INTERACTS_PHYS and activity is not None:
        # Pharos reports activity as -log10 molar.
        if activity >= 7:
            add(0.10, f"potent_activity:{activity:g}")
        elif activity < 5:
            add(-0.20, f"weak_activity:{activity:g}")
        elif activity < 6:
            add(-0.05, f"modest_activity:{activity:g}")
    if predicate == B.PPI and ppi_score is not None:
        s = ppi_score / 1000.0 if ppi_score > 1 else ppi_score
        if s >= 0.7:
            add(0.10, f"high_ppi_score:{s:g}")
        elif s < 0.4:
            add(-0.15, f"low_ppi_score:{s:g}")

    return round(min(max(conf, 0.0), 1.0), 3), notes


CONTEXT_MIN = 3          # surrounding edges needed before absence of support counts
CONTEXT_GAIN = 0.10      # most a supportive context can add
CONTEXT_LOSS = 0.15      # most a contradicting context can take away


def context_adjustment(plaus, ctx_notes, ctx_size):
    """Signed adjustment and note from context plausibility (None if not scored)."""
    if plaus is None:
        return 0.0, None
    supported = ctx_notes not in ("", "no_context")
    if not supported and (ctx_size is None or ctx_size < CONTEXT_MIN):
        return 0.0, None  # nothing around the edge to judge it by: neutral
    d = 0.3 * (plaus - 0.5)
    d = min(d, CONTEXT_GAIN) if d > 0 else max(d, -CONTEXT_LOSS)
    why = ctx_notes if supported else "no_supporting_context"
    return round(d, 3), f"{'+' if d >= 0 else '-'}context:{plaus:.2f}:{why}"


VETO_MARGIN = 0.05  # a vetoed edge is capped this far below the threshold


def context_veto(plaus, cutoff, ctx_size, min_context=CONTEXT_MIN):
    """True when the surrounding graph contradicts the edge strongly enough to
    overrule its evidence (see plausibility.score_plausibility)."""
    if plaus is None or cutoff is None or ctx_size is None:
        return False
    return ctx_size >= min_context and plaus < cutoff


def external_adjustment(predicate, ot_score):
    """Open Targets corroboration of gene-disease edges (None if not checked)."""
    if predicate != B.ASSOC_CONDITION or ot_score is None:
        return 0.0, None
    if ot_score >= 0.1:
        return 0.10, f"+opentargets:{ot_score:.2f}"
    if ot_score <= 0.0:
        return -0.05, "-opentargets:no_association"
    return 0.0, f"=opentargets:{ot_score:.2f}"


def score_edges(edges, threshold=DEFAULT_THRESHOLD, context=None):
    """Add confidence, confidence_notes, verdict and prune_reason to every edge.

    `context` is the frame from plausibility.score_plausibility (same index).
    """
    out = edges.copy()
    if context is not None:
        for c in ("plausibility", "plausibility_notes", "context_size", "veto_cutoff"):
            if c in context:
                out[c] = context[c].reindex(out.index)

    def col(name):
        return out[name].tolist() if name in out else [None] * len(out)

    conf, notes = [], []
    for row in zip(out["predicate"], col("primary_knowledge_source"), col("supporting_sources"),
                   col("knowledge_level"), col("agent_type"), col("publications"),
                   col("activity_value_max"), col("interaction_score_max"),
                   col("similarity_score"), col("plausibility"), col("plausibility_notes"),
                   col("context_size"), col("opentargets_score"), col("veto_cutoff")):
        pred, prim, sup, lvl, agt, pubs, act, ppi, sim, pl, pln, cs, ot, vc = row
        c, n = score_edge(pred, _text(prim), sup, _text(lvl), _text(agt), pubs,
                          _num(act), _num(ppi), _num(sim))
        if pred not in STRUCTURAL and pred != B.SIMILAR:
            for d, note in (context_adjustment(_num(pl), _text(pln), _num(cs)),
                            external_adjustment(pred, _num(ot))):
                if note:
                    c = round(min(max(c + d, 0.0), 1.0), 3)
                    n.append(note)
            if context_veto(_num(pl), _num(vc), _num(cs)):
                c = round(min(c, threshold - VETO_MARGIN), 3)
                n.append(f"-context_veto:{_num(pl):.3f}<{_num(vc):.3f}")
        conf.append(c)
        notes.append(n)

    out["confidence"] = conf
    out["confidence_notes"] = [NOTE_SEP.join(n) for n in notes]
    out["verdict"] = np.where(out["confidence"] >= threshold, "keep", "prune")
    reasons = []
    for v, n in zip(out["verdict"], notes):
        if v == "keep":
            reasons.append("")
            continue
        neg = [x[1:] for x in n if x.startswith("-")]
        reasons.append(NOTE_SEP.join(neg) if neg else "low_source_prior")
    out["prune_reason"] = reasons
    return out


def prune_graph(nodes, edges, threshold=DEFAULT_THRESHOLD, context=None):
    """Return (scored_edges, pruned_edges, pruned_nodes, report)."""
    scored = score_edges(edges, threshold, context)
    kept = scored[scored["verdict"] == "keep"].copy()
    used = set(kept["subject"]) | set(kept["object"])
    pruned_nodes = nodes[nodes["id"].isin(used)].copy()

    dropped = scored[scored["verdict"] == "prune"]
    reason_tokens = Counter(t.split(":")[0] for r in dropped["prune_reason"]
                            for t in r.split(NOTE_SEP) if t)
    report = {
        "threshold": threshold,
        "edges": {"total": int(len(scored)), "kept": int(len(kept)), "pruned": int(len(dropped))},
        "nodes": {"before": int(len(nodes)), "after": int(len(pruned_nodes))},
        "kept_by_predicate": kept["predicate"].value_counts().to_dict(),
        "pruned_by_predicate": dropped["predicate"].value_counts().to_dict(),
        "pruned_by_primary_source": dropped["primary_knowledge_source"].value_counts().to_dict(),
        "top_prune_reasons": dict(reason_tokens.most_common(10)),
    }
    return scored, kept, pruned_nodes, report


def attach_opentargets(edges, ot_path):
    """Add opentargets_score to gene-disease edges from a fetched Open Targets table
    (columns: gene, disease, score; score 0 = checked, no association)."""
    ot = pd.read_csv(ot_path, sep="\t")
    look = {(g, d): float(sc) for g, d, sc in zip(ot["gene"], ot["disease"], ot["score"])}
    checked = set(ot["gene"])
    vals = []
    for s_, p_, o_ in zip(edges["subject"], edges["predicate"], edges["object"]):
        if p_ != B.ASSOC_CONDITION or s_ not in checked:
            vals.append(np.nan)
        else:
            vals.append(look.get((s_, o_), 0.0))
    out = edges.copy()
    out["opentargets_score"] = vals
    return out


def verdict_table(scored, nodes):
    """Every source edge with its verdict, in the form the organizers asked for:
    UniProt IDs for gene and protein ends, plus confidence and reasons."""
    g2p = dict(zip(scored.loc[scored["predicate"] == B.TRANSLATES_TO, "subject"],
                   scored.loc[scored["predicate"] == B.TRANSLATES_TO, "object"]))
    src = scored[scored["primary_knowledge_source"] != B.SELF].copy()
    to_uniprot = lambda x: g2p.get(x, x if str(x).startswith("UniProtKB:") else "")
    src["subject_uniprot"] = src["subject"].map(to_uniprot)
    src["object_uniprot"] = src["object"].map(to_uniprot)
    names = dict(zip(nodes["id"], nodes["name"]))
    src["subject_name"] = src["subject"].map(names)
    src["object_name"] = src["object"].map(names)
    cols = ["id", "subject", "subject_uniprot", "subject_name", "predicate", "object",
            "object_uniprot", "object_name", "primary_knowledge_source", "verdict", "confidence",
            "prune_reason", "confidence_notes"]
    return src[[c for c in cols if c in src]]


def run_prune(kgx_dir, name="pharos_kinase", threshold=DEFAULT_THRESHOLD, use_context=True, log=print):
    kgx_dir = Path(kgx_dir)
    nodes = pd.read_csv(kgx_dir / f"{name}_nodes.tsv", sep="\t", low_memory=False)
    enriched = kgx_dir / f"{name}_enriched_edges.tsv"
    edges = pd.read_csv(enriched if enriched.exists() else kgx_dir / f"{name}_edges.tsv",
                        sep="\t", low_memory=False)
    ot_path = kgx_dir / f"{name}_opentargets.tsv"
    if ot_path.exists():
        edges = attach_opentargets(edges, ot_path)
    context, ctx_summary = None, {}
    if use_context:
        from .plausibility import score_plausibility
        log("context plausibility (out of fold, per predicate)")
        context, ctx_summary, _ = score_plausibility(edges)
    scored, kept, pruned_nodes, report = prune_graph(nodes, edges, threshold, context)
    report["context_models"] = ctx_summary
    report["opentargets_checked"] = bool(ot_path.exists())
    scored.to_csv(kgx_dir / f"{name}_edges_scored.tsv", sep="\t", index=False)
    kept.to_csv(kgx_dir / f"{name}_edges_pruned.tsv", sep="\t", index=False)
    pruned_nodes.to_csv(kgx_dir / f"{name}_nodes_pruned.tsv", sep="\t", index=False)
    verdict_table(scored, nodes).to_csv(kgx_dir / f"{name}_verdicts.tsv", sep="\t", index=False)
    (kgx_dir / f"{name}_prune_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
