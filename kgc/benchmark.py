"""Injected-error benchmark for the pruning agent.

Copies of real edges are planted in the graph with one end swapped, and the
pruning agent is scored on how well it finds them. The planted edges carry the
full provenance of the real edge they were copied from (source, PMIDs,
potency), so they look exactly like a curated record filed under the wrong
entity. Evidence scoring alone cannot tell them apart; that is the point:
the benchmark measures what context adds.

Three kinds of swap, easiest to hardest:
  random     the new end is any type-correct node
  degree     the new end is drawn in proportion to its degree (popular nodes)
  near_miss  the new end is a neighbour of the true end: an interaction
             partner or pathway mate of the true protein (the paralog-style
             mix-up curators actually make), a disease sharing genes with the
             true disease, and so on

The context model is trained on random swaps only (see plausibility.py), so
degree and near-miss results are out of distribution for it.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import biolink as B
from .kgx import edge_id
from .plausibility import TASKS, GraphIndex, score_plausibility
from .prune import DEFAULT_THRESHOLD, score_edges

KINDS = ("random", "degree", "near_miss")


def _near(gi, pred, s, o, rng):
    """A plausible-looking wrong end for (s, pred, o)."""
    if pred == B.INTERACTS_PHYS:
        pool = set(gi.ppi.get(o, set()))
        for w in gi.pw.get(o, set()):
            pool |= gi.pw_members.get(w, set())
        pool = {p for p in pool if gi.ligs.get(p)}  # other ligand-bearing proteins
    elif pred == B.ASSOC_CONDITION:
        pool = set()
        for g in list(gi.dis_genes.get(o, set()))[:50]:
            pool |= gi.gene_dis.get(g, set())
    elif pred == B.PPI:
        pool = set()
        for p in list(gi.ppi.get(o, set()))[:50]:
            pool |= gi.ppi.get(p, set())
    elif pred == B.PARTICIPATES_IN:
        pool = set()
        for p in list(gi.ppi.get(s, set()))[:50]:
            pool |= gi.pw.get(p, set())
    elif pred == B.TREATS:
        pool = set()
        for t in gi.targets.get(s, set()):
            g = gi.p2g.get(t)
            if g:
                pool |= gi.gene_dis.get(g, set())
    else:
        pool = set()
    pool.discard(o)
    pool.discard(s)
    if not pool:
        return None
    pool = sorted(pool)
    return pool[rng.integers(len(pool))]


def inject_errors(edges, rate=0.02, seed=0, min_per=10):
    """Return (edges_with_errors, injected_ids by (predicate, kind))."""
    rng = np.random.default_rng(seed)
    gi = GraphIndex(edges)
    existing = set(zip(edges["subject"], edges["predicate"], edges["object"]))
    rows, planted = [], defaultdict(list)
    for pred in TASKS:
        sub = edges[edges["predicate"] == pred]
        if len(sub) < 20:
            continue
        objects = sub["object"].tolist()
        uniq = sorted(set(objects))
        deg = pd.Series(objects).value_counts()
        w = deg.reindex(uniq).fillna(0).values.astype(float)
        w /= w.sum()
        n = max(min_per, int(rate * len(sub)))
        for kind in KINDS:
            made, tries = 0, 0
            while made < n and tries < 30 * n:
                tries += 1
                tmpl = sub.iloc[rng.integers(len(sub))].to_dict()
                s, o = tmpl["subject"], tmpl["object"]
                if kind == "random":
                    o2 = uniq[rng.integers(len(uniq))]
                elif kind == "degree":
                    o2 = uniq[rng.choice(len(uniq), p=w)]
                else:
                    o2 = _near(gi, pred, s, o, rng)
                if o2 is None or o2 == s or o2 == o:
                    continue
                a, b = (min(s, o2), max(s, o2)) if pred in B.SYMMETRIC else (s, o2)
                if (a, pred, b) in existing or (b, pred, a) in existing:
                    continue
                existing.add((a, pred, b))
                tmpl.update({"subject": a, "object": b, "id": edge_id(a, pred, b) + ":err"})
                rows.append(tmpl)
                planted[(pred, kind)].append(tmpl["id"])
                made += 1
    bad = pd.DataFrame(rows)
    return pd.concat([edges, bad], ignore_index=True), planted


def _auc(y, score):
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else None


def run_benchmark(edges, rate=0.02, seed=0, threshold=DEFAULT_THRESHOLD, log=print):
    noisy, planted = inject_errors(edges, rate=rate, seed=seed)
    log(f"planted {sum(len(v) for v in planted.values())} wrong edges")
    evidence_only = score_edges(noisy, threshold)
    context, _, _ = score_plausibility(noisy, seed=seed)
    full = score_edges(noisy, threshold, context)
    bad_ids = {i for v in planted.values() for i in v}

    res = {"rate": rate, "seed": seed, "threshold": threshold, "by_predicate": {}}
    for pred in TASKS:
        m = noisy["predicate"] == pred
        if not m.any() or not any(p == pred for p, _ in planted):
            continue
        real = m & ~noisy["id"].isin(bad_ids)
        out = {"real_edges": int(real.sum()),
               "real_pruned_share": float((full.loc[real, "verdict"] == "prune").mean())}
        for kind in KINDS:
            ids = set(planted.get((pred, kind), []))
            if not ids:
                continue
            bad = m & noisy["id"].isin(ids)
            sel = real | bad
            y = bad[sel].astype(int).values
            out[kind] = {
                "planted": int(bad.sum()),
                "auroc_evidence_only": _auc(y, -evidence_only.loc[sel, "confidence"].values),
                "auroc_with_context": _auc(y, -full.loc[sel, "confidence"].values),
                "auroc_context_plausibility": _auc(y, -context.loc[sel, "plausibility"].fillna(0.5).values),
                "recall_at_threshold": float((full.loc[bad, "verdict"] == "prune").mean()),
            }
        res["by_predicate"][pred] = out
    return res


def run_benchmark_files(kgx_dir, name="pharos_kinase", rate=0.02, seed=0, log=print):
    kgx_dir = Path(kgx_dir)
    enriched = kgx_dir / f"{name}_enriched_edges.tsv"
    edges = pd.read_csv(enriched if enriched.exists() else kgx_dir / f"{name}_edges.tsv",
                        sep="\t", low_memory=False)
    res = run_benchmark(edges, rate=rate, seed=seed, log=log)
    out = kgx_dir.parent / "results" / f"{name}_pruning_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
