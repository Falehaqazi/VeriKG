"""Leakage-safe split of the link prediction task, plus hard negatives.

Held-out pairs are removed from the training graph under every predicate and
in both directions. Negatives are type-constrained (a real chemical paired
with a real target protein) and degree-matched, so a model cannot score well
just by ranking popular proteins high.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import biolink as B


def pair_key(a, b):
    return (a, b) if a <= b else (b, a)


def audit_leakage(edges, heldout_pairs):
    """Drop every edge that touches a held-out pair, in any direction or predicate.

    Returns (clean_edges, removed_counts_by_primary_source).
    """
    keys = {pair_key(s, o) for s, o in heldout_pairs}
    hit = np.array([pair_key(s, o) in keys for s, o in zip(edges["subject"], edges["object"])],
                   dtype=bool)
    removed = edges.loc[hit, "primary_knowledge_source"].value_counts().to_dict() if hit.any() else {}
    return edges.loc[~hit].copy(), removed


def sample_negatives(pos, candidates, known, degree, k, rng, matched=True):
    """For each positive (s, o), draw k corrupted objects o' never linked to s.

    matched=True draws o' with probability proportional to its training degree,
    so negatives have the same degree distribution as positives and a
    degree-only score cannot separate them. matched=False draws uniformly.
    Returns (positive_index, subject, negative_object) triples.
    """
    cand_all = np.array(candidates)
    if matched:
        w = np.array([degree.get(c, 0) for c in candidates], dtype=float) + 0.5
        w /= w.sum()
    rows = []
    for i, (s, o) in enumerate(pos):
        picked, tries = [], 0
        while len(picked) < k and tries < 50 * k:
            draw = rng.choice(len(cand_all), size=2 * k, p=w) if matched else rng.integers(len(cand_all), size=2 * k)
            for j in draw:
                c = cand_all[j]
                tries += 1
                if c != o and (s, c) not in known and c not in picked:
                    picked.append(c)
                    if len(picked) == k:
                        break
        rows.extend((i, s, c) for c in picked)
    return rows


def make_split(nodes, edges, out_dir, predicate=B.INTERACTS_PHYS, seed=0,
               frac=(0.8, 0.1, 0.1), rank_negatives=99):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    task = edges[edges["predicate"] == predicate]
    pairs = sorted(set(zip(task["subject"], task["object"])))
    order = rng.permutation(len(pairs))
    n_tr = int(frac[0] * len(pairs))
    n_va = int(frac[1] * len(pairs))
    tr = [pairs[i] for i in order[:n_tr]]
    va = [pairs[i] for i in order[n_tr:n_tr + n_va]]
    te = [pairs[i] for i in order[n_tr + n_va:]]

    train_graph, removed = audit_leakage(edges, va + te)

    known = set(pairs)
    proteins = sorted(set(task["object"]))
    deg = {p: 0 for p in proteins}
    for _, o in tr:
        deg[o] += 1

    def write(name, pos):
        neg1 = sample_negatives(pos, proteins, known, deg, 1, rng, matched=True)
        negr = sample_negatives(pos, proteins, known, deg, 1, rng, matched=False)
        df = pd.concat([
            pd.DataFrame(pos, columns=["subject", "object"]).assign(label=1, neg_kind="positive"),
            pd.DataFrame([r[1:] for r in neg1], columns=["subject", "object"]).assign(label=0, neg_kind="degree_matched"),
            pd.DataFrame([r[1:] for r in negr], columns=["subject", "object"]).assign(label=0, neg_kind="random"),
        ], ignore_index=True)
        df.to_csv(out_dir / f"{name}.tsv", sep="\t", index=False)
        return df

    write("val", va)
    write("test", te)
    rank = sample_negatives(te, proteins, known, deg, rank_negatives, rng, matched=True)
    pd.DataFrame(rank, columns=["pos_id", "subject", "object"]).to_csv(
        out_dir / "test_rank_negatives.tsv", sep="\t", index=False)
    train_graph.to_csv(out_dir / "train_graph_edges.tsv", sep="\t", index=False)

    train_chems = {s for s, _ in tr}
    report = {
        "predicate": predicate, "seed": seed, "fractions": list(frac),
        "pairs": {"train": len(tr), "val": len(va), "test": len(te)},
        "leakage_audit_removed_by_source": removed,
        "test_pairs_with_unseen_chemical": float(np.mean([s not in train_chems for s, _ in te])) if te else 0.0,
        "rank_negatives_per_test_positive": rank_negatives,
        "candidate_proteins": len(proteins),
    }
    (out_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
