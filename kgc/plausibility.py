"""Context plausibility: does the rest of the graph agree with this edge?

The evidence score in prune.py asks "how good is the record behind this
edge?". This module asks a different, independent question: "if this record
did not exist, would the surrounding graph still suggest the edge?". A
mis-annotated activity (the right compound filed under a paralog, say) can sit
in a perfectly good ChEMBL document; only context can catch it.

For each predicate a few named, label-free features are computed from the
graph *minus the edge itself*. A logistic regression learns, per predicate,
to tell real edges from corrupted copies (same subject, object swapped for a
random type-correct node). Every real edge is then scored out of fold, so no
edge contributes to its own score. The coefficients and per-edge feature
values are kept, so each plausibility comes with its reasons.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from . import biolink as B

TASKS = [B.INTERACTS_PHYS, B.ASSOC_CONDITION, B.PPI, B.PARTICIPATES_IN, B.TREATS]

FEATURES = {
    B.INTERACTS_PHYS: ["analog_binds", "shared_ligand_profile", "cotarget_interacts", "cotarget_pathway"],
    B.ASSOC_CONDITION: ["interactor_associated", "pathway_partner_associated", "drug_target_link"],
    B.PPI: ["shared_partners", "shared_pathways", "shared_ligands"],
    B.PARTICIPATES_IN: ["interactor_in_pathway"],
    B.TREATS: ["drug_targets_disease_gene", "analog_treats"],
}

FEATURE_TEXT = {
    "analog_binds": "a structurally similar compound binds this protein",
    "shared_ligand_profile": "this compound's other targets share ligands with this protein",
    "cotarget_interacts": "this compound's other targets interact with this protein",
    "cotarget_pathway": "this compound's other targets share a pathway with this protein",
    "interactor_associated": "proteins that interact with this gene's product are linked to the disease",
    "pathway_partner_associated": "genes in the same pathways are linked to the disease",
    "drug_target_link": "a drug for this disease binds this gene's product",
    "shared_partners": "the two proteins share interaction partners",
    "shared_pathways": "the two proteins share pathways",
    "shared_ligands": "the two proteins share ligands",
    "interactor_in_pathway": "interaction partners are in the same pathway",
    "drug_targets_disease_gene": "the drug binds a protein whose gene is linked to the disease",
    "analog_treats": "a structurally similar drug is indicated for the disease",
}


class GraphIndex:
    """Adjacency sets used by the feature functions, built once per graph."""

    def __init__(self, edges):
        self.ligs = defaultdict(set)      # protein -> chemicals
        self.targets = defaultdict(set)   # chemical -> proteins
        self.ppi = defaultdict(set)       # protein -> proteins
        self.pw = defaultdict(set)        # protein -> pathways
        self.pw_members = defaultdict(set)
        self.g2p, self.p2g = {}, {}
        self.dis_genes = defaultdict(set)  # disease -> genes
        self.gene_dis = defaultdict(set)
        self.treats = defaultdict(set)    # disease -> chemicals
        self.sim = defaultdict(dict)      # chemical -> {chemical: similarity}
        cols = ["subject", "predicate", "object"]
        has_sim = "similarity_score" in edges
        sims = edges["similarity_score"].tolist() if has_sim else [None] * len(edges)
        for (s, p, o), sc in zip(edges[cols].itertuples(index=False, name=None), sims):
            self.add(s, p, o, sc)

    def add(self, s, p, o, score=None):
        if p == B.INTERACTS_PHYS:
            self.ligs[o].add(s); self.targets[s].add(o)
        elif p == B.PPI:
            self.ppi[s].add(o); self.ppi[o].add(s)
        elif p == B.PARTICIPATES_IN:
            self.pw[s].add(o); self.pw_members[o].add(s)
        elif p == B.TRANSLATES_TO:
            self.g2p[s] = o; self.p2g[o] = s
        elif p == B.ASSOC_CONDITION:
            self.dis_genes[o].add(s); self.gene_dis[s].add(o)
        elif p == B.TREATS:
            self.treats[o].add(s)
        elif p == B.SIMILAR:
            v = float(score) if score is not None and score == score else 0.6
            self.sim[s][o] = v; self.sim[o][s] = v

    # ---- features; each excludes the edge (s, o) itself ----
    def f_interacts_phys(self, c, p):
        ligs_p = self.ligs.get(p, set()) - {c}
        analog = max((v for c2, v in self.sim.get(c, {}).items() if c2 in ligs_p), default=0.0)
        others = self.targets.get(c, set()) - {p}
        prof = 0.0
        for t in others:
            lt = self.ligs.get(t, set()) - {c}
            if lt and ligs_p:
                prof = max(prof, len(lt & ligs_p) / len(lt | ligs_p))
        ppi_p = self.ppi.get(p, set())
        inter = len(others & ppi_p) / (1 + len(others))
        pw_p = self.pw.get(p, set())
        pwy = float(any(self.pw.get(t, set()) & pw_p for t in others)) if pw_p else 0.0
        return [analog, prof, inter, pwy]

    def f_assoc_condition(self, g, d):
        genes = self.dis_genes.get(d, set()) - {g}
        prot = self.g2p.get(g)
        dprots = {self.g2p[x] for x in genes if x in self.g2p}
        nb = self.ppi.get(prot, set()) if prot else set()
        inter = len(nb & dprots) / (1 + len(nb)) * 10 if nb else 0.0
        pws = self.pw.get(prot, set()) if prot else set()
        pwp = float(any(self.pw.get(x, set()) & pws for x in dprots)) if pws else 0.0
        drugs = self.treats.get(d, set())
        dtl = float(any(prot in self.targets.get(c, set()) for c in drugs)) if prot else 0.0
        return [min(inter, 1.0), pwp, dtl]

    def f_ppi(self, a, b):
        na, nb = self.ppi.get(a, set()) - {b}, self.ppi.get(b, set()) - {a}
        sp = len(na & nb) / (1 + min(len(na), len(nb))) if na and nb else 0.0
        pa, pb = self.pw.get(a, set()), self.pw.get(b, set())
        spw = float(bool(pa & pb))
        la, lb = self.ligs.get(a, set()), self.ligs.get(b, set())
        sl = len(la & lb) / len(la | lb) if la and lb else 0.0
        return [sp, spw, sl]

    def f_participates_in(self, p, w):
        members = self.pw_members.get(w, set()) - {p}
        nb = self.ppi.get(p, set())
        return [len(nb & members) / (1 + len(nb)) * 10 if nb else 0.0]

    def f_treats(self, c, d):
        tg = self.targets.get(c, set())
        genes = {self.p2g.get(t) for t in tg} - {None}
        dg = float(bool(genes & self.dis_genes.get(d, set())))
        others = self.treats.get(d, set()) - {c}
        an = max((v for c2, v in self.sim.get(c, {}).items() if c2 in others), default=0.0)
        return [dg, an]

    def context_size(self, predicate, s, o):
        """How much surrounding graph there is to judge the edge by. Absence of
        support only counts against an edge when there was room for support."""
        if predicate == B.INTERACTS_PHYS:
            return min(len(self.sim.get(s, {})) + len(self.targets.get(s, set())) - 1,
                       len(self.ligs.get(o, set())) - 1)
        if predicate == B.ASSOC_CONDITION:
            prot = self.g2p.get(s)
            return min(len(self.ppi.get(prot, set())) if prot else 0,
                       len(self.dis_genes.get(o, set())) - 1)
        if predicate == B.PPI:
            return min(len(self.ppi.get(s, set())), len(self.ppi.get(o, set()))) - 1
        if predicate == B.PARTICIPATES_IN:
            return min(len(self.ppi.get(s, set())), len(self.pw_members.get(o, set())) - 1)
        if predicate == B.TREATS:
            return min(len(self.targets.get(s, set())), len(self.dis_genes.get(o, set())))
        return 0

    def features(self, predicate, s, o):
        f = {B.INTERACTS_PHYS: self.f_interacts_phys, B.ASSOC_CONDITION: self.f_assoc_condition,
             B.PPI: self.f_ppi, B.PARTICIPATES_IN: self.f_participates_in,
             B.TREATS: self.f_treats}[predicate]
        return f(s, o)


def corrupt(pairs, objects, known, rng, n=None):
    """Same subject, object replaced by a uniformly random type-correct node."""
    n = n or len(pairs)
    objs = np.array(sorted(objects))
    out = []
    idx = rng.integers(len(pairs), size=4 * n)
    for i in idx:
        s = pairs[i][0]
        o = objs[rng.integers(len(objs))]
        if (s, o) not in known and s != o:
            out.append((s, o))
            if len(out) == n:
                break
    return out


VETO_MIN_AUROC = 0.90     # only a context model this good may overrule good evidence
VETO_NEG_QUANTILE = 0.75  # veto edges less plausible than 3/4 of corrupted copies...
VETO_MAX_FLAG = 0.02      # ...unless that would flag more than 2% of the graph's edges
VETO_MIN_CONTEXT = 3


def score_plausibility(edges, folds=5, seed=0, max_train=40000):
    """Out-of-fold context plausibility for every edge of the TASKS predicates.

    Returns (DataFrame indexed like `edges` with columns plausibility,
    plausibility_notes, context_size, veto_cutoff; dict of per-predicate model
    summaries; the GraphIndex).

    veto_cutoff is set only for predicates whose context model separates real
    edges from corrupted copies with out-of-fold AUROC >= VETO_MIN_AUROC. It is
    the VETO_NEG_QUANTILE quantile of the held-out corrupted copies' scores:
    an edge below it looks less plausible than most random mix-ups. The cutoff
    is set from the corrupted copies, not from the graph's own edges, so it
    does not depend on how many errors the graph holds; it is lowered if it
    would flag more than VETO_MAX_FLAG of the edges that have enough context.
    prune.py treats an edge under it as contradicted by its context, however
    good its record looks.
    """
    rng = np.random.default_rng(seed)
    gi = GraphIndex(edges)
    plaus = pd.Series(np.nan, index=edges.index)
    notes = pd.Series("", index=edges.index)
    csize = pd.Series(np.nan, index=edges.index)
    cutoff = pd.Series(np.nan, index=edges.index)
    summary = {}
    for pred in TASKS:
        sub = edges[edges["predicate"] == pred]
        if len(sub) < 20:
            continue
        pairs = list(zip(sub["subject"], sub["object"]))
        known = set(pairs) | {(o, s) for s, o in pairs}
        X = np.array([gi.features(pred, s, o) for s, o in pairs], dtype=float)
        neg = corrupt(pairs, set(sub["object"]) | (set(sub["subject"]) if pred == B.PPI else set()),
                      known, rng)
        Xn = np.array([gi.features(pred, s, o) for s, o in neg], dtype=float)

        oof = np.zeros(len(pairs))
        oof_neg = np.zeros(len(neg))
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        neg_folds = np.array_split(rng.permutation(len(neg)), folds)
        coefs = []
        for k, (tr, te) in enumerate(kf.split(X)):
            ntr = np.concatenate([neg_folds[j] for j in range(folds) if j != k])
            tr = tr if len(tr) <= max_train else rng.choice(tr, max_train, replace=False)
            ntr = ntr if len(ntr) <= max_train else rng.choice(ntr, max_train, replace=False)
            Xtr = np.vstack([X[tr], Xn[ntr]])
            ytr = np.r_[np.ones(len(tr)), np.zeros(len(ntr))]
            m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
            oof[te] = m.predict_proba(X[te])[:, 1]
            oof_neg[neg_folds[k]] = m.predict_proba(Xn[neg_folds[k]])[:, 1]
            coefs.append(m.coef_[0])
        coef = np.mean(coefs, axis=0)
        names = FEATURES[pred]
        plaus.loc[sub.index] = oof.round(4)
        csize.loc[sub.index] = [gi.context_size(pred, s_, o_) for s_, o_ in pairs]
        top = []
        for row in X:
            parts = [n for n, v in zip(names, row) if v > 0]
            top.append(",".join(parts) if parts else "no_context")
        notes.loc[sub.index] = top
        auc = float(roc_auc_score(np.r_[np.ones(len(oof)), np.zeros(len(oof_neg))], np.r_[oof, oof_neg]))
        summary[pred] = {
            "edges": len(sub), "features": names, "coef": dict(zip(names, np.round(coef, 3).tolist())),
            "share_with_any_context": float(np.mean((X > 0).any(axis=1))),
            # held-out real edges vs held-out corrupted copies
            "real_vs_corrupted_auroc_oof": auc,
        }
        enough = csize.loc[sub.index].values >= VETO_MIN_CONTEXT
        if auc >= VETO_MIN_AUROC and enough.sum() >= 100:
            cut = float(np.quantile(oof_neg, VETO_NEG_QUANTILE))
            capped = float(np.mean(oof[enough] < cut)) > VETO_MAX_FLAG
            if capped:
                cut = float(np.quantile(oof[enough], VETO_MAX_FLAG))
            cutoff.loc[sub.index] = cut
            summary[pred]["veto"] = {
                "cutoff": round(cut, 4), "capped_at_max_flag_share": capped,
                "share_of_corrupted_copies_below": round(float(np.mean(oof_neg < cut)), 4),
                "share_of_graph_edges_flagged": round(float(np.mean((oof < cut) & enough)), 4)}
    out = pd.DataFrame({"plausibility": plaus, "plausibility_notes": notes, "context_size": csize,
                        "veto_cutoff": cutoff})
    return out, summary, gi
