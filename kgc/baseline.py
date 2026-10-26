"""CPU baseline for chemical-to-protein link prediction.

Scores a pair (c, p) from the training graph only:
  - ligand overlap: Jaccard between p and the known targets of c (shared ligands)
  - PPI support: how many known targets of c physically interact with p
  - pathway support: shared Reactome pathways between p and known targets of c
  - degree of c and p
A logistic regression fits these on the validation split (whose pairs are not
in the graph), and is scored on test. A degree-only model is reported too:
on degree-matched negatives it should sit near 0.5, which shows the negatives
are doing their job.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from . import biolink as B

FEATURES = ["jac_max", "jac_mean", "ppi_count", "pw_max", "log_deg_c", "log_deg_p"]


class PairFeaturizer:
    def __init__(self, train_edges, task_predicate=B.INTERACTS_PHYS):
        task = train_edges[train_edges["predicate"] == task_predicate]
        # Only proteins with known ligands: keeps the dense P x P matrices small
        # (the full PPI neighbourhood has ~10k proteins and would not fit in RAM).
        self.prot = sorted(set(task["object"]))
        self.pidx = {p: i for i, p in enumerate(self.prot)}
        chems = sorted(set(task["subject"]))
        cidx = {c: i for i, c in enumerate(chems)}
        P, C = len(self.prot), len(chems)

        rows = [cidx[s] for s in task["subject"]]
        cols = [self.pidx[o] for o in task["object"]]
        A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(C, P))
        A.data[:] = 1.0
        self.targets = {c: A.indices[A.indptr[i]:A.indptr[i + 1]] for c, i in cidx.items()}
        self.deg_c = {c: len(v) for c, v in self.targets.items()}
        deg_p = np.asarray(A.sum(axis=0)).ravel()
        self.deg_p = deg_p

        inter = (A.T @ A).toarray()
        union = deg_p[:, None] + deg_p[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            self.jac = np.where(union > 0, inter / union, 0.0)
        np.fill_diagonal(self.jac, 0.0)

        ppi = train_edges[train_edges["predicate"] == B.PPI]
        self.ppi = np.zeros((P, P))
        for s, o in zip(ppi["subject"], ppi["object"]):
            if s in self.pidx and o in self.pidx:
                self.ppi[self.pidx[s], self.pidx[o]] = self.ppi[self.pidx[o], self.pidx[s]] = 1

        pw = train_edges[(train_edges["predicate"] == B.PARTICIPATES_IN)
                         & train_edges["subject"].isin(self.pidx)]
        pws = sorted(set(pw["object"]))
        widx = {w: i for i, w in enumerate(pws)}
        if pws:
            M = sparse.csr_matrix((np.ones(len(pw)), ([self.pidx[s] for s in pw["subject"]],
                                                      [widx[o] for o in pw["object"]])),
                                  shape=(P, len(pws)))
            M.data[:] = 1.0
            self.pw = (M @ M.T).toarray()
            np.fill_diagonal(self.pw, 0.0)
        else:
            self.pw = np.zeros((P, P))

    def transform(self, pairs: pd.DataFrame):
        out = np.zeros((len(pairs), len(FEATURES)))
        for c, grp in pairs.groupby("subject").groups.items():
            t = self.targets.get(c, np.array([], dtype=int))
            idx = np.array(list(grp))
            prots = pairs.loc[idx, "object"].map(self.pidx)
            known = prots.notna().values
            pi = prots.fillna(0).astype(int).values
            if len(t):
                J, Pp, W = self.jac[t][:, pi], self.ppi[t][:, pi], self.pw[t][:, pi]
                out[idx, 0] = J.max(axis=0)
                out[idx, 1] = J.mean(axis=0)
                out[idx, 2] = Pp.sum(axis=0)
                out[idx, 3] = W.max(axis=0)
            out[idx, 4] = np.log1p(self.deg_c.get(c, 0))
            out[idx, 5] = np.where(known, np.log1p(self.deg_p[pi]), 0.0)
            out[idx[~known], :4] = 0.0
        return out


def ranking_metrics(scores_pos, scores_neg_by_pos):
    """Sampled ranking: each positive vs its own degree-matched negatives (ties count half)."""
    ranks = []
    for sp, sn in zip(scores_pos, scores_neg_by_pos):
        sn = np.asarray(sn)
        ranks.append(1 + (sn > sp).sum() + 0.5 * (sn == sp).sum())
    ranks = np.array(ranks)
    return {"hits@10": float(np.mean(ranks <= 10)), "mrr": float(np.mean(1.0 / ranks)),
            "n": int(len(ranks))}


def run_baseline(split_dir, nodes_path, out_path=None, seed=0):
    split_dir = Path(split_dir)
    train = pd.read_csv(split_dir / "train_graph_edges.tsv", sep="\t", low_memory=False)
    val = pd.read_csv(split_dir / "val.tsv", sep="\t")
    test = pd.read_csv(split_dir / "test.tsv", sep="\t")
    rank_neg = pd.read_csv(split_dir / "test_rank_negatives.tsv", sep="\t")
    nodes = pd.read_csv(nodes_path, sep="\t", low_memory=False)
    tdl = dict(zip(nodes["id"], nodes.get("tdl", pd.Series(dtype=str))))

    feat = PairFeaturizer(train)
    Xv = feat.transform(val.reset_index(drop=True))
    yv = val["label"].values
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(Xv, yv)

    test = test.reset_index(drop=True)
    Xt = feat.transform(test)
    test["score_lr"] = model.decision_function(Xt)
    test["score_degree"] = Xt[:, 5]

    results = {"features": FEATURES, "coef": dict(zip(FEATURES, model.coef_[0].round(4).tolist()))}
    for model_name in ("lr", "degree"):
        col = f"score_{model_name}"
        res = {}
        for kind in ("degree_matched", "random"):
            sub = test[test["neg_kind"].isin(["positive", kind])]
            res[kind] = {
                "auroc": float(roc_auc_score(sub["label"], sub[col])),
                "auprc": float(average_precision_score(sub["label"], sub[col])),
            }
        results[model_name] = res

    # Sampled ranking with degree-matched negatives
    pos = test[test["label"] == 1].reset_index(drop=True)
    rank_neg = rank_neg.reset_index(drop=True)
    rank_neg["score"] = model.decision_function(feat.transform(rank_neg))
    pos_scores = model.decision_function(feat.transform(pos))
    by_pos = rank_neg.groupby("pos_id")["score"].apply(np.asarray).to_dict()
    keep = [i for i in range(len(pos)) if i in by_pos]
    results["lr"]["sampled_ranking"] = ranking_metrics(
        pos_scores[keep], [by_pos[i] for i in keep]) if keep else {}

    # Breakdown by target development level of the protein
    by_tdl = {}
    sub = test[test["neg_kind"].isin(["positive", "degree_matched"])].copy()
    sub["tdl"] = sub["object"].map(tdl)
    for level, g in sub.groupby("tdl"):
        if g["label"].nunique() == 2 and len(g) >= 20:
            by_tdl[level] = {"auroc": float(roc_auc_score(g["label"], g["score_lr"])), "n": int(len(g))}
    results["lr"]["by_tdl_degree_matched"] = by_tdl

    if out_path:
        Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
