"""GNN link prediction (PyTorch Geometric): does the agent's graph beat Pharos?

Same model, same split, same negatives; only the graph changes.

  pharos          the Pharos graph as pulled: ligand activities, PPIs, disease
                  associations, pathways. Chemicals are learned embeddings.
  pharos_fp       + Morgan fingerprints as chemical input features
  verikg          + structure-similarity edges between chemicals (the
                  enrichment step), fingerprints as features
  verikg_pruned   verikg after the pruning agent, with pruning re-run on the
                  training graph only so no test edge influences what is cut

Model: two-layer heterogeneous GraphSAGE (one SAGEConv per relation and
direction), hidden size 128, and an MLP decoder on [h_c * h_p, |h_c - h_p|].
Training holds out a fresh 30% of the supervision edges from message passing
every epoch, so the model cannot score a pair by seeing it in its own input.
Negatives in training are drawn in proportion to protein degree, the same way
the evaluation negatives are, so degree alone is no shortcut.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import biolink as B

NODE_TYPES = {
    B.CHEMICAL: "chemical", B.DRUG: "chemical", B.PROTEIN: "protein", B.GENE: "gene",
    B.DISEASE: "disease", B.PATHWAY: "pathway",
}
REL_NAME = {
    B.INTERACTS_PHYS: "binds", B.PPI: "interacts", B.ASSOC_CONDITION: "associated",
    B.PARTICIPATES_IN: "in_pathway", B.TRANSLATES_TO: "translates", B.TREATS: "treats",
    B.SIMILAR: "similar",
}
TASK = ("chemical", "binds", "protein")


def _torch():
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
    return torch, F, nn, HeteroData, HeteroConv, SAGEConv


def index_nodes(nodes):
    """Per-type contiguous indices: {type: {curie: i}}."""
    idx = {}
    for nid, cat in zip(nodes["id"], nodes["category"]):
        t = NODE_TYPES.get(cat)
        if t:
            d = idx.setdefault(t, {})
            d.setdefault(nid, len(d))
    return idx


def build_hetero(nodes, edges, idx, fingerprints=None, weight_col=None):
    """HeteroData with forward and reverse edge types for every predicate.

    fingerprints: optional (ids, bits) for chemical features. Pruned edges
    (verdict == 'prune') are dropped if a verdict column is present.
    """
    torch, _, _, HeteroData, _, _ = _torch()
    data = HeteroData()
    for t, d in idx.items():
        data[t].num_nodes = len(d)
        data[t].node_ids = list(d)
    if fingerprints is not None:
        ids, bits = fingerprints
        x = np.zeros((len(idx["chemical"]), bits.shape[1]), dtype=np.float32)
        pos = {n: i for i, n in enumerate(ids)}
        for n, i in idx["chemical"].items():
            j = pos.get(n)
            if j is not None:
                x[i] = bits[j]
        data["chemical"].x = torch.from_numpy(x)
    e = edges
    if "verdict" in e:
        e = e[e["verdict"] != "prune"]
    cat = dict(zip(nodes["id"], nodes["category"]))
    for pred, grp in e.groupby("predicate"):
        rel = REL_NAME.get(pred)
        if rel is None:
            continue
        st = grp["subject"].map(lambda x: NODE_TYPES.get(cat.get(x)))
        ot = grp["object"].map(lambda x: NODE_TYPES.get(cat.get(x)))
        for (a, b), g in grp.groupby([st, ot]):
            si = g["subject"].map(idx[a]).values
            oi = g["object"].map(idx[b]).values
            ei = torch.tensor(np.vstack([si, oi]), dtype=torch.long)
            data[a, rel, b].edge_index = ei
            if a == b:  # symmetric relation within one type: add the other direction
                data[a, rel, b].edge_index = torch.cat([ei, ei.flip(0)], dim=1)
            else:
                data[b, "rev_" + rel, a].edge_index = ei.flip(0)
    return data


class LinkModel:
    """Built lazily so the module imports without torch."""

    def __new__(cls, data, hidden=128, layers=2, dropout=0.2, use_fp=True):
        torch, F, nn, _, HeteroConv, SAGEConv = _torch()

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.use_fp = use_fp and "x" in data["chemical"]
                self.emb = nn.ModuleDict()
                for t in data.node_types:
                    if t == "chemical" and self.use_fp:
                        continue
                    self.emb[t] = nn.Embedding(data[t].num_nodes, hidden)
                    nn.init.normal_(self.emb[t].weight, std=0.1)
                if self.use_fp:
                    self.fp = nn.Sequential(nn.Linear(data["chemical"].x.shape[1], hidden), nn.ReLU(),
                                            nn.Linear(hidden, hidden))
                self.convs = nn.ModuleList([
                    HeteroConv({et: SAGEConv(hidden, hidden) for et in data.edge_types}, aggr="sum")
                    for _ in range(layers)])
                self.dec = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                self.dropout = dropout

            def encode(self, data, edge_index_dict):
                h = {t: self.emb[t].weight for t in self.emb}
                if self.use_fp:
                    h["chemical"] = self.fp(data["chemical"].x)
                for i, conv in enumerate(self.convs):
                    out = conv(h, edge_index_dict)
                    # keep types that receive no messages; residual connection
                    out = {t: out.get(t, 0) + h[t] for t in h}
                    if i < len(self.convs) - 1:
                        out = {t: F.dropout(F.relu(v), self.dropout, self.training) for t, v in out.items()}
                    h = out
                return h

            def decode(self, h, c, p):
                a, b = h["chemical"][c], h["protein"][p]
                return self.dec(torch.cat([a * b, (a - b).abs()], dim=-1)).squeeze(-1)

        return _M()


def _pairs_to_index(df, idx):
    c = df["subject"].map(idx["chemical"])
    p = df["object"].map(idx["protein"])
    ok = c.notna() & p.notna()
    return c[ok].astype(int).values, p[ok].astype(int).values, ok.values


def train_eval(nodes, train_edges, val, test, rank_neg, fingerprints=None, epochs=120,
               hidden=128, lr=0.005, seed=0, holdout=0.3, neg_ratio=2, log=print,
               patience=4, eval_every=5):
    """Train on the training graph, early-stop on validation AUROC (degree-matched)."""
    torch, F, _, _, _, _ = _torch()
    from sklearn.metrics import average_precision_score, roc_auc_score
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = index_nodes(nodes)
    data = build_hetero(nodes, train_edges, idx, fingerprints)
    model = LinkModel(data, hidden=hidden, use_fp=fingerprints is not None)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    sup = data[TASK].edge_index
    n_sup = sup.shape[1]
    deg = torch.bincount(sup[1], minlength=data["protein"].num_nodes).float() + 0.5
    deg = deg / deg.sum()
    known = set(zip(sup[0].tolist(), sup[1].tolist()))

    def message_graph(mask):
        eid = dict(data.edge_index_dict)
        eid[TASK] = sup[:, mask]
        eid[("protein", "rev_binds", "chemical")] = sup[:, mask].flip(0)
        return eid

    full_graph = data.edge_index_dict

    def scores(df, h):
        c, p, ok = _pairs_to_index(df, idx)
        s = np.full(len(df), np.nan)
        with torch.no_grad():
            s[ok] = model.decode(h, torch.tensor(c), torch.tensor(p)).numpy()
        return s

    def evaluate(df, h):
        s = scores(df, h)
        out = {}
        for kind in ("degree_matched", "random"):
            sub = df["neg_kind"].isin(["positive", kind]).values & ~np.isnan(s)
            y = df["label"].values[sub]
            out[kind] = {"auroc": float(roc_auc_score(y, s[sub])),
                         "auprc": float(average_precision_score(y, s[sub]))}
        return out, s

    best, best_state, bad, hist = -1, None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_sup)
        k = int(holdout * n_sup)
        tgt, keep = perm[:k], torch.ones(n_sup, dtype=torch.bool)
        keep[tgt] = False
        h = model.encode(data, message_graph(keep))
        pc, pp = sup[0, tgt], sup[1, tgt]
        nc = pc.repeat(neg_ratio)
        npr = torch.multinomial(deg, len(nc), replacement=True)
        pos = model.decode(h, pc, pp)
        neg = model.decode(h, nc, npr)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                h = model.encode(data, full_graph)
            res, _ = evaluate(val, h)
            auc = res["degree_matched"]["auroc"]
            hist.append({"epoch": ep, "loss": float(loss), "val_auroc_dm": auc})
            log(f"  epoch {ep:4d} loss {float(loss):.4f} val AUROC (degree-matched) {auc:.4f}"
                f"  [{time.time() - t0:.0f}s]")
            if auc > best + 1e-4:
                best, bad = auc, 0
                best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        h = model.encode(data, full_graph)
    test_res, test_scores = evaluate(test, h)

    # seen vs unseen chemicals (no training edge)
    seen = set(train_edges.loc[train_edges["predicate"] == B.INTERACTS_PHYS, "subject"])
    for name, mask in (("seen_chemical", test["subject"].isin(seen).values),
                       ("unseen_chemical", ~test["subject"].isin(seen).values)):
        sub = mask & test["neg_kind"].isin(["positive", "degree_matched"]).values & ~np.isnan(test_scores)
        y = test["label"].values[sub]
        if len(set(y)) == 2:
            test_res[name] = {"auroc": float(roc_auc_score(y, test_scores[sub])), "n": int(sub.sum())}

    # sampled ranking: each positive vs its 99 degree-matched negatives
    pos = test[test["label"] == 1].reset_index(drop=True)
    ps = scores(pos, h)
    rn = rank_neg.copy()
    rn["score"] = scores(rn, h)
    by = rn.groupby("pos_id")["score"].apply(np.asarray).to_dict()
    ranks = []
    for i, sp in enumerate(ps):
        if i in by and not np.isnan(sp):
            sn = by[i][~np.isnan(by[i])]
            ranks.append(1 + (sn > sp).sum() + 0.5 * (sn == sp).sum())
    ranks = np.array(ranks)
    test_res["sampled_ranking"] = {"hits@10": float(np.mean(ranks <= 10)),
                                   "mrr": float(np.mean(1 / ranks)), "n": int(len(ranks))}
    return {"test": test_res, "best_val_auroc_dm": best, "history": hist,
            "train_seconds": round(time.time() - t0, 1),
            "message_edges": {"/".join(k): int(v.shape[1]) for k, v in data.edge_index_dict.items()}}, \
        (model, data, idx)


def run_comparison(data_dir, name="pharos_kinase", configs=None, epochs=120, seed=0, log=print):
    """Train every config on the same split; write data/results/<name>_gnn.json."""
    from .enrich import load_fingerprints
    from .split import make_split
    data_dir = Path(data_dir)
    kgx = data_dir / "kgx"
    nodes = pd.read_csv(kgx / f"{name}_nodes.tsv", sep="\t", low_memory=False)
    base_edges = pd.read_csv(kgx / f"{name}_edges.tsv", sep="\t", low_memory=False)
    enriched_path = kgx / f"{name}_enriched_edges.tsv"
    enriched = pd.read_csv(enriched_path, sep="\t", low_memory=False) if enriched_path.exists() else None
    fps = load_fingerprints(kgx / f"{name}_fingerprints.npz") \
        if (kgx / f"{name}_fingerprints.npz").exists() else None

    split_dir = data_dir / "split" / name
    if not (split_dir / "train_graph_edges.tsv").exists():
        make_split(nodes, base_edges, split_dir, seed=seed)
    val = pd.read_csv(split_dir / "val.tsv", sep="\t")
    test = pd.read_csv(split_dir / "test.tsv", sep="\t")
    rank_neg = pd.read_csv(split_dir / "test_rank_negatives.tsv", sep="\t")
    held = set(zip(val.loc[val.label == 1, "subject"], val.loc[val.label == 1, "object"])) | \
        set(zip(test.loc[test.label == 1, "subject"], test.loc[test.label == 1, "object"]))

    def train_graph(edges):
        from .split import audit_leakage
        g, _ = audit_leakage(edges, held)
        return g

    configs = configs or ["pharos", "pharos_fp", "verikg", "verikg_pruned"]
    results = {}
    for cfg in configs:
        log(f"[{cfg}]")
        if cfg == "pharos":
            tg, f = train_graph(base_edges), None
        elif cfg == "pharos_fp":
            tg, f = train_graph(base_edges), fps
        elif cfg == "verikg":
            tg, f = train_graph(enriched if enriched is not None else base_edges), fps
        elif cfg == "verikg_pruned":
            from .plausibility import score_plausibility
            from .prune import score_edges
            tg = train_graph(enriched if enriched is not None else base_edges)
            ctx, _, _ = score_plausibility(tg, seed=seed)
            tg = score_edges(tg, context=ctx)
            f = fps
        else:
            raise ValueError(cfg)
        res, _ = train_eval(nodes, tg, val, test, rank_neg, fingerprints=f, epochs=epochs,
                            seed=seed, log=log)
        if cfg == "verikg_pruned":
            res["edges_pruned_from_training_graph"] = int((tg["verdict"] == "prune").sum())
        results[cfg] = res
        log(json.dumps(res["test"], indent=1))
    out = data_dir / "results" / f"{name}_gnn.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def fit_full(nodes, edges, fingerprints=None, epochs=100, hidden=128, lr=0.005, seed=0,
             holdout=0.3, neg_ratio=2, log=print):
    """Train on every edge (no held-out pairs) for a fixed number of epochs, for
    the predictions served in the app. Use the best epoch from run_comparison."""
    torch, F, _, _, _, _ = _torch()
    torch.manual_seed(seed)
    idx = index_nodes(nodes)
    data = build_hetero(nodes, edges, idx, fingerprints)
    model = LinkModel(data, hidden=hidden, use_fp=fingerprints is not None)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sup = data[TASK].edge_index
    n_sup = sup.shape[1]
    deg = torch.bincount(sup[1], minlength=data["protein"].num_nodes).float() + 0.5
    deg = deg / deg.sum()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_sup)
        k = int(holdout * n_sup)
        keep = torch.ones(n_sup, dtype=torch.bool)
        keep[perm[:k]] = False
        eid = dict(data.edge_index_dict)
        eid[TASK] = sup[:, keep]
        eid[("protein", "rev_binds", "chemical")] = sup[:, keep].flip(0)
        h = model.encode(data, eid)
        pc, pp = sup[0, perm[:k]], sup[1, perm[:k]]
        nc = pc.repeat(neg_ratio)
        npr = torch.multinomial(deg, len(nc), replacement=True)
        pos, neg = model.decode(h, pc, pp), model.decode(h, nc, npr)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            log(f"  full fit epoch {ep} loss {float(loss):.4f}")
    model.eval()
    return model, data, idx


def export_predictor(model, data, idx, path):
    """Save final embeddings and decoder weights so the app can score any
    compound-protein pair with numpy alone (no torch at serving time)."""
    torch = _torch()[0]
    with torch.no_grad():
        h = model.encode(data, data.edge_index_dict)
    lin1, lin2 = model.dec[0], model.dec[2]
    np.savez_compressed(
        path,
        chem_ids=np.array(list(idx["chemical"])), prot_ids=np.array(list(idx["protein"])),
        h_chem=h["chemical"].numpy().astype(np.float32), h_prot=h["protein"].numpy().astype(np.float32),
        w1=lin1.weight.detach().numpy(), b1=lin1.bias.detach().numpy(),
        w2=lin2.weight.detach().numpy(), b2=lin2.bias.detach().numpy())


def export_pyg(nodes, edges, path, fingerprints=None):
    """Whole scored graph as a PyG HeteroData file (torch.save), with node ID
    lists, edge confidence and a keep mask on every edge type, so a subgraph
    can be cut with HeteroData.subgraph or edge_subgraph."""
    torch = _torch()[0]
    idx = index_nodes(nodes)
    e = edges.copy()
    e["_keep"] = (e["verdict"] != "prune") if "verdict" in e else True
    tmp = e.drop(columns=["verdict"]) if "verdict" in e else e
    data = build_hetero(nodes, tmp, idx, fingerprints)
    cat = dict(zip(nodes["id"], nodes["category"]))
    for pred, grp in e.groupby("predicate"):
        rel = REL_NAME.get(pred)
        if rel is None:
            continue
        st = grp["subject"].map(lambda x: NODE_TYPES.get(cat.get(x)))
        ot = grp["object"].map(lambda x: NODE_TYPES.get(cat.get(x)))
        for (a, b), g in grp.groupby([st, ot]):
            conf = torch.tensor(g["confidence"].fillna(1.0).values if "confidence" in g else np.ones(len(g)),
                                dtype=torch.float)
            keep = torch.tensor(g["_keep"].values, dtype=torch.bool)
            store = data[a, rel, b]
            if a == b:
                conf, keep = torch.cat([conf, conf]), torch.cat([keep, keep])
            else:
                data[b, "rev_" + rel, a].edge_confidence = conf
                data[b, "rev_" + rel, a].keep = keep
            store.edge_confidence = conf
            store.keep = keep
            store.edge_ids = g["id"].tolist() * (2 if a == b else 1)
    torch.save(data, path)
    return data
