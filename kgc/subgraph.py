"""Relevance filtering: a plain-language question or a list of IDs in, a small
subgraph out, ready for a GNN.

find_seeds        match words in a question to node IDs, symbols and names
extract_subgraph  k-hop neighbourhood around the seeds, highest-confidence edges first
to_arrays         edge_index and type arrays; to_pyg wraps them for PyTorch Geometric
"""

import re

import numpy as np
import pandas as pd

STOPWORDS = {
    "which", "what", "who", "whom", "does", "do", "did", "bind", "binds", "target", "targets",
    "with", "the", "and", "for", "are", "that", "this", "from", "have", "has", "can", "could",
    "show", "find", "list", "give", "all", "any", "about", "into", "between", "related",
    "compounds", "compound", "drugs", "drug", "proteins", "protein", "genes", "gene", "diseases",
    "disease", "pathways", "pathway", "interact", "interacts", "associated", "linked",
}
MAX_GENERIC_MATCHES = 50


def _low(series):
    return series.fillna("").astype(str).str.lower()


def find_seeds(nodes, question, max_per_token=10):
    """Return (seed_ids, matches) where matches maps each word to how it matched.

    Match tiers, best first: exact ID, exact symbol, exact name, name contains
    the word. A word that only matches many names (generic, e.g. "kinase") is
    skipped, so the seeds stay specific.
    """
    ids = nodes["id"].astype(str)
    ids_low = ids.str.lower()
    syms = _low(nodes["symbol"]) if "symbol" in nodes else pd.Series("", index=nodes.index)
    names = _low(nodes["name"]) if "name" in nodes else pd.Series("", index=nodes.index)

    seeds, matches = [], {}
    for tok in re.findall(r"[A-Za-z0-9_.:\-]+", question):
        t = tok.strip(".,:;-").lower()
        if len(t) < 2 or t in STOPWORDS:
            continue
        for tier, hit in (("id", ids_low == t),
                          ("symbol", syms == t),
                          ("name", names == t),
                          ("name_contains", names.str.contains(re.escape(t), regex=True) if len(t) >= 4
                           else pd.Series(False, index=nodes.index))):
            found = sorted(ids[hit])
            if not found:
                continue
            if tier == "name_contains" and len(found) > MAX_GENERIC_MATCHES:
                break
            matches[tok] = {"tier": tier, "ids": found[:max_per_token]}
            seeds.extend(found[:max_per_token])
            break
    return list(dict.fromkeys(seeds)), matches


def extract_subgraph(nodes, edges, seeds, hops=1, max_nodes=2000, keep_only=True):
    """k-hop neighbourhood of the seeds.

    If the edges carry a verdict, pruned edges are ignored (keep_only=True).
    Each hop adds the neighbours joined by the highest-confidence edges first,
    until max_nodes is reached. Returns (nodes, edges) of the induced subgraph.
    """
    e = edges
    if keep_only and "verdict" in e:
        e = e[e["verdict"] == "keep"]
    conf = e["confidence"].tolist() if "confidence" in e else [1.0] * len(e)

    adj = {}
    for s, o, c in zip(e["subject"], e["object"], conf):
        adj.setdefault(s, []).append((o, c))
        adj.setdefault(o, []).append((s, c))

    node_set = set(nodes["id"])
    visited = [s for s in dict.fromkeys(seeds) if s in node_set]
    seen = set(visited)
    frontier = list(visited)
    for _ in range(hops):
        best = {}
        for n in frontier:
            for nb, c in adj.get(n, []):
                if nb not in seen and c > best.get(nb, -1.0):
                    best[nb] = c
        room = max_nodes - len(seen)
        if room <= 0 or not best:
            break
        chosen = [n for n, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:room]]
        seen.update(chosen)
        visited.extend(chosen)
        frontier = chosen

    sub_edges = e[e["subject"].isin(seen) & e["object"].isin(seen)].copy()
    sub_nodes = nodes[nodes["id"].isin(seen)].copy()
    return sub_nodes, sub_edges


def to_arrays(nodes, edges):
    """Homogeneous arrays: node_ids, edge_index (2, E), edge_type (E,), type names."""
    node_ids = nodes["id"].astype(str).tolist()
    idx = {n: i for i, n in enumerate(node_ids)}
    keep = edges["subject"].isin(idx) & edges["object"].isin(idx)
    e = edges[keep]
    types = sorted(set(e["predicate"]))
    tix = {t: i for i, t in enumerate(types)}
    edge_index = np.array([[idx[s] for s in e["subject"]], [idx[o] for o in e["object"]]],
                          dtype=np.int64).reshape(2, -1)
    edge_type = np.array([tix[p] for p in e["predicate"]], dtype=np.int64)
    return {"node_ids": node_ids, "edge_index": edge_index, "edge_type": edge_type,
            "edge_types": types, "edge_ids": e["id"].tolist()}


def save_arrays(arrays, path):
    np.savez_compressed(path,
                        node_ids=np.array(arrays["node_ids"]),
                        edge_index=arrays["edge_index"],
                        edge_type=arrays["edge_type"],
                        edge_types=np.array(arrays["edge_types"]),
                        edge_ids=np.array(arrays["edge_ids"]))


def to_pyg(nodes, edges):
    """PyTorch Geometric Data object. Needs torch and torch_geometric installed."""
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover
        raise ImportError("to_pyg needs torch and torch_geometric; "
                          "use to_arrays/save_arrays without them") from exc
    a = to_arrays(nodes, edges)
    data = Data(edge_index=torch.from_numpy(a["edge_index"]),
                edge_type=torch.from_numpy(a["edge_type"]),
                num_nodes=len(a["node_ids"]))
    data.node_ids, data.edge_types, data.edge_ids = a["node_ids"], a["edge_types"], a["edge_ids"]
    return data
