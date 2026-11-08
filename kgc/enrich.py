"""Enrichment: context the agent adds on top of the source graph.

Structure similarity (offline, no network)
  Every small molecule with a SMILES gets a Morgan fingerprint (radius 2,
  2048 bits). Each molecule is linked to its nearest neighbours by Tanimoto
  similarity with `biolink:chemically_similar_to`. This is the context that
  lets a model reason about a compound it has never seen bind anything:
  62% of held-out test pairs involve a chemical with no training edge, and for
  those the source graph alone carries no signal at all.

External sources (network)
  See kgc/sources/ for the Open Targets adapter, which corroborates or
  contradicts gene-disease edges with an independent association score.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import biolink as B
from .kgx import edge_id

FP_BITS = 2048
FP_RADIUS = 2
FP_NAME = f"morgan_r{FP_RADIUS}_{FP_BITS}"


def _rdkit():
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:  # pragma: no cover
        raise ImportError("structure enrichment needs RDKit: pip install rdkit") from exc
    RDLogger.DisableLog("rdApp.*")
    return Chem, DataStructs, rdFingerprintGenerator


def fingerprints(nodes):
    """Morgan fingerprints for every chemical node with a parsable SMILES.

    Returns (ids, rdkit_fps, packed_bits) where packed_bits is a uint8 array of
    shape (n, FP_BITS / 8), the form stored on disk and used as GNN features.
    """
    Chem, DataStructs, gen_mod = _rdkit()
    gen = gen_mod.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    chem = nodes[nodes["category"].isin(B.CHEMICALS)]
    smiles = chem["smiles"] if "smiles" in chem else pd.Series(dtype=str)
    ids, fps, rows = [], [], []
    for nid, smi in zip(chem["id"], smiles):
        if not isinstance(smi, str) or not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((FP_BITS,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        ids.append(nid)
        fps.append(fp)
        rows.append(np.packbits(arr))
    packed = np.vstack(rows) if rows else np.zeros((0, FP_BITS // 8), dtype=np.uint8)
    return ids, fps, packed


def _sim_chunk(args):
    fps, start, stop, threshold = args
    _, DataStructs, _ = _rdkit()
    I, J, S = [], [], []
    for i in range(start, stop):
        if i + 1 >= len(fps):
            break
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]), dtype=np.float32)
        hit = np.flatnonzero(sims >= threshold)
        if len(hit):
            I.append(np.full(len(hit), i, dtype=np.int32))
            J.append((hit + i + 1).astype(np.int32))
            S.append(sims[hit])
    if not I:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32))
    return np.concatenate(I), np.concatenate(J), np.concatenate(S)


def nearest_neighbours(fps, k=10, threshold=0.6, workers=None, log=None):
    """For each fingerprint, its top-k neighbours with Tanimoto >= threshold.

    Exact search over the upper triangle (each pair compared once) with RDKit's
    bulk Tanimoto, split across processes. A pair is kept if it is in the
    top-k of either molecule. Returns a list of (i, j, similarity), i < j.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor
    n = len(fps)
    workers = workers or max(1, (os.cpu_count() or 1))
    # chunks of equal work: row i costs n - i comparisons
    bounds, target, acc, start = [], n * (n - 1) / 2 / (workers * 8), 0.0, 0
    for i in range(n):
        acc += n - i
        if acc >= target:
            bounds.append((start, i + 1)); start, acc = i + 1, 0.0
    if start < n:
        bounds.append((start, n))
    parts = []
    if workers > 1 and n > 2000:
        with ProcessPoolExecutor(workers) as ex:
            for t, part in enumerate(ex.map(_sim_chunk, [(fps, a, b, threshold) for a, b in bounds])):
                parts.append(part)
                if log:
                    log(f"  similarity: chunk {t + 1}/{len(bounds)}")
    else:
        parts = [_sim_chunk((fps, a, b, threshold)) for a, b in bounds]
    I = np.concatenate([p[0] for p in parts]); J = np.concatenate([p[1] for p in parts])
    S = np.concatenate([p[2] for p in parts])
    if not len(I):
        return []
    df = pd.DataFrame({"a": np.r_[I, J], "b": np.r_[J, I], "s": np.r_[S, S]})
    df = df.sort_values(["a", "s"], ascending=[True, False])
    df = df[df.groupby("a").cumcount() < k]
    lo, hi = np.minimum(df["a"], df["b"]), np.maximum(df["a"], df["b"])
    keep = pd.DataFrame({"i": lo, "j": hi, "s": df["s"]}).drop_duplicates(["i", "j"])
    return list(zip(keep["i"].astype(int), keep["j"].astype(int), keep["s"].astype(float)))


def similarity_edges(ids, pairs, source_version, run_id):
    rows = []
    for i, j, s in pairs:
        a, b = sorted((ids[i], ids[j]))
        rows.append({
            "id": edge_id(a, B.SIMILAR, b), "subject": a, "predicate": B.SIMILAR, "object": b,
            "relation": B.PREDICATE_IRI[B.SIMILAR],
            "primary_knowledge_source": B.SELF, "aggregator_knowledge_source": "",
            "knowledge_level": "observation", "agent_type": "data_analysis_pipeline",
            "publications": "", "supporting_sources": B.SELF, "evidence_count": 1,
            "similarity_score": round(s, 4), "similarity_method": f"tanimoto:{FP_NAME}",
            "source_version": source_version, "ingest_run": run_id,
        })
    return pd.DataFrame(rows)


def run_enrich(kgx_dir, name="pharos_kinase", k=10, threshold=0.6, log=print):
    """Write <name>_enriched_{nodes,edges}.tsv and <name>_fingerprints.npz."""
    kgx_dir = Path(kgx_dir)
    nodes = pd.read_csv(kgx_dir / f"{name}_nodes.tsv", sep="\t", low_memory=False)
    edges = pd.read_csv(kgx_dir / f"{name}_edges.tsv", sep="\t", low_memory=False)
    version = str(edges["source_version"].iloc[0]) if len(edges) else "unknown"
    run_id = str(edges["ingest_run"].iloc[0]) if len(edges) else "unknown"

    log(f"fingerprints for {int(nodes['category'].isin(B.CHEMICALS).sum())} chemicals")
    ids, fps, packed = fingerprints(nodes)
    np.savez_compressed(kgx_dir / f"{name}_fingerprints.npz", ids=np.array(ids), bits=packed,
                        method=np.array(FP_NAME))
    log(f"nearest neighbours (k={k}, Tanimoto >= {threshold}) over {len(ids)} molecules")
    pairs = nearest_neighbours(fps, k=k, threshold=threshold, log=log)
    sim = similarity_edges(ids, pairs, version, run_id)

    enriched = pd.concat([edges, sim], ignore_index=True)
    nodes.to_csv(kgx_dir / f"{name}_enriched_nodes.tsv", sep="\t", index=False)
    enriched.to_csv(kgx_dir / f"{name}_enriched_edges.tsv", sep="\t", index=False)
    report = {
        "fingerprint": FP_NAME, "molecules_with_structure": len(ids),
        "chemicals_without_structure": int(nodes["category"].isin(B.CHEMICALS).sum() - len(ids)),
        "similarity": {"k": k, "threshold": threshold, "edges": len(sim),
                       "molecules_with_a_neighbour": int(len(set(sim["subject"]) | set(sim["object"])))
                       if len(sim) else 0},
        "edges_before": len(edges), "edges_after": len(enriched),
    }
    (kgx_dir / f"{name}_enrich_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_fingerprints(path):
    z = np.load(path, allow_pickle=False)
    return list(z["ids"]), np.unpackbits(z["bits"], axis=1)
