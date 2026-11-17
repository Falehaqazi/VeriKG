"""Where the graph is strong, where it is empty, and where prediction stops working.

Pharos exists to illuminate understudied targets, so it labels every protein with
a target development level: Tclin (an approved drug acts on it), Tchem (ligands
are known), Tbio (biology known, no ligand), Tdark (little known at all). That
label is carried through onto our protein nodes, which lets us ask two questions
the IDG programme actually cares about.

1. coverage_by_tdl: how is ligand evidence distributed, and does pruning fall
   evenly across the four levels? An agent that quietly deleted the understudied
   end of the graph would look good on aggregate metrics while doing harm.

2. prediction_floor: can the link predictor propose ligands for targets that have
   none? Run it and the answer is no, in a specific and measurable way: for
   targets with no ligand neighbourhood the ranking collapses onto a handful of
   compounds and the scores fall to near zero, two orders below the scores for
   targets the graph knows. The scores are therefore honest - the model does not
   invent confidence where it has no evidence - but the ranking underneath them
   is not usable, which is why the app refuses to show it.

`python -m kgc coverage` writes data/results/<name>_coverage.json.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import biolink as B

TDL_ORDER = ["Tclin", "Tchem", "Tbio", "Tdark"]


def coverage_by_tdl(nodes, edges):
    """Ligand evidence and pruning rate per target development level."""
    tdl = dict(zip(nodes["id"], nodes.get("tdl")))
    kept = edges[edges["verdict"] != "prune"] if "verdict" in edges else edges
    binds = edges[edges["predicate"] == B.INTERACTS_PHYS]
    kept_binds = kept[kept["predicate"] == B.INTERACTS_PHYS]
    with_ligand = set(kept_binds["object"])
    per_target = kept_binds.groupby("object").size()

    rows = []
    prot = nodes[nodes["category"] == B.PROTEIN]
    for level in TDL_ORDER:
        ids = set(prot.loc[prot["tdl"] == level, "id"])
        if not ids:
            continue
        sub = binds[binds["object"].isin(ids)]
        pruned = int((sub["verdict"] == "prune").sum()) if "verdict" in sub else 0
        counts = per_target.reindex(sorted(ids)).fillna(0)
        rows.append({
            "tdl": level,
            "proteins": len(ids),
            "with_a_ligand": len(ids & with_ligand),
            "without_any_ligand": len(ids - with_ligand),
            "ligand_edges": int(len(sub)),
            "pruned": pruned,
            "prune_rate": round(pruned / len(sub), 4) if len(sub) else None,
            "median_ligands_per_target_that_has_one": float(counts[counts > 0].median())
            if (counts > 0).any() else 0.0,
        })
    return rows


def prediction_floor(nodes, edges, predictor, sample=300, seed=0):
    """Does the predictor say anything target-specific about ligand-less targets?

    Scores the top compound for a sample of ligand-less Tdark targets and, for
    contrast, for Tclin targets the graph knows well. Returns the share of the
    Tdark sample taken by the single most frequent top-1 compound, how many
    distinct compounds top the sample at all, and the two score distributions.
    """
    tdl = dict(zip(nodes["id"], nodes.get("tdl")))
    kept = edges[edges["verdict"] != "prune"] if "verdict" in edges else edges
    binds = kept[kept["predicate"] == B.INTERACTS_PHYS]
    with_ligand = set(binds["object"])
    in_model = {str(p) for p in predictor.prot_ids}
    rng = np.random.default_rng(seed)

    def pick(level, want_ligand):
        ids = [p for p in sorted(in_model)
               if tdl.get(p) == level and ((p in with_ligand) == want_ligand)]
        if len(ids) > sample:
            ids = [ids[i] for i in rng.permutation(len(ids))[:sample]]
        return ids

    def top1(ids):
        out = []
        for p in ids:
            t = predictor.ligands_for(p, k=1)
            if t:
                out.append((p, str(t[0][0]), float(t[0][1])))
        return out

    dark = top1(pick("Tdark", False))
    clin = top1(pick("Tclin", True))
    if not dark:
        return {"note": "no ligand-less Tdark targets in the model"}
    counts = Counter(c for _, c, _ in dark)
    top_compound, top_n = counts.most_common(1)[0]
    d_scores = np.array([s for _, _, s in dark])
    c_scores = np.array([s for _, _, s in clin]) if clin else np.array([np.nan])
    return {
        "tdark_sampled": len(dark),
        "distinct_top1_compounds": len(counts),
        "most_frequent_top1_share": round(top_n / len(dark), 4),
        "most_frequent_top1_compound": top_compound,
        "tdark_median_top1_score": round(float(np.median(d_scores)), 4),
        "tdark_share_above_0.5": round(float((d_scores > 0.5).mean()), 4),
        "tclin_with_ligands_median_top1_score": round(float(np.median(c_scores)), 4),
        "verdict": ("the ranking for ligand-less targets is degenerate; the scores "
                    "collapse toward zero, so the model signals its own ignorance"),
    }


def run_coverage(data_dir, name="pharos_kinase", log=print):
    data_dir = Path(data_dir)
    kgx = data_dir / "kgx"
    nodes = pd.read_csv(kgx / f"{name}_nodes.tsv", sep="\t", low_memory=False)
    edges_path = kgx / f"{name}_edges_scored.tsv"
    if not edges_path.exists():
        edges_path = kgx / f"{name}_edges.tsv"
    edges = pd.read_csv(edges_path, sep="\t", low_memory=False)

    report = {"by_tdl": coverage_by_tdl(nodes, edges)}
    pred_path = data_dir / "results" / f"{name}_predictor.npz"
    if pred_path.exists():
        from .predictor import Predictor
        report["prediction_floor"] = prediction_floor(nodes, edges, Predictor(pred_path))
    else:
        log("no predictor found; run `python -m kgc predict` for the prediction floor")

    out = data_dir / "results" / f"{name}_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in report["by_tdl"]:
        log(f"  {r['tdl']:6s} {r['proteins']:6d} proteins, {r['without_any_ligand']:6d} with no ligand, "
            f"{r['ligand_edges']:6d} ligand edges, prune rate {r['prune_rate']}")
    if "prediction_floor" in report:
        p = report["prediction_floor"]
        log(f"  prediction floor: {p['distinct_top1_compounds']} distinct top-1 compounds across "
            f"{p['tdark_sampled']} ligand-less Tdark targets; median score {p['tdark_median_top1_score']} "
            f"against {p['tclin_with_ligands_median_top1_score']} for Tclin targets with ligands")
    return report


def supported_for_prediction(edges, protein):
    """True when the graph holds at least one kept ligand edge for this protein,
    which is the condition under which the ranked prediction list means anything."""
    kept = edges[edges["verdict"] != "prune"] if "verdict" in edges else edges
    return bool((kept[(kept["predicate"] == B.INTERACTS_PHYS)
                      & (kept["object"] == protein)]).shape[0])
