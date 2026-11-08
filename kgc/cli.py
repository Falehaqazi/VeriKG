"""Command line: python -m kgc <step> ...

Steps, in order:
  pull      download one Pharos target family to data/raw/
  build     raw JSONL -> KGX nodes/edges TSV in data/kgx/
  validate  structural and Biolink checks on the KGX files
  split     leakage-safe train/val/test split with hard negatives
  baseline  CPU baseline; writes metrics JSON
  all       build, validate, split, baseline (after pull)

Steps that work on the built graph:
  prune     confidence and keep/prune verdict on every edge; writes the whole
            scored graph and the pruned graph
  explain   --subject A --object B: lineage of an edge, or the paths that
            support a link that is not recorded
  subgraph  --question "..." or --seeds ID,ID: k-hop subgraph as KGX plus arrays
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DATA = Path("data")


def _paths(name):
    kgx = DATA / "kgx"
    return {
        "raw": DATA / "raw" / f"{name}.jsonl",
        "kgx": kgx,
        "nodes": kgx / f"{name}_nodes.tsv",
        "edges": kgx / f"{name}_edges.tsv",
        "split": DATA / "split" / name,
        "results": DATA / "results" / f"{name}_baseline.json",
    }


def cmd_pull(a):
    from .pharos import pull_family
    p = _paths(a.name)
    pull_family(p["raw"], family=a.family, max_ligands=a.max_ligands, limit=a.limit)


def cmd_build(a):
    from .kgx import build_kgx
    p = _paths(a.name)
    print(json.dumps(build_kgx(p["raw"], p["kgx"], name=a.name), indent=2))


def cmd_validate(a):
    from .validate import validate
    p = _paths(a.name)
    rep = validate(pd.read_csv(p["nodes"], sep="\t", low_memory=False),
                   pd.read_csv(p["edges"], sep="\t", low_memory=False))
    (p["kgx"] / f"{a.name}_validation.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    if not rep["ok"]:
        sys.exit(1)


def cmd_split(a):
    from .split import make_split
    p = _paths(a.name)
    rep = make_split(pd.read_csv(p["nodes"], sep="\t", low_memory=False),
                     pd.read_csv(p["edges"], sep="\t", low_memory=False),
                     p["split"], seed=a.seed)
    print(json.dumps(rep, indent=2))


def cmd_baseline(a):
    from .baseline import run_baseline
    p = _paths(a.name)
    p["results"].parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run_baseline(p["split"], p["nodes"], p["results"], seed=a.seed), indent=2))


def cmd_all(a):
    for f in (cmd_build, cmd_validate, cmd_split, cmd_baseline):
        f(a)


def _graph(a):
    """Nodes and edges, preferring the scored edges when `prune` has been run."""
    p = _paths(a.name)
    scored = p["kgx"] / f"{a.name}_edges_scored.tsv"
    edges = pd.read_csv(scored if scored.exists() else p["edges"], sep="\t", low_memory=False)
    return pd.read_csv(p["nodes"], sep="\t", low_memory=False), edges


def cmd_prune(a):
    from .prune import run_prune
    print(json.dumps(run_prune(_paths(a.name)["kgx"], a.name, threshold=a.threshold), indent=2))


def cmd_explain(a):
    from .explain import explain_edge, explain_prediction
    if not (a.subject and a.object):
        sys.exit("explain needs --subject and --object")
    nodes, edges = _graph(a)
    res = explain_edge(nodes, edges, a.subject, a.object)
    if not res["found"]:
        res = explain_prediction(edges, a.subject, a.object, nodes=nodes)
    print(res["text"])


def cmd_subgraph(a):
    from .subgraph import extract_subgraph, find_seeds, save_arrays, to_arrays
    nodes, edges = _graph(a)
    if a.seeds:
        seeds = [s.strip() for s in a.seeds.split(",") if s.strip()]
    elif a.question:
        seeds, matches = find_seeds(nodes, a.question)
        print(json.dumps(matches, indent=2))
    else:
        sys.exit("subgraph needs --question or --seeds")
    if not seeds:
        sys.exit("no node matched; try an ID such as UniProtKB:P00533 or a gene symbol")
    sub_nodes, sub_edges = extract_subgraph(nodes, edges, seeds, hops=a.hops, max_nodes=a.max_nodes)
    out = Path(a.out) if a.out else DATA / "subgraph" / a.name
    out.mkdir(parents=True, exist_ok=True)
    sub_nodes.to_csv(out / "nodes.tsv", sep="\t", index=False)
    sub_edges.to_csv(out / "edges.tsv", sep="\t", index=False)
    save_arrays(to_arrays(sub_nodes, sub_edges), out / "arrays.npz")
    print(f"{len(seeds)} seeds -> {len(sub_nodes)} nodes, {len(sub_edges)} edges in {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kgc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["pull", "build", "validate", "split", "baseline", "all",
                                     "prune", "explain", "subgraph"])
    ap.add_argument("--threshold", type=float, default=0.45, help="prune: keep edges at or above this confidence")
    ap.add_argument("--subject", help="explain: subject or chemical ID")
    ap.add_argument("--object", help="explain: object or protein ID")
    ap.add_argument("--question", help="subgraph: plain-language question")
    ap.add_argument("--seeds", help="subgraph: comma-separated node IDs")
    ap.add_argument("--hops", type=int, default=1, help="subgraph: neighbourhood radius")
    ap.add_argument("--max-nodes", type=int, default=2000, help="subgraph: node cap")
    ap.add_argument("--out", help="subgraph: output directory")
    ap.add_argument("--name", default="pharos_kinase", help="dataset name used in file paths")
    ap.add_argument("--family", default="Kinase", help="Pharos target family to pull")
    ap.add_argument("--max-ligands", type=int, default=None,
                    help="cap ligands per target and per type (drug/non-drug); default: all")
    ap.add_argument("--limit", type=int, default=None, help="only pull the first N targets (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    {"pull": cmd_pull, "build": cmd_build, "validate": cmd_validate, "split": cmd_split,
     "baseline": cmd_baseline, "all": cmd_all, "prune": cmd_prune, "explain": cmd_explain,
     "subgraph": cmd_subgraph}[a.step](a)


if __name__ == "__main__":
    main()
