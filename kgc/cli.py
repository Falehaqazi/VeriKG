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

Agent steps (after build):
  enrich       Morgan fingerprints and structure-similarity edges (needs RDKit)
  opentargets  fetch Open Targets scores for every gene-disease edge (network);
               the next `prune` uses them
  benchmark    plant wrong edges in the graph and measure how many pruning finds
  split-scaffold  a second split where no test compound shares a ring scaffold
               with any training compound
  gnn          GNN link prediction on each graph: --configs pharos,pharos_fp,...
               --split random|scaffold (needs PyTorch Geometric)
  predict      train the GNN on the whole graph; export scores for the app and
               the scored graph as PyTorch Geometric HeteroData
  ask          --question "...": plain-language question, answer and Cypher
  table        --file data.csv: any relationship table through the build agent
  update       re-pull when Pharos has a new release and write a changelog
  app          start the web app (Streamlit)
  fetch-data   download the prebuilt graph and results instead of building them
"""

import argparse
import json
import os
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


def cmd_enrich(a):
    from .enrich import run_enrich
    print(json.dumps(run_enrich(_paths(a.name)["kgx"], a.name, k=a.k, threshold=a.sim_threshold), indent=2))


def cmd_opentargets(a):
    from .opentargets import fetch_associations
    print(json.dumps(fetch_associations(_paths(a.name)["kgx"], a.name), indent=2))


def cmd_benchmark(a):
    from .benchmark import run_benchmark_files
    print(json.dumps(run_benchmark_files(_paths(a.name)["kgx"], a.name, rate=a.rate, seed=a.seed), indent=2))


def cmd_split_scaffold(a):
    from .split import make_split, scaffold_groups
    p = _paths(a.name)
    nodes = pd.read_csv(p["nodes"], sep="\t", low_memory=False)
    edges = pd.read_csv(p["edges"], sep="\t", low_memory=False)
    rep = make_split(nodes, edges, DATA / "split" / f"{a.name}_scaffold", seed=a.seed,
                     groups=scaffold_groups(nodes))
    print(json.dumps(rep, indent=2))


def cmd_gnn(a):
    from .gnn import run_comparison
    configs = [c.strip() for c in a.configs.split(",")] if a.configs else None
    res = run_comparison(DATA, a.name, configs=configs, epochs=a.epochs, seed=a.seed, split=a.split)
    print(json.dumps({k: v["test"] for k, v in res.items()}, indent=2))


def cmd_predict(a):
    from .gnn import run_predict
    print(json.dumps(run_predict(DATA, a.name, epochs=a.epochs, seed=a.seed), indent=2))


def cmd_ask(a):
    from .query import KG, answer_text, ask
    if not a.question:
        sys.exit("ask needs --question")
    kg = KG.load(_paths(a.name)["kgx"], a.name)
    print(answer_text(ask(kg, a.question, use_llm=not a.no_llm), max_rows=a.top))


def cmd_table(a):
    from .agent import run_build
    if not a.file:
        sys.exit("table needs --file")
    f = Path(a.file)
    table = pd.read_csv(f, sep="\t" if f.suffix.lower() in (".tsv", ".txt") else ",")
    name = a.out_name or f.stem
    final = run_build({"source": "table", "name": name, "data_dir": str(DATA), "table": table},
                      on_step=lambda step, st: print(st["log"][-1]) if st.get("log") else None)
    print(json.dumps(final.get("outputs", {}), indent=2))


def cmd_update(a):
    from .maintain import run_update
    print(json.dumps(run_update(DATA, a.name, family=a.family, force=a.force), indent=2))


DATA_URL = os.environ.get("VERIKG_DATA_URL",
                          "https://github.com/Falehaqazi/VeriKG/raw/data/verikg_data.zip")
PYG_URL = os.environ.get("VERIKG_PYG_URL",
                         "https://github.com/Falehaqazi/VeriKG/raw/data/verikg_pyg.zip")


def cmd_fetch_data(a):
    """Download the prebuilt graph, scores and results (minutes instead of the
    hour-long Pharos pull)."""
    import io
    import zipfile
    import requests
    for url in ([a.url] if a.url else [DATA_URL, PYG_URL]):
        print(f"downloading {url}")
        r = requests.get(url, timeout=900)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names = [n for n in z.namelist() if n.startswith("data/") and ".." not in n]
            z.extractall(".", members=names)
        print(f"{len(names)} files written under data/")
    scored = DATA / "kgx" / f"{a.name}_edges_scored.tsv"
    pruned = DATA / "kgx" / f"{a.name}_edges_pruned.tsv"
    verdicts = DATA / "kgx" / f"{a.name}_verdicts.tsv"
    if scored.exists() and not (pruned.exists() and verdicts.exists()):  # derived, so left out of the download
        from .prune import verdict_table
        e = pd.read_csv(scored, sep="\t", low_memory=False)
        e[e["verdict"] == "keep"].to_csv(pruned, sep="\t", index=False)
        nodes = pd.read_csv(DATA / "kgx" / f"{a.name}_nodes.tsv", sep="\t", low_memory=False)
        verdict_table(e, nodes).to_csv(verdicts, sep="\t", index=False)
        print(f"wrote {pruned} and {verdicts}")


def cmd_app(a):
    import subprocess
    app = Path(__file__).resolve().parent.parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port),
                    "--server.address", "0.0.0.0"], check=False)


STEPS = {"pull": cmd_pull, "build": cmd_build, "validate": cmd_validate, "split": cmd_split,
         "baseline": cmd_baseline, "all": cmd_all, "prune": cmd_prune, "explain": cmd_explain,
         "subgraph": cmd_subgraph, "enrich": cmd_enrich, "opentargets": cmd_opentargets,
         "benchmark": cmd_benchmark, "split-scaffold": cmd_split_scaffold, "gnn": cmd_gnn,
         "predict": cmd_predict, "ask": cmd_ask, "table": cmd_table, "update": cmd_update,
         "app": cmd_app, "fetch-data": cmd_fetch_data}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kgc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=list(STEPS))
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
    ap.add_argument("--k", type=int, default=10, help="enrich: neighbours per compound")
    ap.add_argument("--sim-threshold", type=float, default=0.6, help="enrich: minimum Tanimoto")
    ap.add_argument("--rate", type=float, default=0.02, help="benchmark: share of edges to plant per predicate")
    ap.add_argument("--configs", help="gnn: comma-separated graphs (pharos, pharos_fp, verikg, verikg_pruned)")
    ap.add_argument("--split", default="random", choices=["random", "scaffold"], help="gnn: which split")
    ap.add_argument("--epochs", type=int, default=120, help="gnn/predict: training epochs")
    ap.add_argument("--no-llm", action="store_true", help="ask: keyword planner only")
    ap.add_argument("--top", type=int, default=15, help="ask: rows to print")
    ap.add_argument("--file", help="table: CSV or TSV of relationships")
    ap.add_argument("--out-name", help="table: dataset name for the output files")
    ap.add_argument("--force", action="store_true", help="update: rebuild even if the version is unchanged")
    ap.add_argument("--port", type=int, default=8501, help="app: port")
    ap.add_argument("--url", help="fetch-data: where to download the prebuilt data from")
    a = ap.parse_args(argv)
    STEPS[a.step](a)


if __name__ == "__main__":
    main()
