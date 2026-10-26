"""Command line: python -m kgc <step> ...

Steps, in order:
  pull      download one Pharos target family to data/raw/
  build     raw JSONL -> KGX nodes/edges TSV in data/kgx/
  validate  structural and Biolink checks on the KGX files
  split     leakage-safe train/val/test split with hard negatives
  baseline  CPU baseline; writes metrics JSON
  all       build, validate, split, baseline (after pull)
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kgc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["pull", "build", "validate", "split", "baseline", "all"])
    ap.add_argument("--name", default="pharos_kinase", help="dataset name used in file paths")
    ap.add_argument("--family", default="Kinase", help="Pharos target family to pull")
    ap.add_argument("--max-ligands", type=int, default=None,
                    help="cap ligands per target and per type (drug/non-drug); default: all")
    ap.add_argument("--limit", type=int, default=None, help="only pull the first N targets (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    {"pull": cmd_pull, "build": cmd_build, "validate": cmd_validate, "split": cmd_split,
     "baseline": cmd_baseline, "all": cmd_all}[a.step](a)


if __name__ == "__main__":
    main()
