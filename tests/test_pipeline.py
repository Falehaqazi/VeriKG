import json

import pandas as pd

from kgc import biolink as B
from kgc.baseline import run_baseline
from kgc.kgx import build_kgx, chemical_curie, gene_curie
from kgc.pharos import pull_family
from kgc.split import audit_leakage, make_split, pair_key
from kgc.validate import validate

from .synthetic import make_records, write_raw


def _build(tmp_path):
    raw = tmp_path / "syn.jsonl"
    write_raw(raw, make_records())
    stats = build_kgx(raw, tmp_path / "kgx", name="syn")
    nodes = pd.read_csv(tmp_path / "kgx" / "syn_nodes.tsv", sep="\t")
    edges = pd.read_csv(tmp_path / "kgx" / "syn_edges.tsv", sep="\t")
    return stats, nodes, edges


def test_ids():
    assert gene_curie({"xrefs": [{"source": "NCBI Gene", "value": "1956"}]}) == "NCBIGene:1956"
    assert gene_curie({"xrefs": [{"source": "HGNC", "value": "HGNC:3236"}]}) == "HGNC:3236"
    assert gene_curie({"xrefs": [], "sym": "EGFR"}) == "HGNC.SYMBOL:EGFR"
    # live Pharos shape: ID in "name", "value" null
    assert gene_curie({"xrefs": [{"source": "HGNC", "name": "HGNC:3236", "value": None}]}) == "HGNC:3236"
    lig = {"ligid": "ABC", "synonyms": [{"name": "LyCHI", "value": "ABC"},
                                        {"name": "ChEMBL", "value": "CHEMBL25"}]}
    assert chemical_curie(lig) == "CHEMBL.COMPOUND:CHEMBL25"
    assert chemical_curie({"ligid": "ABC", "synonyms": []}) == "LYCHI:ABC"


def test_build_and_validate(tmp_path):
    stats, nodes, edges = _build(tmp_path)
    rep = validate(nodes, edges)
    assert rep["ok"], rep["errors"]
    preds = set(edges["predicate"])
    assert {B.TRANSLATES_TO, B.INTERACTS_PHYS, B.PPI, B.ASSOC_CONDITION, B.PARTICIPATES_IN} <= preds
    # indication-type associations are not turned into gene-disease edges
    assert not edges["primary_knowledge_source"].str.contains("drugcentral").any()
    # symmetric PPIs are stored once
    ppi = edges[edges["predicate"] == B.PPI]
    assert len(ppi) == len({pair_key(s, o) for s, o in zip(ppi["subject"], ppi["object"])})
    # every edge carries lineage
    for col in ("primary_knowledge_source", "aggregator_knowledge_source", "knowledge_level",
                "agent_type", "source_version", "ingest_run"):
        assert edges[col].notna().all(), col
    assert stats["issues"]["gene_id_from_symbol_only"] > 0
    assert rep["warnings"].get("nonstandard_prefix:LYCHI", 0) > 0


def test_validator_catches_errors(tmp_path):
    _, nodes, edges = _build(tmp_path)
    bad = edges.copy()
    i = bad.index[bad["predicate"] == B.INTERACTS_PHYS][0]
    bad.loc[i, ["subject", "object"]] = bad.loc[i, ["object", "subject"]].values  # reversed
    bad.loc[bad.index[0], "object"] = "UniProtKB:DOES_NOT_EXIST"
    rep = validate(nodes, bad)
    assert not rep["ok"]
    assert rep["errors"]["dangling_edge"] == 1
    assert rep["errors"][f"domain_violation:{B.INTERACTS_PHYS}"] == 1


def test_split_is_leakage_free(tmp_path):
    _, nodes, edges = _build(tmp_path)
    rep = make_split(nodes, edges, tmp_path / "split", seed=0)
    train = pd.read_csv(tmp_path / "split" / "train_graph_edges.tsv", sep="\t")
    test = pd.read_csv(tmp_path / "split" / "test.tsv", sep="\t")
    held = test[test["label"] == 1]
    train_keys = {pair_key(s, o) for s, o in zip(train["subject"], train["object"])}
    assert not any(pair_key(s, o) in train_keys for s, o in zip(held["subject"], held["object"]))
    # negatives never collide with any true pair
    task = edges[edges["predicate"] == B.INTERACTS_PHYS]
    true = set(zip(task["subject"], task["object"]))
    neg = test[test["label"] == 0]
    assert not any((s, o) in true for s, o in zip(neg["subject"], neg["object"]))
    rank = pd.read_csv(tmp_path / "split" / "test_rank_negatives.tsv", sep="\t")
    assert not any((s, o) in true for s, o in zip(rank["subject"], rank["object"]))
    assert sum(rep["pairs"].values()) == len(true)
    assert sum(rep["leakage_audit_removed_by_source"].values()) == rep["pairs"]["val"] + rep["pairs"]["test"]


def test_audit_catches_reversed_and_other_predicates():
    edges = pd.DataFrame({
        "subject": ["A", "B", "C"], "object": ["B", "A", "D"],
        "predicate": ["p1", "p2", "p1"], "primary_knowledge_source": ["infores:x", "infores:y", "infores:x"],
    })
    clean, removed = audit_leakage(edges, [("A", "B")])
    assert list(clean["subject"]) == ["C"]
    assert removed == {"infores:x": 1, "infores:y": 1}


def test_baseline_learns_structure(tmp_path):
    _build(tmp_path)
    nodes_path = tmp_path / "kgx" / "syn_nodes.tsv"
    make_split(pd.read_csv(nodes_path, sep="\t"),
               pd.read_csv(tmp_path / "kgx" / "syn_edges.tsv", sep="\t"), tmp_path / "split")
    res = run_baseline(tmp_path / "split", nodes_path, tmp_path / "res.json")
    assert res["lr"]["degree_matched"]["auroc"] > 0.75
    assert abs(res["degree"]["degree_matched"]["auroc"] - 0.5) < 0.15
    assert res["lr"]["sampled_ranking"]["n"] > 0
    assert json.loads((tmp_path / "res.json").read_text())["lr"]


class FakeClient:
    """Stands in for the Pharos API: 3 kinases, paged ligands, one foreign activity."""
    endpoint = "fake"

    def __init__(self):
        self.calls = 0

    def query(self, q, v=None):
        self.calls += 1
        v = v or {}
        if "dbVersion" in q:
            return {"dbVersion": "6.13.5"}
        if "targets(filter" in q:
            allt = [{"uniprot": f"Q{i}", "sym": f"K{i}"} for i in range(3)]
            return {"targets": {"count": 3, "targets": allt[v["skip"]:v["skip"] + v["top"]]}}
        if "ligands(" in q:
            u, skip, top = v["uniprot"], v["skip"], v["top"]
            n = 5 if v["isdrug"] else 0
            ligs = [{"ligid": f"L{i}", "name": None, "isdrug": True, "smiles": None, "synonyms": [],
                     "activities": [{"target": {"uniprot": u}, "value": 7.0},
                                    {"target": {"uniprot": "OTHER"}, "value": 9.0}]}
                    for i in range(n)][skip:skip + top]
            return {"target": {"ligands": ligs}}
        return {"target": {"uniprot": v["uniprot"], "sym": "K", "xrefs": [], "diseases": [],
                           "ppis": [], "pathways": []}}


def test_pull_pages_scopes_and_resumes(tmp_path):
    out = tmp_path / "raw.jsonl"
    import kgc.pharos as P
    old = P.pull_ligands

    def small_pages(client, uniprot, isdrug, page=2, max_ligands=None):
        return old(client, uniprot, isdrug, page=page, max_ligands=max_ligands)
    P.pull_ligands = small_pages
    try:
        pull_family(out, client=FakeClient(), log=lambda *_: None)
        recs = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(recs) == 3
        assert all(len(r["ligands"]) == 5 for r in recs)
        assert all(len(l["activities"]) == 1 for r in recs for l in r["ligands"])
        fc = FakeClient()
        pull_family(out, client=fc, log=lambda *_: None)  # resume: nothing new
        assert len(out.read_text().splitlines()) == 3
        assert json.loads(out.with_suffix(".meta.json").read_text())["dbVersion"] == "6.13.5"
    finally:
        P.pull_ligands = old
