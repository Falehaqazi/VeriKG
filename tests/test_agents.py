"""Tests for the agent steps added after the midpoint: enrichment, context
plausibility, the planted-error benchmark, scaffold split, the query agent,
tabular import, Open Targets, maintenance and the numpy predictor.
Synthetic data only; no network."""

import json

import numpy as np
import pandas as pd
import pytest

from kgc import biolink as B
from kgc.benchmark import inject_errors, run_benchmark
from kgc.kgx import build_kgx
from kgc.maintain import changelog_text, diff_graphs
from kgc.opentargets import associations_to_table, fetch_associations, norm_disease
from kgc.plausibility import score_plausibility
from kgc.predictor import Predictor
from kgc.prune import attach_opentargets, external_adjustment, score_edges
from kgc.query import KG, answer_text, ask, plan_rules
from kgc.split import audit_leakage, make_split
from kgc.tabular import build_from_table, suggest_mapping, validate_generic
from kgc.validate import validate

from .synthetic import make_records, write_raw

SMILES = ["CCOc1ccc2nc(S(N)(=O)=O)sc2c1", "CCOc1ccc2nc(S(N)(=O)=O)sc2c1C", "c1ccccc1O",
          "c1ccccc1N", "CC(=O)Oc1ccccc1C(=O)O", "CCN(CC)CC"]


def _built(tmp_path, records=None):
    raw = tmp_path / "syn.jsonl"
    write_raw(raw, records or make_records())
    build_kgx(raw, tmp_path / "kgx", name="syn")
    nodes = pd.read_csv(tmp_path / "kgx" / "syn_nodes.tsv", sep="\t")
    edges = pd.read_csv(tmp_path / "kgx" / "syn_edges.tsv", sep="\t")
    return nodes, edges


def test_every_edge_has_relation_and_every_node_a_class(tmp_path):
    nodes, edges = _built(tmp_path)
    assert edges["relation"].notna().all() and edges["relation"].str.startswith("http").all()
    assert nodes["type_iri"].notna().all()


def test_indications_become_drug_disease_edges(tmp_path):
    recs = make_records()
    drug = next(lig for r in recs for lig in r["ligands"] if lig["isdrug"])
    recs[0]["target"]["diseases"][0]["associations"].append(
        {"type": "DrugCentral Indication", "drug": drug["name"]})
    nodes, edges = _built(tmp_path, recs)
    treats = edges[edges["predicate"] == B.TREATS]
    assert len(treats) >= 1
    assert (treats["primary_knowledge_source"] == "infores:drugcentral").all()
    # an indication is never turned into a gene-disease edge
    gd = edges[edges["predicate"] == B.ASSOC_CONDITION]
    assert set(gd["primary_knowledge_source"]) == {"infores:disgenet"}
    assert validate(nodes, edges)["ok"]


def test_fingerprints_and_similarity_edges():
    pytest.importorskip("rdkit")
    from kgc.enrich import fingerprints, nearest_neighbours, similarity_edges
    nodes = pd.DataFrame({"id": [f"CHEMBL.COMPOUND:C{i}" for i in range(len(SMILES))] + ["X"],
                          "category": [B.CHEMICAL] * len(SMILES) + [B.PROTEIN],
                          "smiles": SMILES + [None]})
    ids, fps, packed = fingerprints(nodes)
    assert len(ids) == len(SMILES) and packed.shape == (len(SMILES), 256)
    pairs = nearest_neighbours(fps, k=3, threshold=0.5)
    assert (0, 1) in {(i, j) for i, j, _ in pairs}  # the methyl analog pair
    assert all(i < j and s >= 0.5 for i, j, s in pairs)
    sim = similarity_edges(ids, pairs, "v", "run")
    assert (sim["predicate"] == B.SIMILAR).all() and (sim["primary_knowledge_source"] == B.SELF).all()
    assert (sim["subject"] < sim["object"]).all()
    scored = score_edges(sim)
    assert (scored["confidence"] == scored["similarity_score"].round(3)).all()


def test_scaffold_groups_share_ring_systems():
    pytest.importorskip("rdkit")
    from kgc.split import scaffold_groups
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"], "category": B.CHEMICAL,
                          "smiles": ["c1ccccc1O", "c1ccccc1N", "CCN(CC)CC", None]})
    g = scaffold_groups(nodes)
    assert g["a"] == g["b"] == "c1ccccc1"
    assert g["c"] != g["d"] and g["c"].startswith("single:")


def test_group_split_keeps_groups_together(tmp_path):
    nodes, edges = _built(tmp_path)
    task = edges[edges["predicate"] == B.INTERACTS_PHYS]
    chems = sorted(set(task["subject"]))
    groups = {c: f"g{i // 7}" for i, c in enumerate(chems)}
    rep = make_split(nodes, edges, tmp_path / "split", groups=groups)
    assert rep["test_pairs_with_unseen_chemical"] == 1.0
    folds = {}
    for name in ("val", "test"):
        df = pd.read_csv(tmp_path / "split" / f"{name}.tsv", sep="\t")
        folds[name] = {groups[c] for c in df.loc[df.label == 1, "subject"]}
    train = pd.read_csv(tmp_path / "split" / "train_graph_edges.tsv", sep="\t")
    tr_groups = {groups[c] for c in train.loc[train.predicate == B.INTERACTS_PHYS, "subject"]}
    assert not (tr_groups & folds["test"]) and not (tr_groups & folds["val"])
    assert not (folds["val"] & folds["test"])


def test_context_plausibility_out_of_fold(tmp_path):
    _, edges = _built(tmp_path)
    ctx, summary, _ = score_plausibility(edges)
    binds = edges["predicate"] == B.INTERACTS_PHYS
    p = ctx.loc[binds, "plausibility"]
    assert p.notna().all() and p.between(0, 1).all()
    # the synthetic graph is clustered, so context must tell real from corrupted
    assert summary[B.INTERACTS_PHYS]["real_vs_corrupted_auroc_oof"] > 0.7
    assert ctx.loc[edges["predicate"] == B.TRANSLATES_TO, "plausibility"].isna().all()


def test_planted_errors_copy_provenance_and_are_found(tmp_path):
    _, edges = _built(tmp_path)
    noisy, planted = inject_errors(edges, rate=0.05, seed=1)
    bad = noisy[noisy["id"].str.endswith(":err")]
    assert len(bad) == sum(len(v) for v in planted.values()) > 0
    assert not bad.duplicated(["subject", "predicate", "object"]).any()
    real = set(zip(edges["subject"], edges["predicate"], edges["object"]))
    assert not any(t in real for t in zip(bad["subject"], bad["predicate"], bad["object"]))
    res = run_benchmark(edges, rate=0.05, seed=1, log=lambda *_: None)
    r = res["by_predicate"][B.INTERACTS_PHYS]["random"]
    assert r["auroc_with_context"] > r["auroc_evidence_only"]


def test_opentargets_table_and_adjustment():
    assert norm_disease("MONDO_0005148") == "MONDO:0005148"
    assert norm_disease("MSH:D001943") == "MESH:D001943"
    pairs = {"HGNC:1": ["MONDO:0000001", "MONDO:0000002"], "HGNC:2": ["MONDO:0000003"]}
    rows = {"HGNC:1": [{"score": 0.4, "disease": {"id": "EFO_1", "dbXRefs": ["MONDO:0000001"]}}]}
    t = associations_to_table(pairs, rows, {"HGNC:1": "ENSG1"})
    assert len(t) == 2  # HGNC:2 not checked: left out, not scored 0
    assert dict(zip(t["disease"], t["score"])) == {"MONDO:0000001": 0.4, "MONDO:0000002": 0.0}
    assert external_adjustment(B.ASSOC_CONDITION, 0.4)[0] > 0
    assert external_adjustment(B.ASSOC_CONDITION, 0.0)[0] < 0
    assert external_adjustment(B.PPI, 0.0) == (0.0, None)


class FakeOT:
    def query(self, q, v=None):
        if "meta" in q:
            return {"meta": {"dataVersion": {"year": 26, "month": 9}}}
        if "mapIds" in q:
            return {"mapIds": {"mappings": [{"term": t, "hits": [{"id": f"ENSG{t}", "entity": "target"}]}
                                            for t in v["terms"]]}}
        return {"target": {"associatedDiseases": {"count": 1, "rows": [
            {"score": 0.3, "disease": {"id": "MONDO_0009000", "dbXRefs": []}}]}}}


def test_opentargets_fetch_feeds_pruning(tmp_path):
    nodes, edges = _built(tmp_path)
    rep = fetch_associations(tmp_path / "kgx", "syn", client=FakeOT(), log=lambda *_: None)
    assert rep["genes_mapped"] == rep["genes"] > 0
    att = attach_opentargets(edges, tmp_path / "kgx" / "syn_opentargets.tsv")
    gd = att[att["predicate"] == B.ASSOC_CONDITION]
    assert gd["opentargets_score"].notna().any()
    scored = score_edges(att)
    notes = ";".join(scored.loc[gd.index, "confidence_notes"])
    assert "opentargets" in notes


def _kg(tmp_path):
    nodes, edges = _built(tmp_path)
    return KG(nodes, score_edges(edges))


def test_query_rules_and_answers(tmp_path):
    kg = _kg(tmp_path)
    assert plan_rules(kg, "Which compounds bind KIN3?").intent == "ligands_of"
    assert plan_rules(kg, "What does compound-12 target?").intent == "targets_of"
    assert plan_rules(kg, "Which kinases are linked to disease 2?").intent == "proteins_for_disease"
    assert plan_rules(kg, "What did you remove about KIN3 and why?").intent == "pruned"
    ans = ask(kg, "Which compounds bind KIN3?", use_llm=False)
    assert ans.rows and ans.cypher.startswith("MATCH")
    assert set(ans.edge_ids) <= set(kg.edges["id"])
    assert ans.plan.entities[0]["id"].startswith("UniProtKB:")
    txt = answer_text(ans)
    assert "Cypher" in txt and "ligands_of" in txt
    multi = ask(kg, "Which compounds act on kinases linked to disease 2?", use_llm=False)
    assert multi.plan.intent == "drugs_for_disease" and multi.rows


def test_query_agent_clarifies_unknown_names(tmp_path):
    from kgc.agent import build_query_agent
    kg = _kg(tmp_path)
    run = build_query_agent(kg)
    s = run({"question": "Which compounds bind zzqqxx?", "use_llm": False, "log": []})
    assert not s["answer"].rows and any(line.startswith("Clarify") for line in s["log"])
    s = run({"question": "Which compounds bind KIN3?", "use_llm": False, "log": []})
    assert s["answer"].rows and [line.split(":")[0] for line in s["log"]][-1] == "Execute"


def test_table_mapping_and_build():
    df = pd.DataFrame({"compound": ["CHEMBL25", "CHEMBL941", "CHEMBL25"],
                       "uniprot": ["P23219", "P00519", "P35354"],
                       "pIC50": [5.1, 7.9, 6.0], "pmid": ["123456", "234567", ""]})
    m = suggest_mapping(df)
    assert (m["subject_column"], m["object_column"]) == ("compound", "uniprot")
    assert m["predicate"] == B.INTERACTS_PHYS and m["score_column"] == "pIC50"
    nodes, edges, issues = build_from_table(df, m)
    assert set(edges["subject"]) == {"CHEMBL.COMPOUND:CHEMBL25", "CHEMBL.COMPOUND:CHEMBL941"}
    assert len(edges) == 3 and edges["publications"].str.contains("PMID").sum() == 2
    assert validate_generic(nodes, edges)["ok"]


def test_build_agent_on_a_table(tmp_path):
    from kgc.agent import run_build
    df = pd.DataFrame({"gene": [f"HGNC:{i}" for i in range(40)],
                       "disease": [f"MONDO:{i % 6:07d}" for i in range(40)]})
    steps = []
    final = run_build({"source": "table", "name": "t", "data_dir": str(tmp_path), "table": df},
                      on_step=lambda name, s: steps.append(name))
    assert steps == ["ingest", "map", "validate", "enrich", "prune", "export"]
    assert (tmp_path / "kgx" / "t_edges_scored.tsv").exists()
    assert set(final["edges"]["verdict"]) <= {"keep", "prune"}


def test_diff_between_releases():
    old = pd.DataFrame({"id": ["a", "b", "c"], "predicate": ["p"] * 3, "publications": ["1", "2", "3"]})
    new = pd.DataFrame({"id": ["b", "c", "d"], "predicate": ["p"] * 3, "publications": ["2", "3|4", "5"]})
    d = diff_graphs(old, new)
    assert (d["added"], d["removed"], d["changed"]) == (1, 1, 1)
    assert d["changed_edges"][0]["id"] == "c"
    assert "1 edges added" in changelog_text(d, "v1", "v2")


def test_numpy_predictor_matches_its_decoder(tmp_path):
    rng = np.random.default_rng(0)
    h = 8
    z = dict(chem_ids=np.array(["c0", "c1", "c2"]), prot_ids=np.array(["p0", "p1"]),
             h_chem=rng.normal(size=(3, h)).astype(np.float32), h_prot=rng.normal(size=(2, h)).astype(np.float32),
             w1=rng.normal(size=(h, 2 * h)).astype(np.float32), b1=np.zeros(h, np.float32),
             w2=rng.normal(size=(1, h)).astype(np.float32), b2=np.zeros(1, np.float32))
    np.savez(tmp_path / "p.npz", **z)
    pr = Predictor(tmp_path / "p.npz")
    s = pr.score("c1", "p0")
    assert 0 < s < 1 and pr.score("nope", "p0") is None
    ranked = pr.targets_for("c1", ["p0", "p1"])
    assert ranked[0][1] >= ranked[1][1]
    assert dict(ranked)["p0"] == pytest.approx(s, rel=1e-5)
    assert [c for c, _ in pr.ligands_for("p0", exclude=["c0"], k=5)] and \
        "c0" not in [c for c, _ in pr.ligands_for("p0", exclude=["c0"], k=5)]


def test_gnn_trains_and_exports(tmp_path):
    pytest.importorskip("torch_geometric")
    from kgc.gnn import export_predictor, fit_full, train_eval
    nodes, edges = _built(tmp_path)
    make_split(nodes, edges, tmp_path / "split")
    sp = tmp_path / "split"
    val, test = pd.read_csv(sp / "val.tsv", sep="\t"), pd.read_csv(sp / "test.tsv", sep="\t")
    rn = pd.read_csv(sp / "test_rank_negatives.tsv", sep="\t")
    held = set(zip(test.loc[test.label == 1, "subject"], test.loc[test.label == 1, "object"])) | \
        set(zip(val.loc[val.label == 1, "subject"], val.loc[val.label == 1, "object"]))
    tg, _ = audit_leakage(edges, held)
    res, _ = train_eval(nodes, tg, val, test, rn, epochs=10, hidden=16, log=lambda *_: None, eval_every=5)
    assert 0 <= res["test"]["degree_matched"]["auroc"] <= 1
    model, data, idx = fit_full(nodes, edges, epochs=3, hidden=16, log=lambda *_: None)
    export_predictor(model, data, idx, tmp_path / "pred.npz")
    pr = Predictor(tmp_path / "pred.npz")
    c, p = edges.loc[edges.predicate == B.INTERACTS_PHYS, ["subject", "object"]].iloc[0]
    assert 0 <= pr.score(c, p) <= 1


def test_cli_lists_every_step():
    from kgc.cli import STEPS
    for step in ("enrich", "opentargets", "benchmark", "split-scaffold", "gnn", "predict", "ask",
                 "table", "update", "app"):
        assert step in STEPS


def test_ligand_with_several_chembl_ids_gets_one_curie():
    from kgc.kgx import chemical_curie, chemical_xrefs
    lig = {"ligid": "L1", "synonyms": [{"name": "ChEMBL", "value": "CHEMBL1336,CHEMBL3928540"},
                                       {"name": "PubChem", "value": "216239"}]}
    assert chemical_curie(lig) == "CHEMBL.COMPOUND:CHEMBL1336"
    assert chemical_xrefs(lig, "CHEMBL.COMPOUND:CHEMBL1336") == ["CHEMBL.COMPOUND:CHEMBL3928540",
                                                                 "PUBCHEM.COMPOUND:216239"]
    from kgc.validate import CURIE
    assert not CURIE.match("CHEMBL.COMPOUND:CHEMBL1,CHEMBL2")


def test_yes_no_question_about_one_link(tmp_path):
    kg = _kg(tmp_path)
    assert plan_rules(kg, "Does compound-12 bind KIN3?").intent == "explain"


def test_prior_comes_from_the_most_reliable_source():
    from kgc.prune import score_edge
    c, notes = score_edge(B.ASSOC_CONDITION, "infores:monarchinitiative",
                          "infores:monarchinitiative|infores:omim|infores:diseases",
                          "knowledge_assertion", "manual_agent", "")
    assert notes[0] == "=source:infores:omim:0.90" and "+corroborated_by:2" in notes
    c2, _ = score_edge(B.ASSOC_CONDITION, "infores:monarchinitiative", "infores:monarchinitiative",
                       "knowledge_assertion", "manual_agent", "")
    assert c > c2
