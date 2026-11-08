import pandas as pd

from kgc import biolink as B
from kgc.explain import explain_edge, explain_prediction
from kgc.kgx import build_kgx, edge_id
from kgc.prune import prune_graph, score_edge, score_edges
from kgc.subgraph import extract_subgraph, find_seeds, to_arrays
from kgc.validate import validate

from .synthetic import make_records, write_raw


def _edge(s, p, o, **kw):
    row = {"id": edge_id(s, p, o), "subject": s, "predicate": p, "object": o,
           "primary_knowledge_source": "infores:chembl", "aggregator_knowledge_source": "infores:pharos",
           "knowledge_level": "knowledge_assertion", "agent_type": "manual_agent",
           "supporting_sources": "infores:chembl", "publications": "PMID:1", "evidence_count": 1,
           "source_version": "pharos:test"}
    row.update(kw)
    return row


def _nodes():
    rows = [("CHEMBL.COMPOUND:A", B.CHEMICAL, "compound A", None),
            ("CHEMBL.COMPOUND:B", B.CHEMICAL, "compound B", None),
            ("UniProtKB:T1", B.PROTEIN, "target one", "KIN1"),
            ("UniProtKB:P1", B.PROTEIN, "protein one", "KIN2"),
            ("REACT:R1", B.PATHWAY, "pathway one", None)]
    return pd.DataFrame(rows, columns=["id", "category", "name", "symbol"])


def _graph_edges():
    A, Bc, T, P = "CHEMBL.COMPOUND:A", "CHEMBL.COMPOUND:B", "UniProtKB:T1", "UniProtKB:P1"
    return pd.DataFrame([
        _edge(A, B.INTERACTS_PHYS, T, activity_value_max=7.5),
        _edge(Bc, B.INTERACTS_PHYS, T, activity_value_max=6.5),
        _edge(Bc, B.INTERACTS_PHYS, P, activity_value_max=6.2),
        _edge(T, B.PPI, P, primary_knowledge_source="infores:string", knowledge_level="statistical_association",
              agent_type="computational_model", supporting_sources="infores:string",
              publications=None, interaction_score_max=0.2),
        _edge(T, B.PARTICIPATES_IN, "REACT:R1", primary_knowledge_source="infores:reactome",
              supporting_sources="infores:reactome", publications=None),
        _edge(P, B.PARTICIPATES_IN, "REACT:R1", primary_knowledge_source="infores:reactome",
              supporting_sources="infores:reactome", publications=None),
    ])


def test_score_edge_signals():
    # structural edges are kept at full confidence
    assert score_edge(B.TRANSLATES_TO, "infores:uniprot", "", "knowledge_assertion", "manual_agent", None)[0] == 1.0
    # curated, potent, cited: high
    hi, _ = score_edge(B.INTERACTS_PHYS, "infores:chembl", "infores:chembl", "knowledge_assertion",
                       "manual_agent", "PMID:1", activity=8.0)
    # text mined, single source, uncited: low
    lo, notes = score_edge(B.ASSOC_CONDITION, "infores:diseases", "infores:diseases", "statistical_association",
                           "text_mining_agent", None)
    assert hi > 0.9 and lo < 0.45
    assert any(n.startswith("-agent:text_mining_agent") for n in notes)
    # weak activity lowers the score, corroboration raises it
    weak, _ = score_edge(B.INTERACTS_PHYS, "infores:chembl", "infores:chembl", "knowledge_assertion",
                         "manual_agent", None, activity=4.0)
    base, _ = score_edge(B.INTERACTS_PHYS, "infores:chembl", "infores:chembl", "knowledge_assertion",
                         "manual_agent", None, activity=5.5)
    assert weak < base
    corr, _ = score_edge(B.ASSOC_CONDITION, "infores:disgenet", "infores:disgenet|infores:ctd|infores:omim",
                         "knowledge_assertion", "automated_agent", None)
    single, _ = score_edge(B.ASSOC_CONDITION, "infores:disgenet", "infores:disgenet",
                           "knowledge_assertion", "automated_agent", None)
    assert corr > single


def test_prune_verdicts_and_outputs():
    nodes, edges = _nodes(), _graph_edges()
    scored, kept, pruned_nodes, report = prune_graph(nodes, edges)
    assert len(scored) == len(edges)
    assert set(scored["verdict"]) <= {"keep", "prune"}
    ppi = scored[scored["predicate"] == B.PPI].iloc[0]
    assert ppi["verdict"] == "prune" and "low_ppi_score" in ppi["prune_reason"]
    assert (kept["verdict"] == "keep").all()
    assert report["edges"]["kept"] + report["edges"]["pruned"] == report["edges"]["total"]
    # every kept edge still has both endpoints
    ids = set(pruned_nodes["id"])
    assert set(kept["subject"]) <= ids and set(kept["object"]) <= ids


def test_prune_on_built_graph_stays_valid(tmp_path):
    raw = tmp_path / "syn.jsonl"
    write_raw(raw, make_records())
    build_kgx(raw, tmp_path / "kgx", name="syn")
    nodes = pd.read_csv(tmp_path / "kgx" / "syn_nodes.tsv", sep="\t")
    edges = pd.read_csv(tmp_path / "kgx" / "syn_edges.tsv", sep="\t")
    scored, kept, pruned_nodes, _ = prune_graph(nodes, edges)
    assert (scored.loc[scored["predicate"] == B.TRANSLATES_TO, "verdict"] == "keep").all()
    assert validate(pruned_nodes, kept)["ok"]


def test_explain_edge_has_lineage():
    nodes, edges = _nodes(), score_edges(_graph_edges())
    res = explain_edge(nodes, edges, "CHEMBL.COMPOUND:A", "UniProtKB:T1")
    assert res["found"]
    assert "infores:chembl" in res["text"] and "PMID:1" in res["text"]
    assert res["edges"][0]["publications"] == ["https://pubmed.ncbi.nlm.nih.gov/1/"]
    assert not explain_edge(nodes, edges, "CHEMBL.COMPOUND:A", "UniProtKB:P1")["found"]


def test_explain_prediction_paths():
    nodes, edges = _nodes(), _graph_edges()
    res = explain_prediction(edges, "CHEMBL.COMPOUND:A", "UniProtKB:P1", nodes=nodes)
    assert not res["already_known"]
    counts = res["path_counts"]
    assert counts["shared_ligand"] == 1 and counts["ppi"] == 1 and counts["pathway"] == 1
    path = res["paths"]["shared_ligand"][0]
    assert path["via_chemical"] == "CHEMBL.COMPOUND:B"
    assert set(path["edges"]) <= set(edges["id"])


def test_subgraph_from_question_and_hops():
    nodes, edges = _nodes(), score_edges(_graph_edges())
    seeds, matches = find_seeds(nodes, "Which compounds bind KIN1?")
    assert seeds == ["UniProtKB:T1"] and matches["KIN1"]["tier"] == "symbol"
    n1, e1 = extract_subgraph(nodes, edges, seeds, hops=1)
    assert {"CHEMBL.COMPOUND:A", "CHEMBL.COMPOUND:B"} <= set(n1["id"])
    # the pruned PPI edge is ignored, so P1 is not a direct neighbour of T1
    assert "UniProtKB:P1" not in set(n1["id"])
    assert (e1["verdict"] == "keep").all()
    arr = to_arrays(n1, e1)
    assert arr["edge_index"].shape == (2, len(e1))
    assert arr["edge_index"].max() < len(arr["node_ids"])


def test_generic_word_is_not_a_seed():
    nodes = pd.DataFrame({"id": [f"UniProtKB:P{i}" for i in range(80)],
                          "category": B.PROTEIN, "name": [f"kinase {i}" for i in range(80)],
                          "symbol": [f"K{i}" for i in range(80)]})
    seeds, _ = find_seeds(nodes, "kinase")
    assert seeds == []
