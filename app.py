"""VeriKG web app: ask the kinase knowledge graph questions in plain language.

    streamlit run app.py          (or: python -m kgc app)

Reads the graph from data/kgx/ (run the pipeline first, or `python -m kgc
fetch-data` for the prebuilt release). Works without an API key; set
NVIDIA_API_KEY to let a language model plan the queries.
"""

import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from kgc import biolink as B
from kgc.agent import build_query_agent, run_build
from kgc.explain import explain_prediction
from kgc.query import CATEGORY_LABEL, EXAMPLES, INTENTS, KG, PRED_LABEL, Plan, edge_card, execute, ground_plan

DATA = Path(os.environ.get("VERIKG_DATA", "data"))
NAME = os.environ.get("VERIKG_NAME", "pharos_kinase")
KGX = DATA / "kgx"
RESULTS = DATA / "results"

st.set_page_config(page_title="VeriKG", layout="wide")


# ------------------------------------------------------------------ loading

@st.cache_resource(show_spinner="Loading the knowledge graph (about 30 s the first time)...")
def load_kg():
    kg = KG.load(KGX, NAME)
    kg.eid_index = {e: i for i, e in enumerate(kg.edges["id"])}
    return kg


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


if not (KGX / f"{NAME}_nodes.tsv").exists():
    st.title("VeriKG")
    st.error(f"No graph found in {KGX}. Build it with `python -m kgc pull` and `python -m kgc all`, "
             "or download the prebuilt graph with `python -m kgc fetch-data`.")
    st.stop()

kg = load_kg()
stats = read_json(KGX / f"{NAME}_stats.json") or {}
prune_rep = read_json(KGX / f"{NAME}_prune_report.json") or {}
has_llm = bool(os.environ.get("NVIDIA_API_KEY") or os.environ.get("LLM_API_KEY"))


def explain_fn(a, b):
    return explain_prediction(kg.edges, a, b, nodes=kg.nodes)


@st.cache_resource
def query_agent():
    return build_query_agent(kg, explain_fn=explain_fn)


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("VeriKG")
    st.caption("A provenance-first knowledge graph of human kinases, their ligands, interaction "
               "partners, diseases and pathways, built and checked by agents.")
    n_keep = int((kg.edges["verdict"] != "prune").sum())
    c1, c2 = st.columns(2)
    c1.metric("Nodes", f"{len(kg.nodes):,}")
    c2.metric("Edges kept", f"{n_keep:,}")
    c1.metric("Edges pruned", f"{len(kg.edges) - n_keep:,}")
    c2.metric("Source", str(stats.get("source_version", "pharos")).replace("pharos:", ""))
    st.divider()
    use_llm = st.toggle("Language model planner", value=has_llm, disabled=not has_llm,
                        help="Needs NVIDIA_API_KEY. Without it a keyword planner is used; "
                             "either way the plan is shown and can be edited.")
    top_k = st.slider("Rows per answer", 5, 100, 25)
    st.divider()
    st.caption("Every answer lists the edges it used. Each edge carries its source, papers, "
               "confidence and the pruning verdict.")


# ------------------------------------------------------------------ helpers

COLORS = {B.PROTEIN: "#cfe2ff", B.GENE: "#d1e7dd", B.CHEMICAL: "#fff3cd", B.DRUG: "#ffe5b4",
          B.DISEASE: "#f8d7da", B.PATHWAY: "#e2d9f3"}


def dot_for(edge_ids, highlight=(), max_edges=60):
    idx = [kg.eid_index[e] for e in dict.fromkeys(edge_ids) if e in kg.eid_index][:max_edges]
    if not idx:
        return None
    sub = kg.edges.iloc[idx]
    nodes = list(dict.fromkeys(list(sub["subject"]) + list(sub["object"])))
    q = lambda s: '"' + str(s).replace('"', "'") + '"'
    lines = ["graph G {", 'graph [overlap=false, splines=true, fontsize=10];',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
             'edge [fontname="Helvetica", fontsize=8];']
    for n in nodes:
        lab = kg.label(n)
        lab = lab if len(lab) <= 32 else lab[:30] + "…"
        pen = ", penwidth=2.5" if n in highlight else ""
        lines.append(f'{q(n)} [label={q(lab)}, fillcolor="{COLORS.get(kg.category(n), "#eeeeee")}"{pen}];')
    for r in sub.to_dict("records"):
        style = ", style=dashed, color=gray" if r["verdict"] == "prune" else ""
        lab = PRED_LABEL.get(r["predicate"], r["predicate"].split(":")[-1])
        lines.append(f'{q(r["subject"])} -- {q(r["object"])} [label={q(lab)}{style}];')
    lines.append("}")
    return "\n".join(lines)


def show_edges(edge_ids, limit=15):
    ids = [e for e in dict.fromkeys(edge_ids) if e in kg.eid_index]
    if not ids:
        return
    st.markdown(f"**Evidence: {len(ids)} edge(s) behind this answer**")
    for eid in ids[:limit]:
        card = edge_card(kg, kg.edges.iloc[kg.eid_index[eid]].to_dict())
        badge = "kept" if card["verdict"] != "prune" else "pruned"
        with st.expander(f"{kg.label(card['subject'])} — {PRED_LABEL.get(card['predicate'], card['predicate'])} — "
                         f"{kg.label(card['object'])}   ·   {card['source']}   ·   {badge} "
                         f"({card['confidence']})"):
            st.markdown(card["explanation"])
            if card["papers"]:
                st.markdown("Papers: " + ", ".join(f"[{u.rstrip('/').split('/')[-1]}]({u})"
                                                   for u in card["papers"][:10]))
            st.caption(f"Edge {card['edge']} · predicate {card['predicate']} · relation {card['relation']}")
    if len(ids) > limit:
        st.caption(f"{len(ids) - limit} more edges not shown.")


def render_answer(ans, log=None):
    st.markdown(ans.summary)
    if ans.rows:
        df = pd.DataFrame(ans.rows)
        hide = [c for c in ("edge", "edges") if c in df]
        st.dataframe(df.drop(columns=hide), width="stretch", hide_index=True)
    dot = dot_for(ans.edge_ids, highlight=[e["id"] for e in ans.plan.entities])
    tabs = st.tabs(["Graph", "Evidence", "Plan and Cypher", "Agent steps"])
    with tabs[0]:
        if dot:
            st.graphviz_chart(dot, width="stretch")
            st.caption("Blue proteins, green genes, yellow compounds, orange approved drugs, red diseases, "
                       "purple pathways. Dashed edges were pruned.")
        else:
            st.caption("No edges to draw for this answer.")
    with tabs[1]:
        show_edges(ans.edge_ids)
    with tabs[2]:
        st.markdown("The plan the agent ran. Edit it and run again if it misread the question.")
        edited = st.text_area("Plan (JSON)", ans.plan.to_json(), height=260, key=f"plan_{id(ans)}")
        if st.button("Run this plan", key=f"run_{id(ans)}"):
            try:
                d = json.loads(edited)
                p = Plan(intent=d["intent"], mentions=d.get("mentions", []), top_k=int(d.get("top_k", 25)),
                         approved_only=bool(d.get("approved_only", False)), planner="edited by user")
                st.session_state["answer"] = execute(kg, ground_plan(kg, p), explain_fn=explain_fn)
                st.session_state["log"] = ["Plan edited by the user; grounded and executed."]
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(f"Could not read the plan: {exc}")
        if ans.cypher:
            st.markdown("Equivalent Cypher, for Neo4j or any openCypher store loaded from the KGX files:")
            st.code(ans.cypher, language="cypher")
        st.caption("Intents the planner can choose: " + "; ".join(f"`{k}` {v}" for k, v in INTENTS.items()))
    with tabs[3]:
        for line in log or []:
            st.markdown(f"- {line}")


# ------------------------------------------------------------------ pages

ask_tab, entity_tab, prune_tab, model_tab, table_tab, dl_tab = st.tabs(
    ["Ask", "Look up", "What was pruned", "Predictions and checks", "Your own table", "Download"])

with ask_tab:
    st.subheader("Ask a question")
    st.caption("Kinases, the compounds that bind them, their interaction partners, linked diseases and "
               "pathways. Names, gene symbols and IDs all work.")
    cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 4].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state["q"] = ex
    q = st.text_input("Question", key="q", placeholder="e.g. Which approved drugs act on kinases linked to "
                                                      "breast cancer?")
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Planning, grounding and querying..."):
            state = query_agent()({"question": q, "use_llm": use_llm, "log": []})
            ans = state["answer"]
            if ans.plan.top_k != top_k and "top" not in q.lower():
                ans.plan.top_k = top_k
                ans = execute(kg, ans.plan, explain_fn=explain_fn)
        st.session_state["answer"], st.session_state["log"] = ans, state.get("log", [])
    if "answer" in st.session_state:
        st.divider()
        render_answer(st.session_state["answer"], st.session_state.get("log"))

with entity_tab:
    st.subheader("Look up an entity")
    name = st.text_input("Gene symbol, compound, drug, disease or ID", placeholder="LRRK2, imatinib, "
                                                                                   "MONDO:0005180 ...")
    if name.strip():
        hits = kg.ground(name)
        if not hits:
            st.warning("Nothing in the graph matches that name.")
        else:
            nid = st.selectbox("Matches", hits[:10], format_func=lambda h: f"{kg.label(h)}  "
                               f"[{CATEGORY_LABEL.get(kg.category(h), kg.category(h))}, {h}]")
            p = ground_plan(kg, Plan(intent="profile", mentions=[{"text": nid}]))
            ans = execute(kg, p)
            st.markdown(f"**{ans.summary}**")
            if ans.rows:
                st.dataframe(pd.DataFrame(ans.rows), width="stretch", hide_index=True)
            e = kg.edges_of(nid, keep_only=False)
            if kg.p2g.get(nid):
                e = pd.concat([e, kg.edges_of(kg.p2g[nid], keep_only=False)])
            st.caption(f"{len(e)} edges touch this entity, {int((e['verdict'] == 'prune').sum())} of them pruned.")

with prune_tab:
    st.subheader("What the pruning agent removed, and why")
    st.markdown(
        "Every edge gets a confidence from named signals: how reliable its source is, whether other sources "
        "or papers back it, how it was produced (curated, statistical, text mined), the measured potency, "
        "whether the rest of the graph supports it, and Open Targets where checked. Edges under the threshold "
        "are pruned: they stay in the full graph with their reasons but are left out of answers.")
    if prune_rep:
        e = prune_rep.get("edges", {})
        c = st.columns(3)
        c[0].metric("Edges scored", f"{e.get('total', 0):,}")
        c[1].metric("Kept", f"{e.get('kept', 0):,}")
        c[2].metric("Pruned", f"{e.get('pruned', 0):,}", help=f"threshold {prune_rep.get('threshold')}")
        a, b = st.columns(2)
        a.markdown("**Pruned by source**")
        a.dataframe(pd.Series(prune_rep.get("pruned_by_primary_source", {}), name="edges"),
                    width="stretch")
        b.markdown("**Most common reasons**")
        b.dataframe(pd.Series(prune_rep.get("top_prune_reasons", {}), name="edges"), width="stretch")
    bench = read_json(RESULTS / f"{NAME}_pruning_benchmark.json")
    if bench:
        st.markdown("**Planted-error test.** Real edges were copied with one end swapped, keeping the original "
                    "source and papers, so they look like a curated record filed under the wrong entity. "
                    "How well does each score rank the planted edges below the real ones (AUROC)?")
        rows = []
        for pred, r in bench["by_predicate"].items():
            for kind in ("random", "degree", "near_miss"):
                if kind in r:
                    rows.append({"edge type": PRED_LABEL.get(pred, pred), "swap": kind.replace("_", "-"),
                                 "planted": r[kind]["planted"],
                                 "evidence only": r[kind]["auroc_evidence_only"],
                                 "evidence + context": r[kind]["auroc_with_context"],
                                 "caught at threshold": r[kind]["recall_at_threshold"],
                                 "real edges pruned": r["real_pruned_share"]})
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
    st.markdown("**Browse pruned edges**")
    pr = kg.edges[kg.edges["verdict"] == "prune"]
    preds = sorted(pr["predicate"].unique())
    pick = st.selectbox("Edge type", preds, format_func=lambda p: PRED_LABEL.get(p, p)) if preds else None
    if pick:
        sub = pr[pr["predicate"] == pick].sort_values("confidence").head(500)
        view = pd.DataFrame({
            "subject": sub["subject"].map(kg.label), "object": sub["object"].map(kg.label),
            "source": sub["primary_knowledge_source"].str.replace("infores:", ""),
            "confidence": sub["confidence"], "reasons": sub["prune_reason"], "edge": sub["id"]})
        st.dataframe(view, width="stretch", hide_index=True)

with model_tab:
    st.subheader("Link prediction: does the agent's graph predict new binders better?")
    st.markdown(
        "Same GNN, same split, same negatives; only the graph changes. Negatives are drawn in proportion to "
        "protein degree, so a model cannot win by ranking popular kinases high. The scaffold split keeps every "
        "compound that shares a ring scaffold with a test compound out of training: the honest test for new "
        "chemistry.")
    rows = []
    for split, path in (("random", RESULTS / f"{NAME}_gnn.json"),
                        ("scaffold", RESULTS / f"{NAME}_scaffold_gnn.json")):
        res = read_json(path) or {}
        for cfg, r in res.items():
            t = r.get("test", {})
            rows.append({"split": split, "graph": cfg,
                         "AUROC": t.get("degree_matched", {}).get("auroc"),
                         "AUPRC": t.get("degree_matched", {}).get("auprc"),
                         "AUROC, unseen compounds": t.get("unseen_chemical", {}).get("auroc"),
                         "Hits@10": t.get("sampled_ranking", {}).get("hits@10"),
                         "MRR": t.get("sampled_ranking", {}).get("mrr")})
    if rows:
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
        st.caption("pharos: the graph as pulled. pharos_fp: + Morgan fingerprints as compound features. "
                   "verikg: + structure-similarity edges. verikg_pruned: verikg after pruning, re-run on the "
                   "training graph only.")
    else:
        st.info("Run `python -m kgc gnn` to fill this table.")
    st.markdown("**Score a pair**")
    if kg.predictions is None:
        st.caption("Run `python -m kgc predict` to enable scores for unrecorded pairs.")
    else:
        a, b = st.columns(2)
        cq = a.text_input("Compound", "gefitinib")
        pq = b.text_input("Protein", "ERBB2")
        ch = kg.ground(cq, prefer={B.CHEMICAL, B.DRUG})
        ph = kg.ground(pq, prefer={B.PROTEIN})
        if ch and ph:
            c_id, p_id = ch[0], kg.g2p.get(ph[0], ph[0])
            sc = kg.predictions.score(c_id, p_id)
            if sc is None:
                st.warning("One of these is not in the model.")
            else:
                st.metric(f"{kg.label(c_id)} → {kg.label(p_id)}", f"{sc:.3f}")
                st.text(explain_fn(c_id, p_id)["text"])

with table_tab:
    st.subheader("Turn your own table into a knowledge graph")
    st.markdown("Upload a CSV or TSV of relationships (a screen, a gene list, a curated sheet). The agent "
                "proposes which columns are the entities, what they are and how they relate; you check the "
                "mapping, then it builds, validates and scores the graph the same way it does for Pharos.")
    up = st.file_uploader("Table", type=["csv", "tsv", "txt"])
    if up is not None:
        from kgc.tabular import CATEGORIES, PREDICATES, suggest_mapping
        sep = "\t" if up.name.lower().endswith((".tsv", ".txt")) else ","
        df = pd.read_csv(up, sep=sep)
        st.dataframe(df.head(10), width="stretch")
        m = suggest_mapping(df)
        cols = list(df.columns)
        c = st.columns(3)
        m["subject_column"] = c[0].selectbox("Subject column", cols, index=cols.index(m["subject_column"])
                                             if m["subject_column"] in cols else 0)
        m["predicate"] = c[1].selectbox("Relationship", PREDICATES, index=PREDICATES.index(m["predicate"])
                                        if m["predicate"] in PREDICATES else 0)
        m["object_column"] = c[2].selectbox("Object column", cols, index=cols.index(m["object_column"])
                                            if m["object_column"] in cols else min(1, len(cols) - 1))
        cats = CATEGORIES + ["biolink:NamedThing"]
        m["subject_category"] = c[0].selectbox("Subject is a", cats, index=cats.index(m["subject_category"])
                                               if m["subject_category"] in cats else len(cats) - 1)
        m["object_category"] = c[2].selectbox("Object is a", cats, index=cats.index(m["object_category"])
                                              if m["object_category"] in cats else len(cats) - 1)
        m["source_name"] = c[1].text_input("Source name", m["source_name"])
        if st.button("Build graph", type="primary"):
            box = st.empty()
            lines = []

            def step(name, s):
                if s.get("log"):
                    lines.append(f"**{name}**: {s['log'][-1]}")
                    box.markdown("\n\n".join(lines))
            out_name = Path(up.name).stem.replace(" ", "_")
            final = run_build({"source": "table", "name": out_name, "data_dir": str(DATA / "uploads"),
                               "table": df, "mapping": m}, on_step=step)
            if final.get("outputs"):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for k, v in final["outputs"].items():
                        z.write(v, Path(v).name)
                st.download_button("Download KGX files", buf.getvalue(), f"{out_name}_kgx.zip")

with dl_tab:
    st.subheader("Download the graph")
    st.markdown("KGX TSV (Biolink categories and predicates, ontology class on every node, relation IRI and "
                "full provenance on every edge). Load into Neo4j, KGX tools, or pandas.")
    for label, fname in (("Nodes", f"{NAME}_nodes.tsv"), ("All edges with verdicts", f"{NAME}_edges_scored.tsv"),
                         ("Pruned graph (kept edges only)", f"{NAME}_edges_pruned.tsv"),
                         ("Verdict table with UniProt IDs", f"{NAME}_verdicts.tsv"),
                         ("PyTorch Geometric HeteroData", f"{NAME}_pyg.pt")):
        p = KGX / fname
        if p.exists():
            st.markdown(f"- **{label}**: `{p}` ({p.stat().st_size / 1e6:.1f} MB)")
    st.caption("Files are large, so they are listed with their paths rather than offered as browser downloads.")
