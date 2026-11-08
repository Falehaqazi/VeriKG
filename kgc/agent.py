"""The VeriKG agents, as explicit LangGraph state machines.

Build agent:  ingest -> map -> validate -> (stop on errors) -> enrich -> prune -> export
Query agent:  plan -> ground -> (clarify if nothing matched) -> execute -> answer

Every transition is inspectable: each node appends a plain-language line to
the run log, so a user sees what the agent did and why, and a run can be
replayed. Language models (when configured) are used only to propose a query
plan or a table mapping; every node that admits data into the graph is
deterministic code. LangGraph is optional at runtime: without it the same
nodes run in sequence.
"""

import json
from pathlib import Path
from typing import Any, Optional, TypedDict

import pandas as pd

from . import biolink as B


class BuildState(TypedDict, total=False):
    source: str                 # "pharos" or "table"
    name: str
    data_dir: str
    family: str
    limit: Optional[int]
    table: Any                  # DataFrame for source == "table"
    mapping: dict
    log: list
    nodes: Any
    edges: Any
    validation: dict
    prune_report: dict
    outputs: dict
    failed: bool


def _say(state, msg):
    state.setdefault("log", []).append(msg)
    return state


def node_ingest(state):
    d = Path(state["data_dir"])
    if state["source"] == "pharos":
        from .pharos import pull_family
        raw = d / "raw" / f"{state['name']}.jsonl"
        pull_family(raw, family=state.get("family", "Kinase"), limit=state.get("limit"),
                    log=lambda m: _say(state, m))
        _say(state, f"Ingest: pulled Pharos family '{state.get('family')}' to {raw.name}, raw responses kept.")
    else:
        from .tabular import suggest_mapping
        if not state.get("mapping"):
            state["mapping"] = suggest_mapping(state["table"])
        m = state["mapping"]
        _say(state, f"Ingest: read a table with {len(state['table'])} rows. Mapping: '{m['subject_column']}' as "
                    f"{m['subject_category']} --{m['predicate']}--> '{m['object_column']}' as {m['object_category']}.")
    return state


def node_map(state):
    d = Path(state["data_dir"])
    if state["source"] == "pharos":
        from .kgx import build_kgx
        stats = build_kgx(d / "raw" / f"{state['name']}.jsonl", d / "kgx", name=state["name"])
        state["nodes"] = pd.read_csv(d / "kgx" / f"{state['name']}_nodes.tsv", sep="\t", low_memory=False)
        state["edges"] = pd.read_csv(d / "kgx" / f"{state['name']}_edges.tsv", sep="\t", low_memory=False)
        _say(state, f"Map: {stats['nodes']} nodes and {stats['edges']} edges in Biolink KGX, "
                    f"every edge with its source, knowledge level and papers.")
    else:
        from .tabular import build_from_table
        n, e, issues = build_from_table(state["table"], state["mapping"])
        state["nodes"], state["edges"] = n, e
        _say(state, f"Map: {len(n)} nodes and {len(e)} edges ({issues['skipped_rows']} rows skipped, "
                    f"{issues['local_ids']} names without a standard identifier).")
    return state


def node_validate(state):
    if state["source"] == "pharos":
        from .validate import validate
        rep = validate(state["nodes"], state["edges"])
    else:
        from .tabular import validate_generic
        rep = validate_generic(state["nodes"], state["edges"])
    state["validation"] = rep
    state["failed"] = not rep["ok"]
    _say(state, f"Validate: {'passed' if rep['ok'] else 'FAILED'}; errors {rep['errors'] or 'none'}; "
                f"warnings {rep['warnings'] or 'none'}.")
    return state


def node_enrich(state):
    n = state["nodes"]
    if "smiles" not in n or not n["category"].isin(B.CHEMICALS).any():
        _say(state, "Enrich: no chemical structures in this graph; structure similarity skipped.")
        return state
    from .enrich import fingerprints, nearest_neighbours, similarity_edges
    ids, fps, _ = fingerprints(n)
    pairs = nearest_neighbours(fps)
    e = state["edges"]
    sim = similarity_edges(ids, pairs, str(e["source_version"].iloc[0]), str(e["ingest_run"].iloc[0]))
    state["edges"] = pd.concat([e, sim], ignore_index=True)
    _say(state, f"Enrich: {len(sim)} structure-similarity links between {len(ids)} compounds.")
    return state


def node_prune(state):
    from .prune import prune_graph
    e = state["edges"]
    if state["source"] == "pharos":
        from .plausibility import score_plausibility
        ctx, summary, _ = score_plausibility(e)
    else:
        from .tabular import topology_plausibility
        p = topology_plausibility(e)
        ctx = pd.DataFrame({"plausibility": p, "plausibility_notes": "topology",
                            "context_size": 3}, index=e.index)
        summary = {}
    scored, kept, _, rep = prune_graph(state["nodes"], e, context=ctx)
    rep["context_models"] = summary
    state["edges"] = scored
    state["prune_report"] = rep
    _say(state, f"Prune: kept {rep['edges']['kept']} of {rep['edges']['total']} edges; "
                f"removed {rep['edges']['pruned']}, each with its reasons.")
    return state


def node_export(state):
    d = Path(state["data_dir"]) / "kgx"
    d.mkdir(parents=True, exist_ok=True)
    name = state["name"]
    state["nodes"].to_csv(d / f"{name}_nodes.tsv", sep="\t", index=False)
    state["edges"].to_csv(d / f"{name}_edges_scored.tsv", sep="\t", index=False)
    kept = state["edges"][state["edges"]["verdict"] == "keep"]
    kept.to_csv(d / f"{name}_edges_pruned.tsv", sep="\t", index=False)
    (d / f"{name}_prune_report.json").write_text(json.dumps(state["prune_report"], indent=2), encoding="utf-8")
    state["outputs"] = {"nodes": str(d / f"{name}_nodes.tsv"),
                        "edges_scored": str(d / f"{name}_edges_scored.tsv"),
                        "edges_pruned": str(d / f"{name}_edges_pruned.tsv")}
    _say(state, f"Export: KGX nodes, scored edges and pruned edges written for '{name}'. Ready for questions.")
    return state


BUILD_STEPS = [("ingest", node_ingest), ("map", node_map), ("validate", node_validate),
               ("enrich", node_enrich), ("prune", node_prune), ("export", node_export)]


def build_graph_agent():
    """Compile the build agent with LangGraph (None if LangGraph is missing)."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None
    g = StateGraph(BuildState)
    for name, fn in BUILD_STEPS:
        g.add_node(name, fn)
    g.set_entry_point("ingest")
    g.add_edge("ingest", "map")
    g.add_edge("map", "validate")
    g.add_conditional_edges("validate", lambda s: "stop" if s.get("failed") else "go",
                            {"stop": END, "go": "enrich"})
    g.add_edge("enrich", "prune")
    g.add_edge("prune", "export")
    g.add_edge("export", END)
    return g.compile()


def run_build(state, on_step=None):
    """Run the build agent; on_step(name, state) is called after every node."""
    agent = build_graph_agent()
    if agent is not None:
        final = dict(state)
        for update in agent.stream(state, stream_mode="updates"):
            for name, st in update.items():
                final.update(st)
                if on_step:
                    on_step(name, final)
        return final
    for name, fn in BUILD_STEPS:
        state = fn(state)
        if on_step:
            on_step(name, state)
        if name == "validate" and state.get("failed"):
            break
    return state


# ---------------------------------------------------------------- query agent

class QueryState(TypedDict, total=False):
    question: str
    use_llm: bool
    plan: Any
    answer: Any
    log: list


def build_query_agent(kg, explain_fn=None):
    from .query import Answer, execute, ground_plan, plan_llm, plan_rules

    def plan_node(s):
        p = plan_llm(s["question"]) if s.get("use_llm", True) else None
        how = "language model" if p else "keyword rules"
        s["plan"] = p or plan_rules(kg, s["question"])
        return _say(s, f"Plan ({how}): intent '{s['plan'].intent}', mentions "
                       f"{[m['text'] for m in s['plan'].mentions]}.")

    def ground_node(s):
        s["plan"] = ground_plan(kg, s["plan"])
        return _say(s, "Ground: " + (", ".join(f"'{e['text']}' -> {e['id']}" for e in s["plan"].entities)
                                     or "nothing matched"))

    def clarify_node(s):
        s["answer"] = Answer(s["plan"], "I could not find those names in the graph. Try a gene symbol "
                                        "(EGFR), a drug (imatinib) or a disease (Parkinson disease).",
                             [], [], [], "")
        return _say(s, "Clarify: asked the user to rephrase.")

    def execute_node(s):
        s["answer"] = execute(kg, s["plan"], explain_fn=explain_fn)
        return _say(s, f"Execute: {len(s['answer'].rows)} rows, {len(s['answer'].edge_ids)} edges used.")

    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        def run(s):
            s = ground_node(plan_node(s))
            return clarify_node(s) if not s["plan"].entities else execute_node(s)
        return run
    g = StateGraph(QueryState)
    g.add_node("plan", plan_node)
    g.add_node("ground", ground_node)
    g.add_node("clarify", clarify_node)
    g.add_node("execute", execute_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "ground")
    g.add_conditional_edges("ground", lambda s: "execute" if s["plan"].entities else "clarify",
                            {"execute": "execute", "clarify": "clarify"})
    g.add_edge("clarify", END)
    g.add_edge("execute", END)
    compiled = g.compile()
    return lambda s: compiled.invoke(s)
