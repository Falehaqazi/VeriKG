"""Plain-language questions over the graph, for people who do not write Cypher.

    question --(planner)--> plan --(grounding)--> plan with IDs --(executor)--> answer

The planner turns a question into a small JSON plan: an intent from a fixed
list, the entity mentions, and a few filters. It is a language model when one
is configured (NVIDIA NIM by default, any OpenAI-compatible endpoint works) and
a deterministic keyword parser otherwise; the app works without any API key.
The model only proposes. Code grounds every mention to a node ID, checks the
intent is one it knows, runs the query and writes the numbers in the answer.
The plan is shown to the user, editable, together with the equivalent Cypher
so a KG-literate user can check or reuse it.
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

from . import biolink as B

INTENTS = {
    "ligands_of": "compounds that bind a protein",
    "targets_of": "proteins a compound binds",
    "proteins_for_disease": "proteins whose genes are linked to a disease",
    "drugs_for_disease": "drugs that act on proteins linked to a disease (multi-hop)",
    "disease_links": "diseases linked to a protein",
    "explain": "why two entities are connected: sources, evidence, verdict",
    "predict_targets": "proteins a compound may bind that are not in the graph yet",
    "predict_ligands": "compounds that may bind a protein, not in the graph yet",
    "similar_compounds": "structurally similar compounds",
    "path": "how two entities are connected through the graph",
    "pruned": "what the pruning agent removed about an entity, and why",
    "profile": "a summary card for one entity",
}

CATEGORY_LABEL = {
    B.PROTEIN: "protein", B.GENE: "gene", B.CHEMICAL: "compound", B.DRUG: "approved drug",
    B.DISEASE: "disease", B.PATHWAY: "pathway",
}
PRED_LABEL = {
    B.INTERACTS_PHYS: "binds", B.PPI: "interacts with", B.ASSOC_CONDITION: "is linked to",
    B.PARTICIPATES_IN: "takes part in", B.TRANSLATES_TO: "encodes", B.TREATS: "is used for",
    B.SIMILAR: "is structurally similar to",
}

EXAMPLES = [
    "Which compounds bind EGFR?",
    "What does imatinib target?",
    "Which kinases are linked to Parkinson disease?",
    "Which approved drugs act on kinases linked to breast cancer?",
    "Why is dasatinib linked to ABL1?",
    "What else might gefitinib bind?",
    "How is LRRK2 connected to Parkinson disease?",
    "What did you remove about CDK2 and why?",
]


def _t(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


# ---------------------------------------------------------------- graph access

class KG:
    """The scored graph plus the lookups the planner and executor need."""

    def __init__(self, nodes, edges, predictions=None):
        self.nodes = nodes.reset_index(drop=True)
        self.edges = edges.reset_index(drop=True)
        if "verdict" not in self.edges:
            self.edges["verdict"] = "keep"
        if "confidence" not in self.edges:
            self.edges["confidence"] = 1.0
        self.node = self.nodes.set_index("id").to_dict("index")
        self.predictions = predictions
        self._index_names()
        self.out = defaultdict(list)
        self.inc = defaultdict(list)
        for i, (s, o) in enumerate(zip(self.edges["subject"], self.edges["object"])):
            self.out[s].append(i)
            self.inc[o].append(i)
        tr = self.edges[self.edges["predicate"] == B.TRANSLATES_TO]
        self.g2p = dict(zip(tr["subject"], tr["object"]))
        self.p2g = dict(zip(tr["object"], tr["subject"]))

    @classmethod
    def load(cls, kgx_dir, name="pharos_kinase"):
        kgx_dir = Path(kgx_dir)
        nodes = pd.read_csv(kgx_dir / f"{name}_nodes.tsv", sep="\t", low_memory=False)
        for cand in (f"{name}_edges_scored.tsv", f"{name}_enriched_edges.tsv", f"{name}_edges.tsv"):
            if (kgx_dir / cand).exists():
                edges = pd.read_csv(kgx_dir / cand, sep="\t", low_memory=False)
                break
        pred_path = kgx_dir.parent / "results" / f"{name}_predictor.npz"
        preds = None
        if pred_path.exists():
            from .predictor import Predictor
            preds = Predictor(pred_path)
        return cls(nodes, edges, preds)

    def _index_names(self):
        self.by_key = defaultdict(set)
        for nid, cat, name, sym in zip(self.nodes["id"], self.nodes["category"],
                                       self.nodes.get("name", pd.Series([""] * len(self.nodes))),
                                       self.nodes.get("symbol", pd.Series([""] * len(self.nodes)))):
            for k in (nid, _t(name), _t(sym), nid.split(":", 1)[-1]):
                k = _norm(k)
                if k:
                    self.by_key[k].add(nid)
        self.keys = list(self.by_key)

    def label(self, nid):
        n = self.node.get(nid) or {}
        sym, name = _t(n.get("symbol")), _t(n.get("name"))
        if sym and name and sym.lower() != name.lower():
            return f"{sym} ({name})"
        return name or sym or nid

    def category(self, nid):
        return _t((self.node.get(nid) or {}).get("category"))

    def edges_of(self, nid, predicate=None, direction="both", keep_only=True):
        ids = (self.out[nid] if direction in ("both", "out") else []) + \
              (self.inc[nid] if direction in ("both", "in") else [])
        e = self.edges.iloc[ids] if ids else self.edges.iloc[0:0]
        if predicate:
            e = e[e["predicate"] == predicate]
        if keep_only:
            e = e[e["verdict"] != "prune"]
        return e

    # --- grounding ---
    def ground(self, text, prefer=None):
        """Return candidate node IDs for a mention, best first."""
        k = _norm(text)
        if not k:
            return []
        hits = list(self.by_key.get(k, []))
        if not hits and len(k) >= 4:
            contain = [key for key in self.keys if key.startswith(k + " ") or f" {k} " in f" {key} "]
            contain.sort(key=len)
            for key in contain[:5]:
                hits.extend(self.by_key[key])
        if not hits and len(k) >= 4:
            for key in get_close_matches(k, self.keys, n=3, cutoff=0.88):
                hits.extend(self.by_key[key])
        order = {B.PROTEIN: 0, B.DRUG: 1, B.CHEMICAL: 2, B.DISEASE: 3, B.GENE: 4, B.PATHWAY: 5}
        if prefer:
            order = {c: (0 if c in prefer else 10 + order.get(c, 9)) for c in order}
        hits = sorted(set(hits), key=lambda h: (order.get(self.category(h), 9),
                                                 -len(self.out[h]) - len(self.inc[h])))
        return hits


def _norm(s):
    s = _t(s).lower().replace("'s ", " ").replace("’s ", " ")
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[^a-z0-9:.\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- planning

@dataclass
class Plan:
    intent: str
    mentions: list = field(default_factory=list)   # [{"text": ..., "role": ...}]
    entities: list = field(default_factory=list)   # grounded: [{"text","id","label","category"}]
    top_k: int = 25
    min_confidence: float = 0.0
    approved_only: bool = False
    planner: str = "rules"
    notes: list = field(default_factory=list)

    def to_json(self):
        d = asdict(self)
        return json.dumps(d, indent=2)


CHEM_WORDS = r"(compound|ligand|inhibitor|molecule|chemical|drug|medicine|binder)s?"
DISEASE_HINT = r"(disease|cancer|syndrome|disorder|carcinoma|leukemia|leukaemia|lymphoma|tumou?r|diabetes|" \
               r"arthritis|fibrosis|deficiency|dementia|sclerosis|infection|asthma)"


# function words a multi-word mention never starts or ends with
TRIM = {"to", "of", "the", "a", "an", "for", "with", "in", "on", "and", "by", "from", "is", "are",
        "does", "do", "which", "what", "how", "why", "about", "that", "be", "between", "any", "all"}


def _mentions_from(kg, q):
    """Longest-first n-gram scan of the question for known entity names."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-.:'’]*", q)
    stop = {"which", "what", "who", "why", "how", "is", "are", "the", "a", "an", "of", "to", "and",
            "for", "with", "in", "on", "does", "do", "bind", "binds", "target", "targets", "linked",
            "kinase", "kinases", "drugs", "drug", "compounds", "compound", "approved", "act", "acts",
            "connected", "else", "might", "could", "remove", "removed", "about", "you", "did", "me",
            "tell", "show", "find", "list", "protein", "proteins", "gene", "genes", "disease",
            "diseases", "between", "new", "other", "similar", "inhibitors", "inhibitor", "be",
            "that", "by", "from", "known", "any", "all", "our", "my", "please", "can", "graph"}
    used, out = set(), []
    for n in range(min(6, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            if any(j in used for j in range(i, i + n)):
                continue
            span = " ".join(words[i:i + n]).strip(".,?!")
            if n == 1 and (span.lower() in stop or len(span) < 3):
                continue
            first, last = words[i].lower().strip(".,?!"), words[i + n - 1].lower().strip(".,?!")
            if n > 1 and (first in TRIM or last in TRIM):
                continue  # "to breast cancer" -> let the shorter span "breast cancer" match
            hits = kg.ground(span)
            if hits:
                out.append((i, span, hits))
                used.update(range(i, i + n))
    out.sort()
    return [{"text": s, "candidates": h[:5]} for _, s, h in out]


def plan_rules(kg, question):
    q = question.strip()
    ql = q.lower()
    ments = _mentions_from(kg, q)
    cats = [kg.category(m["candidates"][0]) for m in ments]
    has = lambda *cs: any(c in cs for c in cats)
    two = len(ments) >= 2

    if re.search(r"\b(remov|prun|drop|unreliable|filtered|cut)\w*", ql):
        intent = "pruned"
    elif re.search(r"\b(why|evidence|lineage|how do (we|you) know|source of|proof)\b", ql) and two:
        intent = "explain"
    elif two and re.match(r"(is|are|does|do|did|can)\b", ql) and \
            re.search(r"\b(bind|target|inhibit|link|associat|interact|treat)\w*", ql):
        intent = "explain"  # "Is sorafenib linked to KDR?": a yes/no about one link
    elif re.search(r"\b(predict|might|could|likely|novel|new targets?|what else|repurpos|off.?target)\w*", ql):
        intent = "predict_targets" if has(B.CHEMICAL, B.DRUG) else "predict_ligands"
    elif re.search(r"\b(similar|analog|analogue|look.?alike)\w*", ql) and has(B.CHEMICAL, B.DRUG):
        intent = "similar_compounds"
    elif re.search(r"\b(connect|path|route|relat|link between|go from)\w*", ql) and two:
        intent = "path"
    elif has(B.DISEASE) and re.search(CHEM_WORDS, ql):
        intent = "drugs_for_disease"
    elif has(B.DISEASE):
        intent = "proteins_for_disease"
    elif has(B.CHEMICAL, B.DRUG) and re.search(r"\b(target|bind|act|inhibit|hit)\w*", ql):
        intent = "targets_of"
    elif has(B.PROTEIN, B.GENE) and re.search(CHEM_WORDS + r"|\bbind", ql):
        intent = "ligands_of"
    elif has(B.PROTEIN, B.GENE) and re.search(DISEASE_HINT + r"|\billness|condition", ql):
        intent = "disease_links"
    elif ments:
        intent = "profile"
    else:
        intent = "profile"
    p = Plan(intent=intent, mentions=[{"text": m["text"]} for m in ments], planner="rules")
    p.approved_only = bool(re.search(r"\b(approved|clinical|marketed|fda)\b", ql))
    m = re.search(r"\btop (\d+)\b", ql)
    if m:
        p.top_k = int(m.group(1))
    return p


SYSTEM_PROMPT = """You turn a biomedical question into a JSON query plan for a knowledge graph of
proteins (kinases and their interaction partners), genes, compounds, approved drugs, diseases and
pathways. Reply with JSON only, no prose. Schema:
{"intent": one of %s,
 "mentions": [{"text": exact entity name as written in the question}],
 "approved_only": true|false, "top_k": integer}
Intents: %s
Use the entity names exactly as they appear in the question. Do not invent IDs."""


def plan_llm(question, model=None, base_url=None, api_key=None, timeout=30):
    """Ask an OpenAI-compatible chat endpoint for a plan. Returns Plan or None."""
    import requests
    api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        return None
    base_url = base_url or os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model = model or os.environ.get("LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
    sys_p = SYSTEM_PROMPT % (json.dumps(list(INTENTS)), json.dumps(INTENTS, indent=0))
    try:
        r = requests.post(f"{base_url}/chat/completions", timeout=timeout,
                          headers={"Authorization": f"Bearer {api_key}"},
                          json={"model": model, "temperature": 0, "max_tokens": 400,
                                "messages": [{"role": "system", "content": "/no_think\n" + sys_p},
                                             {"role": "user", "content": question}]})
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
        js = json.loads(re.search(r"\{.*\}", txt, flags=re.S).group(0))
    except Exception:
        return None
    if js.get("intent") not in INTENTS:
        return None
    return Plan(intent=js["intent"], mentions=[{"text": str(m.get("text", ""))} for m in js.get("mentions", [])
                                               if isinstance(m, dict)],
                approved_only=bool(js.get("approved_only", False)),
                top_k=int(js.get("top_k") or 25), planner=f"llm:{model}")


def make_plan(kg, question, use_llm=True):
    p = plan_llm(question) if use_llm else None
    if p is None:
        p = plan_rules(kg, question)
    return ground_plan(kg, p)


def ground_plan(kg, plan):
    prefer = {
        "ligands_of": {B.PROTEIN}, "predict_ligands": {B.PROTEIN}, "disease_links": {B.PROTEIN},
        "targets_of": {B.CHEMICAL, B.DRUG}, "predict_targets": {B.CHEMICAL, B.DRUG},
        "similar_compounds": {B.CHEMICAL, B.DRUG},
        "proteins_for_disease": {B.DISEASE}, "drugs_for_disease": {B.DISEASE},
    }.get(plan.intent)
    ents = []
    for m in plan.mentions:
        hits = kg.ground(m["text"], prefer=prefer)
        if not hits:
            plan.notes.append(f"could not find '{m['text']}' in the graph")
            continue
        nid = hits[0]
        # a gene symbol matches both the gene and its protein; use the protein
        if kg.category(nid) == B.GENE and nid in kg.g2p:
            nid = kg.g2p[nid]
        ents.append({"text": m["text"], "id": nid, "label": kg.label(nid),
                     "category": CATEGORY_LABEL.get(kg.category(nid), kg.category(nid)),
                     "alternatives": [h for h in hits[1:4]]})
    plan.entities = ents
    return plan


# ---------------------------------------------------------------- execution

@dataclass
class Answer:
    plan: Plan
    summary: str
    columns: list
    rows: list
    edge_ids: list
    cypher: str
    node_ids: list = field(default_factory=list)


def _first(plan, *cats):
    for e in plan.entities:
        if not cats or e["category"] in cats:
            return e
    return None


def _edge_rows(kg, e, other_col, other_is):
    rows = []
    for r in e.to_dict("records"):
        other = r[other_is]
        rows.append({other_col: kg.label(other), "id": other,
                     "confidence": r.get("confidence"),
                     "potency (-log10 M)": r.get("activity_value_max") if r["predicate"] == B.INTERACTS_PHYS else None,
                     "sources": _t(r.get("supporting_sources")).replace("infores:", "").replace("|", ", "),
                     "papers": len([x for x in _t(r.get("publications")).split("|") if x]),
                     "edge": r["id"]})
    return rows


def execute(kg, plan, explain_fn=None):
    it = plan.intent
    k = plan.top_k
    if it not in INTENTS:
        return Answer(plan, f"Unknown intent '{it}'.", [], [], [], "")
    if not plan.entities:
        return Answer(plan, "I could not match any entity in the question to the graph. "
                            "Try a gene symbol (EGFR), a drug name (imatinib) or a disease name.",
                      [], [], [], "")

    if it in ("ligands_of", "targets_of", "disease_links"):
        ent = _first(plan, "protein", "gene") if it != "targets_of" else _first(plan, "compound", "approved drug")
        ent = ent or plan.entities[0]
        nid = ent["id"]
        if it == "ligands_of":
            e = kg.edges_of(nid, B.INTERACTS_PHYS, "in")
            if plan.approved_only:
                e = e[e["subject"].map(kg.category) == B.DRUG]
            e = e.sort_values(["activity_value_max", "confidence"], ascending=False, na_position="last")
            rows = _edge_rows(kg, e.head(k), "compound", "subject")
            summ = (f"{len(e)} {'approved drugs' if plan.approved_only else 'compounds'} bind {ent['label']} "
                    f"(pruned edges excluded). Showing the {min(k, len(e))} most potent.")
            cy = (f"MATCH (c)-[r:`{B.INTERACTS_PHYS}`]->(p {{id:'{nid}'}}) WHERE r.verdict='keep'"
                  f"{' AND c.category=' + repr(B.DRUG) if plan.approved_only else ''} "
                  f"RETURN c.name, r.activity_value_max, r.confidence ORDER BY r.activity_value_max DESC LIMIT {k}")
        elif it == "targets_of":
            e = kg.edges_of(nid, B.INTERACTS_PHYS, "out").sort_values(
                ["activity_value_max", "confidence"], ascending=False, na_position="last")
            rows = _edge_rows(kg, e.head(k), "protein", "object")
            summ = f"{ent['label']} binds {len(e)} proteins in the graph. Strongest first."
            cy = (f"MATCH (c {{id:'{nid}'}})-[r:`{B.INTERACTS_PHYS}`]->(p) WHERE r.verdict='keep' "
                  f"RETURN p.symbol, r.activity_value_max ORDER BY r.activity_value_max DESC LIMIT {k}")
        else:
            gene = kg.p2g.get(nid, nid)
            e = kg.edges_of(gene, B.ASSOC_CONDITION, "out").sort_values("confidence", ascending=False)
            rows = _edge_rows(kg, e.head(k), "disease", "object")
            summ = f"{ent['label']} is linked to {len(e)} diseases (kept edges). Most confident first."
            cy = (f"MATCH (g)-[:`{B.TRANSLATES_TO}`]->(p {{id:'{nid}'}}), (g)-[r:`{B.ASSOC_CONDITION}`]->(d) "
                  f"WHERE r.verdict='keep' RETURN d.name, r.confidence ORDER BY r.confidence DESC LIMIT {k}")
        return Answer(plan, summ, list(rows[0]) if rows else [], rows, e.head(k)["id"].tolist(), cy,
                      [nid] + [r["id"] for r in rows])

    if it in ("proteins_for_disease", "drugs_for_disease"):
        ent = _first(plan, "disease") or plan.entities[0]
        did = ent["id"]
        ge = kg.edges_of(did, B.ASSOC_CONDITION, "in").sort_values("confidence", ascending=False)
        if it == "proteins_for_disease":
            rows = []
            for r in ge.head(k).to_dict("records"):
                prot = kg.g2p.get(r["subject"], r["subject"])
                rows.append({"protein": kg.label(prot), "id": prot, "confidence": r["confidence"],
                             "sources": _t(r.get("supporting_sources")).replace("infores:", "").replace("|", ", "),
                             "edge": r["id"]})
            summ = f"{len(ge)} kinase genes are linked to {ent['label']} (kept edges). Most confident first."
            cy = (f"MATCH (g:`{B.GENE}`)-[r:`{B.ASSOC_CONDITION}`]->(d {{id:'{did}'}}), "
                  f"(g)-[:`{B.TRANSLATES_TO}`]->(p) WHERE r.verdict='keep' "
                  f"RETURN p.symbol, r.confidence ORDER BY r.confidence DESC LIMIT {k}")
            return Answer(plan, summ, list(rows[0]) if rows else [], rows, ge.head(k)["id"].tolist(), cy,
                          [did] + [r["id"] for r in rows])
        # multi-hop: disease <- gene -> protein <- drug ; plus direct indications
        rows, used = [], []
        direct = kg.edges_of(did, B.TREATS, "in")
        indicated = set(direct["subject"])
        best = {}
        for g, gconf, geid in zip(ge["subject"], ge["confidence"], ge["id"]):
            prot = kg.g2p.get(g)
            if not prot:
                continue
            be = kg.edges_of(prot, B.INTERACTS_PHYS, "in")
            if plan.approved_only:
                be = be[be["subject"].map(kg.category) == B.DRUG]
            for c, conf, act, beid in zip(be["subject"], be["confidence"], be.get("activity_value_max"), be["id"]):
                score = float(gconf) * float(conf)
                cur = best.get(c)
                if cur is None or score > cur["score"]:
                    best[c] = {"score": score, "via": prot, "act": act, "edges": [geid, beid]}
        ranked = sorted(best.items(), key=lambda kv: (kv[0] not in indicated,
                                                      kg.category(kv[0]) != B.DRUG, -kv[1]["score"]))
        for c, v in ranked[:k]:
            rows.append({"compound": kg.label(c), "id": c,
                         "approved drug": kg.category(c) == B.DRUG,
                         "indicated for this disease": c in indicated,
                         "via protein": kg.label(v["via"]),
                         "path confidence": round(v["score"], 3),
                         "potency (-log10 M)": v["act"], "edges": ", ".join(v["edges"])})
            used.extend(v["edges"])
        n_drugs = sum(1 for c in best if kg.category(c) == B.DRUG)
        summ = (f"{len(best)} {'approved drugs' if plan.approved_only else 'compounds'} bind a protein whose gene "
                f"is linked to {ent['label']} ({n_drugs} approved drugs; {len(indicated)} already indicated for it). "
                f"Path confidence multiplies the two edge confidences.")
        cy = (f"MATCH (c)-[b:`{B.INTERACTS_PHYS}`]->(p)<-[:`{B.TRANSLATES_TO}`]-(g)-[a:`{B.ASSOC_CONDITION}`]->"
              f"(d {{id:'{did}'}}) WHERE a.verdict='keep' AND b.verdict='keep'"
              f"{' AND c.category=' + repr(B.DRUG) if plan.approved_only else ''} "
              f"RETURN c.name, p.symbol, a.confidence*b.confidence AS path_conf ORDER BY path_conf DESC LIMIT {k}")
        return Answer(plan, summ, list(rows[0]) if rows else [], rows, used, cy,
                      [did] + [r["id"] for r in rows] + [kg.g2p.get(x) for x in ge["subject"].head(k)])

    if it == "explain":
        if len(plan.entities) < 2:
            return Answer(plan, "Explaining a link needs two entities.", [], [], [], "")
        a, b = plan.entities[0]["id"], plan.entities[1]["id"]
        a2, b2 = kg.p2g.get(a, a), kg.p2g.get(b, b)
        m = kg.edges[((kg.edges["subject"].isin([a, a2])) & (kg.edges["object"].isin([b, b2])))
                     | ((kg.edges["subject"].isin([b, b2])) & (kg.edges["object"].isin([a, a2])))]
        if len(m):
            rows = [edge_card(kg, r) for r in m.to_dict("records")]
            summ = "\n\n".join(r["explanation"] for r in rows)
            cy = (f"MATCH (a {{id:'{a}'}})-[r]-(b {{id:'{b}'}}) RETURN type(r), r.primary_knowledge_source, "
                  f"r.publications, r.confidence, r.verdict, r.confidence_notes")
            return Answer(plan, summ, list(rows[0]), rows, m["id"].tolist(), cy, [a, b])
        if explain_fn is not None:
            res = explain_fn(a, b)
            return Answer(plan, res["text"], [], [], [x for v in res.get("paths", {}).values()
                                                     for p in v for x in p["edges"]], "", [a, b])
        plan.intent = "path"
        return execute(kg, plan)

    if it in ("predict_targets", "predict_ligands"):
        ent = plan.entities[0]
        nid = ent["id"]
        if kg.predictions is None:
            return Answer(plan, "No link predictions available; run `python -m kgc predict` first.",
                          [], [], [], "")
        task = kg.edges[kg.edges["predicate"] == B.INTERACTS_PHYS]
        if it == "predict_targets":
            known = set(task.loc[task["subject"] == nid, "object"])
            cands = sorted(set(task["object"]))
            hits = kg.predictions.targets_for(nid, cands, exclude=known, k=k)
            col = "protein"
        else:
            known = set(task.loc[task["object"] == nid, "subject"])
            cands = None
            if plan.approved_only:
                cands = [c for c in kg.predictions.chem_ids if kg.category(c) == B.DRUG]
            hits = kg.predictions.ligands_for(nid, cands, exclude=known, k=k)
            col = "compound"
        rows = [{col: kg.label(x), "id": x, "model score": round(sc, 3), "rank": i + 1}
                for i, (x, sc) in enumerate(hits)]
        summ = (f"Top {len(rows)} predicted {'targets' if col == 'protein' else 'binders'} of {ent['label']} "
                f"that are not recorded in the graph ({len(known)} are). Scores come from the GNN trained on "
                f"the agent's graph. Ask 'why might X bind Y' to see the supporting paths. "
                f"These are hypotheses to test, not facts.")
        return Answer(plan, summ, list(rows[0]) if rows else [], rows, [], "", [nid] + [r["id"] for r in rows])

    if it == "similar_compounds":
        nid = plan.entities[0]["id"]
        e = kg.edges_of(nid, B.SIMILAR).sort_values("similarity_score", ascending=False).head(k)
        rows = []
        for r in e.to_dict("records"):
            o = r["object"] if r["subject"] == nid else r["subject"]
            rows.append({"compound": kg.label(o), "id": o, "Tanimoto": r.get("similarity_score"),
                         "targets in graph": len(kg.edges_of(o, B.INTERACTS_PHYS, "out")), "edge": r["id"]})
        summ = f"{len(rows)} nearest structural neighbours of {plan.entities[0]['label']} (Morgan fingerprints)."
        return Answer(plan, summ, list(rows[0]) if rows else [], rows, e["id"].tolist(),
                      f"MATCH (a {{id:'{nid}'}})-[r:`{B.SIMILAR}`]-(b) RETURN b.name, r.similarity_score "
                      f"ORDER BY r.similarity_score DESC LIMIT {k}", [nid] + [r["id"] for r in rows])

    if it == "path":
        if len(plan.entities) < 2:
            return Answer(plan, "A path needs two entities.", [], [], [], "")
        a, b = plan.entities[0]["id"], plan.entities[1]["id"]
        paths = shortest_paths(kg, a, b, max_paths=k)
        rows, used, nodes = [], [], [a, b]
        for p in paths:
            rows.append({"path": p["text"], "hops": p["hops"], "confidence": round(p["conf"], 3),
                         "edges": ", ".join(p["edges"])})
            used.extend(p["edges"])
            nodes.extend(p["nodes"])
        summ = (f"{len(paths)} shortest paths between {kg.label(a)} and {kg.label(b)} over kept edges, "
                f"best path confidence first." if paths else
                f"No path of up to 4 hops between {kg.label(a)} and {kg.label(b)}.")
        return Answer(plan, summ, list(rows[0]) if rows else [], rows, used,
                      f"MATCH p=allShortestPaths((a {{id:'{a}'}})-[*..4]-(b {{id:'{b}'}})) "
                      f"WHERE all(r IN relationships(p) WHERE r.verdict='keep') RETURN p LIMIT {k}",
                      list(dict.fromkeys(nodes)))

    if it == "pruned":
        nid = plan.entities[0]["id"]
        ids = {nid, kg.p2g.get(nid, nid)}
        e = pd.concat([kg.edges_of(x, keep_only=False) for x in ids])
        e = e[e["verdict"] == "prune"].sort_values("confidence")
        rows = []
        for r in e.head(k).to_dict("records"):
            o = r["object"] if r["subject"] in ids else r["subject"]
            rows.append({"removed link": f"{PRED_LABEL.get(r['predicate'], r['predicate'])} {kg.label(o)}",
                         "confidence": r["confidence"], "why": readable_reasons(r.get("prune_reason")),
                         "source": _t(r.get("primary_knowledge_source")).replace("infores:", ""),
                         "edge": r["id"]})
        summ = (f"The pruning agent removed {len(e)} links about {plan.entities[0]['label']}. "
                "They stay in the full graph with their verdict, so nothing is lost; they are left out of "
                "answers and of the pruned graph." if len(e) else
                f"Nothing about {plan.entities[0]['label']} was pruned.")
        return Answer(plan, summ, list(rows[0]) if rows else [], rows, e.head(k)["id"].tolist(), "",
                      [nid])

    # profile
    nid = plan.entities[0]["id"]
    return profile(kg, nid, plan)


def profile(kg, nid, plan):
    cat = kg.category(nid)
    rows = []
    pid = kg.g2p.get(nid, nid)
    gid = kg.p2g.get(pid, nid)
    for pred, node, direction, label in (
            (B.INTERACTS_PHYS, pid, "in", "compounds binding it"),
            (B.INTERACTS_PHYS, nid, "out", "proteins it binds"),
            (B.PPI, pid, "both", "interaction partners"),
            (B.ASSOC_CONDITION, gid, "out", "linked diseases"),
            (B.ASSOC_CONDITION, nid, "in", "linked genes"),
            (B.PARTICIPATES_IN, pid, "out", "pathways"),
            (B.TREATS, nid, "out", "indications"),
            (B.TREATS, nid, "in", "drugs indicated"),
            (B.SIMILAR, nid, "both", "structural neighbours")):
        e = kg.edges_of(node, pred, direction)
        if len(e):
            top = e.sort_values("confidence", ascending=False).head(5)
            ex = [kg.label(o if s == node else s) for s, o in zip(top["subject"], top["object"])]
            rows.append({"what": label, "count": len(e), "examples": "; ".join(ex)})
    n = kg.node.get(nid) or {}
    extra = [f"{k}: {_t(n.get(k))}" for k in ("tdl", "family", "novelty") if _t(n.get(k))]
    summ = f"{kg.label(nid)} [{CATEGORY_LABEL.get(cat, cat)}, {nid}]" + (f" ({', '.join(extra)})" if extra else "")
    return Answer(plan, summ, ["what", "count", "examples"], rows, [], f"MATCH (n {{id:'{nid}'}})-[r]-(m) "
                  f"WHERE r.verdict='keep' RETURN type(r), count(*)", [nid])


def readable_reasons(reason):
    out = []
    for tok in _t(reason).split(";"):
        if not tok:
            continue
        key, _, val = tok.partition(":")
        out.append({
            "level": "statistical association only",
            "agent": "text-mined or computed",
            "low_source_prior": "low-reliability source",
            "weak_activity": f"weak potency ({val.split(':')[-1]})",
            "modest_activity": f"modest potency ({val.split(':')[-1]})",
            "low_ppi_score": "low interaction score",
            "context": "surrounding graph does not support it",
            "context_veto": "surrounding graph contradicts it (less plausible than most random mix-ups)",
            "opentargets": "no Open Targets association",
            "unknown_source": "unknown source",
        }.get(key, tok))
    return "; ".join(dict.fromkeys(out))


def edge_card(kg, r):
    """Everything about one edge, plus a plain-language explanation."""
    pmids = [p.split(":")[-1] for p in _t(r.get("publications")).split("|") if p]
    notes = [n for n in _t(r.get("confidence_notes")).split(";") if n]
    s, o = kg.label(r["subject"]), kg.label(r["object"])
    verb = PRED_LABEL.get(r["predicate"], r["predicate"])
    lines = [f"**{s} {verb} {o}.**"]
    src = _t(r.get("primary_knowledge_source")).replace("infores:", "")
    agg = _t(r.get("aggregator_knowledge_source")).replace("infores:", "")
    lines.append(f"Reported by **{src}**" + (f", aggregated by {agg} ({_t(r.get('source_version'))})" if agg else "")
                 + f"; {_t(r.get('knowledge_level')).replace('_', ' ')}, by a {_t(r.get('agent_type')).replace('_', ' ')}.")
    if _t(r.get("activity_value_max")):
        lines.append(f"Best measured potency: {float(r['activity_value_max']):g} (-log10 M; "
                     f"{_t(r.get('activity_type_set')).replace('|', ', ')}).")
    if _t(r.get("similarity_score")):
        lines.append(f"Tanimoto similarity {float(r['similarity_score']):.2f} on Morgan fingerprints, "
                     "computed by the enrichment agent.")
    sup = [x.replace("infores:", "") for x in _t(r.get("supporting_sources")).split("|") if x]
    if len(sup) > 1:
        lines.append("Also reported by: " + ", ".join(x for x in sup if x != src) + ".")
    if pmids:
        lines.append(f"{len(pmids)} paper(s): " + ", ".join(f"PMID {p}" for p in pmids[:5])
                     + (" …" if len(pmids) > 5 else "") + ".")
    if _t(r.get("verdict")):
        lines.append(f"Pruning agent: **{r['verdict']}** at confidence {r.get('confidence')}. "
                     f"Why: {readable_notes(notes)}")
    return {"edge": r["id"], "predicate": r["predicate"], "relation": _t(r.get("relation")),
            "subject": r["subject"], "object": r["object"], "source": src,
            "papers": [f"https://pubmed.ncbi.nlm.nih.gov/{p}/" for p in pmids],
            "confidence": r.get("confidence"), "verdict": _t(r.get("verdict")),
            "explanation": " ".join(lines)}


NOTE_TEXT = {
    "source": "source reliability", "corroborated_by": "independent sources agree",
    "literature": "papers", "level": "kind of evidence", "agent": "how it was produced",
    "potent_activity": "potent measured activity", "weak_activity": "weak measured activity",
    "modest_activity": "modest measured activity", "high_ppi_score": "high interaction score",
    "low_ppi_score": "low interaction score", "context": "surrounding graph",
    "opentargets": "Open Targets", "structural": "definitional link", "tanimoto": "structure similarity",
    "context_veto": "context veto: less plausible than most random mix-ups",
    "unknown_source": "unknown source",
}


def readable_notes(notes):
    out = []
    for n in notes:
        sign, body = n[0], n[1:]
        key, _, val = body.partition(":")
        word = NOTE_TEXT.get(key, key)
        if key == "source":
            src, _, prior = val.rpartition(":")
            out.append(f"starts at {prior} for {src.replace('infores:', '')}")
        elif key == "context":
            sc, _, why = val.partition(":")
            why = why.replace("_", " ").replace(",", ", ")
            out.append(f"{'+' if sign == '+' else '−'} surrounding graph {'supports' if sign == '+' else 'does not support'}"
                       f" it (plausibility {sc}; {why})")
        else:
            out.append(f"{'+' if sign == '+' else ('−' if sign == '-' else '')}{word}"
                       f"{' ' + val.replace('_', ' ') if val else ''}")
    return "; ".join(out) + "."


def shortest_paths(kg, a, b, max_hops=4, max_paths=10):
    """Breadth-first shortest paths over kept edges (ignoring similarity edges
    and very high-degree hubs, which make paths meaningless)."""
    from collections import deque
    def nbrs(n):
        for i in kg.out[n] + kg.inc[n]:
            r = kg.edges.iloc[i]
            if r["verdict"] == "prune" or r["predicate"] == B.SIMILAR:
                continue
            m = r["object"] if r["subject"] == n else r["subject"]
            yield m, i
    a_set = {a, kg.p2g.get(a, a)}
    b_set = {b, kg.p2g.get(b, b)}
    prev = {x: [] for x in a_set}
    depth = {x: 0 for x in a_set}
    q = deque(a_set)
    found_at = None
    while q:
        n = q.popleft()
        if found_at is not None and depth[n] >= found_at:
            break
        if depth[n] >= max_hops:
            continue
        deg = len(kg.out[n]) + len(kg.inc[n])
        if deg > 3000 and n not in a_set:
            continue
        for m, i in nbrs(n):
            if m not in depth:
                depth[m] = depth[n] + 1
                prev[m] = [(n, i)]
                q.append(m)
                if m in b_set:
                    found_at = depth[m]
            elif depth[m] == depth[n] + 1:
                prev[m].append((n, i))
    out = []

    def walk(n, acc_nodes, acc_edges):
        if len(out) >= 200:
            return
        if n in a_set:
            out.append((list(reversed(acc_nodes + [n])), list(reversed(acc_edges))))
            return
        for p, i in prev.get(n, [])[:20]:
            walk(p, acc_nodes + [n], acc_edges + [i])
    for t in b_set:
        if t in depth:
            walk(t, [], [])
    res = []
    for nodes, eidx in out:
        conf = float(np.prod([kg.edges.iloc[i]["confidence"] for i in eidx]))
        parts = [kg.label(nodes[0])]
        for i, n in zip(eidx, nodes[1:]):
            r = kg.edges.iloc[i]
            verb = PRED_LABEL.get(r["predicate"], r["predicate"])
            forward = r["object"] == n
            parts.append(f"—{verb}{'→' if forward else '←'} {kg.label(n)}")
        res.append({"nodes": nodes, "edges": [kg.edges.iloc[i]["id"] for i in eidx],
                    "hops": len(eidx), "conf": conf, "text": " ".join(parts), "labels": nodes})
    res.sort(key=lambda r: -r["conf"])
    return res[:max_paths]


def ask(kg, question, use_llm=True, explain_fn=None):
    plan = make_plan(kg, question, use_llm=use_llm)
    return execute(kg, plan, explain_fn=explain_fn)


def answer_text(ans, max_rows=15):
    """Plain-text rendering of an Answer for the command line."""
    p = ans.plan
    lines = [f"Plan ({p.planner}): {p.intent}  " +
             ", ".join(f"'{e['text']}' -> {e['id']}" for e in p.entities)]
    lines += [f"Note: {n}" for n in p.notes]
    lines += ["", ans.summary.replace("**", "")]
    if ans.rows:
        cols = [c for c in ans.columns if c not in ("edge", "edges", "id")] or list(ans.rows[0])
        lines.append("")
        lines.append(" | ".join(cols))
        for r in ans.rows[:max_rows]:
            lines.append(" | ".join(_t(r.get(c)) if not isinstance(r.get(c), float)
                                    else f"{r[c]:.3g}" for c in cols))
        if len(ans.rows) > max_rows:
            lines.append(f"... {len(ans.rows) - max_rows} more")
    if ans.cypher:
        lines += ["", "Cypher: " + ans.cypher]
    return "\n".join(lines)
