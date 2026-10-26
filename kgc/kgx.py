"""Map raw Pharos JSONL to Biolink-compliant KGX TSV (nodes + edges).

Every edge keeps its lineage: primary and aggregator knowledge source,
knowledge level, agent type, PMIDs, source version and the ingest run ID.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from . import biolink as B

LIST_SEP = "|"


def edge_id(s, p, o):
    return "kgc:" + hashlib.sha1(f"{s}|{p}|{o}".encode()).hexdigest()[:16]


def _clean(v):
    return v.strip() if isinstance(v, str) else v


# ---------- identifiers ----------

def gene_curie(target):
    """Prefer NCBIGene, then HGNC, then the symbol (flagged as non-standard)."""
    hgnc = None
    for x in target.get("xrefs") or []:
        # Pharos puts the ID in "name"; "value" is usually null.
        src, val = (x.get("source") or "").lower(), _clean(x.get("value") or x.get("name"))
        if not val:
            continue
        if re.search(r"ncbi gene|entrez|geneid", src) and re.fullmatch(r"\d+", val):
            return f"NCBIGene:{val}"
        if "hgnc" in src and not hgnc:
            num = val.split(":")[-1]
            if num.isdigit():
                hgnc = f"HGNC:{num}"
    if hgnc:
        return hgnc
    sym = target.get("sym")
    return f"HGNC.SYMBOL:{sym}" if sym else None


CHEM_PREFIXES = [
    ("ChEMBL", "CHEMBL.COMPOUND"),
    ("PubChem", "PUBCHEM.COMPOUND"),
    ("DrugCentral", "DRUGCENTRAL"),
    ("Guide to Pharmacology", "GTOPDB"),
    ("unii", "UNII"),
    ("LyCHI", "LYCHI"),
]


def chemical_curie(lig):
    syn = {s["name"]: _clean(s["value"]) for s in lig.get("synonyms") or [] if s.get("value")}
    for name, prefix in CHEM_PREFIXES:
        val = syn.get(name)
        if val:
            val = val.split(":")[-1]
            return f"{prefix}:{val}"
    return f"LYCHI:{lig['ligid']}" if lig.get("ligid") else None


def disease_curie(disease):
    if disease.get("mondoID"):
        return disease["mondoID"]
    for a in disease.get("associations") or []:
        did = _clean(a.get("did"))
        if did and ":" in did:
            return did
    return None


def activity_source(ref):
    r = (ref or "").lower()
    if "chembl" in r:
        return "infores:chembl"
    if "drugcentral" in r or "drug central" in r:
        return "infores:drugcentral"
    if "pharmacology" in r or "iuphar" in r or "gtopdb" in r:
        return "infores:gtopdb"
    return "infores:pharos"


PPI_SOURCES = {
    "string": ("infores:string", "statistical_association", "computational_model"),
    "bioplex": ("infores:bioplex", "knowledge_assertion", "data_analysis_pipeline"),
    "reactome": ("infores:reactome", "knowledge_assertion", "manual_agent"),
}


def ppi_source(kind):
    k = (kind or "").lower()
    for key, val in PPI_SOURCES.items():
        if key in k:
            return val
    return ("infores:pharos", "not_provided", "not_provided")


# ---------- builder ----------

class GraphBuilder:
    def __init__(self, source_version, run_id):
        self.nodes = {}
        self.edges = {}
        self.source_version = source_version
        self.run_id = run_id
        self.issues = Counter()

    def node(self, curie, category, name=None, **props):
        if curie not in self.nodes:
            self.nodes[curie] = {"id": curie, "category": category, "name": name,
                                 "provided_by": B.AGGREGATOR}
        n = self.nodes[curie]
        if name and not n.get("name"):
            n["name"] = name
        for k, v in props.items():
            if v is not None and n.get(k) in (None, ""):
                n[k] = v

    def edge(self, s, p, o, primary, level, agent, pubs=(), sources=(), **props):
        if p in B.SYMMETRIC and s > o:
            s, o = o, s
        eid = edge_id(s, p, o)
        e = self.edges.get(eid)
        if e is None:
            e = self.edges[eid] = {
                "id": eid, "subject": s, "predicate": p, "object": o,
                "primary_knowledge_source": primary,
                "aggregator_knowledge_source": B.AGGREGATOR,
                "knowledge_level": level, "agent_type": agent,
                "_pubs": set(), "_sources": set(), "_n": 0,
                "source_version": self.source_version, "ingest_run": self.run_id,
            }
        e["_pubs"].update(p_ for p_ in pubs if p_)
        e["_sources"].update(x for x in sources if x)
        e["_n"] += 1
        for k, v in props.items():
            if v is None:
                continue
            if k.endswith("_max"):
                e[k] = v if e.get(k) is None else max(e[k], v)
            elif k.endswith("_set"):
                e.setdefault(k, set()).update(v if isinstance(v, (set, list, tuple)) else [v])
            else:
                e.setdefault(k, v)

    # --- per record ---
    def add_target(self, rec):
        t = rec["target"]
        prot = f"UniProtKB:{t['uniprot']}"
        self.node(prot, B.PROTEIN, t.get("name"), symbol=t.get("sym"), tdl=t.get("tdl"),
                  family=t.get("fam"), novelty=t.get("novelty"),
                  type_iri=B.NODE_TYPE_IRI[B.PROTEIN])
        gene = gene_curie(t)
        if gene is None:
            self.issues["target_without_gene_id"] += 1
        else:
            if gene.startswith("HGNC.SYMBOL:"):
                self.issues["gene_id_from_symbol_only"] += 1
            self.node(gene, B.GENE, t.get("sym"), symbol=t.get("sym"),
                      type_iri=B.NODE_TYPE_IRI[B.GENE])
            self.edge(gene, B.TRANSLATES_TO, prot, "infores:uniprot",
                      "knowledge_assertion", "manual_agent")

        self._ligands(prot, rec.get("ligands") or [])
        if gene:
            self._diseases(gene, t.get("diseases") or [])
        self._ppis(prot, t.get("ppis") or [])
        self._pathways(prot, t.get("pathways") or [])

    def _ligands(self, prot, ligands):
        for lig in ligands:
            chem = chemical_curie(lig)
            if chem is None:
                self.issues["ligand_without_id"] += 1
                continue
            if chem.startswith("LYCHI:"):
                self.issues["ligand_id_lychi_only"] += 1
            cat = B.DRUG if lig.get("isdrug") else B.CHEMICAL
            self.node(chem, cat, lig.get("name"), smiles=lig.get("smiles"),
                      lychi=lig.get("ligid"))
            acts = lig.get("activities") or []
            if not acts:
                self.issues["ligand_without_activity_for_target"] += 1
                continue
            for a in acts:
                pubs = [f"PMID:{p['pmid']}" for p in a.get("pubs") or [] if p.get("pmid")]
                src = activity_source(a.get("reference"))
                self.edge(chem, B.INTERACTS_PHYS, prot, src, "knowledge_assertion",
                          "manual_agent", pubs=pubs, sources=[src],
                          activity_value_max=a.get("value"),
                          activity_type_set=a.get("type"),
                          moa_set=a.get("moa"))

    def _diseases(self, gene, diseases):
        for d in diseases:
            dis = disease_curie(d)
            if dis is None:
                self.issues["disease_without_id"] += 1
                continue
            assocs = [a for a in d.get("associations") or []
                      if a.get("type") not in B.INDICATION_TYPES]
            if not assocs:
                continue
            self.node(dis, B.DISEASE, d.get("name"))
            for a in assocs:
                primary, level, agent = B.DISEASE_SOURCES.get(a.get("type"), B.DEFAULT_DISEASE_SOURCE)
                score = next((a.get(k) for k in ("score", "zscore", "conf") if a.get(k) is not None), None)
                self.edge(gene, B.ASSOC_CONDITION, dis, primary, level, agent,
                          sources=[primary], association_score_max=score,
                          evidence_set=a.get("evidence"))

    def _ppis(self, prot, ppis):
        for nb in ppis:
            other = (nb.get("target") or {}).get("uniprot")
            if not other:
                continue
            o = nb["target"]
            oid = f"UniProtKB:{other}"
            self.node(oid, B.PROTEIN, o.get("name"), symbol=o.get("sym"), tdl=o.get("tdl"),
                      family=o.get("fam"), type_iri=B.NODE_TYPE_IRI[B.PROTEIN])
            primary, level, agent = ppi_source(nb.get("type"))
            # type can list several sources, e.g. "BioPlex,STRINGDB"
            all_src = {ppi_source(k)[0] for k in (nb.get("type") or "").split(",")}
            score = None
            for prop in nb.get("props") or []:
                if "score" in (prop.get("name") or "").lower():
                    try:
                        score = float(prop["value"])
                    except (TypeError, ValueError):
                        pass
            self.edge(prot, B.PPI, oid, primary, level, agent, sources=all_src,
                      interaction_score_max=score)

    def _pathways(self, prot, pathways):
        for pw in pathways:
            sid = _clean(pw.get("sourceID"))
            if not sid:
                continue
            pid = f"REACT:{sid}"
            self.node(pid, B.PATHWAY, pw.get("name"))
            self.edge(prot, B.PARTICIPATES_IN, pid, "infores:reactome",
                      "knowledge_assertion", "manual_agent")

    # --- output ---
    def frames(self):
        nodes = pd.DataFrame(list(self.nodes.values()))
        rows = []
        for e in self.edges.values():
            r = {k: v for k, v in e.items() if not k.startswith("_")}
            r["publications"] = LIST_SEP.join(sorted(e["_pubs"]))
            srcs = sorted(e["_sources"] | {e["primary_knowledge_source"]})
            r["supporting_sources"] = LIST_SEP.join(srcs)
            r["evidence_count"] = e["_n"]
            for k, v in list(r.items()):
                if isinstance(v, set):
                    r[k] = LIST_SEP.join(sorted(str(x) for x in v))
            rows.append(r)
        return nodes, pd.DataFrame(rows)


def build_kgx(raw_path, out_dir, name="pharos_kinase", run_id=None):
    raw_path, out_dir = Path(raw_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = raw_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    version = f"pharos:{meta.get('dbVersion', 'unknown')}"
    run_id = run_id or hashlib.sha1(raw_path.read_bytes()).hexdigest()[:12]

    gb = GraphBuilder(version, run_id)
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gb.add_target(json.loads(line))
    nodes, edges = gb.frames()
    nodes.to_csv(out_dir / f"{name}_nodes.tsv", sep="\t", index=False)
    edges.to_csv(out_dir / f"{name}_edges.tsv", sep="\t", index=False)

    stats = {
        "source_version": version, "ingest_run": run_id,
        "nodes": len(nodes), "edges": len(edges),
        "nodes_by_category": nodes["category"].value_counts().to_dict(),
        "edges_by_predicate": edges["predicate"].value_counts().to_dict() if len(edges) else {},
        "issues": dict(gb.issues),
    }
    (out_dir / f"{name}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
