"""Open Targets Platform: an independent check on gene-disease edges.

Pharos aggregates gene-disease links from many resources of very different
quality (text mining, expression atlases, curated databases). Open Targets
scores the same question from its own evidence (genetics, somatic mutations,
known drugs, literature, animal models), so an edge Open Targets also scores
is corroborated, and an edge for a gene Open Targets covers but does not link
to the disease counts mildly against it (prune.external_adjustment).

fetch_associations writes data/kgx/<name>_opentargets.tsv with columns
gene, disease, score, ot_disease, ot_target. Only pairs that are in the graph
are written; a pair for a checked gene that Open Targets does not report gets
score 0, so prune.attach_opentargets can tell "checked, no association" from
"not checked".
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

from . import biolink as B

ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"

MAP_QUERY = """
query($terms: [String!]!) {
  mapIds(queryTerms: $terms, entityNames: ["target"]) {
    mappings { term hits { id entity name } }
  }
}
"""

ASSOC_QUERY = """
query($id: String!, $index: Int!, $size: Int!) {
  target(ensemblId: $id) {
    id approvedSymbol
    associatedDiseases(page: {index: $index, size: $size}) {
      count
      rows { score disease { id dbXRefs } }
    }
  }
}
"""

META_QUERY = "{ meta { dataVersion { year month } apiVersion { x y z } } }"

PREFIX_FIX = {"MSH": "MESH", "MESH": "MESH", "UMLS_CUI": "UMLS", "UMLS": "UMLS", "MONDO": "MONDO",
              "DOID": "DOID", "OMIM": "OMIM", "MIM": "OMIM", "ORPHANET": "ORPHANET", "EFO": "EFO",
              "HP": "HP", "NCIT": "NCIT"}


class OTClient:
    def __init__(self, endpoint=ENDPOINT, pause=0.1, retries=4, timeout=60):
        self.endpoint, self.pause, self.retries, self.timeout = endpoint, pause, retries, timeout
        self.session = requests.Session()

    def query(self, query, variables=None):
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.post(self.endpoint, json={"query": query, "variables": variables or {}},
                                      timeout=self.timeout)
                r.raise_for_status()
                body = r.json()
                if body.get("errors"):
                    raise RuntimeError(json.dumps(body["errors"])[:300])
                time.sleep(self.pause)
                return body["data"]
            except (requests.RequestException, RuntimeError, ValueError) as e:
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Open Targets query failed: {last}")


def norm_disease(curie):
    """'MONDO_0005148' or 'MSH:D001943' -> 'MONDO:0005148' / 'MESH:D001943'."""
    c = str(curie).strip().replace("_", ":", 1) if ":" not in str(curie) else str(curie).strip()
    pre, _, local = c.partition(":")
    pre = PREFIX_FIX.get(pre.upper(), pre.upper())
    return f"{pre}:{local}" if local else c


def disease_keys(ot_disease):
    """Every ID an Open Targets disease row can match in our graph."""
    keys = {norm_disease(ot_disease["id"])}
    for x in ot_disease.get("dbXRefs") or []:
        keys.add(norm_disease(x))
    return keys


def map_symbols(client, symbols, batch=100):
    out = {}
    for i in range(0, len(symbols), batch):
        data = client.query(MAP_QUERY, {"terms": symbols[i:i + batch]})
        for m in data["mapIds"]["mappings"]:
            hit = next((h for h in m.get("hits") or [] if h.get("entity") == "target"
                        and str(h.get("id", "")).startswith("ENSG")), None)
            if hit:
                out[m["term"]] = hit["id"]
    return out


def target_associations(client, ensembl_id, size=500):
    rows, index = [], 0
    while True:
        data = client.query(ASSOC_QUERY, {"id": ensembl_id, "index": index, "size": size})
        t = data.get("target") or {}
        ad = t.get("associatedDiseases") or {}
        batch = ad.get("rows") or []
        rows.extend(batch)
        index += 1
        if len(batch) < size or len(rows) >= ad.get("count", 0):
            return rows


def associations_to_table(pairs_by_gene, ot_rows_by_gene, ensembl_by_gene):
    """pairs_by_gene: {gene: [disease, ...]} from our graph.
    ot_rows_by_gene: {gene: [row from target_associations]}. Returns DataFrame."""
    out = []
    for gene, diseases in pairs_by_gene.items():
        if gene not in ot_rows_by_gene:
            continue  # not checked: leave the gene out entirely
        best, ot_id = {}, {}
        for r in ot_rows_by_gene[gene]:
            for k in disease_keys(r["disease"]):
                if r["score"] > best.get(k, -1):
                    best[k], ot_id[k] = float(r["score"]), r["disease"]["id"]
        for d in diseases:
            out.append({"gene": gene, "disease": d, "score": round(best.get(d, 0.0), 4),
                        "ot_disease": ot_id.get(d, ""), "ot_target": ensembl_by_gene.get(gene, "")})
    return pd.DataFrame(out, columns=["gene", "disease", "score", "ot_disease", "ot_target"])


def fetch_associations(kgx_dir, name="pharos_kinase", client=None, log=print):
    kgx_dir = Path(kgx_dir)
    client = client or OTClient()
    nodes = pd.read_csv(kgx_dir / f"{name}_nodes.tsv", sep="\t", low_memory=False)
    edges = pd.read_csv(kgx_dir / f"{name}_edges.tsv", sep="\t", low_memory=False)
    gd = edges[edges["predicate"] == B.ASSOC_CONDITION]
    pairs = gd.groupby("subject")["object"].apply(lambda s: sorted(set(s))).to_dict()
    sym = dict(zip(nodes["id"], nodes.get("symbol", nodes["name"])))
    genes = sorted(pairs)
    symbols = sorted({str(sym.get(g)) for g in genes if isinstance(sym.get(g), str)})
    meta = client.query(META_QUERY).get("meta", {})
    log(f"Open Targets {meta.get('dataVersion')}: mapping {len(symbols)} gene symbols")
    ens = map_symbols(client, symbols)
    ens_by_gene = {g: ens[sym[g]] for g in genes if sym.get(g) in ens}
    rows = {}
    for i, g in enumerate(genes):
        if g in ens_by_gene:
            rows[g] = target_associations(client, ens_by_gene[g])
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(genes)} genes")
    table = associations_to_table(pairs, rows, ens_by_gene)
    table.to_csv(kgx_dir / f"{name}_opentargets.tsv", sep="\t", index=False)
    report = {"open_targets_data_version": meta.get("dataVersion"), "genes": len(genes),
              "genes_mapped": len(ens_by_gene), "pairs_checked": int(len(table)),
              "pairs_with_association": int((table["score"] > 0).sum()) if len(table) else 0}
    (kgx_dir / f"{name}_opentargets_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
