"""Synthetic Pharos-shaped records for tests only. Not real biology.

Chemicals bind proteins mostly within one hidden cluster, PPIs and pathways
follow the same clusters, so a working baseline must beat 0.5 and a
degree-only score must not.
"""

import json
import random


def make_records(n_prot=60, n_clusters=6, n_chem=900, seed=7):
    rnd = random.Random(seed)
    prots = [f"P{10000 + i}" for i in range(n_prot)]
    cluster = {p: i % n_clusters for i, p in enumerate(prots)}
    by_c = {c: [p for p in prots if cluster[p] == c] for c in range(n_clusters)}
    tdls = ["Tclin", "Tchem", "Tbio", "Tdark"]

    ligs = {p: [] for p in prots}
    for j in range(n_chem):
        c = rnd.randrange(n_clusters)
        k = rnd.choice([1, 1, 2, 2, 3, 4])
        chosen = set()
        for _ in range(k):
            pool = by_c[c] if rnd.random() < 0.9 else prots
            chosen.add(rnd.choice(pool))
        kind = j % 10
        syn = [{"name": "LyCHI", "value": f"LYCHI{j:05d}"}]
        if kind < 7:
            syn.append({"name": "ChEMBL", "value": f"CHEMBL{100000 + j}"})
        elif kind < 9:
            syn.append({"name": "PubChem", "value": str(5000 + j)})
        for p in chosen:
            ligs[p].append({
                "ligid": f"LYCHI{j:05d}", "name": f"compound-{j}", "isdrug": j % 50 == 0,
                "smiles": "C" * (1 + j % 5), "synonyms": syn,
                "activities": [{"actid": j * 10 + len(ligs[p]), "type": "IC50",
                                "value": round(5 + rnd.random() * 4, 2), "moa": None,
                                "reference": "ChEMBL", "target": {"uniprot": p},
                                "pubs": [{"pmid": str(30000000 + j)}]}],
            })

    records = []
    for i, p in enumerate(prots):
        c = cluster[p]
        xrefs = []
        if i % 7 != 0:
            xrefs.append({"source": "NCBI Gene", "value": str(1000 + i)})
        elif i % 14 == 0:
            xrefs.append({"source": "HGNC", "value": f"HGNC:{2000 + i}"})
        neigh = [q for q in by_c[c] if q != p][:4]
        records.append({"target": {
            "tcrdid": i, "uniprot": p, "sym": f"KIN{i}", "name": f"kinase {i}",
            "tdl": tdls[i % 4], "fam": "Kinase", "novelty": 0.1, "xrefs": xrefs,
            "diseases": [{"name": f"disease {c}", "mondoID": f"MONDO:{9000 + c:07d}",
                          "associations": [{"type": "DisGeNET", "score": 0.4, "evidence": "lit"},
                                           {"type": "DrugCentral Indication", "drug": "x"}]}],
            "ppis": [{"type": "STRINGDB", "props": [{"name": "score", "value": "0.8"}],
                      "target": {"uniprot": q, "sym": None, "name": None, "tdl": None, "fam": "Kinase"}}
                     for q in neigh],
            "pathways": [{"type": "Reactome", "name": f"pathway {c}", "sourceID": f"R-HSA-{700 + c}"}],
        }, "ligands": ligs[p]})
    return records


def write_raw(path, records, version="test-6.13.5"):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(str(path).replace(".jsonl", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump({"dbVersion": version, "family": "Kinase"}, f)
