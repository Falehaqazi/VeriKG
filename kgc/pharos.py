"""Pull one target family from the Pharos GraphQL API into raw JSONL.

Raw responses are saved untouched (one line per target), so the KGX mapping
can be changed and re-run without downloading again. The pull is resumable:
targets already in the output file are skipped.
"""

import json
import time
from pathlib import Path

import requests

ENDPOINT = "https://pharos-api.ncats.io/graphql"

LIST_QUERY = """
query($fam: String!, $skip: Int!, $top: Int!) {
  targets(filter: {facets: [{facet: "IDG Family", values: [$fam]}]}) {
    count
    targets(skip: $skip, top: $top) { uniprot sym }
  }
}
"""

TARGET_QUERY = """
query($uniprot: String!, $dtop: Int!, $ptop: Int!, $wtop: Int!) {
  target(q: {uniprot: $uniprot}) {
    tcrdid uniprot sym name tdl fam novelty
    xrefs(source: "HGNC") { source name value }
    diseases(top: $dtop) {
      name mondoID
      associations(top: 100) { type name did evidence zscore conf score source drug reference }
    }
    ppis(top: $ptop) {
      type
      props { name value }
      target { uniprot sym name tdl fam }
    }
    pathways(top: $wtop, type: ["Reactome"]) { type name sourceID }
  }
}
"""

LIGAND_QUERY = """
query($uniprot: String!, $skip: Int!, $top: Int!, $isdrug: Boolean!) {
  target(q: {uniprot: $uniprot}) {
    ligands(skip: $skip, top: $top, isdrug: $isdrug) {
      ligid name isdrug smiles
      synonyms { name value }
      activities { actid type value moa reference target { uniprot } pubs { pmid } }
    }
  }
}
"""

VERSION_QUERY = "{ dbVersion }"


class PharosClient:
    def __init__(self, endpoint=ENDPOINT, pause=0.2, retries=5, timeout=120):
        self.endpoint = endpoint
        self.pause = pause
        self.retries = retries
        self.timeout = timeout
        self.session = requests.Session()

    def query(self, query, variables=None):
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.post(
                    self.endpoint,
                    json={"query": query, "variables": variables or {}},
                    timeout=self.timeout,
                )
                r.raise_for_status()
                body = r.json()
                if body.get("errors"):
                    raise RuntimeError(json.dumps(body["errors"])[:500])
                time.sleep(self.pause)
                return body["data"]
            except (requests.RequestException, RuntimeError, ValueError) as e:
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Pharos query failed after {self.retries} tries: {last}")


def list_family(client, family, page=100):
    out, skip = [], 0
    while True:
        data = client.query(LIST_QUERY, {"fam": family, "skip": skip, "top": page})
        batch = data["targets"]["targets"] or []
        out.extend(batch)
        total = data["targets"]["count"]
        skip += len(batch)
        if not batch or skip >= total:
            return out


def pull_ligands(client, uniprot, isdrug, page=250, max_ligands=None):
    out, skip = [], 0
    while True:
        data = client.query(
            LIGAND_QUERY, {"uniprot": uniprot, "skip": skip, "top": page, "isdrug": isdrug}
        )
        batch = (data["target"] or {}).get("ligands") or []
        for lig in batch:
            # Activities are not always scoped to the parent target; keep only this target's.
            lig["activities"] = [
                a for a in (lig.get("activities") or [])
                if (a.get("target") or {}).get("uniprot") == uniprot
            ]
        out.extend(batch)
        skip += len(batch)
        if len(batch) < page or (max_ligands and skip >= max_ligands):
            return out


def pull_family(out_path, family="Kinase", max_ligands=None, limit=None,
                disease_top=1000, ppi_top=1000, pathway_top=500, client=None, log=print):
    client = client or PharosClient()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            done = {json.loads(line)["target"]["uniprot"] for line in f if line.strip()}

    version = client.query(VERSION_QUERY).get("dbVersion")
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "source": "Pharos GraphQL", "endpoint": client.endpoint, "dbVersion": version,
        "family": family, "max_ligands": max_ligands,
        "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2), encoding="utf-8")

    targets = list_family(client, family)
    if limit:
        targets = targets[:limit]
    log(f"Pharos {version}: {len(targets)} {family} targets, {len(done)} already pulled")

    with out_path.open("a", encoding="utf-8") as f:
        for i, t in enumerate(targets, 1):
            if t["uniprot"] in done:
                continue
            data = client.query(TARGET_QUERY, {
                "uniprot": t["uniprot"], "dtop": disease_top,
                "ptop": ppi_top, "wtop": pathway_top,
            })
            rec = {"target": data["target"]}
            rec["ligands"] = (
                pull_ligands(client, t["uniprot"], True, max_ligands=max_ligands)
                + pull_ligands(client, t["uniprot"], False, max_ligands=max_ligands)
            )
            f.write(json.dumps(rec) + "\n")
            f.flush()
            log(f"[{i}/{len(targets)}] {t['sym'] or t['uniprot']}: "
                f"{len(rec['ligands'])} ligands, "
                f"{len(rec['target'].get('diseases') or [])} diseases, "
                f"{len(rec['target'].get('ppis') or [])} PPIs")
    return out_path
