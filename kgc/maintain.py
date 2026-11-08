"""Maintenance: keep the graph current as sources release new data.

check_source   ask Pharos for its current database version
diff_graphs    what a rebuild added, removed or changed, edge by edge
run_update     pull again when the version moved, rebuild, re-score only what
               changed, and write a changelog a person can read

Edge IDs are a hash of (subject, predicate, object), so the same fact keeps the
same ID across releases and a diff is a set comparison.
"""

import json
import time
from pathlib import Path

import pandas as pd

TRACKED = ["primary_knowledge_source", "publications", "activity_value_max",
           "association_score_max", "interaction_score_max", "evidence_count"]


def check_source(client=None):
    from .pharos import PharosClient, VERSION_QUERY
    client = client or PharosClient()
    return client.query(VERSION_QUERY).get("dbVersion")


def diff_graphs(old, new):
    o = old.set_index("id")
    n = new.set_index("id")
    added = n.index.difference(o.index)
    removed = o.index.difference(n.index)
    common = n.index.intersection(o.index)
    changed = []
    cols = [c for c in TRACKED if c in o.columns and c in n.columns]
    if cols and len(common):
        a = o.loc[common, cols].astype(str)
        b = n.loc[common, cols].astype(str)
        mask = (a != b).any(axis=1)
        for eid in common[mask.values]:
            diffs = {c: (o.at[eid, c], n.at[eid, c]) for c in cols if str(o.at[eid, c]) != str(n.at[eid, c])}
            changed.append({"id": eid, "changes": {k: [str(v[0]), str(v[1])] for k, v in diffs.items()}})
    by_pred = lambda df, ids: df.loc[ids, "predicate"].value_counts().to_dict() if len(ids) else {}
    return {
        "added": len(added), "removed": len(removed), "changed": len(changed),
        "added_by_predicate": by_pred(n, added), "removed_by_predicate": by_pred(o, removed),
        "added_ids": list(added), "removed_ids": list(removed), "changed_edges": changed,
    }


def changelog_text(diff, old_version, new_version):
    lines = [f"Update {old_version} -> {new_version}:",
             f"  {diff['added']} edges added, {diff['removed']} removed, {diff['changed']} changed."]
    for k, v in diff["added_by_predicate"].items():
        lines.append(f"  + {v} {k}")
    for k, v in diff["removed_by_predicate"].items():
        lines.append(f"  - {v} {k}")
    return "\n".join(lines)


def run_update(data_dir, name="pharos_kinase", family="Kinase", force=False, log=print):
    """Re-pull and rebuild if the source moved; return the diff summary."""
    from .kgx import build_kgx
    from .pharos import pull_family
    data_dir = Path(data_dir)
    raw = data_dir / "raw" / f"{name}.jsonl"
    meta = json.loads(raw.with_suffix(".meta.json").read_text()) if raw.with_suffix(".meta.json").exists() else {}
    old_version = meta.get("dbVersion", "unknown")
    current = check_source()
    log(f"local: {old_version}, Pharos now: {current}")
    if current == old_version and not force:
        return {"updated": False, "version": current}
    stamp = time.strftime("%Y%m%d")
    new_name = f"{name}_{stamp}"
    new_raw = data_dir / "raw" / f"{new_name}.jsonl"
    pull_family(new_raw, family=family, log=log)
    build_kgx(new_raw, data_dir / "kgx", name=new_name)
    old_e = pd.read_csv(data_dir / "kgx" / f"{name}_edges.tsv", sep="\t", low_memory=False)
    new_e = pd.read_csv(data_dir / "kgx" / f"{new_name}_edges.tsv", sep="\t", low_memory=False)
    diff = diff_graphs(old_e, new_e)
    text = changelog_text(diff, old_version, current)
    log(text)
    out = data_dir / "results" / f"{name}_changelog_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in diff.items() if not k.endswith("_ids")} | {"text": text},
                              indent=2), encoding="utf-8")
    return {"updated": True, "version": current, "new_name": new_name, "summary": text}
