# VeriKG: an agentic, provenance-first biomedical knowledge graph

Entry for the Bio-Agent Knowledge Graph Construction Challenge, Structured Data
Integration and Link Prediction track.

VeriKG builds a Biolink-compliant knowledge graph from Pharos/TCRD,
keeps the lineage of every edge, and measures link prediction on a
leakage-audited split with hard negatives. Enrichment, pruning and the
LangGraph agents build on this base.

## Status

| Stage | State |
| --- | --- |
| Pharos pull (one target family, resumable; queries verified against the live API, Pharos 3.19) | done |
| KGX export with Biolink categories, predicates, provenance | done |
| Structural validator (domain/range, dangling edges, provenance) | done |
| Leakage-safe split, degree-matched negatives | done |
| CPU baseline (ligand overlap, PPI, pathway features) | done |
| Enrichment (Open Targets, OptimusKG, ChEMBL, Monarch) | next |
| Pruning agent, explanations, UI, Brev Launchable | planned |

## Quickstart

Needs Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m kgc pull --limit 5      # smoke test: first 5 of the 635 kinases
python -m kgc all                 # build, validate, split, baseline on those 5

python -m kgc pull                # all 635 kinases, about an hour; resumable
python -m kgc all
```

Outputs land in `data/` (git-ignored):

| Path | Contents |
| --- | --- |
| `data/raw/pharos_kinase.jsonl` | Untouched API responses, one target per line, plus `.meta.json` with the Pharos version |
| `data/kgx/pharos_kinase_{nodes,edges}.tsv` | The graph in KGX TSV |
| `data/kgx/pharos_kinase_stats.json` | Counts by category and predicate, plus mapping issues |
| `data/kgx/pharos_kinase_validation.json` | Validator report |
| `data/split/pharos_kinase/` | Train graph, val and test pairs, ranking negatives, split report |
| `data/results/pharos_kinase_baseline.json` | Baseline metrics |

Run the tests with `python -m pytest -q tests`. They use synthetic records
only; no network needed.

## Design decisions

- **Raw first.** The API response is stored untouched, so ID mapping can change without downloading again.
- **Most specific classes.** Genes are `biolink:Gene` typed SO:0001217 (protein-coding gene), proteins are `biolink:Protein` typed PR:000000001, linked by `biolink:translates_to`.
- **Lineage on every edge.** `primary_knowledge_source` and `aggregator_knowledge_source` (infores CURIEs), `knowledge_level`, `agent_type`, `publications`, `source_version`, `ingest_run`, and `supporting_sources`.
- **Activities scoped to their target.** Pharos can return a ligand's activities against other targets; the pull keeps only the parent target's.
- **Drug indications are not gene-disease evidence.** Pharos lists DrugCentral indications among disease associations; they are excluded from `gene_associated_with_condition`.
- **Fair evaluation.** Held-out pairs are removed from the training graph under every predicate and both directions. Negatives are type-constrained and degree-matched; a degree-only score is reported as a check and should sit near 0.5.
- **Sampled ranking.** Hits@10 and MRR rank each test positive against 99 degree-matched negatives.

See [docs/schema.md](docs/schema.md) for every class and predicate used.

## Data and licenses

Code: MIT. Data come from Pharos/TCRD and its upstream sources, each under
its own license; see the [Pharos data sources](https://pharos.nih.gov/about).
Please respect them when redistributing derived graphs.
