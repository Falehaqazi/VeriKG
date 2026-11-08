# VeriKG: an agentic, provenance-first biomedical knowledge graph

Entry for the Bio-Agent Knowledge Graph Construction Challenge, Structured Data
Integration and Link Prediction track.

VeriKG turns Pharos/TCRD into a Biolink-compliant knowledge graph of the human
kinome, then lets agents enrich it, score and prune every edge with a
plain-language reason, predict missing ligand-target links, and answer
questions from people who do not write graph queries. Every answer lists the
edges it used, and every edge carries its source, papers, confidence and
verdict.

## Try it

**On Brev:** deploy the VeriKG Launchable and open the link for port 8501
(setup in [docs/brev.md](docs/brev.md)).

**Locally** (Python 3.10 or newer):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m kgc fetch-data        # prebuilt graph, scores and results
python -m kgc app               # http://localhost:8501
```

Or from the command line:

```bash
python -m kgc ask --question "Which approved drugs act on kinases linked to breast cancer?"
python -m kgc ask --question "Why is dasatinib linked to ABL1?"
python -m kgc ask --question "What did you remove about CDK2 and why?"
```

No API key is needed. With `NVIDIA_API_KEY` set, a NIM model plans the
queries; code still grounds every name to a node ID and runs the query, and
the plan is shown and editable.

## The agents

Both agents are explicit [LangGraph](https://github.com/langchain-ai/langgraph)
state machines (`kgc/agent.py`); every step writes a line to a run log the app
shows.

| Agent | Steps | What a user gets |
| --- | --- | --- |
| Build | ingest, map to Biolink, validate (stop on errors), enrich, prune, export | A KGX graph from Pharos, or from any CSV a researcher uploads, with a verdict on every edge |
| Query | plan, ground names to IDs, clarify if nothing matched, execute | An answer table, the edges behind it drawn as a graph, each edge's lineage, the plan, and the equivalent Cypher |
| Maintain | check the Pharos release, re-pull, diff edge by edge | A changelog of what a new release added, removed or changed |

The query agent covers twelve intents: ligands of a protein, targets of a
compound, proteins for a disease, drugs for a disease (multi-hop through
the disease's genes), diseases of a protein, why two entities are linked,
predicted new targets or ligands, similar compounds, paths between two
entities, what was pruned and why, and an entity summary.

## Results (Pharos 3.19, all 635 kinases)

### The graph

| | |
| --- | --- |
| Nodes | 76,593: 58,702 small molecules, 125 approved drugs, 13,776 proteins, 635 genes, 2,303 diseases, 1,052 pathways |
| Source edges | 198,491: 80,909 ligand-protein, 95,012 protein-protein, 14,114 gene-disease, 7,689 protein-pathway, 635 gene-protein, 132 drug-indication |
| Added by enrichment | 295,484 structure-similarity edges (Morgan fingerprints, Tanimoto >= 0.6, 10 nearest neighbours) |
| Validator | 0 errors, 0 warnings |

### Pruning

Every edge gets a confidence in [0, 1] from named signals: the reliability of
its most reliable source, corroboration by the others, papers, how the knowledge was
produced (curated, statistical, text mined, computed), measured potency or
interaction score, and context plausibility, a model of whether the rest of
the graph, without the edge, still points to it (scored out of fold, so no
edge contributes to its own score). Edges under 0.45 are pruned. They stay in
the full graph with their reasons and are left out of answers. Of 493,975
edges, 13,462 are pruned:

| | Edges | Pruned | Main reasons |
| --- | --- | --- | --- |
| Ligand-protein | 80,909 | 1,205 (1.5%) | the record names no upstream source and the surrounding graph does not support it (899); the surrounding graph contradicts it (306, the context veto) |
| Protein-protein | 95,012 | 8,233 (8.7%) | STRING only, computed, and the two proteins share no partners, pathways or ligands |
| Gene-disease | 14,114 | 4,024 (28.5%) | statistical association only (Expression Atlas 2,201, DisGeNET 762, TIGA 493, text mining 348, eRAM 220) with no support from the gene's interaction partners or pathways |
| Pathway, gene-protein, indication | 8,456 | 0 | curated |

**Planted-error test.** For each edge type, 2% of its edges are copied three
times, once per kind of swap, with one end swapped and the original source,
papers and potency kept, so each planted edge looks like a curated record
filed under the wrong entity. Evidence scoring alone
cannot see these (AUROC 0.50, as designed); context can:

| Edge type | Planted | AUROC, evidence only | AUROC, evidence + context (random / degree / near-miss swaps) | Planted edges pruned | Real edges pruned | Precision of pruning |
| --- | --- | --- | --- | --- | --- | --- |
| Ligand-protein | 4,854 | 0.50 | 0.953 / 0.928 / 0.928 | 32.0% | 1.2% | 61.0% |
| Gene-disease | 846 | 0.52 | 0.669 / 0.573 / 0.709 | 61.7% | 26.4% | 12.3% |
| Protein-protein | 5,700 | 0.50 | 0.655 / 0.656 / 0.612 | 22.7% | 8.3% | 14.2% |
| Protein-pathway | 459 | 0.50 | 0.813 / 0.734 / 0.678 | 0.0% | 0.0% | nothing pruned |
| Drug-indication | 30 | 0.50 | 0.771 / 0.666 / 0.297 | 0.0% | 0.0% | nothing pruned |

The context veto applies where the context model is strong (ligand-protein
edges: out-of-fold AUROC 0.98 against corrupted copies). It flags an edge
that is less plausible than three quarters of random mix-ups and has enough
surrounding graph to judge by; on the real graph that is 306 edges (0.4%;
226 from ChEMBL, 82 without a paper), candidates for review rather than
proven errors. Near-miss swaps (the compound
filed under an interaction partner or pathway mate of the true kinase) are
the hardest, as they should be.

### Link prediction

Same model (two-layer heterogeneous GraphSAGE), same split, same negatives;
only the graph changes. Negatives are drawn in proportion to protein degree,
so ranking popular kinases high earns nothing (a degree-only score gets
0.51). Hits@10 and MRR rank each true pair against 99 such negatives.

| Split | Graph | AUROC | AUPRC | AUROC, compounds unseen in training | Hits@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| random | `pharos` | 0.736 | 0.766 | 0.561 | 0.390 | 0.244 |
| random | `pharos_fp` | 0.948 | 0.942 | 0.943 | 0.766 | 0.462 |
| random | `verikg` | 0.981 | 0.974 | 0.985 | 0.903 | 0.654 |
| random | `verikg_pruned` | 0.980 | 0.977 | 0.986 | 0.902 | 0.669 |
| random | feature baseline (logistic regression) | 0.709 | 0.749 | | 0.514 | 0.327 |
| scaffold | `pharos` | 0.506 | 0.501 | 0.506 | 0.089 | 0.050 |
| scaffold | `pharos_fp` | 0.949 | 0.947 | 0.949 | 0.787 | 0.521 |
| scaffold | `verikg` | 0.958 | 0.957 | 0.958 | 0.795 | 0.578 |
| scaffold | `verikg_pruned` | 0.956 | 0.952 | 0.956 | 0.794 | 0.518 |
| scaffold | feature baseline (logistic regression) | 0.486 | 0.485 | | 0.285 | 0.110 |

`pharos` is the graph as pulled; `pharos_fp` adds Morgan fingerprints as
compound features; `verikg` adds the structure-similarity edges; and
`verikg_pruned` is verikg after the pruning agent, re-run on the training
graph alone so no test edge influences what is cut. Single runs (seed 0);
differences under about 0.01 should not be read as real.

The enrichment is what carries prediction: on the random split it lifts
AUROC from 0.736 to 0.981 and Hits@10 from 0.39 to 0.90. Pruning leaves
link prediction where it was (0.980 against 0.981 on the random split, 0.956
against 0.958 on the scaffold split): it removes 13,462 edges to make the
graph more precise without costing the model anything measurable.

The **scaffold split** is the honest test for new chemistry: no test compound
shares a Bemis-Murcko scaffold with any training compound, so every test pair
involves a compound the model has never seen bound to anything. The source
graph alone has no signal there (`pharos` at 0.506, the feature baseline at
0.486), which is why enrichment matters. Accuracy falls as test compounds
get less similar to anything in training. The similarity edges help where a
neighbour at Tanimoto 0.6 or more exists and cost a little between 0.4 and
0.6 (the 164 pairs under 0.4 are too few to read):

| Max Tanimoto of the test compound to any training compound | <0.4 | 0.4-0.6 | 0.6-0.8 | >=0.8 |
| --- | --- | --- | --- | --- |
| Test pairs (positives and negatives) | 164 | 2,184 | 9,736 | 4,080 |
| `pharos` AUROC | 0.573 | 0.507 | 0.506 | 0.503 |
| `pharos_fp` AUROC | 0.720 | 0.839 | 0.958 | 0.974 |
| `verikg` AUROC | 0.764 | 0.806 | 0.969 | 0.979 |
| `verikg_pruned` AUROC | 0.708 | 0.819 | 0.966 | 0.976 |

## Standards

- **Biolink** categories and predicates throughout, with domain and range
  checked by the validator; genes typed `SO:0001217`, proteins `PR:000000001`,
  and an ontology class on every node (`type_iri`).
- **Relation IRI on every edge** (RO terms where they exist).
- **Lineage on every edge**: `primary_knowledge_source` and
  `aggregator_knowledge_source` (infores CURIEs), `knowledge_level`,
  `agent_type`, `publications`, `supporting_sources`, `source_version`,
  `ingest_run`; computed edges name `infores:verikg` as their source.
- **KGX TSV** for every graph; a **PyTorch Geometric** `HeteroData` file with
  edge confidence and a keep mask on every edge type; numpy arrays for any
  extracted subgraph.

See [docs/schema.md](docs/schema.md) for every class and predicate.

## Build it yourself

```bash
pip install -r requirements.txt
pip install -r requirements-gnn.txt   # only for the GNN steps

python -m kgc pull                    # all 635 kinases from the Pharos API, about an hour; resumable
python -m kgc all                     # build, validate, split, feature baseline
python -m kgc enrich                  # fingerprints and similarity edges, about 10 min
python -m kgc opentargets             # optional: Open Targets scores for gene-disease edges (network)
python -m kgc prune                   # confidence, verdict and reasons on every edge
python -m kgc benchmark               # planted-error test
python -m kgc split-scaffold          # scaffold split
python -m kgc gnn                     # link prediction, random split
python -m kgc gnn --split scaffold
python -m kgc predict                 # model for the app, plus the PyG export
python -m kgc app
```

`python -m kgc pull --limit 5` followed by the same steps is a quick smoke
test. `python -m kgc table --file my_table.csv` runs the build agent on any
relationship table. `python -m kgc update` checks Pharos for a new release.

Main outputs (`data/`, git-ignored):

| Path | Contents |
| --- | --- |
| `raw/pharos_kinase.jsonl` | Untouched API responses, one target per line, plus `.meta.json` with the Pharos version |
| `kgx/pharos_kinase_{nodes,edges}.tsv` | The source graph in KGX TSV |
| `kgx/pharos_kinase_edges_scored.tsv` | Every edge, enrichment included, with confidence, notes, verdict and reasons |
| `kgx/pharos_kinase_edges_pruned.tsv` | The pruned graph (kept edges only) |
| `kgx/pharos_kinase_verdicts.tsv` | Every Pharos edge with its verdict, confidence and reasons, gene and protein ends as UniProt IDs |
| `kgx/pharos_kinase_pyg.pt` | PyTorch Geometric HeteroData of the scored graph |
| `results/` | Baselines, GNN comparisons, planted-error benchmark |

Run the tests with `python -m pytest -q tests`. They use synthetic records
only; no network needed.

## Design decisions

- **Raw first.** API responses are stored untouched, so ID mapping can change without downloading again.
- **Activities scoped to their target, and credited to their source.** Pharos can return a ligand's activities against other targets; only the parent target's are kept. An activity whose reference is a journal citation on a ChEMBL compound is credited to ChEMBL, mechanism-of-action rows to DrugCentral.
- **Drug indications are not gene-disease evidence.** Pharos lists DrugCentral indications among disease associations; they become drug-disease edges instead.
- **One ID per compound.** When Pharos lists several ChEMBL IDs for one ligand, the first is the node ID and the rest are kept as `xref`.
- **Nothing is deleted.** Pruning writes a verdict and reasons; the full graph and the pruned graph are both written so they can be compared.
- **Fair evaluation.** Held-out pairs are removed from the training graph under every predicate and both directions; negatives are type-constrained and degree-matched; a scaffold split tests new chemistry; pruning for the GNN comparison is re-run on the training graph only.
- **Language models propose, code decides.** A model may plan a query or suggest a table mapping; every step that admits data or computes an answer is deterministic code.

## Data and licenses

Code: MIT. Data come from Pharos/TCRD and its upstream sources (ChEMBL,
DrugCentral, Guide to Pharmacology, STRING, BioPlex, Reactome, DisGeNET,
JensenLab, Expression Atlas and others), each under its own license; see the
[Pharos data sources](https://pharos.nih.gov/about). The prebuilt data
download is provided for evaluating this entry; please respect the upstream
licenses when redistributing derived graphs.
