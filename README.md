# VeriKG prebuilt data

The graph, scores and results that `python -m kgc fetch-data` downloads, so
the app can run without the hour-long Pharos pull. Code and documentation are
on the `main` branch.

| File | Contents |
| --- | --- |
| `verikg_data.zip` | `data/kgx/`: nodes, source edges, every edge with confidence and verdict (KGX TSV), fingerprints, reports; `data/results/`: baselines, GNN comparisons, planted-error benchmark, the app's link predictor |
| `verikg_pyg.zip` | `data/kgx/pharos_kinase_pyg.pt`: the scored graph as PyTorch Geometric HeteroData |

`fetch-data` also writes the pruned graph and the verdict table from the
scored edges.

Built from Pharos 3.19 (pharos319), all 635 kinases. Data come from
Pharos/TCRD and its upstream sources (ChEMBL, DrugCentral, Guide to
Pharmacology, STRING, BioPlex, Reactome, DisGeNET, JensenLab, Expression Atlas
and others), each under its own license; see the
[Pharos data sources](https://pharos.nih.gov/about). Provided for evaluating
this entry.
