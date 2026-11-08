# Graph schema

Every class and predicate the graph emits. The validator (`kgc/validate.py`)
enforces the domain and range below; source of truth is `kgc/biolink.py`.

## Node categories

| Category | ID prefix (preferred first) | Ontology class (`type_iri`) |
| --- | --- | --- |
| `biolink:Gene` | HGNC (Pharos exposes HGNC IDs; NCBIGene used if present; fallback: HGNC.SYMBOL, flagged) | SO:0001217 protein_coding_gene |
| `biolink:Protein` | UniProtKB | PR:000000001 protein |
| `biolink:SmallMolecule` | CHEMBL.COMPOUND, PUBCHEM.COMPOUND, DRUGCENTRAL, GTOPDB, UNII (fallback: LYCHI, flagged); further IDs of the same ligand in `xref` | CHEBI:25367 molecule |
| `biolink:Drug` | same as SmallMolecule; used when Pharos marks the ligand as an approved drug | CHEBI:23888 drug |
| `biolink:Disease` | MONDO, else the source's own ID (DOID, UMLS, OMIM, MESH) | MONDO:0000001 disease |
| `biolink:Pathway` | REACT | PW:0000001 pathway |

Genes and proteins carry `in_taxon` NCBITaxon:9606. Chemicals carry `smiles`.

## Predicates

| Predicate | Subject | Object | Relation IRI | Built from |
| --- | --- | --- | --- | --- |
| `biolink:translates_to` | Gene | Protein | SO translates_to | Pharos target record |
| `biolink:directly_physically_interacts_with` | SmallMolecule, Drug | Protein | RO:0002436 | Ligand activities (ChEMBL, DrugCentral, Guide to Pharmacology); **link prediction task** |
| `biolink:physically_interacts_with` | Protein | Protein | RO:0002434 | Pharos PPIs (STRING, BioPlex, Reactome); symmetric, stored once |
| `biolink:gene_associated_with_condition` | Gene | Disease | Biolink | Pharos disease associations, excluding drug indications |
| `biolink:treats_or_applied_or_studied_to_treat` | SmallMolecule, Drug | Disease | Biolink | DrugCentral indications, which Pharos lists among disease associations |
| `biolink:participates_in` | Protein | Pathway | RO:0000056 | Reactome pathways |
| `biolink:chemically_similar_to` | SmallMolecule, Drug | SmallMolecule, Drug | Biolink | Enrichment: Morgan fingerprint Tanimoto >= 0.6, 10 nearest neighbours; symmetric, stored once |

## Edge properties

| Property | Meaning |
| --- | --- |
| `relation` | Ontology relation behind the predicate, as a full IRI |
| `primary_knowledge_source` | infores of the original source (`infores:verikg` for computed edges) |
| `aggregator_knowledge_source` | `infores:pharos` |
| `knowledge_level`, `agent_type` | Biolink provenance enums |
| `publications` | PMIDs, pipe-separated |
| `supporting_sources` | every infores that contributed |
| `evidence_count` | number of source records merged into the edge |
| `activity_value_max`, `activity_type_set`, `moa_set` | ligand activity, -log10 molar as Pharos reports it |
| `association_score_max`, `evidence_set` | disease association strength |
| `interaction_score_max` | PPI score |
| `similarity_score`, `similarity_method` | Tanimoto and fingerprint, for similarity edges |
| `source_version`, `ingest_run` | Pharos version and the hash of the raw file |

Added by the pruning agent (`*_edges_scored.tsv`):

| Property | Meaning |
| --- | --- |
| `confidence` | 0 to 1, from the signals in `confidence_notes` |
| `confidence_notes` | signed signals: `=source:<infores>:<prior>`, `+literature:3`, `-level:statistical_association`, `-context:0.12:no_supporting_context`, `-context_veto:0.031<0.049`, ... |
| `plausibility`, `plausibility_notes`, `context_size` | out-of-fold context model score, the context features that fired, and how much surrounding graph there was |
| `opentargets_score` | Open Targets association score for gene-disease edges, when fetched |
| `verdict`, `prune_reason` | `keep` or `prune` (confidence below 0.45), and the negative signals behind a prune |
