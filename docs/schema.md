# Graph schema

Every class and predicate the graph emits. The validator (`kgc/validate.py`)
enforces the domain and range below; source of truth is `kgc/biolink.py`.

## Node categories

| Category | ID prefix (preferred first) | Ontology class |
| --- | --- | --- |
| `biolink:Gene` | HGNC (Pharos exposes HGNC IDs; NCBIGene used if present; fallback: HGNC.SYMBOL, flagged) | SO:0001217 protein_coding_gene |
| `biolink:Protein` | UniProtKB | PR:000000001 protein |
| `biolink:SmallMolecule` | CHEMBL.COMPOUND, PUBCHEM.COMPOUND, DRUGCENTRAL, GTOPDB, UNII (fallback: LYCHI, flagged) | |
| `biolink:Drug` | same as SmallMolecule; used when Pharos marks the ligand as an approved drug | |
| `biolink:Disease` | MONDO, else the source's own ID (DOID, UMLS, OMIM) | |
| `biolink:Pathway` | REACT | |

## Predicates

| Predicate | Subject | Object | Built from |
| --- | --- | --- | --- |
| `biolink:translates_to` | Gene | Protein | Pharos target record |
| `biolink:directly_physically_interacts_with` | SmallMolecule, Drug | Protein | Ligand activities (ChEMBL, DrugCentral, Guide to Pharmacology); **link prediction task** |
| `biolink:physically_interacts_with` | Protein | Protein | Pharos PPIs (STRING, BioPlex, Reactome); symmetric, stored once |
| `biolink:gene_associated_with_condition` | Gene | Disease | Pharos disease associations, excluding drug indications |
| `biolink:participates_in` | Protein | Pathway | Reactome pathways |

## Edge properties

| Property | Meaning |
| --- | --- |
| `primary_knowledge_source` | infores of the original source |
| `aggregator_knowledge_source` | `infores:pharos` |
| `knowledge_level`, `agent_type` | Biolink provenance enums |
| `publications` | PMIDs, pipe-separated |
| `supporting_sources` | every infores that contributed |
| `evidence_count` | number of source records merged into the edge |
| `activity_value_max`, `activity_type_set`, `moa_set` | ligand activity, -log10 molar as Pharos reports it |
| `association_score_max`, `evidence_set` | disease association strength |
| `interaction_score_max` | PPI score |
| `source_version`, `ingest_run` | Pharos version and the hash of the raw file |
