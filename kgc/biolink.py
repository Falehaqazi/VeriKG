"""Biolink categories, predicates and provenance values used by this graph.

Kept deliberately small: every class and predicate the graph emits is listed
here, with the domain and range the validator enforces.
"""

PROTEIN = "biolink:Protein"
GENE = "biolink:Gene"
CHEMICAL = "biolink:SmallMolecule"
DRUG = "biolink:Drug"
DISEASE = "biolink:Disease"
PATHWAY = "biolink:Pathway"

# Most specific ontology class for each category (SO for gene, PR for protein, as
# the brief asks; the rest are Biolink's own exact mappings).
NODE_TYPE_IRI = {
    GENE: "SO:0001217",       # protein_coding_gene
    PROTEIN: "PR:000000001",  # protein
    CHEMICAL: "CHEBI:25367",  # molecule
    DRUG: "CHEBI:23888",      # drug
    DISEASE: "MONDO:0000001",  # disease
    PATHWAY: "PW:0000001",    # pathway
}
TAXON_HUMAN = "NCBITaxon:9606"

TRANSLATES_TO = "biolink:translates_to"
INTERACTS_PHYS = "biolink:directly_physically_interacts_with"
PPI = "biolink:physically_interacts_with"
ASSOC_CONDITION = "biolink:gene_associated_with_condition"
TREATS = "biolink:treats_or_applied_or_studied_to_treat"
PARTICIPATES_IN = "biolink:participates_in"
SIMILAR = "biolink:chemically_similar_to"

# Ontology relation behind each predicate, written as a full IRI to the KGX
# `relation` column (SO's translates_to is a named property, not an OBO number).
OBO = "http://purl.obolibrary.org/obo/"
BL = "https://w3id.org/biolink/vocab/"
PREDICATE_IRI = {
    TRANSLATES_TO: OBO + "so#translates_to",
    INTERACTS_PHYS: OBO + "RO_0002436",   # molecularly interacts with
    PPI: OBO + "RO_0002434",              # interacts with
    ASSOC_CONDITION: BL + "gene_associated_with_condition",
    TREATS: BL + "treats_or_applied_or_studied_to_treat",
    PARTICIPATES_IN: OBO + "RO_0000056",  # participates in
    SIMILAR: BL + "chemically_similar_to",
}

CHEMICALS = {CHEMICAL, DRUG}

# predicate -> (allowed subject categories, allowed object categories)
DOMAIN_RANGE = {
    TRANSLATES_TO: ({GENE}, {PROTEIN}),
    INTERACTS_PHYS: (CHEMICALS, {PROTEIN}),
    PPI: ({PROTEIN}, {PROTEIN}),
    ASSOC_CONDITION: ({GENE}, {DISEASE}),
    TREATS: (CHEMICALS, {DISEASE}),
    PARTICIPATES_IN: ({PROTEIN, GENE}, {PATHWAY}),
    SIMILAR: (CHEMICALS, CHEMICALS),
}

# Symmetric predicates: (a, b) and (b, a) are the same fact.
SYMMETRIC = {PPI, SIMILAR}

KNOWLEDGE_LEVELS = {
    "knowledge_assertion", "statistical_association", "prediction",
    "observation", "not_provided",
}
AGENT_TYPES = {
    "manual_agent", "automated_agent", "text_mining_agent",
    "data_analysis_pipeline", "computational_model", "not_provided",
}

AGGREGATOR = "infores:pharos"
# Edges this pipeline computes itself (structure similarity) name it as source.
SELF = "infores:verikg"

# Pharos association "type" -> (primary infores, knowledge_level, agent_type)
DISEASE_SOURCES = {
    "UniProt Disease": ("infores:uniprot", "knowledge_assertion", "manual_agent"),
    "OMIM": ("infores:omim", "knowledge_assertion", "manual_agent"),
    "DisGeNET": ("infores:disgenet", "knowledge_assertion", "automated_agent"),
    "eRAM": ("infores:eram", "knowledge_assertion", "automated_agent"),
    "CTD": ("infores:ctd", "knowledge_assertion", "manual_agent"),
    "Monarch": ("infores:monarchinitiative", "knowledge_assertion", "automated_agent"),
    "JensenLab Knowledge UniProtKB-KW": ("infores:diseases", "knowledge_assertion", "manual_agent"),
    "JensenLab Knowledge MedlinePlus": ("infores:diseases", "knowledge_assertion", "manual_agent"),
    "JensenLab Text Mining": ("infores:diseases", "statistical_association", "text_mining_agent"),
    "JensenLab Experiment TIGA": ("infores:tiga", "statistical_association", "data_analysis_pipeline"),
    "Expression Atlas": ("infores:gxa", "statistical_association", "data_analysis_pipeline"),
}
DEFAULT_DISEASE_SOURCE = ("infores:pharos", "not_provided", "not_provided")

# Drug-indication associations are reported by Pharos as a disease association type.
INDICATION_TYPES = {"DrugCentral Indication"}
