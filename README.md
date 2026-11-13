# Plasmodium falciparum Drug–Target Interaction Pipeline

This repository brings together data collection and graph preparation for drug–target interactions (DTIs) involving *Plasmodium falciparum* proteins. The workflow collects activity and compound records, prepares and explores the resulting datasets, represents compounds and proteins as separate graphs, and provides protein graph visualizations.

The project is organized as research scripts and their data products. There is not currently one root-level command that runs every stage end to end; each stage has its own script and input expectations.

## Pipeline at a glance

1. **Fetch source data** — collect ChEMBL assay/activity records and PubChem assay, compound, and protein-target information.
2. **Preprocess interactions** — extract standardized activity fields, identify active and inactive records, and combine compatible rows for downstream analysis.
3. **Explore the data (EDA)** — summarize activity and assay coverage, inspect compound chemical space, and compare activity distributions across proteins and sources.
4. **Build drug graphs** — convert compound SMILES into atom-and-bond graphs.
5. **Build protein graphs** — construct residue-level graphs from protein structure data associated with UniProt accessions; optional edge embeddings are also prepared.
6. **Visualize** — export interactive HTML views of protein structure graphs.

Drug and protein graphs are constructed separately. This repository contains the graph components and interaction tables; a single integrated drug–protein heterogeneous graph or model-training pipeline is not represented by the current top-level structure.

## Repository structure

```text
DataChembl/
├── data_fetch/                       # ChEMBL and PubChem collection scripts
├── data/
│   ├── proteins&chembl/              # ChEMBL, assay, compound, and protein source tables
│   └── pubchem_pfalciparum/           # PubChem compound, protein, and activity data
├── preprocessing/                    # Activity extraction and row-combination scripts
│   ├── active_extract.csv            # Prepared active records
│   ├── inactive_extract.csv          # Prepared inactive records
│   └── eda/                          # Chemical-space analysis artifacts
├── eda/                              # Dataset, assay, activity, and chemical-space analyses
│   ├── eda_results/                  # Summaries, plots, and analysis tables
│   ├── assaysInfo/                   # Assay metadata summaries
│   └── eda.ipynb                     # Exploratory notebook
├── drug_graphs_construction/         # SMILES-to-molecular-graph code and graph outputs
│   ├── build_graphs.py
│   ├── functions.py
│   └── graphs/                       # Serialized drug graphs, when generated
├── graph_construction_proteins/      # Protein structure-to-graph pipeline
│   ├── main.py                       # Protein graph pipeline entry point
│   ├── functions.py                  # Graph conversion and feature functions
│   ├── embeddings.py                 # Protein edge embedding utilities
│   ├── pdb_files/                    # Cached/downloaded protein structures
│   └── graphproteins/                # Serialized protein graph outputs
├── processed_graphs_embeddings/      # Per-protein serialized embedding artifacts
├── vizualize/                        # Protein visualization script and HTML outputs
├── STITCH/                           # STITCH data and compound/protein comparisons
├── final/                            # Consolidated interaction and EDA outputs
├── output/                           # Graph normalization summaries and plots
├── gatheringProteinGraphEmbeddings.py
├── data.csv                          # Root-level interaction table
└── sample_data.csv                   # Small sample table
```

Some listed directories contain generated artifacts or cached data. Their contents may vary as the pipeline is rerun. The protein graph module also has its own [README](graph_construction_proteins/README.md) with details specific to graph construction.

## Stage guide

### 1. Data collection — `data_fetch/`

The scripts collect data from ChEMBL and PubChem, including organism-specific assays, compounds, activity outcomes, and target information. Examples include:

- `extract_organism_assays.py` — filters ChEMBL assays for *P. falciparum* (taxonomy ID 5833) and retrieves activities.
- `pubchem_extract.py` and `pubchem_extract_from_pro.py` — PubChem-oriented collection workflows.
- `DTI_extraction_data.py` — additional DTI dataset extraction and harmonization logic.

Collection scripts may call external APIs and need network access. Check each script's command-line arguments and expected input columns before running it; output paths and schemas are script-specific.

### 2. Preprocessing — `preprocessing/`

This stage prepares activity records for analysis. `extract_standard_types.py` inspects standardized activity fields, while `extract_inactive_from_unspecified.py` and `combine_active_inactive_rows.py` help derive and combine activity classes. The directory includes prepared active/inactive CSVs and a summary of standardized activity types.

The `eda/` folder also contains a chemical-space analysis artifact created during preprocessing. Related copies of preprocessing scripts appear in the root `eda/` directory; use the version appropriate to the dataset and workflow you are running.

### 3. Exploratory analysis — `eda/`

Scripts and the notebook cover global and per-protein summaries, assay/source checks, activity metrics, PubChem outcomes, and chemical-space analysis. Results are generally written beneath `eda/eda_results/` (with additional historical outputs in `eda/extra/` and `eda/final/`).

### 4. Drug graphs — `drug_graphs_construction/`

`build_graphs.py` reads a CSV with SMILES and a compound identifier, converts each valid molecule to a PyTorch Geometric graph using functions in `functions.py`, and saves serialized graph files in `graphs/`. The current feature code encodes atom properties and bond properties. Verify the CSV column names with the script options (`--smiles-col`, `--id-col`) before running.

### 5. Protein graphs and embeddings — `graph_construction_proteins/`

`main.py` orchestrates protein graph creation from UniProt protein entries. `functions.py` handles graph construction and conversion, and `pdb_files/` stores structure files used by the pipeline. The module's README documents its graph features and command-line options. `embeddings.py` and the root-level `gatheringProteinGraphEmbeddings.py` contain utilities for preparing graph embeddings; serialized embedding artifacts are stored in `processed_graphs_embeddings/`.

### 6. Visualization — `vizualize/`

`protein.py` builds a Graphein protein graph from a local PDB structure and writes an interactive Plotly HTML visualization. Existing HTML files are under `vizualize/viz_proteins/`. The directory name is intentionally shown as it exists in the repository (`vizualize`).

## Running the scripts

Run scripts from the repository root or their own folder as required by their relative paths. For help with options, invoke a script with `--help` when supported. For example:

```bash
python data_fetch/extract_organism_assays.py --help
python drug_graphs_construction/build_graphs.py --help
python graph_construction_proteins/main.py --help
python vizualize/protein.py --help
```

Input schemas and outputs differ by stage. Inspect the relevant script before connecting stages, especially where a script expects a particular SMILES, compound-ID, or UniProt column. Protein graph generation and PubChem/ChEMBL collection may download data or structures and can require network access.

## Data and generated artifacts

The repository includes source tables, intermediate CSVs, analysis outputs, cached structures, serialized graphs, and embeddings. Some outputs are large or reproducible. `.gitignore` excludes several generated or local artifacts (including `*.pt`, PDB files, `output/`, and most of `data/`); check repository tracking status before assuming every local file is committed or available in a fresh checkout.

When reproducing results, record the source dataset, script, and options used. Activity labels and available measurements can differ by source, so retain source and assay identifiers when comparing or combining records.
