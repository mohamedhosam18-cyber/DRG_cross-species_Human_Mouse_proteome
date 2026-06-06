# Cross-Species Proteomic Analysis of Human and Mouse Dorsal Root Ganglia

## Overview

This repository contains the complete computational workflow used for a cross-species proteomic comparison of human and mouse dorsal root ganglia (DRG).

The analysis was designed to investigate conserved and divergent molecular signatures between human and mouse sensory ganglia using publicly available mass spectrometry datasets. By integrating ortholog mapping, differential abundance analysis, pathway enrichment, and gene set enrichment analysis (GSEA), the workflow provides a systems-level comparison of sensory neuron biology across species.

The repository accompanies the manuscript:

**Cross-Species Proteomic Analysis of Human and Mouse Dorsal Root Ganglia Reveals Conserved and Divergent Molecular Signatures**

---

## Scientific Background

The dorsal root ganglion (DRG) contains the cell bodies of primary sensory neurons responsible for transmitting nociceptive, mechanosensory, proprioceptive, thermal, and visceral information from the periphery to the central nervous system.

Rodent models are widely used in sensory neuroscience and pain research; however, accumulating evidence suggests substantial molecular differences between human and mouse DRG. Understanding these similarities and differences is critical for improving translational interpretation of preclinical studies and identifying biological pathways that may be conserved or species-specific.

This workflow was developed to perform a systematic proteomic comparison between human and mouse DRG tissue using ortholog-resolved protein abundance measurements.

---

## Study Objectives

The analysis aimed to:

- Compare global proteomic profiles between human and mouse DRG.
- Identify proteins enriched in each species.
- Quantify conserved and divergent molecular signatures.
- Characterize pathway-level similarities and differences.
- Perform functional enrichment and GSEA analyses.
- Generate publication-quality visualizations.
- Provide a reproducible framework for future cross-species sensory neuroscience studies.

---

## Data Sources

### Human DRG Proteomics

Repository:

PRIDE Archive

Accession:

PXD061129

Input file:

```text
ST11-ExpressionMatrix.csv
```

Samples used:

Human dorsal root ganglion samples only.

---

### Mouse DRG Proteomics

Repository:

PRIDE Archive

Accession:

PXD031322

Input file:

```text
report.tsv
```

Samples used:

Untreated control DRG samples only.

---

## Data Availability

Raw proteomic datasets are not included in this repository due to file size limitations and repository storage constraints.

Users wishing to reproduce the analysis should download the original datasets from the corresponding public repositories and place them into the expected folder structure described below.

---

## Expected Project Structure

```text
DRG_cross_species_FINAL/
│
├── data_raw/
│   ├── ST11-ExpressionMatrix.csv
│   └── report.tsv
│
├── ortholog_mapping/
│   ├── ortholog_raw.txt
│   └── human_id_symbol.txt
│
├── data_processed/
│
├── results/
│
├── figures/
│
└── Scripts/
```

---

## Analysis Workflow

Scripts should be executed in numerical order.

### Data Preparation

#### 01_prepare_human_Data.py

- Loads human DRG proteomic matrix.
- Extracts DRG samples.
- Removes missing entries.
- Collapses duplicate genes.
- Performs log transformation if required.
- Generates cleaned human dataset.

#### 02_prepare_mouse_Data.py

- Loads mouse proteomic dataset.
- Extracts untreated control samples.
- Aggregates proteins to gene level.
- Removes invalid entries.
- Generates cleaned mouse dataset.

---

### Ortholog Mapping

#### 03_build_ortholog_table.py

- Builds human–mouse ortholog mapping table.

#### 04_map_to_orthologs.py

- Maps species-specific proteins to ortholog pairs.

---

### Matrix Construction

#### 05_build_merged_matrix.py

- Constructs merged cross-species abundance matrix.

---

### Differential Proteomic Analysis

#### 06_cross_species_analysis.py

- Performs species comparison.
- Computes fold changes.
- Performs statistical testing.
- Generates differential abundance results.

#### 07_volcano_plot.py

- Generates volcano plots.

#### 08_extract_top_DE_genes.py

- Extracts top differentially abundant proteins.

---

### Functional Enrichment

#### 09_enrichment_analysis_full.py

- Performs enrichment analysis.

#### 10_cross_species_enrichment_comparison.py

- Compares enrichment profiles.

#### 11_enrichment_visualization.py

- Generates enrichment figures.

#### 12_cross_species_enrichment_professional.py

- Creates publication-quality enrichment visualizations.

---

### Dimensionality Reduction

#### 13_final_dual_pca_with_distribution.py

- Performs PCA analysis.
- Visualizes species separation.
- Generates distribution plots.

---

### Gene Set Enrichment Analysis

#### 14_run_full_GSEA.py

- Runs GSEA.

#### 15_GSEA_visualization.py

- Visualizes GSEA outputs.

#### 16_GSEA_upset.py

- Generates GSEA overlap visualizations.

---

### Global Visualization

#### 17_heatmap.py

- Generates cross-species heatmaps.

#### 18_Scatter_plot_concordance_analysis.py

- Computes and visualizes concordance between species.

#### 19_thresholded_upset.py

- Generates threshold-based overlap plots.

---

### Biological Validation

#### 20_final_transcriptomics_panel_validation.py

- Compares proteomic findings with transcriptomic markers.

#### 21_Radar_plots_module_activity_convergence.py

- Generates systems-level radar plots summarizing pathway convergence and divergence.

---

## Main Outputs

The workflow generates:

- Differential abundance tables
- Volcano plots
- Heatmaps
- PCA visualizations
- Concordance analyses
- Enrichment analyses
- Gene Set Enrichment Analyses
- UpSet plots
- Radar plots
- Publication-ready figures

---

## Important Notes

This repository contains the computational workflow and analysis code used in the study.

Because raw proteomic datasets are extremely large, they are not hosted directly within the repository.

Users reproducing the workflow must download the original datasets from the corresponding public repositories and update local paths if necessary.

The analysis is intended as a tissue-level cross-species comparison and does not resolve individual neuronal subtypes.

---

## Citation

If you use this workflow, data processing strategy, or analysis framework, please cite:

Madkoor M.

Cross-Species Proteomic Analysis of Human and Mouse Dorsal Root Ganglia Reveals Conserved and Divergent Molecular Signatures.

---

## Author

Mohamed Madkoor

Medical Neurosciences, Charité – Universitätsmedizin Berlin, Germany
