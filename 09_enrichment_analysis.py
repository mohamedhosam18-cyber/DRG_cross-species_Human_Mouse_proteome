import pandas as pd
import numpy as np
from pathlib import Path
import gseapy as gp

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()

human_file = base / "results" / "top_gene_lists" / "top100_human_higher_genes.txt"
mouse_file = base / "results" / "top_gene_lists" / "top100_mouse_higher_genes.txt"

results_dir = base / "results" / "enrichment"
results_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# LOAD GENE LISTS
# =========================================
human_genes = (
    pd.read_csv(human_file, header=None)[0]
    .dropna()
    .astype(str)
    .str.upper()
    .str.strip()
    .tolist()
)

mouse_genes = (
    pd.read_csv(mouse_file, header=None)[0]
    .dropna()
    .astype(str)
    .str.upper()
    .str.strip()
    .tolist()
)

print("Human genes:", len(human_genes))
print("Mouse genes:", len(mouse_genes))

# =========================================
# GENE SETS
# =========================================
human_sets = {
    "GO_BP": "GO_Biological_Process_2023",
    "GO_MF": "GO_Molecular_Function_2023",
    "KEGG": "KEGG_2021_Human",
    "Reactome": "Reactome_2022",
}

mouse_sets = {
    "GO_BP": "GO_Biological_Process_2023",
    "GO_MF": "GO_Molecular_Function_2023",
    "KEGG": "KEGG_2019_Mouse",
    "Reactome": "Reactome_2022",
}

# =========================================
# RUN HUMAN ORA
# =========================================
for label, gene_set in human_sets.items():
    print(f"Running HUMAN {label} ...")
    enr = gp.enrichr(
        gene_list=human_genes,
        gene_sets=gene_set,
        organism="human",
        outdir=None
    )
    res = enr.results.copy()
    out_csv = results_dir / f"human_{label}_enrichment.csv"
    res.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

# =========================================
# RUN MOUSE ORA
# =========================================
for label, gene_set in mouse_sets.items():
    print(f"Running MOUSE {label} ...")
    enr = gp.enrichr(
        gene_list=mouse_genes,
        gene_sets=gene_set,
        organism="mouse",
        outdir=None
    )
    res = enr.results.copy()
    out_csv = results_dir / f"mouse_{label}_enrichment.csv"
    res.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

print("ORA enrichment analysis complete.")
