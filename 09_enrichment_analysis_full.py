import pandas as pd
from pathlib import Path
import gseapy as gp

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()

gene_dir = base / "string_analysis" / "gene_lists"
results_dir = base / "results" / "enrichment"
results_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# INPUT GENE FILES
# choose the lists you want to enrich
# =========================================
human_file = gene_dir / "top150_human_higher_string.txt"
mouse_file = gene_dir / "top150_mouse_higher_string.txt"

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

print("Human genes loaded:", len(human_genes))
print("Mouse genes loaded:", len(mouse_genes))

# =========================================
# ENRICHR LIBRARIES
# =========================================
human_sets = {
    "GO_BP": "GO_Biological_Process_2023",
    "KEGG": "KEGG_2021_Human",
    "Reactome": "Reactome_2022",
    "WikiPathways": "WikiPathways_2024_Human",
}

mouse_sets = {
    "GO_BP": "GO_Biological_Process_2023",
    "KEGG": "KEGG_2019_Mouse",
    "Reactome": "Reactome_2022",
    "WikiPathways": "WikiPathways_2024_Mouse",
}

# =========================================
# FUNCTION
# =========================================
def run_enrichr(gene_list, gene_sets_dict, organism_label, prefix):
    for short_name, gene_set in gene_sets_dict.items():
        print(f"\nRunning {prefix} {short_name} using {gene_set} ...")

        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_set,
                organism=organism_label,
                outdir=None
            )

            res = enr.results.copy()

            out_csv = results_dir / f"{prefix}_{short_name}_enrichment.csv"
            res.to_csv(out_csv, index=False)

            print(f"Saved: {out_csv}")
            print(f"Rows: {res.shape[0]}")

        except Exception as e:
            print(f"FAILED: {prefix} {short_name}")
            print("Reason:", e)

# =========================================
# RUN HUMAN
# =========================================
run_enrichr(
    gene_list=human_genes,
    gene_sets_dict=human_sets,
    organism_label="human",
    prefix="human"
)

# =========================================
# RUN MOUSE
# =========================================
run_enrichr(
    gene_list=mouse_genes,
    gene_sets_dict=mouse_sets,
    organism_label="mouse",
    prefix="mouse"
)

print("\nAll enrichment analyses completed.")
