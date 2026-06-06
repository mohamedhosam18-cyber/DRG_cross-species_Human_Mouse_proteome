import pandas as pd
from pathlib import Path
import gseapy as gp

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()

input_file = base / "results" / "cross_species_results.csv"
rnk_file = base / "results" / "cross_species_GSEA.rnk"
results_dir = base / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# PARAMETERS
# =========================================
gene_sets = {
    "GO_BP": "GO_Biological_Process_2023",
    "KEGG": "KEGG_2021_Human",
    "Reactome": "Reactome_2022",
}

permutation_num = 1000
min_size = 15
max_size = 500
seed = 42

# =========================================
# LOAD INPUT
# =========================================
df = pd.read_csv(input_file)

print("Input shape:", df.shape)
print("Columns:", df.columns.tolist())

required_cols = ["Gene", "MeanDiff_norm"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# =========================================
# CLEAN + PREPARE RANKING
# =========================================
df = df.dropna(subset=["Gene", "MeanDiff_norm"]).copy()
df["Gene"] = df["Gene"].astype(str).str.strip().str.upper()

# if FDR exists, keep the best entry per duplicated gene
if "FDR" in df.columns:
    df = df.sort_values(["FDR", "MeanDiff_norm"], ascending=[True, False]).copy()
else:
    df = df.sort_values("MeanDiff_norm", ascending=False).copy()

df = df.drop_duplicates(subset=["Gene"], keep="first").copy()

# sort from most human-enriched to most mouse-enriched
df = df.sort_values("MeanDiff_norm", ascending=False).copy()

# save ranking file
df[["Gene", "MeanDiff_norm"]].to_csv(
    rnk_file,
    sep="\t",
    index=False,
    header=False
)

print("Saved ranking file:", rnk_file)
print("Final ranking shape:", df.shape)

print("\nTop 10 human-side genes:")
print(df[["Gene", "MeanDiff_norm"]].head(10))

print("\nTop 10 mouse-side genes:")
print(df[["Gene", "MeanDiff_norm"]].tail(10))

# =========================================
# RUN GSEA PRERANK
# =========================================
for label, gs in gene_sets.items():
    outdir = results_dir / f"GSEA_{label}"
    print(f"\nRunning GSEA for {label} ...")
    print("Gene set database:", gs)
    print("Output folder:", outdir)

    gp.prerank(
        rnk=str(rnk_file),
        gene_sets=gs,
        outdir=str(outdir),
        permutation_num=permutation_num,
        min_size=min_size,
        max_size=max_size,
        seed=seed,
        verbose=True
    )

print("\nFull GSEA pipeline completed.")
print("Generated folders:")
for label in gene_sets:
    print(results_dir / f"GSEA_{label}")
