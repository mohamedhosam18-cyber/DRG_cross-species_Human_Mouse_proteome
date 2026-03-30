import pandas as pd
from pathlib import Path

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()

human_file = base / "data_processed/human_ortholog.csv"
mouse_file = base / "data_processed/mouse_ortholog.csv"
output_file = base / "results/merged_matrix.csv"

# =========================================
# LOAD
# =========================================
human = pd.read_csv(human_file)
mouse = pd.read_csv(mouse_file)

print("Human ortholog shape:", human.shape)
print("Mouse ortholog shape:", mouse.shape)

# =========================================
# FIND SHARED GENES
# =========================================
shared = sorted(set(human["Ortholog_gene"]).intersection(set(mouse["Ortholog_gene"])))
print("Shared genes:", len(shared))

# =========================================
# KEEP ONLY SHARED GENES
# =========================================
human2 = human[human["Ortholog_gene"].isin(shared)].copy()
mouse2 = mouse[mouse["Ortholog_gene"].isin(shared)].copy()

# =========================================
# SORT BY GENE
# =========================================
human2 = human2.sort_values("Ortholog_gene").reset_index(drop=True)
mouse2 = mouse2.sort_values("Ortholog_gene").reset_index(drop=True)

# =========================================
# CHECK SAME ORDER
# =========================================
if not human2["Ortholog_gene"].equals(mouse2["Ortholog_gene"]):
    raise ValueError("Gene order mismatch between human and mouse matrices.")

# =========================================
# MERGE SIDE-BY-SIDE
# =========================================
human_sample_cols = [c for c in human2.columns if c.startswith("H_S")]
mouse_sample_cols = [c for c in mouse2.columns if c.startswith("M_S")]

merged = pd.concat(
    [
        human2[["Ortholog_gene"] + human_sample_cols],
        mouse2[mouse_sample_cols]
    ],
    axis=1
)

# =========================================
# SAVE
# =========================================
merged.to_csv(output_file, index=False)

print("Saved:", output_file)
print("Merged shape:", merged.shape)
print("Columns:", merged.columns.tolist())
print(merged.head(10))
