import pandas as pd
from pathlib import Path

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()

human_file = base / "data_processed/human_clean.csv"
mouse_file = base / "data_processed/mouse_clean.csv"
ortholog_file = base / "ortholog_mapping/human_mouse_orthologs.csv"

human_out = base / "data_processed/human_ortholog.csv"
mouse_out = base / "data_processed/mouse_ortholog.csv"

# =========================================
# LOAD
# =========================================
human = pd.read_csv(human_file)
mouse = pd.read_csv(mouse_file)
orth = pd.read_csv(ortholog_file)

print("Human shape:", human.shape)
print("Mouse shape:", mouse.shape)
print("Ortholog shape:", orth.shape)

# =========================================
# CLEAN GENE COLUMNS
# =========================================
human["Gene"] = human["Gene"].astype(str).str.strip().str.upper()
mouse["Gene"] = mouse["Gene"].astype(str).str.strip().str.upper()

orth["Human_gene"] = orth["Human_gene"].astype(str).str.strip().str.upper()
orth["Mouse_gene"] = orth["Mouse_gene"].astype(str).str.strip().str.upper()

# =========================================
# HUMAN: KEEP ONLY ORTHOLOG-SUPPORTED GENES
# =========================================
human2 = human[human["Gene"].isin(set(orth["Human_gene"]))].copy()
human2 = human2.rename(columns={"Gene": "Ortholog_gene"})

# =========================================
# MOUSE: MAP MOUSE GENE -> HUMAN GENE SPACE
# =========================================
mouse_map = orth[["Mouse_gene", "Human_gene"]].drop_duplicates().copy()

mouse2 = mouse.merge(
    mouse_map,
    left_on="Gene",
    right_on="Mouse_gene",
    how="inner"
).copy()

mouse2 = mouse2.drop(columns=["Gene", "Mouse_gene"])
mouse2 = mouse2.rename(columns={"Human_gene": "Ortholog_gene"})

# =========================================
# REMOVE POSSIBLE DUPLICATES AFTER MAPPING
# =========================================
human_sample_cols = [c for c in human2.columns if c.startswith("H_S")]
mouse_sample_cols = [c for c in mouse2.columns if c.startswith("M_S")]

human2 = (
    human2.groupby("Ortholog_gene", as_index=False)[human_sample_cols]
    .mean()
)

mouse2 = (
    mouse2.groupby("Ortholog_gene", as_index=False)[mouse_sample_cols]
    .mean()
)

# =========================================
# SAVE
# =========================================
human2.to_csv(human_out, index=False)
mouse2.to_csv(mouse_out, index=False)

print("Saved human ortholog matrix:", human_out)
print("Saved mouse ortholog matrix:", mouse_out)
print("Human ortholog shape:", human2.shape)
print("Mouse ortholog shape:", mouse2.shape)

shared = set(human2["Ortholog_gene"]).intersection(set(mouse2["Ortholog_gene"]))
print("Shared ortholog genes between processed matrices:", len(shared))
print("First 10 shared genes:", sorted(list(shared))[:10])
