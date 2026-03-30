import pandas as pd
from pathlib import Path

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
ortholog_file = base / "ortholog_mapping/ortholog_raw.txt"
symbol_file = base / "ortholog_mapping/human_id_symbol.txt"
output_file = base / "ortholog_mapping/human_mouse_orthologs.csv"

# =========================================
# LOAD
# =========================================
orth = pd.read_csv(ortholog_file)
sym = pd.read_csv(symbol_file)

print("Ortholog columns:", orth.columns.tolist())
print("Symbol columns:", sym.columns.tolist())
print("Ortholog shape:", orth.shape)
print("Symbol shape:", sym.shape)

# =========================================
# RENAME COLUMNS
# =========================================
orth = orth.rename(columns={
    "Gene stable ID": "Human_gene_id",
    "Mouse gene name": "Mouse_gene",
    "Mouse gene stable ID": "Mouse_gene_id",
    "Mouse homology type": "Orthology_type",
    "Mouse orthology confidence [0 low, 1 high]": "Orthology_confidence"
})

sym = sym.rename(columns={
    "Gene stable ID": "Human_gene_id",
    "HGNC symbol": "Human_gene"
})

# =========================================
# KEEP NEEDED COLUMNS
# =========================================
orth = orth[[
    "Human_gene_id",
    "Mouse_gene",
    "Mouse_gene_id",
    "Orthology_type",
    "Orthology_confidence"
]].copy()

sym = sym[[
    "Human_gene_id",
    "Human_gene"
]].copy()

# =========================================
# CLEAN
# =========================================
sym["Human_gene"] = sym["Human_gene"].astype(str).str.strip().str.upper()
orth["Mouse_gene"] = orth["Mouse_gene"].astype(str).str.strip().str.upper()

sym = sym.replace({"": pd.NA, "NAN": pd.NA, "None": pd.NA})
sym = sym.dropna(subset=["Human_gene"])

# =========================================
# MERGE
# =========================================
df = orth.merge(sym, on="Human_gene_id", how="left")

print("After merge:", df.shape)

# =========================================
# FILTER TO HIGH-CONFIDENCE ONE-TO-ONE
# =========================================
df["Orthology_confidence"] = pd.to_numeric(df["Orthology_confidence"], errors="coerce")

df = df[
    (df["Orthology_type"] == "ortholog_one2one") &
    (df["Orthology_confidence"] == 1)
].copy()

# =========================================
# REMOVE MISSING NAMES
# =========================================
df = df.dropna(subset=["Human_gene", "Mouse_gene"])

df = df[
    (df["Human_gene"] != "NAN") &
    (df["Mouse_gene"] != "NAN")
].copy()

# =========================================
# REMOVE DUPLICATE / AMBIGUOUS MAPPINGS
# =========================================
df = df.drop_duplicates(subset=["Human_gene", "Mouse_gene"])
df = df.drop_duplicates(subset=["Human_gene"], keep=False)
df = df.drop_duplicates(subset=["Mouse_gene"], keep=False)

# =========================================
# FINAL TABLE
# =========================================
df = df[[
    "Human_gene",
    "Mouse_gene",
    "Human_gene_id",
    "Mouse_gene_id"
]].copy()

df = df.sort_values("Human_gene").reset_index(drop=True)

# =========================================
# SAVE
# =========================================
df.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
print(f"Final shape: {df.shape}")
print(df.head(10))
