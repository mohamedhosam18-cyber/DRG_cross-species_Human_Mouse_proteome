import pandas as pd
import numpy as np
from pathlib import Path

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
input_file = base / "data_raw/report.tsv"
output_file = base / "data_processed/mouse_clean.csv"

# =========================================
# LOAD DATA
# =========================================
df = pd.read_csv(input_file, sep="\t", low_memory=False)
print(f"Original shape: {df.shape}")

required_cols = ["Run", "Genes", "Genes.MaxLFQ"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required_cols].copy()

# =========================================
# KEEP ONLY CTRL1-CTRL5
# =========================================
wanted_runs = [
    "OFL001513-YLL-Single-DIA-15K-Ctrl1",
    "OFL001513-YLL-Single-DIA-15K-Ctrl2",
    "OFL001513-YLL-Single-DIA-15K-Ctrl3",
    "OFL001513-YLL-Single-DIA-15K-Ctrl4",
    "OFL001513-YLL-Single-DIA-15K-Ctrl5"
]

df = df[df["Run"].isin(wanted_runs)].copy()
print(f"After Ctrl selection: {df.shape}")

# =========================================
# CLEAN GENE NAMES
# =========================================
df = df.dropna(subset=["Genes"])
df["Gene"] = (
    df["Genes"]
    .astype(str)
    .str.split(";").str[0]
    .str.upper()
    .str.strip()
)
df = df[df["Gene"] != ""]

# =========================================
# NUMERIC CONVERSION
# =========================================
df["Genes.MaxLFQ"] = pd.to_numeric(df["Genes.MaxLFQ"], errors="coerce")
df = df.dropna(subset=["Genes.MaxLFQ"])

# =========================================
# AGGREGATE TO GENE x RUN
# =========================================
gene_run = (
    df.groupby(["Gene", "Run"], as_index=False)["Genes.MaxLFQ"]
    .mean()
)

print(f"After gene/run aggregation: {gene_run.shape}")

# =========================================
# PIVOT TO WIDE MATRIX
# =========================================
wide = gene_run.pivot(index="Gene", columns="Run", values="Genes.MaxLFQ").reset_index()
wide = wide.dropna(subset=wanted_runs, how="all")

print(f"After pivot: {wide.shape}")

# =========================================
# LOG2 TRANSFORM
# =========================================
for col in wanted_runs:
    wide[col] = np.log2(wide[col] + 1)

# =========================================
# RENAME SAMPLES
# =========================================
rename_map = {
    "OFL001513-YLL-Single-DIA-15K-Ctrl1": "M_S1",
    "OFL001513-YLL-Single-DIA-15K-Ctrl2": "M_S2",
    "OFL001513-YLL-Single-DIA-15K-Ctrl3": "M_S3",
    "OFL001513-YLL-Single-DIA-15K-Ctrl4": "M_S4",
    "OFL001513-YLL-Single-DIA-15K-Ctrl5": "M_S5",
}
wide = wide.rename(columns=rename_map)

# =========================================
# SAVE
# =========================================
wide.to_csv(output_file, index=False)

print("Saved:", output_file)
print(f"Final shape:", wide.shape)
print("Final columns:", wide.columns.tolist())
