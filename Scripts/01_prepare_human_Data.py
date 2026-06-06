import pandas as pd
import numpy as np
from pathlib import Path

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
input_file = base / "data_raw/ST11-ExpressionMatrix.csv"
output_file = base / "data_processed/human_clean.csv"

# =========================================
# LOAD DATA
# =========================================
df = pd.read_csv(input_file)
print(f"Original shape: {df.shape}")

# =========================================
# KEEP RELEVANT COLUMNS
# =========================================
df = df.rename(columns={"genes": "Gene"})
df = df.dropna(subset=["Gene"])

ganglia_cols = [c for c in df.columns if ".ganglia" in c]
df = df[["Gene"] + ganglia_cols]

print(f"After selecting ganglia: {df.shape}")
print("Ganglia columns:", ganglia_cols)

# =========================================
# CLEAN GENE NAMES
# =========================================
df["Gene"] = (
    df["Gene"]
    .astype(str)
    .str.split(";").str[0]
    .str.upper()
    .str.strip()
)
df = df[df["Gene"] != ""]

# =========================================
# CONVERT TO NUMERIC
# =========================================
for col in ganglia_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================================
# REMOVE ALL-NA ROWS
# =========================================
df = df.dropna(subset=ganglia_cols, how="all")

# =========================================
# COLLAPSE DUPLICATES TO GENE LEVEL
# =========================================
df = df.groupby("Gene", as_index=False)[ganglia_cols].mean()
print(f"After grouping: {df.shape}")

# =========================================
# CHECK SCALE
# =========================================
sample_values = df[ganglia_cols].values.flatten()
sample_values = sample_values[~np.isnan(sample_values)]

print(f"Value range: min={np.min(sample_values)}, max={np.max(sample_values)}")

if np.max(sample_values) > 50:
    print("Data appears NOT log-transformed -> applying log2(x + 1)")
    df[ganglia_cols] = np.log2(df[ganglia_cols] + 1)
else:
    print("Data appears already log-transformed -> keeping as is")

# =========================================
# RENAME SAMPLES
# =========================================
rename_map = {col: f"H_S{i+1}" for i, col in enumerate(ganglia_cols)}
df = df.rename(columns=rename_map)

# =========================================
# SAVE
# =========================================
df.to_csv(output_file, index=False)

print("Saved:", output_file)
print("Final shape:", df.shape)
print("Final columns:", df.columns.tolist())
