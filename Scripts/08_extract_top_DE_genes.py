import pandas as pd
from pathlib import Path

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
input_file = base / "results/cross_species_results.csv"
out_dir = base / "results" / "top_gene_lists"
out_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# PARAMETERS
# =========================================
top_sizes = [50, 100, 120]

# =========================================
# LOAD
# =========================================
df = pd.read_csv(input_file)

print("Input shape:", df.shape)
required_cols = ["Gene", "MeanDiff_norm", "pvalue", "FDR", "status"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# =========================================
# CLEAN
# =========================================
df = df.dropna(subset=["Gene", "MeanDiff_norm", "pvalue", "FDR", "status"]).copy()
df["Gene"] = df["Gene"].astype(str).str.upper().str.strip()

# Keep only significant directional sets
human = df[df["status"] == "Higher in Human"].copy()
mouse = df[df["status"] == "Higher in Mouse"].copy()

print("Higher in Human:", human.shape[0])
print("Higher in Mouse:", mouse.shape[0])

# =========================================
# SORTING RULES
# =========================================
# Human-higher:
#   1) smaller FDR
#   2) smaller pvalue
#   3) larger MeanDiff_norm
human = human.sort_values(
    ["FDR", "pvalue", "MeanDiff_norm"],
    ascending=[True, True, False]
).reset_index(drop=True)

# Mouse-higher:
#   1) smaller FDR
#   2) smaller pvalue
#   3) more negative MeanDiff_norm
mouse = mouse.sort_values(
    ["FDR", "pvalue", "MeanDiff_norm"],
    ascending=[True, True, True]
).reset_index(drop=True)

# =========================================
# SAVE FUNCTION
# =========================================
def save_gene_set(df_subset, prefix):
    csv_file = out_dir / f"{prefix}.csv"
    txt_file = out_dir / f"{prefix}_genes.txt"

    df_subset.to_csv(csv_file, index=False)
    df_subset["Gene"].to_csv(txt_file, index=False, header=False)

    print(f"Saved: {csv_file}")
    print(f"Saved: {txt_file}")

# =========================================
# EXPORT TOP LISTS
# =========================================
for n in top_sizes:
    top_human = human.head(n).copy()
    top_mouse = mouse.head(n).copy()

    # Main biologically correct names
    save_gene_set(top_human, f"top{n}_human_higher")
    save_gene_set(top_mouse, f"top{n}_mouse_higher")

    # Optional convenience aliases:
    # human-down = mouse-higher
    # mouse-down = human-higher
    save_gene_set(top_mouse, f"top{n}_human_down")
    save_gene_set(top_human, f"top{n}_mouse_down")

# =========================================
# SAVE FULL SIGNIFICANT SETS TOO
# =========================================
save_gene_set(human, "all_significant_human_higher")
save_gene_set(mouse, "all_significant_mouse_higher")
save_gene_set(mouse, "all_significant_human_down")
save_gene_set(human, "all_significant_mouse_down")

# =========================================
# PREVIEW
# =========================================
print("\nTop 10 Human-higher genes:")
print(human[["Gene", "MeanDiff_norm", "pvalue", "FDR"]].head(10))

print("\nTop 10 Mouse-higher genes:")
print(mouse[["Gene", "MeanDiff_norm", "pvalue", "FDR"]].head(10))
