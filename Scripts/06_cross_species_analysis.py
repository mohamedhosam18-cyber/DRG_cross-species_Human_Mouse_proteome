import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
input_file = base / "results/merged_matrix.csv"
output_file = base / "results/cross_species_results.csv"
aligned_output_file = base / "results/merged_matrix_median_aligned.csv"

# =========================================
# PARAMETERS
# =========================================
fdr_cutoff = 0.05
effect_cutoff = 1.0
min_non_na_per_group = 2

# =========================================
# LOAD
# =========================================
df = pd.read_csv(input_file)

print("Merged matrix shape:", df.shape)

human_cols = [c for c in df.columns if c.startswith("H_S")]
mouse_cols = [c for c in df.columns if c.startswith("M_S")]
sample_cols = human_cols + mouse_cols

print("Human columns:", human_cols)
print("Mouse columns:", mouse_cols)

# =========================================
# SAMPLE-WISE MEDIAN ALIGNMENT
# IMPORTANT:
# This is used only for cross-species comparative testing,
# because the two source datasets are on different global scales.
# =========================================
aligned = df.copy()

for col in sample_cols:
    median_val = aligned[col].median(skipna=True)
    aligned[col] = aligned[col] - median_val

aligned.to_csv(aligned_output_file, index=False)
print("Saved median-aligned matrix:", aligned_output_file)

# =========================================
# CALCULATE STATS
# =========================================
results = []

for _, row in aligned.iterrows():
    gene = row["Ortholog_gene"]

    human_vals = pd.to_numeric(row[human_cols], errors="coerce").astype(float).values
    mouse_vals = pd.to_numeric(row[mouse_cols], errors="coerce").astype(float).values

    human_vals = human_vals[~np.isnan(human_vals)]
    mouse_vals = mouse_vals[~np.isnan(mouse_vals)]

    n_human = len(human_vals)
    n_mouse = len(mouse_vals)

    if n_human == 0 or n_mouse == 0:
        human_mean = np.nan
        mouse_mean = np.nan
        mean_diff = np.nan
        pval = np.nan
    else:
        human_mean = np.mean(human_vals)
        mouse_mean = np.mean(mouse_vals)
        mean_diff = human_mean - mouse_mean

        if n_human >= min_non_na_per_group and n_mouse >= min_non_na_per_group:
            try:
                _, pval = ttest_ind(human_vals, mouse_vals, equal_var=False, nan_policy="omit")
            except Exception:
                pval = np.nan
        else:
            pval = np.nan

    results.append({
        "Gene": gene,
        "Human_Mean_norm": human_mean,
        "Mouse_Mean_norm": mouse_mean,
        "MeanDiff_norm": mean_diff,
        "n_human": n_human,
        "n_mouse": n_mouse,
        "pvalue": pval
    })

res = pd.DataFrame(results)

# =========================================
# CLEAN + FDR
# =========================================
res = res.replace([np.inf, -np.inf], np.nan)

valid_mask = res["pvalue"].notna()
res["FDR"] = np.nan

if valid_mask.sum() > 0:
    res.loc[valid_mask, "FDR"] = multipletests(
        res.loc[valid_mask, "pvalue"],
        method="fdr_bh"
    )[1]

# =========================================
# STATUS
# =========================================
res["status"] = "Not significant"

res.loc[
    (res["MeanDiff_norm"] >= effect_cutoff) & (res["FDR"] < fdr_cutoff),
    "status"
] = "Higher in Human"

res.loc[
    (res["MeanDiff_norm"] <= -effect_cutoff) & (res["FDR"] < fdr_cutoff),
    "status"
] = "Higher in Mouse"

# =========================================
# SORT
# =========================================
res = res.sort_values(
    ["FDR", "pvalue", "MeanDiff_norm"],
    ascending=[True, True, False]
).reset_index(drop=True)

# =========================================
# SAVE
# =========================================
res.to_csv(output_file, index=False)

print("Saved:", output_file)
print("Results shape:", res.shape)
print("Valid p-values:", valid_mask.sum())
print("Status counts:")
print(res["status"].value_counts(dropna=False))
print(res.head(10))
