import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()

merged_raw_file = base / "results" / "merged_matrix.csv"
merged_aligned_file = base / "results" / "merged_matrix_median_aligned.csv"
de_file = base / "results" / "cross_species_results.csv"

human_orth_file = base / "data_processed" / "human_ortholog.csv"
mouse_orth_file = base / "data_processed" / "mouse_ortholog.csv"

fig_dir = base / "figures" / "heatmaps"
fig_dir.mkdir(parents=True, exist_ok=True)

results_dir = base / "results" / "markers"
results_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# OUTPUT FILES
# =========================================
TOP100_HUMAN_ALL_OUT = fig_dir / "Heatmap_top100_human_higher_all_samples.png"
TOP100_MOUSE_ALL_OUT = fig_dir / "Heatmap_top100_mouse_higher_all_samples.png"
TOP100_HUMAN_MEAN_OUT = fig_dir / "Heatmap_top100_human_higher_species_mean.png"
TOP100_MOUSE_MEAN_OUT = fig_dir / "Heatmap_top100_mouse_higher_species_mean.png"

MARKER_DETECTION_OUT = fig_dir / "DRG_marker_detection_matrix.png"
MARKER_ALL_OUT = fig_dir / "DRG_marker_expression_all_samples_species_specific.png"
MARKER_MEAN_OUT = fig_dir / "DRG_marker_expression_species_mean_species_specific.png"

MARKER_DETECTION_CSV = results_dir / "marker_detection_matrix.csv"
MARKER_MATRIX_CSV = results_dir / "marker_expression_matrix_species_specific.csv"
MARKER_MEAN_CSV = results_dir / "marker_species_mean_matrix_species_specific.csv"

# =========================================
# LITERATURE-GROUNDED MARKER PANEL
# =========================================
marker_blocks = {
    "Pan-neuronal / axonal / synaptic": [
        "RBFOX3", "TUBB3", "PRPH", "NEFL", "NEFM", "NEFH",
        "GAP43", "STMN2", "MAP1B", "DPYSL2", "SNAP25", "STX1A", "VAMP1", "NCAM1"
    ],
    "Peptidergic nociceptor": [
        "CALCA", "TAC1", "TRPV1", "NTRK1", "SCN9A", "SCN10A", "SCN11A", "ADCYAP1"
    ],
    "Non-peptidergic / purinergic / trophic nociceptor": [
        "P2RX3", "RET", "GFRA1", "GFRA2", "GFRA3", "OSMR", "MRGPRD", "SLC17A8"
    ],
    "Mechanoreceptor / proprioceptor / myelinated sensory": [
        "PIEZO2", "NTRK2", "NTRK3", "PVALB", "RUNX3", "ASIC3", "HCN1", "KCNQ2", "KCNQ3"
    ],
    "Glial / myelin / satellite glia": [
        "SOX10", "MBP", "MPZ", "PLP1", "PMP22", "MAG", "GFAP", "S100B", "FABP7", "KCNJ10", "GLUL", "SLC1A3"
    ]
}

ordered_markers = []
marker_group_rows = []
for group, genes in marker_blocks.items():
    for g in genes:
        if g not in ordered_markers:
            ordered_markers.append(g)
            marker_group_rows.append({"Gene": g, "MarkerGroup": group})

marker_group_df = pd.DataFrame(marker_group_rows)
marker_group_map = marker_group_df.set_index("Gene")["MarkerGroup"].to_dict()

# =========================================
# HELPERS
# =========================================
def clean_gene_series(series):
    return series.astype(str).str.upper().str.strip()

def make_numeric(df):
    return df.apply(pd.to_numeric, errors="coerce")

def zscore_rows_keep_nan(df):
    df = df.copy()

    def _z(row):
        vals = row.astype(float)
        mu = vals.mean(skipna=True)
        sd = vals.std(skipna=True)
        if pd.isna(sd) or sd == 0:
            out = vals.copy()
            out.loc[vals.notna()] = 0.0
            return out
        return (vals - mu) / sd

    return df.apply(_z, axis=1)

def fill_gene_row_nans(df):
    return df.apply(lambda row: row.fillna(row.mean(skipna=True)), axis=1)

def style_heatmap_axis(ax, title):
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10, rotation=90)
    ax.tick_params(axis="y", labelsize=9)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

def save_simple_heatmap(df, title, outfile, figsize=(12, 10), cmap="coolwarm",
                        cbar_label="Value", mask=None, center=None, vmin=None, vmax=None):
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        df,
        cmap=cmap,
        linewidths=0.0,
        linecolor=None,
        cbar_kws={"label": cbar_label},
        mask=mask,
        center=center,
        vmin=vmin,
        vmax=vmax,
        yticklabels=False   # turn off auto behavior first
    )

    # force all row labels to appear
    ax.set_yticks(np.arange(df.shape[0]) + 0.5)
    ax.set_yticklabels(df.index.tolist(), fontsize=8, fontweight="bold", rotation=0)

    style_heatmap_axis(ax, title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=700, bbox_inches="tight")
    plt.close()

def add_group_separators(ax, row_names):
    prev_group = None
    for i, gene in enumerate(row_names):
        current_group = marker_group_map.get(gene, None)
        if prev_group is None:
            prev_group = current_group
        elif current_group != prev_group:
            ax.axhline(i, color="black", linewidth=1.2)
            prev_group = current_group

def save_species_specific_marker_heatmap(df, title, outfile, figsize=(14, 18),
                                         cmap="coolwarm", cbar_label="Row-wise z-score",
                                         center=None, vmin=None, vmax=None):
    mask = df.isna()
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        df,
        cmap=cmap,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        mask=mask,
        center=center,
        vmin=vmin,
        vmax=vmax
    )
    style_heatmap_axis(ax, title)
    add_group_separators(ax, df.index.tolist())
    plt.tight_layout()
    plt.savefig(outfile, dpi=700, bbox_inches="tight")
    plt.close()

def prepare_expression_block(df, gene_col, genes, sample_cols):
    block = df[df[gene_col].isin(genes)].copy()
    block = block.set_index(gene_col)[sample_cols]
    block = make_numeric(block)
    block = block.loc[[g for g in genes if g in block.index]]
    block = block.dropna(axis=0, how="all")
    block = fill_gene_row_nans(block)
    block = block.dropna(axis=0, how="any")
    return block

def collapse_for_species_mean_heatmap(expr_df):
    """
    Minimal fix ONLY for the 2-column species mean heatmaps.
    Keep the across-samples heatmaps unchanged.
    """
    out = expr_df.copy()

    if out.index.duplicated().any():
        out = out.groupby(out.index).mean()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=0, how="any")

    return out

def save_top100_all_sample_heatmap(expr_df, title, outfile, col_colors):
    z = zscore_rows_keep_nan(expr_df)

    cg = sns.clustermap(
        z,
        cmap="coolwarm",
        linewidths=0.0,
        col_colors=col_colors,
        figsize=(14, 18),
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Row-wise z-score"},
        dendrogram_ratio=(0.2, 0.15),
        row_cluster=True,
        col_cluster=False   # keep H and M order fixed
    )

    cg.fig.suptitle(title, fontsize=20, fontweight="bold", y=1.02)
    cg.ax_heatmap.set_xlabel("")
    cg.ax_heatmap.set_ylabel("Gene", fontsize=12, fontweight="bold")
    cg.ax_heatmap.tick_params(axis="x", labelsize=10, rotation=90)
    cg.ax_heatmap.tick_params(axis="y", labelsize=9)

    for lab in cg.ax_heatmap.get_xticklabels():
        lab.set_fontweight("bold")
    for lab in cg.ax_heatmap.get_yticklabels():
        lab.set_fontweight("bold")

    cg.savefig(outfile, dpi=700, bbox_inches="tight")
    plt.close("all")

# =========================================
# LOAD DATA
# =========================================
merged_raw = pd.read_csv(merged_raw_file)
merged_aligned = pd.read_csv(merged_aligned_file)
de = pd.read_csv(de_file)

for df in [merged_raw, merged_aligned]:
    df["Ortholog_gene"] = clean_gene_series(df["Ortholog_gene"])

de["Gene"] = clean_gene_series(de["Gene"])

human_cols = [c for c in merged_aligned.columns if c.startswith("H_S")]
mouse_cols = [c for c in merged_aligned.columns if c.startswith("M_S")]
all_cols = human_cols + mouse_cols

print("Merged raw shape:", merged_raw.shape)
print("Merged aligned shape:", merged_aligned.shape)
print("DE shape:", de.shape)
print("Human columns:", human_cols)
print("Mouse columns:", mouse_cols)

# fixed sample colors in fixed order
col_colors = pd.Series(
    ["#d97706"] * len(human_cols) + ["#2e8b57"] * len(mouse_cols),
    index=all_cols
)

# =========================================
# TOP100 HUMAN-HIGHER ACROSS ALL SAMPLES
# IMPORTANT: keep old logic unchanged
# =========================================
de_human_top100 = (
    de[de["status"] == "Higher in Human"]
    .dropna(subset=["Gene", "MeanDiff_norm", "FDR"])
    .sort_values(["FDR", "MeanDiff_norm"], ascending=[True, False])
    .head(100)
    .copy()
)

top100_human_genes = de_human_top100["Gene"].tolist()
top100_human_expr = prepare_expression_block(merged_aligned, "Ortholog_gene", top100_human_genes, all_cols)

save_top100_all_sample_heatmap(
    top100_human_expr,
    "Top 100 Human-higher proteins across all samples",
    TOP100_HUMAN_ALL_OUT,
    col_colors
)

# minimal fix only for species mean heatmap
top100_human_expr_meanfix = collapse_for_species_mean_heatmap(top100_human_expr)

top100_human_mean = pd.DataFrame({
    "Human_mean": top100_human_expr_meanfix[human_cols].mean(axis=1),
    "Mouse_mean": top100_human_expr_meanfix[mouse_cols].mean(axis=1)
})

max_abs_human = np.nanmax(np.abs(top100_human_mean.values))
save_simple_heatmap(
    top100_human_mean,
    "Top 100 Human-higher proteins: species mean (aligned values)",
    TOP100_HUMAN_MEAN_OUT,
    figsize=(7, max(12, 0.20 * len(top100_human_mean))),
    cmap="coolwarm",
    cbar_label="Aligned mean abundance",
    center=0,
    vmin=-max_abs_human,
    vmax=max_abs_human
)
# =========================================
# TOP100 MOUSE-HIGHER ACROSS ALL SAMPLES
# IMPORTANT: keep old logic unchanged
# =========================================
de_mouse_top100 = (
    de[de["status"] == "Higher in Mouse"]
    .dropna(subset=["Gene", "MeanDiff_norm", "FDR"])
    .sort_values(["FDR", "MeanDiff_norm"], ascending=[True, True])
    .head(100)
    .copy()
)

top100_mouse_genes = de_mouse_top100["Gene"].tolist()
top100_mouse_expr = prepare_expression_block(merged_aligned, "Ortholog_gene", top100_mouse_genes, all_cols)

save_top100_all_sample_heatmap(
    top100_mouse_expr,
    "Top 100 Mouse-higher proteins across all samples",
    TOP100_MOUSE_ALL_OUT,
    col_colors
)

# minimal fix only for species mean heatmap
top100_mouse_expr_meanfix = collapse_for_species_mean_heatmap(top100_mouse_expr)

top100_mouse_mean = pd.DataFrame({
    "Human_mean": top100_mouse_expr_meanfix[human_cols].mean(axis=1),
    "Mouse_mean": top100_mouse_expr_meanfix[mouse_cols].mean(axis=1)
})

max_abs_mouse = np.nanmax(np.abs(top100_mouse_mean.values))
save_simple_heatmap(
    top100_mouse_mean,
    "Top 100 Mouse-higher proteins: species mean (aligned values)",
    TOP100_MOUSE_MEAN_OUT,
    figsize=(7, max(12, 0.20 * len(top100_mouse_mean))),
    cmap="coolwarm",
    cbar_label="Aligned mean abundance",
    center=0,
    vmin=-max_abs_mouse,
    vmax=max_abs_mouse
)
# =========================================
# LOAD SPECIES-SPECIFIC ORTHOLOG MATRICES FOR MARKERS
# keep old logic unchanged
# =========================================
human_orth = pd.read_csv(human_orth_file)
mouse_orth = pd.read_csv(mouse_orth_file)

human_orth["Ortholog_gene"] = clean_gene_series(human_orth["Ortholog_gene"])
mouse_orth["Ortholog_gene"] = clean_gene_series(mouse_orth["Ortholog_gene"])

human_orth = human_orth.drop_duplicates(subset=["Ortholog_gene"]).set_index("Ortholog_gene")
mouse_orth = mouse_orth.drop_duplicates(subset=["Ortholog_gene"]).set_index("Ortholog_gene")

human_orth_cols = [c for c in human_orth.columns if c.startswith("H_S")]
mouse_orth_cols = [c for c in mouse_orth.columns if c.startswith("M_S")]

human_full = make_numeric(human_orth[human_orth_cols])
mouse_full = make_numeric(mouse_orth[mouse_orth_cols])

# sample-wise median alignment for visualization only
for col in human_orth_cols:
    human_full[col] = human_full[col] - human_full[col].median(skipna=True)

for col in mouse_orth_cols:
    mouse_full[col] = mouse_full[col] - mouse_full[col].median(skipna=True)

human_sub = human_full.reindex(ordered_markers)
mouse_sub = mouse_full.reindex(ordered_markers)

combined_marker = pd.concat([human_sub, mouse_sub], axis=1)

# detection matrix
marker_detection = pd.DataFrame(index=ordered_markers)
marker_detection["Human_detected"] = human_orth.reindex(ordered_markers)[human_orth_cols].notna().any(axis=1).astype(int)
marker_detection["Mouse_detected"] = mouse_orth.reindex(ordered_markers)[mouse_orth_cols].notna().any(axis=1).astype(int)
marker_detection.to_csv(MARKER_DETECTION_CSV)

save_simple_heatmap(
    marker_detection,
    "DRG marker detection matrix",
    MARKER_DETECTION_OUT,
    figsize=(8, 22),
    cmap="YlGnBu",
    cbar_label="Detected (1) / Not detected (0)"
)

# all-sample species-specific marker heatmap
marker_z = zscore_rows_keep_nan(combined_marker)
marker_z.to_csv(MARKER_MATRIX_CSV)
save_species_specific_marker_heatmap(
    marker_z,
    "DRG marker expression across all samples (blank = not detected)",
    MARKER_ALL_OUT,
    figsize=(14, 20),
    cmap="coolwarm",
    cbar_label="Row-wise z-score",
    center=0,
    vmin=-2.5,
    vmax=2.5
)

# species mean marker heatmap on aligned means
marker_mean = pd.DataFrame(index=ordered_markers)
marker_mean["Human_mean"] = human_sub.mean(axis=1, skipna=True)
marker_mean["Mouse_mean"] = mouse_sub.mean(axis=1, skipna=True)

marker_mean.to_csv(MARKER_MEAN_CSV)
max_abs_marker = np.nanmax(np.abs(marker_mean.values))

save_species_specific_marker_heatmap(
    marker_mean,
    "DRG marker expression: species mean (blank = not detected)",
    MARKER_MEAN_OUT,
    figsize=(7, 20),
    cmap="coolwarm",
    cbar_label="Aligned mean abundance",
    center=0,
    vmin=-max_abs_marker,
    vmax=max_abs_marker
)

# =========================================
# REPORT
# =========================================
print("\nSaved files:")
print(TOP100_HUMAN_ALL_OUT)
print(TOP100_MOUSE_ALL_OUT)
print(TOP100_HUMAN_MEAN_OUT)
print(TOP100_MOUSE_MEAN_OUT)
print(MARKER_DETECTION_OUT)
print(MARKER_ALL_OUT)
print(MARKER_MEAN_OUT)

print("\nSaved tables:")
print(MARKER_DETECTION_CSV)
print(MARKER_MATRIX_CSV)
print(MARKER_MEAN_CSV)

print("\nTop 100 Human-higher genes used:", len(top100_human_genes))
print("Top 100 Mouse-higher genes used:", len(top100_mouse_genes))

marker_detection_reset = marker_detection.reset_index().rename(columns={"index": "Gene"})

print("\nMarkers present in human only:")
human_only = marker_detection_reset[
    (marker_detection_reset["Human_detected"] == 1) &
    (marker_detection_reset["Mouse_detected"] == 0)
]["Gene"].tolist()
print(human_only)

print("\nMarkers present in mouse only:")
mouse_only = marker_detection_reset[
    (marker_detection_reset["Human_detected"] == 0) &
    (marker_detection_reset["Mouse_detected"] == 1)
]["Gene"].tolist()
print(mouse_only)

print("\nMarkers absent in both:")
absent_both = marker_detection_reset[
    (marker_detection_reset["Human_detected"] == 0) &
    (marker_detection_reset["Mouse_detected"] == 0)
]["Gene"].tolist()
print(absent_both)
