import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adjustText import adjust_text

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()

merged_raw_file = base / "results" / "merged_matrix.csv"
merged_aligned_file = base / "results" / "merged_matrix_median_aligned.csv"
de_file = base / "results" / "cross_species_results.csv"

out_dir = base / "figures" / "paper_style_final_fixed"
out_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# GLOBAL STYLE
# =========================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

# =========================================
# SETTINGS
# marker panel kept broad enough to match your target figure
# =========================================
marker_genes = [
    "TRPV1", "SCN9A", "SCN10A", "SCN11A",
    "P2RX3", "RET", "OSMR",
    "NTRK1", "NTRK2", "NTRK3", "GFRA2",
    "HCN1",
    "NEFH", "NEFL", "PRPH", "GAP43",
    "MBP", "MPZ", "GFAP", "SOX10", "S100B",
    "MAG", "PLP1", "FABP7", "KCNJ10", "GLUL", "SLC1A3"
]

ranked_curve_markers_human = [
    "NEFL", "PRPH", "S100B", "GAP43", "SCN10A",
    "NTRK3", "NTRK1", "GFRA2", "SCN9A", "TRPV1", "P2RX3", "RET"
]

ranked_curve_markers_mouse = [
    "MPZ", "MBP", "NEFL", "NEFH", "PRPH",
    "S100B", "SCN11A", "GFAP", "SCN9A",
    "SOX10", "TRPV1", "NTRK1", "MAG", "PLP1"
]

# this is the key fix: select 16 strongest non-marker genes on each side
N_HUMAN_EXTREME = 16
N_MOUSE_EXTREME = 16

COLOR_HUMAN = "#cc6f12"
COLOR_MOUSE = "#2f8a57"
COLOR_MARKER = "#7a1fa2"
COLOR_OTHER = "gray"

# =========================================
# HELPERS
# =========================================
def clean_gene(series):
    return series.astype(str).str.upper().str.strip()

def style_axis(ax, spine_width=2.6):
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
        spine.set_color("black")
    ax.tick_params(axis="both", labelsize=13, width=1.8, length=6)
    ax.grid(False)

def annotate_ranked(ax, ranked_df, genes_to_use, color):
    texts = []

    for gene in genes_to_use:
        sub = ranked_df[ranked_df["Gene"] == gene]
        if sub.empty:
            continue

        x = sub["Rank"].iloc[0]
        y = sub["MeanAbundance"].iloc[0]

        ax.scatter(x, y, s=34, color=color, edgecolor="black", linewidth=0.6, zorder=5)

        txt = ax.text(
            x, y, gene,
            fontsize=11,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=1.2, alpha=0.96),
            zorder=6
        )
        texts.append(txt)

    if texts:
        adjust_text(
            texts,
            ax=ax,
            expand=(1.18, 1.28),
            force_text=(0.45, 0.70),
            force_points=(0.20, 0.40),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0)
        )

def annotate_scatter(ax, df, genes_to_label, color_map):
    texts = []

    for gene in genes_to_label:
        sub = df[df["Gene"] == gene]
        if sub.empty:
            continue

        x = sub["Human_mean"].iloc[0]
        y = sub["Mouse_mean"].iloc[0]
        c = color_map.get(gene, "black")

        txt = ax.text(
            x, y, gene,
            fontsize=11,
            fontweight="bold",
            color=c,
            zorder=8
        )
        texts.append(txt)

    if texts:
        adjust_text(
            texts,
            ax=ax,
            expand=(1.14, 1.24),
            force_text=(0.48, 0.75),
            force_points=(0.28, 0.48),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.9)
        )

# =========================================
# LOAD DATA
# =========================================
merged_raw = pd.read_csv(merged_raw_file)
merged_aligned = pd.read_csv(merged_aligned_file)
de = pd.read_csv(de_file)

merged_raw["Ortholog_gene"] = clean_gene(merged_raw["Ortholog_gene"])
merged_aligned["Ortholog_gene"] = clean_gene(merged_aligned["Ortholog_gene"])
de["Gene"] = clean_gene(de["Gene"])

human_cols = [c for c in merged_raw.columns if c.startswith("H_S")]
mouse_cols = [c for c in merged_raw.columns if c.startswith("M_S")]

# raw matrix for ranked abundance
expr_raw = merged_raw.set_index("Ortholog_gene")[human_cols + mouse_cols].apply(pd.to_numeric, errors="coerce")

# aligned matrix for the main scatter
expr_aligned = merged_aligned.set_index("Ortholog_gene")[human_cols + mouse_cols].apply(pd.to_numeric, errors="coerce")

# =========================================
# 1) RANKED ABUNDANCE FIGURE
# =========================================
mean_df_raw = pd.DataFrame(index=expr_raw.index)
mean_df_raw["Human_mean"] = expr_raw[human_cols].mean(axis=1)
mean_df_raw["Mouse_mean"] = expr_raw[mouse_cols].mean(axis=1)
mean_df_raw = mean_df_raw.reset_index().rename(columns={"Ortholog_gene": "Gene"})

human_rank = (
    mean_df_raw[["Gene", "Human_mean"]]
    .dropna()
    .rename(columns={"Human_mean": "MeanAbundance"})
    .sort_values("MeanAbundance", ascending=False)
    .reset_index(drop=True)
)
human_rank["Rank"] = np.arange(1, len(human_rank) + 1)

mouse_rank = (
    mean_df_raw[["Gene", "Mouse_mean"]]
    .dropna()
    .rename(columns={"Mouse_mean": "MeanAbundance"})
    .sort_values("MeanAbundance", ascending=False)
    .reset_index(drop=True)
)
mouse_rank["Rank"] = np.arange(1, len(mouse_rank) + 1)

human_rank_markers = [g for g in ranked_curve_markers_human if g in human_rank["Gene"].values]
mouse_rank_markers = [g for g in ranked_curve_markers_mouse if g in mouse_rank["Gene"].values]

fig, axes = plt.subplots(1, 2, figsize=(24, 14.5), constrained_layout=True)

ax = axes[0]
ax.plot(human_rank["Rank"], human_rank["MeanAbundance"], color=COLOR_HUMAN, lw=3.2, zorder=2)
ax.scatter(human_rank["Rank"], human_rank["MeanAbundance"], color="black", s=7, alpha=0.45, zorder=1)
annotate_ranked(ax, human_rank, human_rank_markers, COLOR_HUMAN)
ax.set_xlabel("Protein abundance rank", fontsize=16, fontweight="bold")
ax.set_ylabel("Mean protein level", fontsize=16, fontweight="bold")
style_axis(ax, spine_width=2.4)

ax = axes[1]
ax.plot(mouse_rank["Rank"], mouse_rank["MeanAbundance"], color=COLOR_MOUSE, lw=3.2, zorder=2)
ax.scatter(mouse_rank["Rank"], mouse_rank["MeanAbundance"], color="black", s=7, alpha=0.45, zorder=1)
annotate_ranked(ax, mouse_rank, mouse_rank_markers, COLOR_MOUSE)
ax.set_xlabel("Protein abundance rank", fontsize=16, fontweight="bold")
ax.set_ylabel("Mean protein level", fontsize=16, fontweight="bold")
style_axis(ax, spine_width=2.4)

fig.suptitle("Ranked abundance of ortholog-mapped DRG proteomes", fontsize=26, fontweight="bold", y=1.02)
plt.savefig(out_dir / "ranked_proteome_abundance_human_mouse_final.png", dpi=700, bbox_inches="tight")
plt.close()

# =========================================
# 2) MAIN CROSS-SPECIES SCATTER
# IMPORTANT FIX:
# - aligned means for axes
# - old script's label selection logic from log2FC + pvalue
# - marker colors stay purple
# - human extremes orange, mouse extremes green
# =========================================
mean_df_aligned = pd.DataFrame(index=expr_aligned.index)
mean_df_aligned["Human_mean"] = expr_aligned[human_cols].mean(axis=1)
mean_df_aligned["Mouse_mean"] = expr_aligned[mouse_cols].mean(axis=1)
mean_df_aligned = mean_df_aligned.reset_index().rename(columns={"Ortholog_gene": "Gene"})

needed_de_cols = ["Gene", "MeanDiff_norm", "pvalue", "status"]
missing = [c for c in needed_de_cols if c not in de.columns]
if missing:
    raise ValueError(f"Missing DE columns: {missing}")

plot_df = mean_df_aligned.merge(
    de[needed_de_cols],
    on="Gene",
    how="left"
)

plot_df = plot_df.dropna(subset=["Human_mean", "Mouse_mean"]).copy()

pearson_r = np.corrcoef(plot_df["Human_mean"], plot_df["Mouse_mean"])[0, 1]

def point_category(row):
    if row["Gene"] in marker_genes:
        return "marker"
    if pd.notna(row.get("status")):
        if row["status"] == "Higher in Human":
            return "human_up"
        if row["status"] == "Higher in Mouse":
            return "mouse_up"
    return "other"

plot_df["Category"] = plot_df.apply(point_category, axis=1)

# this is the critical part copied from the correct logic:
extreme_human = (
    plot_df[plot_df["Category"] != "marker"]
    .dropna(subset=["MeanDiff_norm", "pvalue"])
    .sort_values(["MeanDiff_norm", "pvalue"], ascending=[False, True])
    .head(N_HUMAN_EXTREME)["Gene"]
    .tolist()
)

extreme_mouse = (
    plot_df[plot_df["Category"] != "marker"]
    .dropna(subset=["MeanDiff_norm", "pvalue"])
    .sort_values(["MeanDiff_norm", "pvalue"], ascending=[True, True])
    .head(N_MOUSE_EXTREME)["Gene"]
    .tolist()
)

genes_to_label = list(dict.fromkeys(marker_genes + extreme_human + extreme_mouse))
genes_to_label = [g for g in genes_to_label if g in plot_df["Gene"].values]

label_color_map = {}
for g in marker_genes:
    label_color_map[g] = COLOR_MARKER
for g in extreme_human:
    if g not in label_color_map:
        label_color_map[g] = COLOR_HUMAN
for g in extreme_mouse:
    if g not in label_color_map:
        label_color_map[g] = COLOR_MOUSE

fig, ax = plt.subplots(figsize=(11.5, 12.5), constrained_layout=True)

other_df = plot_df[plot_df["Category"] == "other"]
human_up_df = plot_df[plot_df["Category"] == "human_up"]
mouse_up_df = plot_df[plot_df["Category"] == "mouse_up"]
marker_df = plot_df[plot_df["Category"] == "marker"]

ax.scatter(
    other_df["Human_mean"], other_df["Mouse_mean"],
    s=16, alpha=0.18, color=COLOR_OTHER, edgecolor="none", zorder=1
)
ax.scatter(
    human_up_df["Human_mean"], human_up_df["Mouse_mean"],
    s=24, alpha=0.60, color=COLOR_HUMAN, edgecolor="none", zorder=2, label="Human-biased"
)
ax.scatter(
    mouse_up_df["Human_mean"], mouse_up_df["Mouse_mean"],
    s=24, alpha=0.60, color=COLOR_MOUSE, edgecolor="none", zorder=2, label="Mouse-biased"
)
ax.scatter(
    marker_df["Human_mean"], marker_df["Mouse_mean"],
    s=52, alpha=0.98, color=COLOR_MARKER, edgecolor="black", linewidth=0.5, zorder=4, label="Key markers"
)

xmin = min(plot_df["Human_mean"].min(), plot_df["Mouse_mean"].min())
xmax = max(plot_df["Human_mean"].max(), plot_df["Mouse_mean"].max())
pad = (xmax - xmin) * 0.08
xmin -= pad
xmax += pad

ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", color="black", lw=1.8, zorder=0)

annotate_scatter(ax, plot_df, genes_to_label, label_color_map)

ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)

xrange = xmax - xmin
yrange = xmax - xmin

ax.text(
    xmin + 0.08 * xrange, xmax - 0.10 * yrange,
    "Higher in Mouse",
    fontsize=17, fontweight="bold", color=COLOR_MOUSE,
    ha="left", va="top"
)

ax.text(
    xmax - 0.08 * xrange, xmin + 0.10 * yrange,
    "Higher in Human",
    fontsize=17, fontweight="bold", color=COLOR_HUMAN,
    ha="right", va="bottom"
)

summary_text = (
    f"Shared ortholog proteins: {len(plot_df)}\n"
    f"Pearson r = {pearson_r:.2f}"
)

ax.text(
    0.03, 0.98,
    summary_text,
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    ha="left",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.28",
        fc="white",
        ec="black",
        lw=1.4,
        alpha=0.96
    ),
    zorder=10
)

ax.set_xlabel("Human mean aligned protein level", fontsize=18, fontweight="bold")
ax.set_ylabel("Mouse mean aligned protein level", fontsize=18, fontweight="bold")
ax.set_title("Cross-species comparison of ortholog-mapped DRG proteomes", fontsize=27, fontweight="bold", pad=16)
style_axis(ax, spine_width=2.8)

leg = ax.legend(
    frameon=True,
    edgecolor="black",
    fontsize=13,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98)
)
leg.get_frame().set_linewidth(1.6)
leg.get_frame().set_alpha(0.95)

plt.savefig(out_dir / "human_vs_mouse_mean_scatter_annotated_final_fixed.png", dpi=700, bbox_inches="tight")
plt.close()

# =========================================
# SAVE TABLES
# =========================================
human_rank.to_csv(out_dir / "human_ranked_abundance_table_final.csv", index=False)
mouse_rank.to_csv(out_dir / "mouse_ranked_abundance_table_final.csv", index=False)
plot_df.to_csv(out_dir / "human_vs_mouse_scatter_table_final_fixed.csv", index=False)

pd.DataFrame({"Gene": extreme_human}).to_csv(out_dir / "human_extreme_labeled_genes.csv", index=False)
pd.DataFrame({"Gene": extreme_mouse}).to_csv(out_dir / "mouse_extreme_labeled_genes.csv", index=False)
pd.DataFrame({"Gene": genes_to_label}).to_csv(out_dir / "all_labeled_genes.csv", index=False)

# =========================================
# REPORT
# =========================================
print("\nDONE")
print(f"\nPearson r = {pearson_r:.4f}")

print("\nHuman extreme labeled genes:")
print(extreme_human)

print("\nMouse extreme labeled genes:")
print(extreme_mouse)

print("\nAll labeled genes:")
print(genes_to_label)

print("\nSaved:")
print(out_dir / "ranked_proteome_abundance_human_mouse_final.png")
print(out_dir / "human_vs_mouse_mean_scatter_annotated_final_fixed.png")
print(out_dir / "human_extreme_labeled_genes.csv")
print(out_dir / "mouse_extreme_labeled_genes.csv")
print(out_dir / "all_labeled_genes.csv")
