import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
input_file = base / "results" / "merged_matrix.csv"

fig_dir = base / "figures" / "final_dual_pca"
fig_dir.mkdir(parents=True, exist_ok=True)

results_dir = base / "results" / "final_dual_pca"
results_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# STYLE
# =========================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 1.6
plt.rcParams["xtick.major.width"] = 1.3
plt.rcParams["ytick.major.width"] = 1.3
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

COLOR_HUMAN = "#cc6f12"
COLOR_MOUSE = "#2f8a57"

# =========================================
# HELPERS
# =========================================
def clean_gene(series):
    return series.astype(str).str.upper().str.strip()

def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.tick_params(axis="both", labelsize=11, width=1.3)

def save_scree_plot(explained, out_file, title):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    pcs = ["PC1", "PC2", "PC3"]
    ax.bar(pcs, explained, edgecolor="black", linewidth=1.0)
    ax.plot(pcs, explained, marker="o", linewidth=1.5, color="black")
    for i, v in enumerate(explained):
        ax.text(i, v + max(explained) * 0.015, f"{v:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Explained variance (%)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_file, dpi=500, bbox_inches="tight")
    plt.close()

def save_2d_plot(scores, explained, out_file, title):
    human = scores[scores["Group"] == "Human"].copy()
    mouse = scores[scores["Group"] == "Mouse"].copy()

    fig, ax = plt.subplots(figsize=(9.5, 8))

    ax.scatter(
        human["PC1"], human["PC2"],
        s=150, color=COLOR_HUMAN, label="Human",
        edgecolors="black", linewidths=1.0, zorder=3
    )
    ax.scatter(
        mouse["PC1"], mouse["PC2"],
        s=150, color=COLOR_MOUSE, label="Mouse",
        edgecolors="black", linewidths=1.0, zorder=3
    )

    for _, row in scores.iterrows():
        ax.annotate(
            row["Sample"],
            (row["PC1"], row["PC2"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.2f}% variance)", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"PC2 ({explained[1]:.2f}% variance)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.legend(frameon=True, edgecolor="black", fontsize=12)
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_file, dpi=500, bbox_inches="tight")
    plt.close()

def save_3d_plot(scores, explained, out_file, title):
    human = scores[scores["Group"] == "Human"].copy()
    mouse = scores[scores["Group"] == "Mouse"].copy()

    fig = plt.figure(figsize=(12.5, 10.2))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        human["PC1"], human["PC2"], human["PC3"],
        s=160, color=COLOR_HUMAN, label="Human",
        edgecolors="black", linewidths=0.9, depthshade=True
    )
    ax.scatter(
        mouse["PC1"], mouse["PC2"], mouse["PC3"],
        s=160, color=COLOR_MOUSE, label="Mouse",
        edgecolors="black", linewidths=0.9, depthshade=True
    )

    for _, row in scores.iterrows():
        ax.text(
            row["PC1"], row["PC2"], row["PC3"],
            row["Sample"],
            fontsize=9,
            fontweight="bold"
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.2f}% variance)", fontsize=12, fontweight="bold", labelpad=16)
    ax.set_ylabel(f"PC2 ({explained[1]:.2f}% variance)", fontsize=12, fontweight="bold", labelpad=16)
    ax.set_zlabel(f"PC3 ({explained[2]:.2f}% variance)", fontsize=12, fontweight="bold", labelpad=24)

    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    ax.view_init(elev=24, azim=-58)

    try:
        ax.set_box_aspect((1.15, 1.0, 0.95))
    except Exception:
        pass

    ax.legend(loc="upper left", frameon=True, edgecolor="black", fontsize=11)

    # Extra figure-side label so PC3 is unmistakably visible in PNG
    fig.text(
        0.94, 0.50,
        f"PC3 ({explained[2]:.2f}% variance)",
        rotation=90,
        ha="center", va="center",
        fontsize=12, fontweight="bold"
    )

    plt.subplots_adjust(left=0.03, right=0.90, top=0.90, bottom=0.05)
    plt.savefig(out_file, dpi=500, bbox_inches="tight")
    plt.close()

def save_pairwise_plot(scores, explained, out_file, title):
    human = scores[scores["Group"] == "Human"].copy()
    mouse = scores[scores["Group"] == "Mouse"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    pairs = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]
    var_map = {"PC1": explained[0], "PC2": explained[1], "PC3": explained[2]}

    for ax, (xpc, ypc) in zip(axes, pairs):
        ax.scatter(
            human[xpc], human[ypc],
            s=130, color=COLOR_HUMAN, edgecolors="black", linewidths=1.0, zorder=3
        )
        ax.scatter(
            mouse[xpc], mouse[ypc],
            s=130, color=COLOR_MOUSE, edgecolors="black", linewidths=1.0, zorder=3
        )

        for _, row in scores.iterrows():
            ax.annotate(
                row["Sample"],
                (row[xpc], row[ypc]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold"
            )

        ax.set_xlabel(f"{xpc} ({var_map[xpc]:.2f}%)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{ypc} ({var_map[ypc]:.2f}%)", fontsize=12, fontweight="bold")
        ax.set_title(f"{xpc} vs {ypc}", fontsize=14, fontweight="bold", pad=10)
        style_axes(ax)

    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_file, dpi=500, bbox_inches="tight")
    plt.close()

def run_pca(X_array, sample_index):
    pca = PCA(n_components=3)
    coords = pca.fit_transform(X_array)
    explained = pca.explained_variance_ratio_ * 100

    scores = pd.DataFrame({
        "Sample": sample_index,
        "Group": ["Human" if s.startswith("H_") else "Mouse" for s in sample_index],
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "PC3": coords[:, 2]
    })

    return pca, scores, explained

# =========================================
# LOAD DATA
# =========================================
df = pd.read_csv(input_file)
df["Ortholog_gene"] = clean_gene(df["Ortholog_gene"])

human_cols = [c for c in df.columns if c.startswith("H_S")]
mouse_cols = [c for c in df.columns if c.startswith("M_S")]
sample_cols = human_cols + mouse_cols

if len(sample_cols) < 3:
    raise ValueError("Need at least 3 samples for PCA.")

X = df[sample_cols].T
X.index = sample_cols

n_genes_before = X.shape[1]

# strict NA removal for reproducible PCA
mask = ~X.isna().any(axis=0)
X = X.loc[:, mask].copy()

n_genes_after = X.shape[1]
n_genes_removed = n_genes_before - n_genes_after

if X.shape[1] < 3:
    raise ValueError("Too few genes remain after filtering.")

# =========================================
# DISTRIBUTION SHIFT PLOT
# sample medians and spread before scaling
# =========================================
dist_rows = []
for sample in sample_cols:
    species = "Human" if sample.startswith("H_") else "Mouse"
    vals = pd.to_numeric(X.loc[sample], errors="coerce").dropna()
    dist_rows.append({
        "Sample": sample,
        "Species": species,
        "Median": vals.median(),
        "Mean": vals.mean(),
        "Std": vals.std(),
        "IQR": vals.quantile(0.75) - vals.quantile(0.25)
    })

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(results_dir / "distribution_shift_summary.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

# sample medians
for species, color in [("Human", COLOR_HUMAN), ("Mouse", COLOR_MOUSE)]:
    sub = dist_df[dist_df["Species"] == species]
    axes[0].scatter(sub["Sample"], sub["Median"], s=140, color=color, edgecolors="black", linewidths=1.0)

axes[0].set_title("Per-sample median abundance before scaling", fontsize=15, fontweight="bold", pad=10)
axes[0].set_ylabel("Median abundance", fontsize=12, fontweight="bold")
axes[0].tick_params(axis="x", rotation=45, labelsize=10)
style_axes(axes[0])

# boxplot of pooled distributions
human_vals = X.loc[human_cols].values.flatten()
mouse_vals = X.loc[mouse_cols].values.flatten()
human_vals = human_vals[~np.isnan(human_vals)]
mouse_vals = mouse_vals[~np.isnan(mouse_vals)]

bp = axes[1].boxplot(
    [human_vals, mouse_vals],
    labels=["Human", "Mouse"],
    patch_artist=True,
    widths=0.55
)
bp["boxes"][0].set(facecolor=COLOR_HUMAN, alpha=0.45, edgecolor="black", linewidth=1.2)
bp["boxes"][1].set(facecolor=COLOR_MOUSE, alpha=0.45, edgecolor="black", linewidth=1.2)
for part in ["whiskers", "caps", "medians"]:
    for item in bp[part]:
        item.set(color="black", linewidth=1.2)

axes[1].set_title("Global abundance distribution before scaling", fontsize=15, fontweight="bold", pad=10)
axes[1].set_ylabel("Protein abundance", fontsize=12, fontweight="bold")
style_axes(axes[1])

plt.tight_layout()
plt.savefig(fig_dir / "distribution_shift_human_vs_mouse.png", dpi=500, bbox_inches="tight")
plt.close()

# =========================================
# PCA MODE 1 — WITHOUT SCALING
# just mean-center features
# =========================================
X_centered = X - X.mean(axis=0)
pca_raw, scores_raw, explained_raw = run_pca(X_centered.values, X.index)

scores_raw.to_csv(results_dir / "pca_scores_unscaled.csv", index=False)

loadings_raw = pd.DataFrame(
    pca_raw.components_.T,
    columns=["PC1", "PC2", "PC3"],
    index=X.columns
).reset_index().rename(columns={"index": "Gene"})
loadings_raw.to_csv(results_dir / "pca_loadings_unscaled.csv", index=False)

save_scree_plot(explained_raw, fig_dir / "PCA_scree_plot_unscaled.png", "Scree plot of PCA without scaling")
save_2d_plot(scores_raw, explained_raw, fig_dir / "PCA_2D_PC1_PC2_unscaled.png", "PCA without scaling")
save_3d_plot(scores_raw, explained_raw, fig_dir / "PCA_3D_PC1_PC2_PC3_unscaled.png", "3D PCA without scaling")
save_pairwise_plot(scores_raw, explained_raw, fig_dir / "PCA_pairwise_panels_unscaled.png", "Pairwise PCA views without scaling")

# =========================================
# PCA MODE 2 — WITH Z-SCORING
# =========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_z, scores_z, explained_z = run_pca(X_scaled, X.index)

scores_z.to_csv(results_dir / "pca_scores_zscore.csv", index=False)

loadings_z = pd.DataFrame(
    pca_z.components_.T,
    columns=["PC1", "PC2", "PC3"],
    index=X.columns
).reset_index().rename(columns={"index": "Gene"})
loadings_z.to_csv(results_dir / "pca_loadings_zscore.csv", index=False)

save_scree_plot(explained_z, fig_dir / "PCA_scree_plot_zscore.png", "Scree plot of z-score PCA")
save_2d_plot(scores_z, explained_z, fig_dir / "PCA_2D_PC1_PC2_zscore.png", "Z-score PCA")
save_3d_plot(scores_z, explained_z, fig_dir / "PCA_3D_PC1_PC2_PC3_zscore.png", "3D z-score PCA")
save_pairwise_plot(scores_z, explained_z, fig_dir / "PCA_pairwise_panels_zscore.png", "Pairwise PCA views after z-scoring")

# =========================================
# COMPARISON SUMMARY TABLE
# =========================================
summary_compare = pd.DataFrame([
    {
        "Mode": "Unscaled_centered",
        "Genes_before_filtering": n_genes_before,
        "Genes_after_filtering": n_genes_after,
        "Genes_removed_due_to_NA": n_genes_removed,
        "PC1_variance_pct": explained_raw[0],
        "PC2_variance_pct": explained_raw[1],
        "PC3_variance_pct": explained_raw[2],
        "PC1_PC2_PC3_cumulative_pct": explained_raw.sum()
    },
    {
        "Mode": "Zscore_scaled",
        "Genes_before_filtering": n_genes_before,
        "Genes_after_filtering": n_genes_after,
        "Genes_removed_due_to_NA": n_genes_removed,
        "PC1_variance_pct": explained_z[0],
        "PC2_variance_pct": explained_z[1],
        "PC3_variance_pct": explained_z[2],
        "PC1_PC2_PC3_cumulative_pct": explained_z.sum()
    }
])
summary_compare.to_csv(results_dir / "pca_mode_comparison_summary.csv", index=False)

# =========================================
# TEXT SUMMARY
# =========================================
with open(results_dir / "pca_dual_mode_summary.txt", "w") as f:
    f.write("FINAL DUAL-MODE PCA SUMMARY\n")
    f.write("=" * 45 + "\n")
    f.write(f"Input file: {input_file}\n")
    f.write(f"Human samples: {len(human_cols)}\n")
    f.write(f"Mouse samples: {len(mouse_cols)}\n")
    f.write(f"Genes before filtering: {n_genes_before}\n")
    f.write(f"Genes after filtering: {n_genes_after}\n")
    f.write(f"Genes removed due to NA: {n_genes_removed}\n\n")

    f.write("UNSCALED PCA (centered only)\n")
    f.write(f"PC1: {explained_raw[0]:.4f}%\n")
    f.write(f"PC2: {explained_raw[1]:.4f}%\n")
    f.write(f"PC3: {explained_raw[2]:.4f}%\n")
    f.write(f"Cumulative: {explained_raw.sum():.4f}%\n\n")

    f.write("Z-SCORE PCA\n")
    f.write(f"PC1: {explained_z[0]:.4f}%\n")
    f.write(f"PC2: {explained_z[1]:.4f}%\n")
    f.write(f"PC3: {explained_z[2]:.4f}%\n")
    f.write(f"Cumulative: {explained_z.sum():.4f}%\n")

# =========================================
# REPORT
# =========================================
print("\nFINAL DUAL-MODE PCA COMPLETED")
print(f"Input file: {input_file}")
print(f"Human samples: {len(human_cols)}")
print(f"Mouse samples: {len(mouse_cols)}")
print(f"Genes before filtering: {n_genes_before}")
print(f"Genes after filtering: {n_genes_after}")
print(f"Genes removed due to NA: {n_genes_removed}")

print("\nUNSCALED PCA:")
print(f"PC1: {explained_raw[0]:.4f}%")
print(f"PC2: {explained_raw[1]:.4f}%")
print(f"PC3: {explained_raw[2]:.4f}%")

print("\nZ-SCORE PCA:")
print(f"PC1: {explained_z[0]:.4f}%")
print(f"PC2: {explained_z[1]:.4f}%")
print(f"PC3: {explained_z[2]:.4f}%")

print("\nSaved figures:")
print(fig_dir / "distribution_shift_human_vs_mouse.png")
print(fig_dir / "PCA_scree_plot_unscaled.png")
print(fig_dir / "PCA_2D_PC1_PC2_unscaled.png")
print(fig_dir / "PCA_3D_PC1_PC2_PC3_unscaled.png")
print(fig_dir / "PCA_pairwise_panels_unscaled.png")
print(fig_dir / "PCA_scree_plot_zscore.png")
print(fig_dir / "PCA_2D_PC1_PC2_zscore.png")
print(fig_dir / "PCA_3D_PC1_PC2_PC3_zscore.png")
print(fig_dir / "PCA_pairwise_panels_zscore.png")

print("\nSaved tables:")
print(results_dir / "distribution_shift_summary.csv")
print(results_dir / "pca_scores_unscaled.csv")
print(results_dir / "pca_loadings_unscaled.csv")
print(results_dir / "pca_scores_zscore.csv")
print(results_dir / "pca_loadings_zscore.csv")
print(results_dir / "pca_mode_comparison_summary.csv")
print(results_dir / "pca_dual_mode_summary.txt")
