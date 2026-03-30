import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adjustText import adjust_text
from scipy.stats import gaussian_kde

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
input_file = base / "results/cross_species_results.csv"

fig_dir = base / "figures" / "volcano"
fig_dir.mkdir(parents=True, exist_ok=True)

lab_dir = base / "results" / "volcano_labels"
lab_dir.mkdir(parents=True, exist_ok=True)

out_mouse = fig_dir / "volcano_mouse_up_down.png"
out_human = fig_dir / "volcano_human_up_down.png"
out_combined = fig_dir / "combined_species_bias_density.png"

out_mouse_labels = lab_dir / "volcano_mouse_labeled_genes.csv"
out_human_labels = lab_dir / "volcano_human_labeled_genes.csv"
out_combined_labels = lab_dir / "combined_density_labeled_genes.csv"

# =========================================
# PARAMETERS
# =========================================
effect_threshold = 1.0
fdr_threshold = 0.05
n_labels_each_side = 18

# =========================================
# LOAD
# =========================================
df = pd.read_csv(input_file)

required_cols = ["Gene", "MeanDiff_norm", "FDR", "status"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Gene", "MeanDiff_norm", "FDR", "status"]).copy()
df = df[df["FDR"] > 0].copy()

df["Gene"] = df["Gene"].astype(str).str.upper().str.strip()
df["minus_log10_FDR"] = -np.log10(df["FDR"])

# =========================================
# GROUPS
# MeanDiff_norm > 0 means Human higher
# MeanDiff_norm < 0 means Mouse higher
# =========================================
df["Group"] = "Not significant"
df.loc[
    (df["MeanDiff_norm"] >= effect_threshold) & (df["FDR"] < fdr_threshold),
    "Group"
] = "Higher in Human"

df.loc[
    (df["MeanDiff_norm"] <= -effect_threshold) & (df["FDR"] < fdr_threshold),
    "Group"
] = "Higher in Mouse"

# =========================================
# LABEL SELECTION
# =========================================
top_human = (
    df[df["Group"] == "Higher in Human"]
    .sort_values(["FDR", "MeanDiff_norm"], ascending=[True, False])
    .head(n_labels_each_side)
    .copy()
)

top_mouse = (
    df[df["Group"] == "Higher in Mouse"]
    .sort_values(["FDR", "MeanDiff_norm"], ascending=[True, True])
    .head(n_labels_each_side)
    .copy()
)

combined_labeled = pd.concat([top_human, top_mouse], axis=0)

top_human.to_csv(out_human_labels, index=False)
top_mouse.to_csv(out_mouse_labels, index=False)
combined_labeled.to_csv(out_combined_labels, index=False)

# =========================================
# STYLE
# =========================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 1.6
plt.rcParams["xtick.major.width"] = 1.3
plt.rcParams["ytick.major.width"] = 1.3

color_ns = "gray"
color_human = "darkorange"
color_mouse = "forestgreen"

def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

def add_labels(ax, label_df):
    texts = []
    for _, row in label_df.iterrows():
        texts.append(
            ax.text(
                row["MeanDiff_norm"],
                row["minus_log10_FDR"],
                row["Gene"],
                fontsize=10,
                weight="bold",
                color="black",
                zorder=6
            )
        )

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8, alpha=0.8),
        expand_text=(1.2, 1.4),
        expand_points=(1.2, 1.4),
        force_text=(0.8, 1.0),
        force_points=(0.3, 0.5)
    )

# =========================================
# 1) MOUSE VOLCANO
# right = up in mouse
# left = down in mouse
# To make this visually intuitive for mouse, we flip x
# x_mouse = -MeanDiff_norm
# =========================================
mouse_df = df.copy()
mouse_df["Effect_mouse"] = -mouse_df["MeanDiff_norm"]

mouse_labels = top_mouse.copy()
mouse_labels["Effect_mouse"] = -mouse_labels["MeanDiff_norm"]

human_down_in_mouse_labels = top_human.copy()
human_down_in_mouse_labels["Effect_mouse"] = -human_down_in_mouse_labels["MeanDiff_norm"]

mouse_labeled_all = pd.concat([mouse_labels, human_down_in_mouse_labels], axis=0)

fig, ax = plt.subplots(figsize=(12, 9))

# not significant
sub = mouse_df[mouse_df["Group"] == "Not significant"]
ax.scatter(
    sub["Effect_mouse"], sub["minus_log10_FDR"],
    c=color_ns, s=28, alpha=0.60, edgecolors="none", zorder=2
)

# up in mouse
sub = mouse_df[mouse_df["Group"] == "Higher in Mouse"]
ax.scatter(
    sub["Effect_mouse"], sub["minus_log10_FDR"],
    c=color_mouse, s=30, alpha=0.82, edgecolors="none", zorder=3
)

# down in mouse (= human higher)
sub = mouse_df[mouse_df["Group"] == "Higher in Human"]
ax.scatter(
    sub["Effect_mouse"], sub["minus_log10_FDR"],
    c=color_human, s=30, alpha=0.82, edgecolors="none", zorder=3
)

ax.axhline(-np.log10(fdr_threshold), linestyle="--", linewidth=1.5, color="black", alpha=0.7)
ax.axvline(effect_threshold, linestyle="--", linewidth=1.5, color="black", alpha=0.7)
ax.axvline(-effect_threshold, linestyle="--", linewidth=1.5, color="black", alpha=0.7)

# labels
texts = []
for _, row in mouse_labeled_all.iterrows():
    texts.append(
        ax.text(
            row["Effect_mouse"],
            row["minus_log10_FDR"],
            row["Gene"],
            fontsize=10,
            weight="bold",
            color="black",
            zorder=6
        )
    )

adjust_text(
    texts,
    ax=ax,
    arrowprops=dict(arrowstyle="-", color="black", lw=0.8, alpha=0.8),
    expand_text=(1.2, 1.4),
    expand_points=(1.2, 1.4),
    force_text=(0.8, 1.0),
    force_points=(0.3, 0.5)
)

ax.set_xlabel("Normalized abundance difference relative to Mouse", fontsize=15, weight="bold")
ax.set_ylabel("-log10(FDR)", fontsize=15, weight="bold")
ax.set_title("Mouse volcano: upregulated and downregulated proteins", fontsize=18, weight="bold", pad=14)

# small side annotations instead of legend
ymax = mouse_df["minus_log10_FDR"].max()
xmin, xmax = ax.get_xlim()
ax.text(xmin + 0.02*(xmax-xmin), ymax*0.98, "Down in Mouse", fontsize=14, weight="bold",
        color=color_human, ha="left", va="top")
ax.text(xmax - 0.02*(xmax-xmin), ymax*0.98, "Up in Mouse", fontsize=14, weight="bold",
        color=color_mouse, ha="right", va="top")

style_axes(ax)
plt.tight_layout()
plt.savefig(out_mouse, dpi=500, bbox_inches="tight")
plt.close()

# =========================================
# 2) HUMAN VOLCANO
# right = up in human
# left = down in human
# x_human = MeanDiff_norm
# =========================================
human_df = df.copy()

human_labels = top_human.copy()
mouse_down_in_human_labels = top_mouse.copy()

human_labeled_all = pd.concat([human_labels, mouse_down_in_human_labels], axis=0)

fig, ax = plt.subplots(figsize=(12, 9))

sub = human_df[human_df["Group"] == "Not significant"]
ax.scatter(
    sub["MeanDiff_norm"], sub["minus_log10_FDR"],
    c=color_ns, s=28, alpha=0.60, edgecolors="none", zorder=2
)

sub = human_df[human_df["Group"] == "Higher in Human"]
ax.scatter(
    sub["MeanDiff_norm"], sub["minus_log10_FDR"],
    c=color_human, s=30, alpha=0.82, edgecolors="none", zorder=3
)

sub = human_df[human_df["Group"] == "Higher in Mouse"]
ax.scatter(
    sub["MeanDiff_norm"], sub["minus_log10_FDR"],
    c=color_mouse, s=30, alpha=0.82, edgecolors="none", zorder=3
)

ax.axhline(-np.log10(fdr_threshold), linestyle="--", linewidth=1.5, color="black", alpha=0.7)
ax.axvline(effect_threshold, linestyle="--", linewidth=1.5, color="black", alpha=0.7)
ax.axvline(-effect_threshold, linestyle="--", linewidth=1.5, color="black", alpha=0.7)

texts = []
for _, row in human_labeled_all.iterrows():
    texts.append(
        ax.text(
            row["MeanDiff_norm"],
            row["minus_log10_FDR"],
            row["Gene"],
            fontsize=10,
            weight="bold",
            color="black",
            zorder=6
        )
    )

adjust_text(
    texts,
    ax=ax,
    arrowprops=dict(arrowstyle="-", color="black", lw=0.8, alpha=0.8),
    expand_text=(1.2, 1.4),
    expand_points=(1.2, 1.4),
    force_text=(0.8, 1.0),
    force_points=(0.3, 0.5)
)

ax.set_xlabel("Normalized abundance difference relative to Human", fontsize=15, weight="bold")
ax.set_ylabel("-log10(FDR)", fontsize=15, weight="bold")
ax.set_title("Human volcano: upregulated and downregulated proteins", fontsize=18, weight="bold", pad=14)

ymax = human_df["minus_log10_FDR"].max()
xmin, xmax = ax.get_xlim()
ax.text(xmin + 0.02*(xmax-xmin), ymax*0.98, "Down in Human", fontsize=14, weight="bold",
        color=color_mouse, ha="left", va="top")
ax.text(xmax - 0.02*(xmax-xmin), ymax*0.98, "Up in Human", fontsize=14, weight="bold",
        color=color_human, ha="right", va="top")

style_axes(ax)
plt.tight_layout()
plt.savefig(out_human, dpi=500, bbox_inches="tight")
plt.close()

# =========================================
# 3) COMBINED WIDE DENSITY PLOT
# scatter + density contours
# =========================================
comb_df = df.copy()

x = comb_df["MeanDiff_norm"].values
y = comb_df["minus_log10_FDR"].values

# KDE only if enough variation
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

order = z.argsort()
x_sorted, y_sorted, z_sorted = x[order], y[order], z[order]

fig, ax = plt.subplots(figsize=(15, 9))

# base points ordered by density
ax.scatter(
    x_sorted, y_sorted,
    c=z_sorted, s=20, cmap="inferno",
    alpha=0.75, edgecolors="none", zorder=2
)

# overlay significant points by species direction
sub = comb_df[comb_df["Group"] == "Higher in Human"]
ax.scatter(
    sub["MeanDiff_norm"], sub["minus_log10_FDR"],
    c=color_human, s=28, alpha=0.85, edgecolors="none", zorder=3
)

sub = comb_df[comb_df["Group"] == "Higher in Mouse"]
ax.scatter(
    sub["MeanDiff_norm"], sub["minus_log10_FDR"],
    c=color_mouse, s=28, alpha=0.85, edgecolors="none", zorder=3
)

# contour grid
xmin, xmax = np.percentile(x, [0.5, 99.5])
ymin, ymax = np.percentile(y, [0.5, 99.5])

xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([xx.ravel(), yy.ravel()])
f = np.reshape(gaussian_kde(xy)(positions).T, xx.shape)

ax.contour(
    xx, yy, f,
    levels=8,
    colors="black",
    linewidths=0.7,
    alpha=0.35,
    zorder=1
)

ax.axhline(-np.log10(fdr_threshold), linestyle="--", linewidth=1.5, color="black", alpha=0.7)
ax.axvline(effect_threshold, linestyle="--", linewidth=1.5, color="black", alpha=0.7)
ax.axvline(-effect_threshold, linestyle="--", linewidth=1.5, color="black", alpha=0.7)

add_labels(ax, combined_labeled)

ax.set_xlabel("Normalized abundance difference (Human vs Mouse)", fontsize=15, weight="bold")
ax.set_ylabel("-log10(FDR)", fontsize=15, weight="bold")
ax.set_title("Combined cross-species differential abundance plot", fontsize=18, weight="bold", pad=14)

# side annotations
ymax2 = comb_df["minus_log10_FDR"].max()
xmin2, xmax2 = ax.get_xlim()
ax.text(xmin2 + 0.02*(xmax2-xmin2), ymax2*0.98, "Higher in Mouse", fontsize=14, weight="bold",
        color=color_mouse, ha="left", va="top")
ax.text(xmax2 - 0.02*(xmax2-xmin2), ymax2*0.98, "Higher in Human", fontsize=14, weight="bold",
        color=color_human, ha="right", va="top")

style_axes(ax)
plt.tight_layout()
plt.savefig(out_combined, dpi=500, bbox_inches="tight")
plt.close()

print("Saved figures:")
print(out_mouse)
print(out_human)
print(out_combined)

print("\nSaved label tables:")
print(out_mouse_labels)
print(out_human_labels)
print(out_combined_labels)

print("\nTop labeled Human genes:")
print(top_human[["Gene", "MeanDiff_norm", "FDR"]].head(16))

print("\nTop labeled Mouse genes:")
print(top_mouse[["Gene", "MeanDiff_norm", "FDR"]].head(16))
