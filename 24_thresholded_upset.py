import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================
# SETTINGS
# =========================================
HUMAN_THRESHOLD = 3   # detected in >= 3 of 8 human samples
MOUSE_THRESHOLD = 2   # detected in >= 2 of 5 mouse samples

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()

human_file = base / "data_processed" / "human_ortholog.csv"
mouse_file = base / "data_processed" / "mouse_ortholog.csv"

results_dir = base / "results" / "thresholded_overlap"
results_dir.mkdir(parents=True, exist_ok=True)

fig_dir = base / "figures" / "thresholded_overlap"
fig_dir.mkdir(parents=True, exist_ok=True)

out_counts_csv = results_dir / "thresholded_overlap_counts.csv"
out_detection_table_csv = results_dir / "thresholded_detection_table.csv"
out_plot = fig_dir / "thresholded_species_overlap_upset.png"

# =========================================
# LOAD
# =========================================
human = pd.read_csv(human_file)
mouse = pd.read_csv(mouse_file)

human["Ortholog_gene"] = human["Ortholog_gene"].astype(str).str.upper().str.strip()
mouse["Ortholog_gene"] = mouse["Ortholog_gene"].astype(str).str.upper().str.strip()

human_cols = [c for c in human.columns if c.startswith("H_S")]
mouse_cols = [c for c in mouse.columns if c.startswith("M_S")]

print("Human shape:", human.shape)
print("Mouse shape:", mouse.shape)
print("Human sample columns:", human_cols)
print("Mouse sample columns:", mouse_cols)

# =========================================
# NUMERIC CONVERSION
# =========================================
human[human_cols] = human[human_cols].apply(pd.to_numeric, errors="coerce")
mouse[mouse_cols] = mouse[mouse_cols].apply(pd.to_numeric, errors="coerce")

# =========================================
# DETECTION COUNTS
# =========================================
human_det = pd.DataFrame({
    "Gene": human["Ortholog_gene"],
    "Human_non_na_count": human[human_cols].notna().sum(axis=1)
}).drop_duplicates(subset=["Gene"])

mouse_det = pd.DataFrame({
    "Gene": mouse["Ortholog_gene"],
    "Mouse_non_na_count": mouse[mouse_cols].notna().sum(axis=1)
}).drop_duplicates(subset=["Gene"])

# union of all ortholog genes seen in either matrix
all_genes = sorted(set(human_det["Gene"]).union(set(mouse_det["Gene"])))

det = pd.DataFrame({"Gene": all_genes})
det = det.merge(human_det, on="Gene", how="left")
det = det.merge(mouse_det, on="Gene", how="left")

det["Human_non_na_count"] = det["Human_non_na_count"].fillna(0).astype(int)
det["Mouse_non_na_count"] = det["Mouse_non_na_count"].fillna(0).astype(int)

# =========================================
# THRESHOLDED DETECTION CALLS
# =========================================
det["Human_detected"] = det["Human_non_na_count"] >= HUMAN_THRESHOLD
det["Mouse_detected"] = det["Mouse_non_na_count"] >= MOUSE_THRESHOLD

# classification
det["Category"] = "Neither"
det.loc[(det["Human_detected"]) & (~det["Mouse_detected"]), "Category"] = "Human_only"
det.loc[(~det["Human_detected"]) & (det["Mouse_detected"]), "Category"] = "Mouse_only"
det.loc[(det["Human_detected"]) & (det["Mouse_detected"]), "Category"] = "Shared"

# save full table
det.to_csv(out_detection_table_csv, index=False)

# =========================================
# COUNTS
# =========================================
human_total = int(det["Human_detected"].sum())
mouse_total = int(det["Mouse_detected"].sum())
shared = int((det["Category"] == "Shared").sum())
human_only = int((det["Category"] == "Human_only").sum())
mouse_only = int((det["Category"] == "Mouse_only").sum())

counts_df = pd.DataFrame({
    "Category": ["Human_total", "Mouse_total", "Shared", "Human_only", "Mouse_only"],
    "Count": [human_total, mouse_total, shared, human_only, mouse_only]
})
counts_df.to_csv(out_counts_csv, index=False)

print("\nThresholded overlap counts:")
print(counts_df)

# =========================================
# PLOT: CUSTOM TWO-SET UPSET STYLE
# =========================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 1.6
plt.rcParams["xtick.major.width"] = 1.3
plt.rcParams["ytick.major.width"] = 1.3

fig = plt.figure(figsize=(14, 9))

# layout
ax_top = plt.axes([0.34, 0.56, 0.58, 0.34])   # intersection bars
ax_mat = plt.axes([0.34, 0.18, 0.58, 0.22])   # membership matrix
ax_left = plt.axes([0.08, 0.18, 0.18, 0.22])  # set-size bars

# -----------------------------------------
# TOP INTERSECTION BARS
# order: Human_only, Shared, Mouse_only
# -----------------------------------------
bar_labels = ["Human only", "Shared", "Mouse only"]
bar_values = [human_only, shared, mouse_only]
bar_x = np.arange(len(bar_labels))

bar_colors = ["#d97706", "#4b5563", "#2e8b57"]

bars = ax_top.bar(
    bar_x,
    bar_values,
    color=bar_colors,
    edgecolor="black",
    linewidth=1.1,
    alpha=0.95
)

for x, y in zip(bar_x, bar_values):
    ax_top.text(
        x, y + max(bar_values) * 0.02,
        str(y),
        ha="center", va="bottom",
        fontsize=12, fontweight="bold"
    )

ax_top.set_ylabel("Intersection size", fontsize=14, fontweight="bold")
ax_top.set_xticks([])
ax_top.set_title(
    "Thresholded cross-species protein detection overlap",
    fontsize=18,
    fontweight="bold",
    pad=12
)

for spine in ax_top.spines.values():
    spine.set_linewidth(1.6)

# -----------------------------------------
# MEMBERSHIP MATRIX
# rows: Mouse, Human
# -----------------------------------------
ax_mat.set_xlim(-0.5, len(bar_labels) - 0.5)
ax_mat.set_ylim(-0.5, 1.5)

# background gray circles
for i in bar_x:
    for y in [0, 1]:
        ax_mat.plot(i, y, "o", color="#d1d5db", markersize=9, zorder=1)

# Human only
ax_mat.plot(0, 1, "o", color="black", markersize=10, zorder=3)

# Shared
ax_mat.plot(1, 1, "o", color="black", markersize=10, zorder=3)
ax_mat.plot(1, 0, "o", color="black", markersize=10, zorder=3)
ax_mat.plot([1, 1], [0, 1], color="black", linewidth=1.6, zorder=2)

# Mouse only
ax_mat.plot(2, 0, "o", color="black", markersize=10, zorder=3)

ax_mat.set_xticks(bar_x)
ax_mat.set_xticklabels(bar_labels, fontsize=12, fontweight="bold")
ax_mat.set_yticks([0, 1])
ax_mat.set_yticklabels(["Mouse", "Human"], fontsize=12, fontweight="bold")
ax_mat.set_xlabel("Detection overlap category", fontsize=14, fontweight="bold")

for spine in ax_mat.spines.values():
    spine.set_linewidth(1.6)

# -----------------------------------------
# LEFT SET-SIZE BARS
# -----------------------------------------
set_names = ["Mouse detected", "Human detected"]
set_sizes = [mouse_total, human_total]
set_y = [0, 1]
set_colors = ["#2e8b57", "#d97706"]

ax_left.barh(
    set_y,
    set_sizes,
    color=set_colors,
    edgecolor="black",
    linewidth=1.1,
    alpha=0.95
)

for y, val in zip(set_y, set_sizes):
    ax_left.text(
        val + max(set_sizes) * 0.02,
        y,
        str(val),
        va="center", ha="left",
        fontsize=12, fontweight="bold"
    )

ax_left.set_yticks(set_y)
ax_left.set_yticklabels(set_names, fontsize=12, fontweight="bold")
ax_left.set_xlabel("Set size", fontsize=13, fontweight="bold")

for spine in ax_left.spines.values():
    spine.set_linewidth(1.6)

# -----------------------------------------
# FOOTNOTE / INTERPRETATION
# -----------------------------------------
fig.text(
    0.34, 0.05,
    f"Detection thresholds: Human >= {HUMAN_THRESHOLD}/8 samples, Mouse >= {MOUSE_THRESHOLD}/5 samples",
    fontsize=11,
    fontweight="bold"
)

plt.savefig(out_plot, dpi=700, bbox_inches="tight")
plt.close()

print("\nSaved files:")
print(out_counts_csv)
print(out_detection_table_csv)
print(out_plot)
