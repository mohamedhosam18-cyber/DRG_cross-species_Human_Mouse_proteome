import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import pi
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()

human_file = base / "data_processed" / "human_ortholog.csv"
mouse_file = base / "data_processed" / "mouse_ortholog.csv"

out_dir = base / "figures" / "final_radar_panels_labels_outside"
out_dir.mkdir(parents=True, exist_ok=True)

results_dir = base / "results" / "final_radar_panels_labels_outside"
results_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# STYLE
# =========================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

HUMAN_COLOR = "#cc6f12"
MOUSE_COLOR = "#2f8a57"

GRID_COLOR = "#bdbdbd"
SPINE_COLOR = "black"
TITLE_COLOR = "#1a1a1a"
SUBTITLE_COLOR = "#555555"

# =========================================
# MODULES
# =========================================
modules = {
    "Nociceptor excitability": [
        "TRPV1", "SCN9A", "SCN10A", "SCN11A", "NTRK1", "TAC1", "CALCA"
    ],
    "Purinergic / trophic nociceptor": [
        "P2RX3", "RET", "OSMR", "GFRA1", "GFRA2", "GFRA3", "NTRK2", "NTRK3"
    ],
    "Myelinated axon / structural neuron": [
        "NEFH", "NEFL", "PRPH", "GAP43", "DPYSL2", "MAP1B", "STMN2", "TUBB3"
    ],
    "Glia / myelin support": [
        "SOX10", "MBP", "MPZ", "PLP1", "MAG", "GFAP", "S100B", "FABP7", "KCNJ10", "GLUL", "SLC1A3", "PMP22"
    ],
    "Mitochondrial respiration": [
        "ATP5PF", "ATP5PB", "COX5A", "COX6B1", "NDUFA4", "NDUFB8", "UQCRC1", "SDHB"
    ],
    "Synaptic vesicle cycle": [
        "STX1B", "STXBP1", "SYN1", "SNAP25", "VAMP1", "SYT1", "RAB3A", "CPLX1"
    ],
    "Trafficking / endolysosomal": [
        "LAMP2", "RAB35", "ARL8A", "RAB6B", "ARF4", "COPB1", "AP2M1", "CLTC"
    ],
    "Proteostasis / RNA handling": [
        "HSPA4L", "PCBP2", "APEX1", "POLR2A", "HSP90AB1", "RPS3", "EIF3A", "UCHL1"
    ],
    "Glutathione / redox": [
        "MPST", "GSTP1", "PRDX1", "TXN", "GCLC", "GSS", "GPX4", "SOD1"
    ],
    "RHO / cytoskeleton signaling": [
        "GNAI2", "RDX", "ACTB", "PFN2", "CFL1", "ROCK2", "RHOB", "ARPC3"
    ],
    "Matrisome / ECM context": [
        "COL1A1", "COL1A2", "COL3A1", "LAMB1", "FN1", "VCAN", "SPARC", "TNC"
    ],
}

# shorter display labels
display_labels = {
    "Nociceptor excitability": "Nociceptor\nexcitability",
    "Purinergic / trophic nociceptor": "Purinergic /\ntrophic",
    "Myelinated axon / structural neuron": "Myelinated axon /\nstructural",
    "Glia / myelin support": "Glia /\nmyelin",
    "Mitochondrial respiration": "Mitochondrial\nrespiration",
    "Synaptic vesicle cycle": "Synaptic vesicle\ncycle",
    "Trafficking / endolysosomal": "Trafficking /\nendolysosomal",
    "Proteostasis / RNA handling": "Proteostasis /\nRNA handling",
    "Glutathione / redox": "Glutathione /\nredox",
    "RHO / cytoskeleton signaling": "RHO /\ncytoskeleton",
    "Matrisome / ECM context": "Matrisome /\nECM",
}

# =========================================
# HELPERS
# =========================================
def clean_gene_series(series):
    return series.astype(str).str.upper().str.strip()

def make_numeric(df):
    return df.apply(pd.to_numeric, errors="coerce")

def median_align_by_sample(df):
    out = df.copy()
    for col in out.columns:
        med = out[col].median(skipna=True)
        out[col] = out[col] - med
    return out

def close_radar(values):
    vals = list(values)
    return vals + vals[:1]

def get_angles(n):
    angles = [i / float(n) * 2 * pi for i in range(n)]
    return angles + angles[:1]

def global_minmax_scale(two_col_df):
    vals = two_col_df.values.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return two_col_df * np.nan
    vmin = np.min(vals)
    vmax = np.max(vals)
    if vmax == vmin:
        return pd.DataFrame(
            np.full(two_col_df.shape, 0.5),
            index=two_col_df.index,
            columns=two_col_df.columns
        )
    return (two_col_df - vmin) / (vmax - vmin)

def positive_scale(series):
    vals = pd.Series(series).astype(float).fillna(0.0)
    vmax = vals.max()
    if vmax <= 0:
        return pd.Series(np.zeros(len(vals)), index=vals.index)
    return vals / vmax

def style_radar(ax):
    ax.spines["polar"].set_color(SPINE_COLOR)
    ax.spines["polar"].set_linewidth(1.5)
    ax.grid(color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.7)
    ax.set_facecolor("white")
    ax.set_rlabel_position(0)

def add_custom_labels(ax, angles, labels, r=1.085, fontsize=10):
    """
    Place category labels outside the circle manually.
    """
    for ang, lab in zip(angles[:-1], labels):
        ang_deg = np.degrees(ang)

        if 0 <= ang_deg < 180:
            ha = "left"
        else:
            ha = "right"

        # top / bottom fine tuning
        if abs(ang_deg - 90) < 8 or abs(ang_deg - 270) < 8:
            ha = "center"

        ax.text(
            ang,
            r,
            lab,
            fontsize=fontsize,
            fontweight="bold",
            ha=ha,
            va="center",
            color="black",
            clip_on=False
        )

def add_title_and_subtitle(fig, title, subtitle):
    fig.text(
        0.5, 0.965,
        title,
        ha="center", va="center",
        fontsize=20, fontweight="bold",
        color=TITLE_COLOR
    )
    fig.text(
        0.5, 0.935,
        subtitle,
        ha="center", va="center",
        fontsize=11.5, fontweight="bold",
        color=SUBTITLE_COLOR
    )

def add_clean_legend(fig, labels):
    handles = [
        Patch(facecolor=HUMAN_COLOR, edgecolor=HUMAN_COLOR, alpha=0.18, label=labels[0]),
        Patch(facecolor=MOUSE_COLOR, edgecolor=MOUSE_COLOR, alpha=0.18, label=labels[1]),
    ]

    legend_labels = [labels[0], labels[1], "", ""]
    leg = fig.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.975),
        frameon=True,
        fontsize=10.5,
        edgecolor="black",
        borderpad=0.4,
        labelspacing=0.5
    )
    leg.get_frame().set_linewidth(1.2)
    leg.get_frame().set_alpha(0.97)

# =========================================
# LOAD
# =========================================
human = pd.read_csv(human_file)
mouse = pd.read_csv(mouse_file)

human["Ortholog_gene"] = clean_gene_series(human["Ortholog_gene"])
mouse["Ortholog_gene"] = clean_gene_series(mouse["Ortholog_gene"])

human = human.drop_duplicates(subset=["Ortholog_gene"]).set_index("Ortholog_gene")
mouse = mouse.drop_duplicates(subset=["Ortholog_gene"]).set_index("Ortholog_gene")

human_cols = [c for c in human.columns if c.startswith("H_S")]
mouse_cols = [c for c in mouse.columns if c.startswith("M_S")]

human_expr = make_numeric(human[human_cols])
mouse_expr = make_numeric(mouse[mouse_cols])

human_aligned = median_align_by_sample(human_expr)
mouse_aligned = median_align_by_sample(mouse_expr)

# =========================================
# MODULE SUMMARY
# =========================================
summary_rows = []
sample_rows = []

for module_name, genes in modules.items():
    genes_upper = [g.upper().strip() for g in genes]

    human_found = [g for g in genes_upper if g in human_aligned.index]
    mouse_found = [g for g in genes_upper if g in mouse_aligned.index]

    module_gene_universe = list(dict.fromkeys(genes_upper))
    module_size = len(module_gene_universe)

    human_detected = 0
    mouse_detected = 0

    if len(human_found) > 0:
        human_detected = int(human_expr.loc[human_found].notna().any(axis=1).sum())
    if len(mouse_found) > 0:
        mouse_detected = int(mouse_expr.loc[mouse_found].notna().any(axis=1).sum())

    human_coverage = human_detected / module_size if module_size > 0 else np.nan
    mouse_coverage = mouse_detected / module_size if module_size > 0 else np.nan

    human_score_per_sample = pd.Series(np.nan, index=human_cols, dtype=float)
    mouse_score_per_sample = pd.Series(np.nan, index=mouse_cols, dtype=float)

    if len(human_found) > 0:
        human_score_per_sample = human_aligned.loc[human_found].mean(axis=0, skipna=True)
    if len(mouse_found) > 0:
        mouse_score_per_sample = mouse_aligned.loc[mouse_found].mean(axis=0, skipna=True)

    human_mean = human_score_per_sample.mean(skipna=True)
    mouse_mean = mouse_score_per_sample.mean(skipna=True)

    human_sd = human_score_per_sample.std(skipna=True)
    mouse_sd = mouse_score_per_sample.std(skipna=True)

    human_minus_mouse = human_mean - mouse_mean

    summary_rows.append({
        "Module": module_name,
        "Module_size": module_size,
        "Human_genes_found": len(human_found),
        "Mouse_genes_found": len(mouse_found),
        "Human_detected_genes": human_detected,
        "Mouse_detected_genes": mouse_detected,
        "Human_coverage": human_coverage,
        "Mouse_coverage": mouse_coverage,
        "Human_mean_aligned": human_mean,
        "Mouse_mean_aligned": mouse_mean,
        "Human_sd_aligned": human_sd,
        "Mouse_sd_aligned": mouse_sd,
        "Human_minus_Mouse": human_minus_mouse
    })

    for s in human_cols:
        sample_rows.append({
            "Module": module_name,
            "Species": "Human",
            "Sample": s,
            "Score_aligned": human_score_per_sample.get(s, np.nan)
        })

    for s in mouse_cols:
        sample_rows.append({
            "Module": module_name,
            "Species": "Mouse",
            "Sample": s,
            "Score_aligned": mouse_score_per_sample.get(s, np.nan)
        })

summary_df = pd.DataFrame(summary_rows)
sample_df = pd.DataFrame(sample_rows)

summary_df = summary_df[
    (summary_df["Human_genes_found"] >= 2) | (summary_df["Mouse_genes_found"] >= 2)
].copy()

summary_df["AbsDiff"] = summary_df["Human_minus_Mouse"].abs()
summary_df["MeanCoverage"] = summary_df[["Human_coverage", "Mouse_coverage"]].mean(axis=1)
summary_df = summary_df.sort_values(["AbsDiff", "MeanCoverage"], ascending=[False, False]).reset_index(drop=True)

module_order = summary_df["Module"].tolist()

summary_df.to_csv(results_dir / "radar_module_summary.csv", index=False)
sample_df.to_csv(results_dir / "radar_module_sample_scores.csv", index=False)

categories = module_order
display_categories = [display_labels.get(x, x) for x in categories]
N = len(categories)
angles = get_angles(N)

# =========================================
# RADAR 1 — ACTIVITY
# =========================================
activity_df = summary_df.set_index("Module")[["Human_mean_aligned", "Mouse_mean_aligned"]].loc[categories]
activity_scaled = global_minmax_scale(activity_df)

all_means = activity_df.values.flatten()
all_means = all_means[~np.isnan(all_means)]
vmin = np.min(all_means) if len(all_means) else 0
vmax = np.max(all_means) if len(all_means) else 1
den = (vmax - vmin) if vmax != vmin else 1

human_lower = ((summary_df["Human_mean_aligned"] - summary_df["Human_sd_aligned"]) - vmin) / den
human_upper = ((summary_df["Human_mean_aligned"] + summary_df["Human_sd_aligned"]) - vmin) / den
mouse_lower = ((summary_df["Mouse_mean_aligned"] - summary_df["Mouse_sd_aligned"]) - vmin) / den
mouse_upper = ((summary_df["Mouse_mean_aligned"] + summary_df["Mouse_sd_aligned"]) - vmin) / den

human_lower = np.clip(human_lower, 0, 1)
human_upper = np.clip(human_upper, 0, 1)
mouse_lower = np.clip(mouse_lower, 0, 1)
mouse_upper = np.clip(mouse_upper, 0, 1)

human_vals = close_radar(activity_scaled["Human_mean_aligned"].loc[categories].fillna(0).tolist())
mouse_vals = close_radar(activity_scaled["Mouse_mean_aligned"].loc[categories].fillna(0).tolist())

human_lower_vals = close_radar(human_lower.tolist())
human_upper_vals = close_radar(human_upper.tolist())
mouse_lower_vals = close_radar(mouse_lower.tolist())
mouse_upper_vals = close_radar(mouse_upper.tolist())

fig = plt.figure(figsize=(11.2, 10.4))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
style_radar(ax)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([])  # remove default labels entirely
add_custom_labels(ax, angles, display_categories, r=1.085, fontsize=10)

plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], fontsize=9)
plt.ylim(0, 1)

ax.fill_between(angles, human_lower_vals, human_upper_vals, color=HUMAN_COLOR, alpha=0.08)
ax.fill_between(angles, mouse_lower_vals, mouse_upper_vals, color=MOUSE_COLOR, alpha=0.08)

ax.plot(angles, human_vals, linewidth=2.8, color=HUMAN_COLOR)
ax.fill(angles, human_vals, alpha=0.12, color=HUMAN_COLOR)

ax.plot(angles, mouse_vals, linewidth=2.8, color=MOUSE_COLOR)
ax.fill(angles, mouse_vals, alpha=0.12, color=MOUSE_COLOR)

add_title_and_subtitle(
    fig,
    "Radar 1 | Module activity",
    "Median-aligned mean abundance across DRG biological programs"
)
add_clean_legend(fig, ["Human", "Mouse"])

plt.subplots_adjust(left=0.08, right=0.80, top=0.86, bottom=0.07)
plt.savefig(out_dir / "radar_1_module_activity.png", dpi=700, bbox_inches="tight")
plt.close()

# =========================================
# RADAR 2 — COVERAGE
# =========================================
coverage_df = summary_df.set_index("Module")[["Human_coverage", "Mouse_coverage"]].loc[categories]

human_cov = close_radar(coverage_df["Human_coverage"].fillna(0).tolist())
mouse_cov = close_radar(coverage_df["Mouse_coverage"].fillna(0).tolist())

fig = plt.figure(figsize=(11.2, 10.4))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
style_radar(ax)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([])
add_custom_labels(ax, angles, display_categories, r=1.085, fontsize=10)

plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], fontsize=9)
plt.ylim(0, 1)

ax.plot(angles, human_cov, linewidth=2.8, color=HUMAN_COLOR)
ax.fill(angles, human_cov, alpha=0.12, color=HUMAN_COLOR)

ax.plot(angles, mouse_cov, linewidth=2.8, color=MOUSE_COLOR)
ax.fill(angles, mouse_cov, alpha=0.12, color=MOUSE_COLOR)

add_title_and_subtitle(
    fig,
    "Radar 2 | Detection coverage",
    "Fraction of module genes detected in each species"
)
add_clean_legend(fig, ["Human coverage", "Mouse coverage"])

plt.subplots_adjust(left=0.08, right=0.80, top=0.86, bottom=0.07)
plt.savefig(out_dir / "radar_2_detection_coverage.png", dpi=700, bbox_inches="tight")
plt.close()

# =========================================
# RADAR 3 — DIVERGENCE
# =========================================
effect = summary_df.set_index("Module")["Human_minus_Mouse"].loc[categories]
human_adv = positive_scale(effect.clip(lower=0))
mouse_adv = positive_scale((-effect).clip(lower=0))

human_adv_vals = close_radar(human_adv.tolist())
mouse_adv_vals = close_radar(mouse_adv.tolist())

fig = plt.figure(figsize=(11.2, 10.4))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
style_radar(ax)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([])
add_custom_labels(ax, angles, display_categories, r=1.085, fontsize=10)

plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], fontsize=9)
plt.ylim(0, 1)

ax.plot(angles, human_adv_vals, linewidth=2.8, color=HUMAN_COLOR)
ax.fill(angles, human_adv_vals, alpha=0.12, color=HUMAN_COLOR)

ax.plot(angles, mouse_adv_vals, linewidth=2.8, color=MOUSE_COLOR)
ax.fill(angles, mouse_adv_vals, alpha=0.12, color=MOUSE_COLOR)

add_title_and_subtitle(
    fig,
    "Radar 3 | Species divergence",
    "Relative human-biased and mouse-biased module signals"
)
add_clean_legend(fig, ["Human-biased", "Mouse-biased"])

plt.subplots_adjust(left=0.08, right=0.80, top=0.86, bottom=0.07)
plt.savefig(out_dir / "radar_3_species_divergence.png", dpi=700, bbox_inches="tight")
plt.close()

# =========================================
# PRETTY TABLE
# =========================================
pretty_df = summary_df.copy()
pretty_df["Human_coverage_pct"] = (100 * pretty_df["Human_coverage"]).round(1)
pretty_df["Mouse_coverage_pct"] = (100 * pretty_df["Mouse_coverage"]).round(1)
pretty_df["Human_mean_aligned"] = pretty_df["Human_mean_aligned"].round(3)
pretty_df["Mouse_mean_aligned"] = pretty_df["Mouse_mean_aligned"].round(3)
pretty_df["Human_minus_Mouse"] = pretty_df["Human_minus_Mouse"].round(3)
pretty_df.to_csv(results_dir / "radar_module_summary_pretty.csv", index=False)

# =========================================
# REPORT
# =========================================
print("\nDONE")
print("\nModules retained:")
print(module_order)

print("\nSaved figures:")
print(out_dir / "radar_1_module_activity.png")
print(out_dir / "radar_2_detection_coverage.png")
print(out_dir / "radar_3_species_divergence.png")

print("\nSaved tables:")
print(results_dir / "radar_module_summary.csv")
print(results_dir / "radar_module_sample_scores.csv")
print(results_dir / "radar_module_summary_pretty.csv")
