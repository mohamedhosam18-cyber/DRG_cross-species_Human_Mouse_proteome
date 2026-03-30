import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import fisher_exact

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()

proteomics_file = base / "results" / "cross_species_results.csv"

tx_dir = base / "transcriptomics"
tx_results_dir = tx_dir / "results"
tx_final_dir = tx_dir / "results_final_panels"
tx_final_dir.mkdir(parents=True, exist_ok=True)

human_tx_file = tx_results_dir / "human_drg_transcriptome.csv"
mouse_expr_file = tx_results_dir / "mouse_atlas_expression.csv"
mouse_meta_file = tx_results_dir / "mouse_atlas_metadata.csv"  # optional

# outputs
summary_csv = tx_final_dir / "validation_summary.csv"
top30_human_csv = tx_final_dir / "top30_human_validated_genes.csv"
top30_mouse_csv = tx_final_dir / "top30_mouse_validated_genes.csv"
panel_stats_csv = tx_final_dir / "panel_enrichment_statistics.csv"
panel_presence_csv = tx_final_dir / "panel_gene_presence_matrix.csv"

overlap_pct_png = tx_final_dir / "overlap_percent_proteomics.png"
top30_human_png = tx_final_dir / "top30_human_validated_genes.png"
top30_mouse_png = tx_final_dir / "top30_mouse_validated_genes.png"
panel_fdr_heatmap_png = tx_final_dir / "panel_enrichment_fdr_heatmap.png"
panel_support_heatmap_png = tx_final_dir / "panel_support_fraction_heatmap.png"
panel_barplot_png = tx_final_dir / "panel_enrichment_barplot.png"
panel_presence_heatmap_png = tx_final_dir / "panel_gene_presence_heatmap.png"

report_txt = tx_final_dir / "validation_report.txt"

# =========================================
# STYLE
# =========================================
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

COLOR_HUMAN = "#d97706"
COLOR_MOUSE = "#2e8b57"

# =========================================
# CURATED DRG / SENSORY PANELS
# You can edit these later if needed
# =========================================
marker_panels = {
    "Pan-neuronal / axonal": [
        "RBFOX3", "TUBB3", "PRPH", "NEFL", "NEFM", "NEFH",
        "GAP43", "STMN2", "MAP1B", "DPYSL2", "SNAP25", "STX1A", "VAMP1", "NCAM1"
    ],
    "Peptidergic nociceptor": [
        "CALCA", "TAC1", "TRPV1", "NTRK1", "SCN9A", "SCN10A", "SCN11A", "ADCYAP1"
    ],
    "Non-peptidergic / purinergic": [
        "P2RX3", "RET", "GFRA1", "GFRA2", "GFRA3", "OSMR", "MRGPRD", "SLC17A8"
    ],
    "Mechanoreceptor / proprioceptor": [
        "PIEZO2", "NTRK2", "NTRK3", "PVALB", "RUNX3", "ASIC3", "HCN1", "KCNQ2", "KCNQ3"
    ],
    "Glia / myelin": [
        "SOX10", "MBP", "MPZ", "PLP1", "PMP22", "MAG", "GFAP", "S100B", "FABP7", "KCNJ10", "GLUL", "SLC1A3"
    ],
    "NF200 / myelinated neuron": [
        "NEFH", "NEFL", "NEFM", "PRPH", "NTRK2", "NTRK3", "RUNX3", "PVALB"
    ],
    "CGRP / peptidergic pain": [
        "CALCA", "TAC1", "TRPV1", "NTRK1", "SCN9A", "SCN10A"
    ],
    "P2X3 / non-peptidergic pain": [
        "P2RX3", "RET", "GFRA2", "OSMR", "MRGPRD", "SCN11A"
    ]
}

# =========================================
# HELPERS
# =========================================
def clean_gene_series(series):
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
    )

def bh_fdr(pvals):
    """
    Benjamini-Hochberg correction
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        val = min(val, prev, 1.0)
        adjusted[i] = val
        prev = val

    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out

def load_human_transcriptome():
    if not human_tx_file.exists():
        raise FileNotFoundError(f"Missing processed human transcriptome file: {human_tx_file}")

    df = pd.read_csv(human_tx_file)

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Gene"})
    elif "Gene" not in df.columns:
        raise ValueError("Human transcriptome file must contain 'Gene' or 'Unnamed: 0'.")

    df["Gene"] = clean_gene_series(df["Gene"])
    return df

def load_mouse_expression():
    if not mouse_expr_file.exists():
        raise FileNotFoundError(f"Missing processed mouse expression file: {mouse_expr_file}")

    df = pd.read_csv(mouse_expr_file)

    if "Gene" not in df.columns:
        raise ValueError("Mouse expression file must contain 'Gene' column.")

    df["Gene"] = clean_gene_series(df["Gene"])
    return df

def get_mouse_valid_cell_cols(mouse_expr):
    cols = [c for c in mouse_expr.columns if c != "Gene"]

    if mouse_meta_file.exists():
        try:
            meta = pd.read_csv(mouse_meta_file)
            if "Field" in meta.columns and (meta["Field"] == "Content").any():
                row = meta[meta["Field"] == "Content"].iloc[0]
                valid_cols = [
                    c for c in row.index
                    if c != "Field"
                    and c in mouse_expr.columns
                    and str(row[c]).strip().lower() == "cell"
                ]
                if len(valid_cols) > 0:
                    return valid_cols
        except Exception as e:
            print("Warning: could not refine mouse cell columns from metadata:", e)

    return cols

def top_validated_expression(df, gene_col, expr_cols, overlap_genes, top_n=30):
    sub = df[df[gene_col].isin(overlap_genes)].copy()
    sub[expr_cols] = sub[expr_cols].apply(pd.to_numeric, errors="coerce")
    sub["Mean_expression"] = sub[expr_cols].mean(axis=1, skipna=True)
    sub = sub[[gene_col, "Mean_expression"]].dropna()
    sub = sub.sort_values("Mean_expression", ascending=False).head(top_n).copy()
    return sub

def fisher_panel_enrichment(universe_genes, proteomics_genes, panel_genes):
    """
    One-sided Fisher's exact test:
    Are proteomics genes enriched in this panel relative to universe?
    """
    universe_genes = set(universe_genes)
    proteomics_genes = set(proteomics_genes) & universe_genes
    panel_genes = set(panel_genes) & universe_genes

    a = len(proteomics_genes & panel_genes)               # in proteomics and in panel
    b = len(proteomics_genes - panel_genes)               # in proteomics not in panel
    c = len(panel_genes - proteomics_genes)               # in panel not in proteomics
    d = len(universe_genes - proteomics_genes - panel_genes)  # neither

    table = [[a, b], [c, d]]

    # if no panel genes are present in universe, return neutral
    if len(panel_genes) == 0:
        return {
            "a_overlap": 0,
            "b_prot_not_panel": b,
            "c_panel_not_prot": 0,
            "d_neither": d,
            "odds_ratio": np.nan,
            "p_value": np.nan,
            "panel_genes_in_universe": 0,
            "proteomics_genes_in_universe": len(proteomics_genes)
        }

    odds_ratio, p_value = fisher_exact(table, alternative="greater")

    return {
        "a_overlap": a,
        "b_prot_not_panel": b,
        "c_panel_not_prot": c,
        "d_neither": d,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "panel_genes_in_universe": len(panel_genes),
        "proteomics_genes_in_universe": len(proteomics_genes)
    }

def save_horizontal_barplot(df, x, y, outfile, title, xlabel, color):
    plt.figure(figsize=(9, 7.5))
    ax = sns.barplot(data=df, x=x, y=y, color=color, edgecolor="black")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Gene", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================
# LOAD DATA
# =========================================
if not proteomics_file.exists():
    raise FileNotFoundError(f"Missing proteomics file: {proteomics_file}")

prot = pd.read_csv(proteomics_file)
if "Gene" not in prot.columns:
    raise ValueError("Proteomics file must contain a 'Gene' column.")

prot["Gene"] = clean_gene_series(prot["Gene"])
proteomics_genes = set(prot["Gene"].dropna().unique())

human_tx = load_human_transcriptome()
mouse_expr = load_mouse_expression()

human_expr_cols = [c for c in human_tx.columns if c != "Gene"]
mouse_expr_cols = get_mouse_valid_cell_cols(mouse_expr)

human_genes = set(human_tx["Gene"].dropna().unique())
mouse_genes = set(mouse_expr["Gene"].dropna().unique())

# =========================================
# OVERLAP SUMMARY
# =========================================
human_overlap = proteomics_genes & human_genes
mouse_overlap = proteomics_genes & mouse_genes

summary = pd.DataFrame([
    {
        "Species": "Human",
        "Proteomics_genes": len(proteomics_genes),
        "Transcriptomics_genes": len(human_genes),
        "Overlap_genes": len(human_overlap),
        "Proteomics_only_genes": len(proteomics_genes - human_genes),
        "Overlap_percent_of_proteomics": len(human_overlap) / len(proteomics_genes) * 100
    },
    {
        "Species": "Mouse",
        "Proteomics_genes": len(proteomics_genes),
        "Transcriptomics_genes": len(mouse_genes),
        "Overlap_genes": len(mouse_overlap),
        "Proteomics_only_genes": len(proteomics_genes - mouse_genes),
        "Overlap_percent_of_proteomics": len(mouse_overlap) / len(proteomics_genes) * 100
    }
])
summary.to_csv(summary_csv, index=False)

# =========================================
# TOP VALIDATED GENES
# =========================================
top30_human = top_validated_expression(
    human_tx, "Gene", human_expr_cols, human_overlap, top_n=30
)
top30_mouse = top_validated_expression(
    mouse_expr, "Gene", mouse_expr_cols, mouse_overlap, top_n=30
)

top30_human.to_csv(top30_human_csv, index=False)
top30_mouse.to_csv(top30_mouse_csv, index=False)

save_horizontal_barplot(
    top30_human,
    x="Mean_expression",
    y="Gene",
    outfile=top30_human_png,
    title="Top proteomics-supported genes in human DRG transcriptome",
    xlabel="Mean transcriptomic expression",
    color=COLOR_HUMAN
)

save_horizontal_barplot(
    top30_mouse,
    x="Mean_expression",
    y="Gene",
    outfile=top30_mouse_png,
    title="Top proteomics-supported genes in mouse DRG atlas",
    xlabel="Mean atlas expression",
    color=COLOR_MOUSE
)

# =========================================
# PANEL-LEVEL VALIDATION + STATISTICS
# =========================================
panel_rows = []
presence_rows = []

for panel_name, genes in marker_panels.items():
    panel_clean = sorted(set([g.upper().strip() for g in genes]))

    # human panel enrichment
    h_stats = fisher_panel_enrichment(human_genes, proteomics_genes, panel_clean)
    h_panel_in_tx = set(panel_clean) & human_genes
    h_panel_in_prot = set(panel_clean) & proteomics_genes
    h_panel_overlap = set(panel_clean) & human_genes & proteomics_genes

    panel_rows.append({
        "Species": "Human",
        "Panel": panel_name,
        "Panel_size_defined": len(panel_clean),
        "Panel_genes_in_transcriptome": len(h_panel_in_tx),
        "Panel_genes_in_proteomics": len(h_panel_in_prot),
        "Panel_genes_supported_by_both": len(h_panel_overlap),
        "Panel_support_fraction_of_transcriptome_panel": (
            len(h_panel_overlap) / len(h_panel_in_tx) if len(h_panel_in_tx) > 0 else np.nan
        ),
        "Odds_ratio": h_stats["odds_ratio"],
        "P_value": h_stats["p_value"]
    })

    # mouse panel enrichment
    m_stats = fisher_panel_enrichment(mouse_genes, proteomics_genes, panel_clean)
    m_panel_in_tx = set(panel_clean) & mouse_genes
    m_panel_in_prot = set(panel_clean) & proteomics_genes
    m_panel_overlap = set(panel_clean) & mouse_genes & proteomics_genes

    panel_rows.append({
        "Species": "Mouse",
        "Panel": panel_name,
        "Panel_size_defined": len(panel_clean),
        "Panel_genes_in_transcriptome": len(m_panel_in_tx),
        "Panel_genes_in_proteomics": len(m_panel_in_prot),
        "Panel_genes_supported_by_both": len(m_panel_overlap),
        "Panel_support_fraction_of_transcriptome_panel": (
            len(m_panel_overlap) / len(m_panel_in_tx) if len(m_panel_in_tx) > 0 else np.nan
        ),
        "Odds_ratio": m_stats["odds_ratio"],
        "P_value": m_stats["p_value"]
    })

    # gene-level presence matrix
    for gene in panel_clean:
        presence_rows.append({
            "Panel": panel_name,
            "Gene": gene,
            "Human_transcriptome": int(gene in human_genes),
            "Human_proteomics": int(gene in proteomics_genes),
            "Human_supported_by_both": int(gene in human_genes and gene in proteomics_genes),
            "Mouse_transcriptome": int(gene in mouse_genes),
            "Mouse_proteomics": int(gene in proteomics_genes),
            "Mouse_supported_by_both": int(gene in mouse_genes and gene in proteomics_genes)
        })

panel_stats = pd.DataFrame(panel_rows)
presence_df = pd.DataFrame(presence_rows)

# FDR correction separately per species
panel_stats["FDR"] = np.nan
for species in panel_stats["Species"].unique():
    mask = panel_stats["Species"] == species
    pvals = panel_stats.loc[mask, "P_value"].astype(float).values
    panel_stats.loc[mask, "FDR"] = bh_fdr(pvals)

panel_stats["minus_log10_FDR"] = -np.log10(panel_stats["FDR"].replace(0, 1e-300))
panel_stats.to_csv(panel_stats_csv, index=False)
presence_df.to_csv(panel_presence_csv, index=False)

# =========================================
# PLOTS
# =========================================
# overlap percentage
plt.figure(figsize=(6.5, 5.2))
ax = sns.barplot(
    data=summary,
    x="Species",
    y="Overlap_percent_of_proteomics",
    palette={"Human": COLOR_HUMAN, "Mouse": COLOR_MOUSE}
)
ax.set_title("Transcriptomic support for proteomics genes", fontsize=16, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Overlap (% of proteomics genes)", fontsize=12, fontweight="bold")
for i, row in summary.reset_index(drop=True).iterrows():
    ax.text(i, row["Overlap_percent_of_proteomics"] + 1.0,
            f"{row['Overlap_percent_of_proteomics']:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(overlap_pct_png, dpi=300, bbox_inches="tight")
plt.close()

# panel FDR heatmap
fdr_heat = panel_stats.pivot(index="Panel", columns="Species", values="minus_log10_FDR")
plt.figure(figsize=(7, 6.5))
ax = sns.heatmap(
    fdr_heat,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label": "-log10(FDR)"}
)
ax.set_title("DRG marker panel enrichment", fontsize=16, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(panel_fdr_heatmap_png, dpi=300, bbox_inches="tight")
plt.close()

# panel support fraction heatmap
support_heat = panel_stats.pivot(index="Panel", columns="Species", values="Panel_support_fraction_of_transcriptome_panel")
plt.figure(figsize=(7, 6.5))
ax = sns.heatmap(
    support_heat,
    annot=True,
    fmt=".2f",
    cmap="Oranges",
    linewidths=0.5,
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Support fraction"}
)
ax.set_title("Fraction of panel genes supported by proteomics", fontsize=16, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(panel_support_heatmap_png, dpi=300, bbox_inches="tight")
plt.close()

# panel enrichment barplot
plot_df = panel_stats.copy()
plt.figure(figsize=(10, 6.5))
ax = sns.barplot(
    data=plot_df,
    x="minus_log10_FDR",
    y="Panel",
    hue="Species",
    palette={"Human": COLOR_HUMAN, "Mouse": COLOR_MOUSE}
)
ax.set_title("Panel enrichment statistics", fontsize=16, fontweight="bold", pad=12)
ax.set_xlabel("-log10(FDR)", fontsize=12, fontweight="bold")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(panel_barplot_png, dpi=300, bbox_inches="tight")
plt.close()

# gene-level presence heatmap
presence_heat = presence_df.set_index(["Panel", "Gene"])[[
    "Human_supported_by_both", "Mouse_supported_by_both"
]]
plt.figure(figsize=(7.5, max(8, 0.22 * presence_heat.shape[0])))
ax = sns.heatmap(
    presence_heat,
    cmap="YlGn",
    linewidths=0.2,
    linecolor="white",
    cbar_kws={"label": "Supported by transcriptome + proteomics"},
    vmin=0,
    vmax=1
)
ax.set_title("Gene-level panel support matrix", fontsize=16, fontweight="bold", pad=12)
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(panel_presence_heatmap_png, dpi=300, bbox_inches="tight")
plt.close()

# =========================================
# REPORT
# =========================================
with open(report_txt, "w") as f:
    f.write("FINAL TRANSCRIPTOMICS PANEL VALIDATION REPORT\n")
    f.write("=" * 55 + "\n\n")

    f.write("OVERLAP SUMMARY\n")
    f.write(summary.to_string(index=False))
    f.write("\n\n")

    f.write("PANEL ENRICHMENT STATISTICS\n")
    f.write(panel_stats.to_string(index=False))
    f.write("\n\n")

    f.write("INTERPRETATION NOTES\n")
    f.write("- Overlap percentage is descriptive validation.\n")
    f.write("- Fisher's exact test assesses whether proteomics genes are enriched in curated DRG/sensory panels relative to the transcriptomic universe.\n")
    f.write("- FDR is Benjamini-Hochberg corrected within species across panels.\n")

print("\nFINAL PANEL VALIDATION COMPLETE")
print(f"Summary table: {summary_csv}")
print(f"Top30 human: {top30_human_csv}")
print(f"Top30 mouse: {top30_mouse_csv}")
print(f"Panel statistics: {panel_stats_csv}")
print(f"Presence matrix: {panel_presence_csv}")
print(f"Report: {report_txt}")
