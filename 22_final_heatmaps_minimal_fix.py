import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib import gridspec

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()

merged_aligned_file = base / "results" / "merged_matrix_median_aligned.csv"
de_file = base / "results" / "cross_species_results.csv"

out_dir = base / "figures" / "FINAL_main_heatmap_refined"
out_dir.mkdir(parents=True, exist_ok=True)

OUT_PNG = out_dir / "FINAL_main_heatmap_refined.png"
OUT_SVG = out_dir / "FINAL_main_heatmap_refined.svg"
OUT_PDF = out_dir / "FINAL_main_heatmap_refined.pdf"
OUT_ORDER = out_dir / "FINAL_main_heatmap_gene_order.csv"
OUT_MATRIX = out_dir / "FINAL_main_heatmap_expression_matrix.csv"
OUT_Z = out_dir / "FINAL_main_heatmap_row_zscore_matrix.csv"
OUT_SUMMARY = out_dir / "FINAL_main_heatmap_summary.txt"

# =========================================
# STYLE
# =========================================
sns.set_style("white")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

# =========================================
# HELPERS
# =========================================
def clean_gene_series(series):
    return series.astype(str).str.upper().str.strip()

def make_numeric(df):
    return df.apply(pd.to_numeric, errors="coerce")

def zscore_rows_keep_nan(df):
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

def fill_row_nans(df):
    return df.apply(lambda row: row.fillna(row.mean(skipna=True)), axis=1)

def dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}. Available columns: {list(df.columns)}")

def draw_group_brackets(
    ax,
    group_ranges,
    total_rows,
    x_text=0.10,
    x_line_start=0.42,
    x_bracket=0.86,
    x_end=0.98,
    text_fs=13.0,
    line_w=2.2
):
    """
    Draw paper-style right-facing tournament brackets:
    label ---- [bracket] close to gene names

    group_ranges:
        list of tuples (group_name, start_row, end_row)
        start_row inclusive, end_row exclusive
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(total_rows, 0)
    ax.axis("off")

    for group_name, start, end in group_ranges:
        y0 = start
        y1 = end
        ym = (y0 + y1) / 2.0

        # label
        ax.text(
            x_text,
            ym,
            group_name,
            ha="right",
            va="center",
            fontsize=text_fs,
            fontweight="bold",
            linespacing=1.08
        )

        # line from label toward bracket
        ax.plot(
            [x_text + 0.02, x_line_start],
            [ym, ym],
            color="black",
            lw=line_w,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            clip_on=False
        )

        # horizontal line from label-side line to bracket spine
        ax.plot(
            [x_line_start, x_bracket],
            [ym, ym],
            color="black",
            lw=line_w,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            clip_on=False
        )

        # short top arm toward genes
        ax.plot(
            [x_bracket, x_end],
            [y0, y0],
            color="black",
            lw=line_w,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            clip_on=False
        )

        # short bottom arm toward genes
        ax.plot(
            [x_bracket, x_end],
            [y1, y1],
            color="black",
            lw=line_w,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            clip_on=False
        )

        # vertical bracket spine close to genes
        ax.plot(
            [x_bracket, x_bracket],
            [y0, y1],
            color="black",
            lw=line_w,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            clip_on=False
        )
# =========================================
# LOAD EXPRESSION MATRIX
# =========================================
merged = pd.read_csv(merged_aligned_file)
merged.columns = [str(c).strip() for c in merged.columns]
require_columns(merged, ["Ortholog_gene"], "merged_matrix_median_aligned.csv")

merged["Ortholog_gene"] = clean_gene_series(merged["Ortholog_gene"])

human_cols = [c for c in merged.columns if str(c).startswith("H_S")]
mouse_cols = [c for c in merged.columns if str(c).startswith("M_S")]
all_cols = human_cols + mouse_cols

if len(human_cols) == 0 or len(mouse_cols) == 0:
    raise ValueError(
        f"Could not detect H_S/M_S sample columns. Found columns: {list(merged.columns)}"
    )

expr = merged.set_index("Ortholog_gene")[all_cols].copy()
expr = make_numeric(expr)

if expr.index.duplicated().any():
    expr = expr.groupby(expr.index).mean()

# =========================================
# LOAD DE TABLE
# =========================================
de = pd.read_csv(de_file)
de.columns = [str(c).strip() for c in de.columns]
require_columns(de, ["Gene", "status", "MeanDiff_norm", "FDR"], "cross_species_results.csv")

de["Gene"] = clean_gene_series(de["Gene"])
de["status"] = de["status"].astype(str).str.strip()

de_human_top100 = (
    de.loc[de["status"] == "Higher in Human", ["Gene", "MeanDiff_norm", "FDR"]]
    .dropna(subset=["Gene", "MeanDiff_norm", "FDR"])
    .sort_values(["FDR", "MeanDiff_norm"], ascending=[True, False])
    .head(100)
    .copy()
)

de_mouse_top100 = (
    de.loc[de["status"] == "Higher in Mouse", ["Gene", "MeanDiff_norm", "FDR"]]
    .dropna(subset=["Gene", "MeanDiff_norm", "FDR"])
    .sort_values(["FDR", "MeanDiff_norm"], ascending=[True, True])
    .head(100)
    .copy()
)

human_top100 = de_human_top100["Gene"].tolist()
mouse_top100 = de_mouse_top100["Gene"].tolist()

# =========================================
# MANUAL BIOLOGICAL GROUPS
# =========================================
anchor_blocks = {
    "Shared sensory-neuronal\nbackbone": [
        "TUBB3", "PRPH", "NEFL", "NEFH", "GAP43", "UCHL1", "NCAM1", "MAP1B", "STMN2"
    ],
    "Sensory signaling and\nexcitability markers": [
        "TRPV1", "NTRK1", "SCN9A", "SCN10A", "SCN11A", "P2RX3", "RET",
        "GFRA1", "GFRA2", "GFRA3", "OSMR", "PIEZO2", "NTRK2", "NTRK3",
        "PVALB", "RUNX3"
    ],
    "Glial and myelin-associated\nmarkers": [
        "SOX10", "MBP", "MPZ", "PLP1", "PMP22", "MAG", "GFAP", "S100B",
        "FABP7", "KCNJ10", "GLUL", "SLC1A3"
    ]
}

mouse_blocks = {
    "Myelin and axon-glial\nsupport": [
        "MAG", "PRX", "SLC12A2", "SUSD2"
    ],
    "Synaptic vesicle and\naxonal machinery": [
        "SNAP25", "STXBP1", "STX1B", "STXBP6", "SYN1", "DNM1", "NSF", "VAT1", "RAB6B",
        "PACSIN2", "SEC31A", "DLG1", "GIT1", "AP2A2", "AAK1", "ADAM22", "NCDN"
    ],
    "Cytoskeletal and axonal\nstructural programs": [
        "TUBB", "TUBB2A", "TUBA4A", "SPTBN1", "DST", "PLEC", "CLASP1", "EML1",
        "KIFAP3", "SEPTIN7", "CORO7", "PLS1", "ACTA1", "MYH1"
    ],
    "Mitochondrial and oxidative-\nenergetic programs": [
        "COX6B1", "NDUFA4", "NDUFAB1", "NDUFS6", "ETFA", "ETFB", "PDHB", "DLD",
        "GOT2", "ATP5F1A", "ATP5PF", "ATP5ME", "ATP6V1A", "ATP6V1B2", "ATP6V0A1",
        "TIMM13", "ATP1A1", "CMPK1", "AACS", "IVD", "ADPGK", "GPD1", "PYGM"
    ],
    "Proteostasis and stress-\nresponse programs": [
        "HSPA4L", "DNAJA4", "UBQLN1", "UBL7", "NEDD4", "OTUB1", "SELENOO", "BOLA1",
        "TXNDC12", "GLRX3", "TXNRD1", "HUWE1", "HSP90AB1", "PSMA7", "SGTA", "ARIH1"
    ],
    "Neuronal signaling and\nmaintenance programs": [
        "MAPK10", "PRKAR2B", "PRKCB", "NKTR", "GMFB", "AS3MT", "PREPL", "MCRIP1",
        "PCBP2", "PCBP3", "PPP3CA", "PPP2CA", "KCNA1", "KCNAB1", "PVALB", "RANBP1",
        "PRPS1", "NARS1", "SPR", "TCP11L1", "THEM4", "ABCB9", "ASRGL1", "ADSS1",
        "HSD17B7", "LYPLA2", "ARF5"
    ]
}

human_blocks = {
    "Trafficking and lysosomal-\nendosomal programs": [
        "LAMP2", "ARL8A", "ARF4", "RAB35", "RAB5B", "RAB5C", "RAB3D", "RAB33B",
        "ACP2", "CD63", "CHMP2A", "SCAMP1", "MAP1LC3A", "SH3GLB1", "ATL3", "BCAP31",
        "GPAA1", "MESD", "APRT", "TSFM", "THUMPD1", "STXBP3"
    ],
    "Redox, detoxification, and\nstress-buffering programs": [
        "GSTK1", "CYB5R1", "EPHX1", "MPST", "SOD3", "TST", "APEX1", "FAHD1",
        "TRAP1", "CYGB", "FTH1", "GSTM2", "PTGR1", "PCYOX1", "RDH14"
    ],
    "RNA processing, transcription,\nand translational control": [
        "POLR2A", "CPSF7", "EEF1A2", "CARS1", "RPS11", "SRP14", "SRSF4", "XPOT",
        "EWSR1", "TSN", "TUFM", "ZSWIM8", "RP2", "HP1BP3", "LMNA", "LMNB2", "HMGB2",
        "PPP2CB", "MYL9", "RDX", "YWHAH", "RAP2B", "CCT6B"
    ],
    "Extracellular matrix, adhesion,\nand tissue-interface programs": [
        "BGN", "ITGA1", "PODN", "ALCAM", "EMILIN1", "OLFML3", "THSD4", "AGRN",
        "CLDND1", "CTNNA2", "LANCL2", "LANCL3", "NHLRC2", "TNS3", "LASP1", "PAPLN",
        "APOD"
    ],
    "Metabolic, lipid, and\nmembrane-maintenance programs": [
        "LPCAT2", "SLC25A24", "SLC1A4", "SCCPDH", "DBT", "PGM2", "SPTLC2", "ACOT13",
        "PDXK", "ALDOC", "GGH", "GNAI2", "CNDP2", "TALDO1", "PHYKPL", "TMEM11",
        "S100A1"
    ],
    "Stress, damage-response,\nand immune-associated signals": [
        "NFKB1", "BAX", "SAMHD1", "PSME3", "TIPRL", "IGKC"
    ]
}

# =========================================
# BUILD FINAL ORDER
# =========================================
final_order = []
records = []
seen = set()

def add_group(group_name, genes, section, allowed_set=None):
    for g in genes:
        if allowed_set is not None and g not in allowed_set:
            continue
        if g in expr.index and g not in seen:
            final_order.append(g)
            records.append({"Gene": g, "Group": group_name, "Section": section})
            seen.add(g)

for group_name, genes in anchor_blocks.items():
    add_group(group_name, genes, "anchor")

mouse_set = set(mouse_top100)
for group_name, genes in mouse_blocks.items():
    add_group(group_name, genes, "mouse", allowed_set=mouse_set)

for g in mouse_top100:
    if g in expr.index and g not in seen:
        final_order.append(g)
        records.append({"Gene": g, "Group": "Additional\nproteins", "Section": "mouse"})
        seen.add(g)

human_set = set(human_top100)
for group_name, genes in human_blocks.items():
    add_group(group_name, genes, "human", allowed_set=human_set)

for g in human_top100:
    if g in expr.index and g not in seen:
        final_order.append(g)
        records.append({"Gene": g, "Group": "Additional\nproteins", "Section": "human"})
        seen.add(g)

final_order = dedupe_keep_order(final_order)
ann = pd.DataFrame(records).drop_duplicates(subset=["Gene"], keep="first")
require_columns(ann, ["Gene", "Group", "Section"], "annotation table")

# =========================================
# BUILD FINAL MATRIX
# =========================================
heat = expr.reindex(final_order)[all_cols]
heat = fill_row_nans(heat)
heat = heat.dropna(axis=0, how="all")

ann_ordered = ann.set_index("Gene").reindex(heat.index).copy()
ann_ordered.index.name = "Gene"

heat_z = zscore_rows_keep_nan(heat)

heat.to_csv(OUT_MATRIX)
heat_z.to_csv(OUT_Z)
ann_ordered.reset_index().to_csv(OUT_ORDER, index=False)

# =========================================
# BUILD DISPLAY MATRIX WITH WHITE GAPS
# =========================================
display_rows = []
display_gene_labels = []
group_ranges = []

unique_groups = ann_ordered["Group"].dropna().drop_duplicates().tolist()
current_row = 0

for gi, group in enumerate(unique_groups):
    sub = ann_ordered[ann_ordered["Group"] == group].copy()
    genes = sub.index.astype(str).tolist()
    genes = [g for g in genes if g in heat_z.index]

    start = current_row

    for gene in genes:
        display_rows.append(heat_z.loc[gene].values)
        display_gene_labels.append(gene)
        current_row += 1

    end = current_row
    group_ranges.append((group, start, end))

    if gi < len(unique_groups) - 1:
        display_rows.append([np.nan] * len(all_cols))
        display_gene_labels.append("")
        current_row += 1

display_df = pd.DataFrame(display_rows, columns=all_cols)

# =========================================
# PLOT
# =========================================
n_rows = display_df.shape[0]
fig_h = max(22, n_rows * 0.16)
fig_w = 26

fig = plt.figure(figsize=(fig_w, fig_h))
gs = gridspec.GridSpec(
    1, 4,
    width_ratios=[2.2, 2.0, 18.2, 0.9],
    wspace=0.04
)

ax_group = plt.subplot(gs[0, 0])
ax_gene = plt.subplot(gs[0, 1])
ax_heat = plt.subplot(gs[0, 2])
ax_cbar = plt.subplot(gs[0, 3])

# -----------------------------------------
# BRACKET LABELS
# -----------------------------------------
draw_group_brackets(
    ax_group,
    group_ranges,
    total_rows=n_rows,
    x_text=0.60,
    x_line_start=0.64,
    x_bracket=0.86,
    x_end=1.02,
    text_fs=13.2,
    line_w=2.4
)
# -----------------------------------------
# GENE NAMES
# -----------------------------------------
ax_gene.set_xlim(0, 1)
ax_gene.set_ylim(n_rows, 0)
ax_gene.axis("off")

gene_fontsize = 8.2 if n_rows <= 170 else 7.6

for i, gene in enumerate(display_gene_labels):
    if gene != "":
        ax_gene.text(
            0.98, i + 0.5, gene,
            ha="right",
            va="center",
            fontsize=gene_fontsize,
            fontweight="bold"
        )

# -----------------------------------------
# HEATMAP
# -----------------------------------------
mask = display_df.isna()
cmap = sns.color_palette("vlag", as_cmap=True)

sns.heatmap(
    display_df,
    ax=ax_heat,
    cmap=cmap,
    center=0,
    vmin=-2.5,
    vmax=2.5,
    mask=mask,
    linewidths=0,
    cbar=True,
    cbar_ax=ax_cbar,
    cbar_kws={"label": "Row z-score"},
    yticklabels=False,
    xticklabels=True
)

ax_heat.set_title(
    "Cross-species organization of the DRG proteome",
    fontsize=20,
    fontweight="bold",
    pad=16
)
ax_heat.set_xlabel("")
ax_heat.set_ylabel("")

ax_heat.set_xticklabels(all_cols, rotation=90, fontsize=11, fontweight="bold")
ax_heat.tick_params(axis="x", bottom=True, top=False, labelbottom=True, pad=6)

for lbl, col in zip(ax_heat.get_xticklabels(), all_cols):
    if str(col).startswith("H_S"):
        lbl.set_color("#C97900")
    else:
        lbl.set_color("#2E8B57")

for spine in ax_heat.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.9)
    spine.set_color("black")

ax_cbar.tick_params(labelsize=10)
ax_cbar.yaxis.label.set_size(12)
ax_cbar.yaxis.label.set_weight("bold")

# -----------------------------------------
# SAVE
# -----------------------------------------
plt.savefig(OUT_PNG, dpi=1200, bbox_inches="tight")
plt.savefig(OUT_SVG, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

# =========================================
# SUMMARY
# =========================================
with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
    f.write("FINAL MAIN HEATMAP SUMMARY\n")
    f.write("=" * 60 + "\n")
    f.write(f"Expression source: {merged_aligned_file}\n")
    f.write(f"DE source: {de_file}\n")
    f.write(f"Output folder: {out_dir}\n\n")
    f.write(f"Top100 human-higher: {len(human_top100)}\n")
    f.write(f"Top100 mouse-higher: {len(mouse_top100)}\n")
    f.write(f"Final unique genes displayed: {heat.shape[0]}\n\n")
    f.write("Groups in order:\n")
    for g in unique_groups:
        n = int((ann_ordered['Group'] == g).sum())
        f.write(f"- {g.replace(chr(10), ' ')}: {n} genes\n")

print("DONE")
print(OUT_PNG)
print(OUT_SVG)
print(OUT_PDF)
print(OUT_ORDER)
print(OUT_MATRIX)
print(OUT_Z)
print(OUT_SUMMARY)
