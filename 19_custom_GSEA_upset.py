import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
from collections import defaultdict

# =========================================
# SETTINGS
# =========================================
TOP_N = 10
FDR_CUTOFF = 0.25
MAX_INTERSECTIONS = 20
MIN_INTERSECTION_SIZE = 3

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
gsea_go_dir = base / "results" / "GSEA_GO_BP"
fig_dir = base / "figures" / "gsea"
fig_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# SHORT DISPLAY LABELS
# only for figure readability
# =========================================
SHORT_LABELS = {
    "Mitochondrial Electron Transport, NADH To Ubiquinone": "ETC: NADH to ubiquinone",
    "Oxidative Phosphorylation": "Oxidative phosphorylation",
    "Proton Motive Force-Driven ATP Synthesis": "ATP synthesis (PMF-driven)",
    "Peptidyl-Serine Dephosphorylation": "Peptidyl-serine dephosphorylation",
    "Proton Motive Force-Driven Mitochondrial ATP Synthesis": "Mitochondrial ATP synthesis",
    "Mitochondrial ATP Synthesis Coupled Electron Transport": "ATP synthesis + ETC",
    "Aerobic Electron Transport Chain": "Aerobic ETC",
    "Cellular Respiration": "Cellular respiration",
    "Striated Muscle Contraction": "Striated muscle contraction",
    "Muscle Contraction": "Muscle contraction",
    "Sarcomere Organization": "Sarcomere organization",
    "Heart Contraction": "Heart contraction",
    "Cardiac Muscle Contraction": "Cardiac muscle contraction",
    "Regulation Of Response To External Stimulus": "Response to external stimulus",
    "Double-Strand Break Repair Via Homologous Recombination": "DSB repair via HR",
    "Regulation Of Defense Response": "Defense response regulation",
    "Nucleic Acid Catabolic Process": "Nucleic acid catabolism",
    "Mitochondrial Gene Expression": "Mitochondrial gene expression",
    "Mitochondrial Translation": "Mitochondrial translation",
    "Protein Homotetramerization": "Protein homotetramerization",
    "Double-Strand Break Repair": "Double-strand break repair",
}

# =========================================
# HELPERS
# =========================================
def find_report_file(folder: Path) -> Path:
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {folder}")

    for f in csvs:
        name = f.name.lower()
        if "gseapy.gene_set.prerank.report" in name:
            return f
    for f in csvs:
        name = f.name.lower()
        if "report" in name:
            return f
    return csvs[0]

def split_leading_genes(x):
    if pd.isna(x):
        return []
    genes = [g.strip().upper() for g in str(x).split(";") if g.strip()]
    genes = [g for g in genes if g != "NAN"]
    return list(dict.fromkeys(genes))

def shorten_label(term):
    term = str(term).replace("_", " ")
    if " (GO:" in term:
        term = term.split(" (GO:")[0]
    return SHORT_LABELS.get(term, term)

def prepare_subset(df, direction="human"):
    df = df.copy()

    # flexible column mapping
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "term":
            col_map[c] = "Term"
        elif cl == "nes":
            col_map[c] = "NES"
        elif cl in ["fdr q-val", "fdr_q-val", "fdr"]:
            col_map[c] = "FDR"
        elif cl in ["lead_genes", "lead genes"]:
            col_map[c] = "Lead_genes"

    df = df.rename(columns=col_map)

    required = ["Term", "NES", "FDR", "Lead_genes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[["Term", "NES", "FDR", "Lead_genes"]].copy()
    df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
    df["FDR"] = pd.to_numeric(df["FDR"], errors="coerce")
    df = df.dropna(subset=["Term", "NES", "FDR", "Lead_genes"]).copy()
    df = df[df["FDR"] < FDR_CUTOFF].copy()

    if direction == "human":
        df = df[df["NES"] > 0].sort_values("NES", ascending=False).head(TOP_N).copy()
    else:
        df = df[df["NES"] < 0].sort_values("NES", ascending=True).head(TOP_N).copy()

    df["Label"] = df["Term"].apply(shorten_label)
    df["Genes"] = df["Lead_genes"].apply(split_leading_genes)
    df = df[df["Genes"].map(len) > 0].copy()

    return df

def build_intersections(contents):
    gene_to_sets = defaultdict(list)

    for set_name, genes in contents.items():
        for g in genes:
            gene_to_sets[g].append(set_name)

    combo_counts = defaultdict(int)
    for _, sets_ in gene_to_sets.items():
        key = tuple(sorted(set(sets_)))
        combo_counts[key] += 1

    combo_df = pd.DataFrame(
        [{"sets": k, "count": v} for k, v in combo_counts.items()]
    ).sort_values("count", ascending=False)

    combo_df = combo_df[combo_df["count"] >= MIN_INTERSECTION_SIZE].copy()
    return combo_df

def style_spines(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")

def make_custom_upset(df, title, outfile, color="#2f5d8a"):
    contents = {row["Label"]: set(row["Genes"]) for _, row in df.iterrows()}
    set_names = list(contents.keys())

    if len(set_names) < 2:
        print(f"Not enough sets for {title}")
        return

    set_sizes = {k: len(v) for k, v in contents.items()}
    combo_df = build_intersections(contents)

    if combo_df.empty:
        print(f"No intersections found for {title}")
        return

    combo_df = combo_df.head(MAX_INTERSECTIONS).reset_index(drop=True)

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[2.2, 4.8],
        height_ratios=[2.3, 2.0],
        wspace=0.04,
        hspace=0.05
    )

    ax_sets = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_mat = fig.add_subplot(gs[1, 1])

    # Left set-size bars
    y_positions = list(range(len(set_names)))[::-1]
    ordered_names = set_names[::-1]
    ordered_sizes = [set_sizes[n] for n in ordered_names]

    ax_sets.barh(
        y_positions,
        ordered_sizes,
        color=color,
        edgecolor="black",
        linewidth=1.2
    )
    ax_sets.set_yticks(y_positions)
    ax_sets.set_yticklabels(ordered_names, fontsize=13, fontweight="bold")
    ax_sets.invert_xaxis()
    ax_sets.set_xlabel("Set size", fontsize=15, fontweight="bold")
    ax_sets.tick_params(axis="x", labelsize=11, width=1.2)
    ax_sets.tick_params(axis="y", pad=6)
    style_spines(ax_sets)

    # Top intersection bars
    x = range(len(combo_df))
    ax_bar.bar(
        x,
        combo_df["count"],
        color=color,
        edgecolor="black",
        linewidth=1.2
    )
    ax_bar.set_ylabel("Intersection size", fontsize=15, fontweight="bold")
    ax_bar.set_xticks([])
    ax_bar.tick_params(axis="y", labelsize=11, width=1.2)

    for i, c in enumerate(combo_df["count"]):
        ax_bar.text(
            i, c + 0.15, str(c),
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )
    style_spines(ax_bar)

    # Matrix
    name_to_y = {name: y for name, y in zip(ordered_names, y_positions)}

    for i in x:
        for name in ordered_names:
            ax_mat.plot(
                i, name_to_y[name],
                "o",
                color="#d0d0d0",
                markersize=7,
                zorder=1
            )

    for i, row in combo_df.iterrows():
        sets_in_combo = row["sets"]
        ys = sorted([name_to_y[s] for s in sets_in_combo])

        for s in sets_in_combo:
            ax_mat.plot(
                i, name_to_y[s],
                "o",
                color="black",
                markersize=8,
                zorder=3
            )

        if len(ys) > 1:
            ax_mat.plot(
                [i, i],
                [min(ys), max(ys)],
                color="black",
                linewidth=1.6,
                zorder=2
            )

    ax_mat.set_xticks(list(x))
    ax_mat.set_xticklabels([str(i + 1) for i in x], fontsize=11, fontweight="bold")
    ax_mat.set_xlabel("Gene overlap combinations", fontsize=15, fontweight="bold")
    ax_mat.set_yticks(y_positions)
    ax_mat.set_yticklabels([])
    ax_mat.tick_params(axis="x", width=1.2)
    style_spines(ax_mat)

    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.965)

    plt.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.10)
    plt.savefig(outfile, dpi=700, bbox_inches="tight")
    plt.close()

# =========================================
# LOAD GSEA GO REPORT
# =========================================
report_file = find_report_file(gsea_go_dir)
df = pd.read_csv(report_file)

print("Report file:", report_file)
print("Input shape:", df.shape)
print("Columns:", df.columns.tolist())

# =========================================
# HUMAN-ENRICHED GO PATHWAYS
# =========================================
human_df = prepare_subset(df, direction="human")
print("\nTop human-enriched GO pathways used:")
print(human_df[["Term", "NES", "FDR"]])

make_custom_upset(
    human_df,
    "Leading-edge gene overlap across top human-enriched GO pathways",
    fig_dir / "GSEA_GO_BP_human_leading_edge_custom_upset.png",
    color="#cc6f12"
)

# =========================================
# MOUSE-ENRICHED GO PATHWAYS
# =========================================
mouse_df = prepare_subset(df, direction="mouse")
print("\nTop mouse-enriched GO pathways used:")
print(mouse_df[["Term", "NES", "FDR"]])

make_custom_upset(
    mouse_df,
    "Leading-edge gene overlap across top mouse-enriched GO pathways",
    fig_dir / "GSEA_GO_BP_mouse_leading_edge_custom_upset.png",
    color="#2f8a57"
)

print("\nSaved:")
print(fig_dir / "GSEA_GO_BP_human_leading_edge_custom_upset.png")
print(fig_dir / "GSEA_GO_BP_mouse_leading_edge_custom_upset.png")
