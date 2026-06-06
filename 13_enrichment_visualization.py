import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
results_dir = base / "results" / "enrichment"
figures_dir = base / "figures" / "enrichment_dotplots_professional"
figures_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# FILES TO PLOT
# =========================================
files = [
    ("human_GO_BP_enrichment.csv", "Human GO Biological Process", "human_GO_BP_dotplot_pro.png"),
    ("mouse_GO_BP_enrichment.csv", "Mouse GO Biological Process", "mouse_GO_BP_dotplot_pro.png"),
    ("human_KEGG_enrichment.csv", "Human KEGG Pathways", "human_KEGG_dotplot_pro.png"),
    ("mouse_KEGG_enrichment.csv", "Mouse KEGG Pathways", "mouse_KEGG_dotplot_pro.png"),
    ("human_Reactome_enrichment.csv", "Human Reactome Pathways", "human_Reactome_dotplot_pro.png"),
    ("mouse_Reactome_enrichment.csv", "Mouse Reactome Pathways", "mouse_Reactome_dotplot_pro.png"),
    ("human_WikiPathways_enrichment.csv", "Human WikiPathways", "human_WikiPathways_dotplot_pro.png"),
    ("mouse_WikiPathways_enrichment.csv", "Mouse WikiPathways", "mouse_WikiPathways_dotplot_pro.png"),
]

# =========================================
# SETTINGS
# =========================================
TOP_N = 10
WRAP_WIDTH = 34
MIN_SIZE = 120
MAX_SIZE = 900

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 1.6
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2

# =========================================
# HELPERS
# =========================================
def wrap_term(x, width=WRAP_WIDTH):
    return "\n".join(textwrap.wrap(str(x), width=width))

def clean_term(term):
    term = str(term)

    if " (GO:" in term:
        term = term.split(" (GO:")[0]

    if " R-HSA-" in term:
        term = term.split(" R-HSA-")[0]

    if " WP" in term:
        parts = term.rsplit(" WP", 1)
        if len(parts) == 2:
            term = parts[0]

    term = term.replace("_", " ").strip()
    return term

def parse_overlap(x):
    try:
        left, right = str(x).split("/")
        count = int(left)
        total = int(right)
        signal = count / total if total > 0 else np.nan
        return count, total, signal
    except Exception:
        return np.nan, np.nan, np.nan

def scale_sizes(values, min_size=MIN_SIZE, max_size=MAX_SIZE):
    vals = pd.Series(values).astype(float)

    if vals.nunique(dropna=True) <= 1:
        return pd.Series(np.repeat((min_size + max_size) / 2, len(vals)), index=vals.index)

    vmin = vals.min()
    vmax = vals.max()
    return min_size + (vals - vmin) * (max_size - min_size) / (vmax - vmin)

def pick_size_legend_values(counts):
    counts = sorted(pd.Series(counts).dropna().astype(int).unique())
    if len(counts) == 0:
        return []
    if len(counts) <= 4:
        return counts

    vals = [
        counts[0],
        int(np.percentile(counts, 35)),
        int(np.percentile(counts, 65)),
        counts[-1]
    ]
    vals = sorted(list(dict.fromkeys(vals)))
    return vals

# =========================================
# MAIN PLOTTER
# =========================================
def make_dotplot(csv_name, title, out_name):
    file_path = results_dir / csv_name
    if not file_path.exists():
        print(f"Missing: {file_path}")
        return

    df = pd.read_csv(file_path)

    required_cols = ["Term", "Adjusted P-value"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing columns {missing} in {csv_name}")
        return

    df["Adjusted P-value"] = pd.to_numeric(df["Adjusted P-value"], errors="coerce")
    df = df.dropna(subset=["Term", "Adjusted P-value"]).copy()

    # overlap parsing
    if "Overlap" in df.columns:
        parsed = df["Overlap"].apply(parse_overlap)
        df["GeneCount"] = [x[0] for x in parsed]
        df["TermSize"] = [x[1] for x in parsed]
        df["Signal"] = [x[2] for x in parsed]
    else:
        df["GeneCount"] = np.nan
        df["TermSize"] = np.nan
        df["Signal"] = np.nan

    df["CleanTerm"] = df["Term"].apply(clean_term)
    df["WrappedTerm"] = df["CleanTerm"].apply(wrap_term)

    # top enriched terms
    df = df.sort_values("Adjusted P-value", ascending=True).head(TOP_N).copy()

    if df.empty:
        print(f"No rows to plot in {csv_name}")
        return

    # fallback if signal unavailable
    if df["Signal"].isna().all():
        df["Signal"] = -np.log10(df["Adjusted P-value"])

    df["DotSize"] = scale_sizes(df["GeneCount"]) if df["GeneCount"].notna().sum() > 0 else (MIN_SIZE + MAX_SIZE) / 2

    # reverse so strongest term shows at top
    df = df.iloc[::-1].reset_index(drop=True)

    y_pos = np.arange(len(df))

    # =====================================
    # FIGURE LAYOUT
    # =====================================
    fig = plt.figure(figsize=(16, 9))

    # main plot
    ax = fig.add_axes([0.33, 0.12, 0.42, 0.76])

    # colorbar axis
    cax = fig.add_axes([0.77, 0.16, 0.022, 0.70])

    # dummy axis for size legend area
    lax = fig.add_axes([0.82, 0.22, 0.14, 0.45])
    lax.axis("off")

    # lollipop lines
    for y, signal in zip(y_pos, df["Signal"]):
        ax.hlines(y=y, xmin=0, xmax=signal, color="#b9c3ce", linewidth=2.0, alpha=0.75, zorder=1)

    # dots
    sc = ax.scatter(
        df["Signal"],
        y_pos,
        s=df["DotSize"],
        c=df["Adjusted P-value"],
        cmap="YlGnBu_r",
        edgecolors="#7a8794",
        linewidths=1.0,
        alpha=0.98,
        zorder=3
    )

    # y axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["WrappedTerm"], fontsize=12, fontweight="bold")

    # x axis
    ax.set_xlabel("Signal (overlap count / pathway size)", fontsize=16, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=22, fontweight="bold", pad=14)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.grid(axis="x", linestyle="--", alpha=0.20)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    # colorbar
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label("FDR", fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.invert_yaxis()

    # size legend
    legend_counts = pick_size_legend_values(df["GeneCount"])
    handles = []
    labels = []

    if len(legend_counts) > 0:
        for count in legend_counts:
            size = float(scale_sizes(pd.Series([count])).iloc[0])
            handles.append(
                plt.scatter([], [], s=size, color="#bfd08a", edgecolors="#88985b", linewidths=1.0)
            )
            labels.append(str(count))

        leg = lax.legend(
            handles,
            labels,
            title="Gene count",
            scatterpoints=1,
            frameon=False,
            fontsize=11,
            title_fontsize=13,
            loc="upper left",
            labelspacing=1.4,
            borderpad=0.6,
            handletextpad=1.0
        )
        leg.get_title().set_fontweight("bold")

    plt.savefig(figures_dir / out_name, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved: {figures_dir / out_name}")

# =========================================
# RUN ALL
# =========================================
for csv_name, title, out_name in files:
    make_dotplot(csv_name, title, out_name)

print("All professional enrichment dotplots completed.")
