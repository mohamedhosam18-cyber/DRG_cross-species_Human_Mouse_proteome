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
figures_dir = base / "figures" / "enrichment"
figures_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# FILES TO PLOT
# =========================================
files = [
    ("human_GO_BP_enrichment.csv", "Human GO Biological Process", "human_GO_BP_barplot.png"),
    ("mouse_GO_BP_enrichment.csv", "Mouse GO Biological Process", "mouse_GO_BP_barplot.png"),
    ("human_KEGG_enrichment.csv", "Human KEGG Pathways", "human_KEGG_barplot.png"),
    ("mouse_KEGG_enrichment.csv", "Mouse KEGG Pathways", "mouse_KEGG_barplot.png"),
    ("human_Reactome_enrichment.csv", "Human Reactome Pathways", "human_Reactome_barplot.png"),
    ("mouse_Reactome_enrichment.csv", "Mouse Reactome Pathways", "mouse_Reactome_barplot.png"),
]

# =========================================
# HELPERS
# =========================================
def wrap_term(x, width=42):
    return "\n".join(textwrap.wrap(str(x), width=width))

def parse_overlap_count(x):
    try:
        return int(str(x).split("/")[0])
    except:
        return np.nan

def clean_term(term):
    term = str(term)
    if " (GO:" in term:
        term = term.split(" (GO:")[0]
    return term

def make_barplot(csv_name, title, out_name):
    file_path = results_dir / csv_name
    if not file_path.exists():
        print(f"Missing: {file_path}")
        return

    df = pd.read_csv(file_path)

    if "Adjusted P-value" not in df.columns:
        print(f"Adjusted P-value column missing in {csv_name}")
        return

    df["Adjusted P-value"] = pd.to_numeric(df["Adjusted P-value"], errors="coerce")
    df = df.dropna(subset=["Term", "Adjusted P-value"]).copy()

    # Keep strongest enriched terms
    df = df.sort_values("Adjusted P-value", ascending=True).head(12).copy()

    if df.empty:
        print(f"No rows to plot in {csv_name}")
        return

    df["minuslog10_FDR"] = -np.log10(df["Adjusted P-value"])
    df["CleanTerm"] = df["Term"].apply(clean_term)
    df["WrappedTerm"] = df["CleanTerm"].apply(wrap_term)

    if "Overlap" in df.columns:
        df["Count"] = df["Overlap"].apply(parse_overlap_count)
    else:
        df["Count"] = np.nan

    # reverse so the strongest term is at the top visually
    df = df.iloc[::-1].copy()

    # choose color by species
    if title.startswith("Human"):
        bar_color = "#d97706"   # orange
    else:
        bar_color = "#2e8b57"   # green

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2

    fig, ax = plt.subplots(figsize=(12, 8.5))

    bars = ax.barh(
        df["WrappedTerm"],
        df["minuslog10_FDR"],
        color=bar_color,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.92
    )

    # annotate gene counts at the bar ends
    for bar, count in zip(bars, df["Count"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        if pd.notna(count):
            ax.text(
                x + 0.03,
                y,
                f"n={int(count)}",
                va="center",
                fontsize=10,
                weight="bold"
            )

    ax.set_xlabel("-log10 adjusted p-value (FDR)", fontsize=14, weight="bold")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=18, weight="bold", pad=14)

    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=11)

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(figures_dir / out_name, dpi=500, bbox_inches="tight")
    plt.close()

    print(f"Saved: {figures_dir / out_name}")

# =========================================
# RUN ALL
# =========================================
for csv_name, title, out_name in files:
    make_barplot(csv_name, title, out_name)

print("All enrichment barplots completed.")
