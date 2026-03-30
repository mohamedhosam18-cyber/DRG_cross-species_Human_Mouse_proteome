import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

# =========================================
# SETTINGS
# =========================================
TOP_N_GO = 8
TOP_N_KEGG = 8
TOP_N_REACTOME = 6
WRAP_WIDTH = 28

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
results_dir = base / "results" / "enrichment"
fig_dir = base / "figures" / "enrichment"
fig_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# FILE SETS
# =========================================
comparisons = [
    (
        results_dir / "human_GO_BP_enrichment.csv",
        results_dir / "mouse_GO_BP_enrichment.csv",
        fig_dir / "cross_species_GO_BP_comparison.png",
        "Cross-species comparison of enriched GO Biological Processes",
        TOP_N_GO
    ),
    (
        results_dir / "human_KEGG_enrichment.csv",
        results_dir / "mouse_KEGG_enrichment.csv",
        fig_dir / "cross_species_KEGG_comparison.png",
        "Cross-species comparison of enriched KEGG pathways",
        TOP_N_KEGG
    ),
    (
        results_dir / "human_Reactome_enrichment.csv",
        results_dir / "mouse_Reactome_enrichment.csv",
        fig_dir / "cross_species_Reactome_comparison.png",
        "Cross-species comparison of enriched Reactome pathways",
        TOP_N_REACTOME
    ),
]

# =========================================
# HELPERS
# =========================================
def wrap_term(term, width=WRAP_WIDTH):
    return "\n".join(textwrap.wrap(str(term), width=width))

def clean_term(term):
    term = str(term)

    # remove GO IDs
    if " (GO:" in term:
        term = term.split(" (GO:")[0]

    # remove Reactome IDs like R-HSA-123456
    if " R-HSA-" in term:
        term = term.split(" R-HSA-")[0]

    term = term.replace("_", " ")
    return term

def parse_count(overlap):
    try:
        return int(str(overlap).split("/")[0])
    except Exception:
        return np.nan

def prepare_df(df):
    df = df.copy()

    if "Adjusted P-value" not in df.columns:
        raise ValueError("Missing 'Adjusted P-value' column")

    df["Adjusted P-value"] = pd.to_numeric(df["Adjusted P-value"], errors="coerce")
    df = df.dropna(subset=["Term", "Adjusted P-value"]).copy()

    df["minuslog10_FDR"] = -np.log10(df["Adjusted P-value"])
    df["CleanTerm"] = df["Term"].apply(clean_term)
    df["Label"] = df["CleanTerm"].apply(wrap_term)

    if "Overlap" in df.columns:
        df["GeneCount"] = df["Overlap"].apply(parse_count)
    else:
        df["GeneCount"] = np.nan

    return df

def plot_mirrored_comparison(human_top, mouse_top, title, out_plot):
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 1.6
    plt.rcParams["xtick.major.width"] = 1.3
    plt.rcParams["ytick.major.width"] = 1.3

    fig, ax = plt.subplots(figsize=(16, 14))

    n_mouse = len(mouse_top)
    n_human = len(human_top)

    mouse_y = np.arange(n_mouse)
    human_y = np.arange(n_human) + n_mouse + 2

    # Mouse on left
    ax.barh(
        mouse_y,
        -mouse_top["minuslog10_FDR"],
        color="#2e8b57",
        edgecolor="black",
        linewidth=0.9,
        alpha=0.92
    )

    # Human on right
    ax.barh(
        human_y,
        human_top["minuslog10_FDR"],
        color="#d97706",
        edgecolor="black",
        linewidth=0.9,
        alpha=0.92
    )

    # Y labels
    y_labels = list(mouse_top["Label"]) + list(human_top["Label"])
    ax.set_yticks(list(mouse_y) + list(human_y))
    ax.set_yticklabels(y_labels, fontsize=12, fontweight="bold")

    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")

    ax.tick_params(axis="y", pad=8)

    # Center line
    ax.axvline(0, color="black", linewidth=1.8)

    # X tick labels as absolute values
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x):.1f}" for x in xticks], fontsize=12, fontweight="bold")

    ax.set_xlabel("-log10 adjusted p-value (FDR)", fontsize=16, weight="bold")
    ax.set_title(title, fontsize=21, weight="bold", pad=18)

    mouse_max = mouse_top["minuslog10_FDR"].max() if len(mouse_top) else 1
    human_max = human_top["minuslog10_FDR"].max() if len(human_top) else 1

    ax.text(
        -mouse_max * 0.72,
        max(n_mouse - 0.25, 0),
        "Mouse-enriched",
        fontsize=15,
        weight="bold",
        color="#2e8b57",
        ha="center"
    )

    ax.text(
        human_max * 0.72,
        n_mouse + 1.8 + max(n_human - 0.25, 0),
        "Human-enriched",
        fontsize=15,
        weight="bold",
        color="#d97706",
        ha="center"
    )

    # Gene count annotations
    for y, val, count in zip(mouse_y, mouse_top["minuslog10_FDR"], mouse_top["GeneCount"]):
        if pd.notna(count):
            ax.text(
                -val - 0.05,
                y,
                f"n={int(count)}",
                va="center",
                ha="right",
                fontsize=11,
                weight="bold"
            )

    for y, val, count in zip(human_y, human_top["minuslog10_FDR"], human_top["GeneCount"]):
        if pd.notna(count):
            ax.text(
                val + 0.05,
                y,
                f"n={int(count)}",
                va="center",
                ha="left",
                fontsize=11,
                weight="bold"
            )

    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=500, bbox_inches="tight")
    plt.close()

def run_one_comparison(human_file, mouse_file, out_plot, title, top_n):
    if not human_file.exists():
        raise FileNotFoundError(f"Missing file: {human_file}")
    if not mouse_file.exists():
        raise FileNotFoundError(f"Missing file: {mouse_file}")

    human = pd.read_csv(human_file)
    mouse = pd.read_csv(mouse_file)

    human = prepare_df(human)
    mouse = prepare_df(mouse)

    human_top = human.sort_values("Adjusted P-value", ascending=True).head(top_n).copy()
    mouse_top = mouse.sort_values("Adjusted P-value", ascending=True).head(top_n).copy()

    # Reverse for plotting
    human_top = human_top.iloc[::-1].reset_index(drop=True)
    mouse_top = mouse_top.iloc[::-1].reset_index(drop=True)

    plot_mirrored_comparison(human_top, mouse_top, title, out_plot)

    print(f"Saved: {out_plot}")
    print("\nTop mouse terms:")
    print(mouse_top[["CleanTerm", "Adjusted P-value", "Overlap"]].head(top_n))
    print("\nTop human terms:")
    print(human_top[["CleanTerm", "Adjusted P-value", "Overlap"]].head(top_n))
    print("-" * 80)

# =========================================
# RUN ALL
# =========================================
for human_file, mouse_file, out_plot, title, top_n in comparisons:
    run_one_comparison(human_file, mouse_file, out_plot, title, top_n)

print("All cross-species enrichment comparison figures completed.")
