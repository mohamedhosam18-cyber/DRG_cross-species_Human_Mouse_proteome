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
TOP_N_WIKI = 8
WRAP_WIDTH = 30
MIN_SIZE = 120
MAX_SIZE = 900

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
results_dir = base / "results" / "enrichment"
fig_dir = base / "figures" / "enrichment_cross_species_professional"
fig_dir.mkdir(parents=True, exist_ok=True)

# =========================================
# FILE SETS
# =========================================
comparisons = [
    (
        results_dir / "human_GO_BP_enrichment.csv",
        results_dir / "mouse_GO_BP_enrichment.csv",
        fig_dir / "cross_species_GO_BP_dotplot_pro.png",
        "Cross-species comparison of enriched GO Biological Processes",
        TOP_N_GO
    ),
    (
        results_dir / "human_KEGG_enrichment.csv",
        results_dir / "mouse_KEGG_enrichment.csv",
        fig_dir / "cross_species_KEGG_dotplot_pro.png",
        "Cross-species comparison of enriched KEGG pathways",
        TOP_N_KEGG
    ),
    (
        results_dir / "human_Reactome_enrichment.csv",
        results_dir / "mouse_Reactome_enrichment.csv",
        fig_dir / "cross_species_Reactome_dotplot_pro.png",
        "Cross-species comparison of enriched Reactome pathways",
        TOP_N_REACTOME
    ),
    (
        results_dir / "human_WikiPathways_enrichment.csv",
        results_dir / "mouse_WikiPathways_enrichment.csv",
        fig_dir / "cross_species_WikiPathways_dotplot_pro.png",
        "Cross-species comparison of enriched WikiPathways",
        TOP_N_WIKI
    ),
]

# =========================================
# GLOBAL STYLE
# =========================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.6
plt.rcParams["xtick.major.width"] = 1.3
plt.rcParams["ytick.major.width"] = 1.3

# =========================================
# HELPERS
# =========================================
def wrap_term(term, width=WRAP_WIDTH):
    return "\n".join(textwrap.wrap(str(term), width=width))

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

def parse_overlap(overlap):
    try:
        left, right = str(overlap).split("/")
        count = int(left)
        total = int(right)
        signal = count / total if total > 0 else np.nan
        return count, total, signal
    except Exception:
        return np.nan, np.nan, np.nan

def prepare_df(df):
    df = df.copy()

    if "Adjusted P-value" not in df.columns:
        raise ValueError("Missing 'Adjusted P-value' column")

    df["Adjusted P-value"] = pd.to_numeric(df["Adjusted P-value"], errors="coerce")
    df = df.dropna(subset=["Term", "Adjusted P-value"]).copy()

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
    df["Label"] = df["CleanTerm"].apply(wrap_term)

    if df["Signal"].isna().all():
        df["Signal"] = -np.log10(df["Adjusted P-value"])

    return df

def scale_sizes(values, min_size=MIN_SIZE, max_size=MAX_SIZE):
    vals = pd.Series(values).astype(float)

    if vals.nunique(dropna=True) <= 1:
        return pd.Series(np.repeat((min_size + max_size) / 2, len(vals)), index=vals.index)

    vmin = vals.min()
    vmax = vals.max()
    return min_size + (vals - vmin) * (max_size - min_size) / (vmax - vmin)

def pick_size_legend_values(values):
    vals = sorted(pd.Series(values).dropna().astype(int).unique())
    if len(vals) == 0:
        return []
    if len(vals) <= 4:
        return vals
    picks = [
        vals[0],
        int(np.percentile(vals, 35)),
        int(np.percentile(vals, 65)),
        vals[-1]
    ]
    return sorted(list(dict.fromkeys(picks)))

def build_combined_scale(human_top, mouse_top):
    combined_counts = pd.concat([human_top["GeneCount"], mouse_top["GeneCount"]], ignore_index=True)
    combined_sizes = scale_sizes(combined_counts)
    split_idx = len(mouse_top)
    mouse_sizes = combined_sizes.iloc[:split_idx].reset_index(drop=True)
    human_sizes = combined_sizes.iloc[split_idx:].reset_index(drop=True)
    return mouse_sizes, human_sizes

def plot_mirrored_dotplot(human_top, mouse_top, title, out_plot):
    # reverse already handled before calling if needed
    n_mouse = len(mouse_top)
    n_human = len(human_top)

    mouse_y = np.arange(n_mouse)
    human_y = np.arange(n_human) + n_mouse + 2

    # shared size scaling across both sides
    mouse_sizes, human_sizes = build_combined_scale(human_top, mouse_top)

    fig = plt.figure(figsize=(18, 13))

    # main axis
    ax = fig.add_axes([0.24, 0.10, 0.48, 0.80])

    # colorbar axis - moved farther right
    cax = fig.add_axes([0.76, 0.18, 0.022, 0.66])

    # size legend axis - moved farther right than colorbar
    lax = fig.add_axes([0.84, 0.28, 0.13, 0.34])
    lax.axis("off")

    # lollipop lines
    for y, signal in zip(mouse_y, mouse_top["Signal"]):
        ax.hlines(y=y, xmin=-signal, xmax=0, color="#b9c3ce", linewidth=2.0, alpha=0.72, zorder=1)

    for y, signal in zip(human_y, human_top["Signal"]):
        ax.hlines(y=y, xmin=0, xmax=signal, color="#b9c3ce", linewidth=2.0, alpha=0.72, zorder=1)

    # dots
    mouse_sc = ax.scatter(
        -mouse_top["Signal"],
        mouse_y,
        s=mouse_sizes,
        c=mouse_top["Adjusted P-value"],
        cmap="YlGnBu_r",
        edgecolors="#7a8794",
        linewidths=1.0,
        alpha=0.98,
        zorder=3
    )

    ax.scatter(
        human_top["Signal"],
        human_y,
        s=human_sizes,
        c=human_top["Adjusted P-value"],
        cmap="YlGnBu_r",
        edgecolors="#7a8794",
        linewidths=1.0,
        alpha=0.98,
        zorder=3
    )

    # y labels
    y_labels = list(mouse_top["Label"]) + list(human_top["Label"])
    ax.set_yticks(list(mouse_y) + list(human_y))
    ax.set_yticklabels(y_labels, fontsize=12, fontweight="bold")

    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")

    ax.tick_params(axis="y", pad=10)
    ax.tick_params(axis="x", labelsize=12)

    # center line
    ax.axvline(0, color="black", linewidth=1.8)

    # absolute x tick labels
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x):.2f}" for x in xticks], fontsize=12, fontweight="bold")

    ax.set_xlabel("Signal (overlap count / pathway size)", fontsize=16, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=22, fontweight="bold", pad=16)

    mouse_max = mouse_top["Signal"].max() if len(mouse_top) else 1
    human_max = human_top["Signal"].max() if len(human_top) else 1
    xlim = max(mouse_max, human_max) * 1.28
    ax.set_xlim(-xlim, xlim)

    ax.text(
        -xlim * 0.58,
        max(n_mouse - 0.35, 0),
        "Mouse-enriched",
        fontsize=16,
        fontweight="bold",
        color="#2e8b57",
        ha="center"
    )

    ax.text(
        xlim * 0.58,
        n_mouse + 1.8 + max(n_human - 0.35, 0),
        "Human-enriched",
        fontsize=16,
        fontweight="bold",
        color="#d97706",
        ha="center"
    )

    ax.grid(axis="x", linestyle="--", alpha=0.20)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    # colorbar
    cbar = plt.colorbar(mouse_sc, cax=cax)
    cbar.set_label("FDR", fontsize=13, fontweight="bold", labelpad=12)
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.invert_yaxis()

    # size legend
    legend_counts = pick_size_legend_values(
        pd.concat([mouse_top["GeneCount"], human_top["GeneCount"]], ignore_index=True)
    )

    if len(legend_counts) > 0:
        handles = []
        labels = []
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
            labelspacing=1.5,
            borderpad=0.6,
            handletextpad=1.0
        )
        leg.get_title().set_fontweight("bold")

    plt.savefig(out_plot, dpi=600, bbox_inches="tight")
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

    # strongest on top
    human_top = human_top.iloc[::-1].reset_index(drop=True)
    mouse_top = mouse_top.iloc[::-1].reset_index(drop=True)

    plot_mirrored_dotplot(human_top, mouse_top, title, out_plot)

    print(f"Saved: {out_plot}")
    print("\nTop mouse terms:")
    cols = [c for c in ["CleanTerm", "Adjusted P-value", "Overlap", "Signal"] if c in mouse_top.columns]
    print(mouse_top[cols].head(top_n))
    print("\nTop human terms:")
    cols = [c for c in ["CleanTerm", "Adjusted P-value", "Overlap", "Signal"] if c in human_top.columns]
    print(human_top[cols].head(top_n))
    print("-" * 80)

# =========================================
# RUN ALL
# =========================================
for human_file, mouse_file, out_plot, title, top_n in comparisons:
    run_one_comparison(human_file, mouse_file, out_plot, title, top_n)

print("All cross-species enrichment comparison figures completed.")
