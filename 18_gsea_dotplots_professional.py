import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

# =========================================
# SETTINGS
# =========================================
TOP_N = 8
FDR_CUTOFF = 0.25
WRAP_WIDTH = 38
MIN_SIZE = 120
MAX_SIZE = 900

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
results_dir = base / "results"
fig_dir = base / "figures" / "gsea_dotplots_professional"
fig_dir.mkdir(parents=True, exist_ok=True)

gsea_dirs = {
    "GO_BP": results_dir / "GSEA_GO_BP",
    "KEGG": results_dir / "GSEA_KEGG",
    "Reactome": results_dir / "GSEA_Reactome",
    "WikiPathways": results_dir / "GSEA_WikiPathways",
}

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

def wrap_label(term, width=WRAP_WIDTH):
    return "\n".join(textwrap.wrap(str(term), width=width))

def clean_term(term):
    term = str(term).replace("_", " ")

    if " (GO:" in term:
        term = term.split(" (GO:")[0]

    if " R-HSA-" in term:
        term = term.split(" R-HSA-")[0]

    if " WP" in term:
        parts = term.rsplit(" WP", 1)
        if len(parts) == 2:
            term = parts[0]

    return term.strip()

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

def count_leading_genes(x):
    if pd.isna(x):
        return np.nan
    genes = [g.strip() for g in str(x).split(";") if g.strip()]
    return len(genes)

def load_gsea_table(folder: Path):
    report_file = find_report_file(folder)
    df = pd.read_csv(report_file)

    # flexible column mapping
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "term":
            col_map[c] = "Term"
        elif cl == "es":
            col_map[c] = "ES"
        elif cl == "nes":
            col_map[c] = "NES"
        elif cl in ["fdr q-val", "fdr_q-val", "fdr"]:
            col_map[c] = "FDR"
        elif cl in ["lead_genes", "lead genes"]:
            col_map[c] = "Lead_genes"

    df = df.rename(columns=col_map)

    needed = ["Term", "NES", "FDR"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing GSEA columns {missing} in {report_file}")

    out = df.copy()
    out["NES"] = pd.to_numeric(out["NES"], errors="coerce")
    out["FDR"] = pd.to_numeric(out["FDR"], errors="coerce")

    out = out.dropna(subset=["Term", "NES", "FDR"]).copy()
    out["Term"] = out["Term"].apply(clean_term)
    out["WrappedTerm"] = out["Term"].apply(wrap_label)

    if "Lead_genes" in out.columns:
        out["LeadCount"] = out["Lead_genes"].apply(count_leading_genes)
    else:
        out["LeadCount"] = np.nan

    return out, report_file

def get_top_terms(df):
    sig = df[df["FDR"] < FDR_CUTOFF].copy()

    if len(sig) < 4:
        sig = df.sort_values("FDR", ascending=True).head(16).copy()

    human = sig[sig["NES"] > 0].sort_values(["FDR", "NES"], ascending=[True, False]).head(TOP_N).copy()
    mouse = sig[sig["NES"] < 0].sort_values(["FDR", "NES"], ascending=[True, True]).head(TOP_N).copy()

    mouse = mouse.iloc[::-1].reset_index(drop=True)
    human = human.iloc[::-1].reset_index(drop=True)

    return human, mouse

def build_combined_scale(human_top, mouse_top):
    combined_counts = pd.concat([mouse_top["LeadCount"], human_top["LeadCount"]], ignore_index=True)
    combined_sizes = scale_sizes(combined_counts)
    split_idx = len(mouse_top)
    mouse_sizes = combined_sizes.iloc[:split_idx].reset_index(drop=True)
    human_sizes = combined_sizes.iloc[split_idx:].reset_index(drop=True)
    return mouse_sizes, human_sizes

def plot_mirrored_gsea_dotplot(human_top, mouse_top, title, out_plot):
    n_mouse = len(mouse_top)
    n_human = len(human_top)

    mouse_y = np.arange(n_mouse)
    human_y = np.arange(n_human) + n_mouse + 2

    mouse_sizes, human_sizes = build_combined_scale(human_top, mouse_top)

    fig = plt.figure(figsize=(18, 13))

    # main axis
    ax = fig.add_axes([0.24, 0.10, 0.48, 0.80])

    # colorbar axis
    cax = fig.add_axes([0.76, 0.18, 0.022, 0.66])

    # size legend axis
    lax = fig.add_axes([0.85, 0.28, 0.11, 0.34])
    lax.axis("off")

    # lollipop lines
    for y, nes in zip(mouse_y, mouse_top["NES"].abs()):
        ax.hlines(y=y, xmin=-nes, xmax=0, color="#b9c3ce", linewidth=2.0, alpha=0.72, zorder=1)

    for y, nes in zip(human_y, human_top["NES"]):
        ax.hlines(y=y, xmin=0, xmax=nes, color="#b9c3ce", linewidth=2.0, alpha=0.72, zorder=1)

    # dots
    mouse_sc = ax.scatter(
        mouse_top["NES"],
        mouse_y,
        s=mouse_sizes,
        c=mouse_top["FDR"],
        cmap="YlGnBu_r",
        edgecolors="#7a8794",
        linewidths=1.0,
        alpha=0.98,
        zorder=3
    )

    ax.scatter(
        human_top["NES"],
        human_y,
        s=human_sizes,
        c=human_top["FDR"],
        cmap="YlGnBu_r",
        edgecolors="#7a8794",
        linewidths=1.0,
        alpha=0.98,
        zorder=3
    )

    # y labels
    y_labels = list(mouse_top["WrappedTerm"]) + list(human_top["WrappedTerm"])
    ax.set_yticks(list(mouse_y) + list(human_y))
    ax.set_yticklabels(y_labels, fontsize=12, fontweight="bold")

    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")

    ax.tick_params(axis="y", pad=10)
    ax.tick_params(axis="x", labelsize=12)

    # center line
    ax.axvline(0, color="black", linewidth=1.8)

    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=16, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=22, fontweight="bold", pad=16)

    max_abs_nes = max(
        mouse_top["NES"].abs().max() if len(mouse_top) else 1,
        human_top["NES"].abs().max() if len(human_top) else 1
    ) * 1.25
    ax.set_xlim(-max_abs_nes, max_abs_nes)

    ax.text(
        -max_abs_nes * 0.58,
        max(n_mouse - 0.35, 0),
        "Mouse-enriched",
        fontsize=16,
        fontweight="bold",
        color="#2e8b57",
        ha="center"
    )

    ax.text(
        max_abs_nes * 0.58,
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
        pd.concat([mouse_top["LeadCount"], human_top["LeadCount"]], ignore_index=True)
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
            title="Leading-edge\ngene count",
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

# =========================================
# RUN
# =========================================
for label, folder in gsea_dirs.items():
    df, report_file = load_gsea_table(folder)

    print(f"\n=== {label} ===")
    print("Report file:", report_file)
    print("Input shape:", df.shape)

    human_top, mouse_top = get_top_terms(df)

    plot_mirrored_gsea_dotplot(
        human_top,
        mouse_top,
        title=f"Cross-species GSEA summary: {label}",
        out_plot=fig_dir / f"final_GSEA_{label}_dotplot.png"
    )

    print("Saved dotplot:", fig_dir / f"final_GSEA_{label}_dotplot.png")

print("\nFinal GSEA dotplots completed.")
