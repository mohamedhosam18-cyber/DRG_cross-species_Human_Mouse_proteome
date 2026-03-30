import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import textwrap

# =========================================
# SETTINGS
# =========================================
TOP_N = 8
FDR_CUTOFF = 0.25
WRAP_WIDTH = 40

# =========================================
# PATHS
# =========================================
base = Path("~/Desktop/DRG_cross_species_FINAL").expanduser()
results_dir = base / "results"
fig_dir = base / "figures" / "gsea"
fig_dir.mkdir(parents=True, exist_ok=True)

gsea_dirs = {
    "GO_BP": results_dir / "GSEA_GO_BP",
    "KEGG": results_dir / "GSEA_KEGG",
    "Reactome": results_dir / "GSEA_Reactome",
    "WikiPathways": results_dir / "GSEA_WikiPathways",
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

def wrap_label(term, width=WRAP_WIDTH):
    return "\n".join(textwrap.wrap(str(term), width=width))

def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
        spine.set_color("black")
    ax.tick_params(axis="x", labelsize=12, width=1.4)
    ax.tick_params(axis="y", labelsize=12, width=1.4)

def bold_ylabels(ax):
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")
        lab.set_fontsize(12)

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

def load_gsea_table(folder: Path):
    report_file = find_report_file(folder)
    df = pd.read_csv(report_file)

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
    if "ES" in out.columns:
        out["ES"] = pd.to_numeric(out["ES"], errors="coerce")

    out = out.dropna(subset=["Term", "NES", "FDR"]).copy()
    out["Term"] = out["Term"].apply(clean_term)
    out["WrappedTerm"] = out["Term"].apply(wrap_label)
    out["Direction"] = np.where(out["NES"] > 0, "Human-enriched", "Mouse-enriched")
    out["absNES"] = out["NES"].abs()

    return out, report_file

def save_filtered_tables(df, prefix):
    sig = df[df["FDR"] < FDR_CUTOFF].copy()
    sig = sig.sort_values(["FDR", "absNES"], ascending=[True, False]).copy()
    out_file = results_dir / f"{prefix}_GSEA_filtered.csv"
    sig.to_csv(out_file, index=False)
    print("Saved filtered table:", out_file)

def get_top_terms(df):
    sig = df[df["FDR"] < FDR_CUTOFF].copy()

    if len(sig) < 4:
        sig = df.sort_values("FDR", ascending=True).head(16).copy()

    human = sig[sig["NES"] > 0].sort_values("NES", ascending=False).head(TOP_N).copy()
    mouse = sig[sig["NES"] < 0].sort_values("NES", ascending=True).head(TOP_N).copy()

    mouse = mouse.iloc[::-1].reset_index(drop=True)
    human = human.iloc[::-1].reset_index(drop=True)

    return human, mouse

# =========================================
# FINAL MIRRORED BARPLOT
# =========================================
def make_barplot(df, title, outfile):
    human, mouse = get_top_terms(df)

    mouse_y = np.arange(len(mouse))
    human_y = np.arange(len(human)) + len(mouse) + 2

    fig, ax = plt.subplots(figsize=(18, 11))

    human_color = "#cc6f12"
    mouse_color = "#2f8a57"

    if len(mouse) > 0:
        ax.barh(
            mouse_y,
            mouse["NES"],
            color=mouse_color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.95
        )

    if len(human) > 0:
        ax.barh(
            human_y,
            human["NES"],
            color=human_color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.95
        )

    all_labels = list(mouse["WrappedTerm"]) + list(human["WrappedTerm"])
    all_y = list(mouse_y) + list(human_y)

    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels)
    bold_ylabels(ax)

    ax.axvline(0, color="black", linewidth=2.2)

    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=18, weight="bold")
    ax.set_title(title, fontsize=23, weight="bold", pad=18)

    ax.grid(axis="x", linestyle="--", alpha=0.18)
    ax.set_axisbelow(True)
    style_axes(ax)

    legend_handles = [
        Patch(facecolor=human_color, edgecolor="black", label="Human-enriched (NES > 0)"),
        Patch(facecolor=mouse_color, edgecolor="black", label="Mouse-enriched (NES < 0)")
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
        frameon=True,
        fontsize=12
    )
    leg.get_frame().set_linewidth(1.5)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_alpha(1.0)

    plt.tight_layout()
    plt.savefig(outfile, dpi=700, bbox_inches="tight")
    plt.close()

# =========================================
# RUN
# =========================================
for label, folder in gsea_dirs.items():
    df, report_file = load_gsea_table(folder)

    print(f"\n=== {label} ===")
    print("Report file:", report_file)
    print("Input shape:", df.shape)
    print(df[["Term", "NES", "FDR"]].head(5))

    save_filtered_tables(df, label)

    make_barplot(
        df,
        title=f"Cross-species GSEA summary: {label}",
        outfile=fig_dir / f"final_GSEA_{label}_barplot.png"
    )

    print("Saved barplot:", fig_dir / f"final_GSEA_{label}_barplot.png")

print("\nFinal GSEA barplots completed.")
