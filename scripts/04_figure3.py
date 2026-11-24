"""Figure 3: permutation null of the dominance projection ratio and observed value in the four populations
(reads the outputs of 01_projection.py). usage: python scripts/04_figure3.py"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hybridproj import data

OUT = data.ROOT / "results"
t = pd.read_csv(OUT / "table1_projection.csv").set_index("population")
order = [("Spring wheat", " (Years 1 + 2)"), ("Winter wheat", " (Exp. I)"), ("Maize", ""), ("Rice", "")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(2, 2, figsize=(10, 6.9), sharex=True)
for ax, (pop, suffix), lab in zip(axes.ravel(), order, "abcd"):
    n = pd.read_csv(OUT / f"permutations_{pop.replace(' ', '_')}.csv").permuted_ratio.to_numpy()
    o = t.loc[pop, "observed"]
    ax.hist(n, bins=np.arange(n.min() - 0.001, n.max() + 0.002, 0.001), color="#1f77b4", edgecolor="none")
    ax.axvline(o, color="black", ls="--", lw=2)
    ax.set_xlim(0, 1); ax.set_title(pop + suffix, fontsize=13)
    ax.text((n.mean() + o) / 2, 0.95, f"Observed = {o:.3f}\nNull mean = {n.mean():.3f}\nB = 1,000", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9.5, bbox=dict(boxstyle="round", fc="white", ec="0.8"))
    ax.text(-0.13, 1.06, f"({lab})", transform=ax.transAxes, fontweight="bold", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="0.9"); ax.set_axisbelow(True)
for ax in axes[:, 0]: ax.set_ylabel("Permutation frequency")
for ax in axes[1]: ax.set_xlabel("Projection ratio")
fig.tight_layout(); fig.savefig(OUT / "Figure3.png", dpi=300); fig.savefig(OUT / "Figure3.pdf")
print("saved results/Figure3.png")
