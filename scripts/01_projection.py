"""Table 1 and Figure 3: projection of the dominance/heterozygosity marker structure of the hybrids onto the
centered parental main-effect (GCA-contrast) space, with a permutation null (B = 1,000).

Spring wheat: Years 1 and 2 combined (hybrid-year records); other populations: one row per hybrid.
Also reports the additive positive control, the center-only sensitivity estimate and, for winter wheat,
the ratio of the dominance relationship matrix supplied with the source study and of the same coding
rebuilt from the parental SNPs.

usage: python scripts/01_projection.py
outputs: results/table1_projection.csv, results/permutations_<population>.csv"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from hybridproj import core, data

OUT = data.ROOT / "results"; OUT.mkdir(exist_ok=True)
SEEDS = {"Spring wheat": 123, "Winter wheat": 20260906, "Maize": 20260912, "Rice": 20260913}


def source_study_dominance(xg):
    """Genotypic dominance coding of the winter wheat source study (code across Exp.R of
    github.com/ysz98/Genomic-prediction-of-hybrid-wheat); genotype codes -1, -0.5, 0, 0.5, 1."""
    u = np.zeros_like(xg); n = len(xg)
    for j in range(xg.shape[1]):
        x = xg[:, j]
        p11 = ((x == -1).sum() + .5 * (x == -.5).sum()) / n
        p22 = ((x == 1).sum() + .5 * (x == .5).sum()) / n
        p12 = ((x == 0).sum() + .5 * (x == -.5).sum() + .5 * (x == .5).sum()) / n
        th = p11 + p22 - (p11 - p22) ** 2
        if th == 0:
            continue
        d11, d12, d22 = -2 * p12 * p22 / th, 4 * p11 * p22 / th, -2 * p11 * p12 / th
        v = np.zeros(n); v[x == -1] = d11; v[x == 0] = d12; v[x == 1] = d22; v[x == -.5] = (d11 + d12) / 2; v[x == .5] = (d12 + d22) / 2
        u[:, j] = v
    return u


rows = []
sets = {"Spring wheat": data.spring_wheat(), "Winter wheat": data.winter_wheat(), "Maize": data.maize(), "Rice": data.rice()}
for pop, ss in sets.items():
    p1 = np.concatenate([s.p1 for s in ss]); p2 = np.concatenate([s.p2 for s in ss])
    fa = np.vstack([s.projection_features()[0] for s in ss]); fd = np.vstack([s.projection_features()[1] for s in ss])
    P, rank = core.gca_projector(p1, p2)
    za = core.center(fa, scale=True); zd = core.center(fd, scale=True)
    res, null = core.projection_test(P, zd @ zd.T, B=1000, seed=SEEDS[pop])
    zc = core.center(fd, scale=False)
    rows.append(dict(population=pop, units=len(p1), gca_rank=rank, additive_control=core.projection_ratio(P, za),
                     **res, center_only=core.projection_ratio(P, zc)))
    pd.DataFrame({"permuted_ratio": null}).to_csv(OUT / f"permutations_{pop.replace(' ', '_')}.csv", index=False)
    print(pop, {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)

# winter wheat: dominance coding of the source study (supplied matrix, and rebuilt from the SNPs)
ww = sets["Winter wheat"][0]
P, _ = core.gca_projector(ww.p1, ww.p2)
d = data.DATA / "winter_wheat"
ph = pd.read_csv(d / "Yield within Exp I.txt", sep=r"\s+"); hyb = (ph.Type != "Line").to_numpy()
Dm = pd.read_csv(d / "Dominance relationship matrix for Exp I.txt", sep=r"\s+", index_col=0).to_numpy(float)[np.ix_(hyb, hyb)]
gp = pd.read_csv(d / "ExpI_parents_gametic_p.csv", index_col=0)
lines = ph.Genotype[~hyb].values
xg = np.vstack([2 * gp.loc[lines].to_numpy() - 1, ww.g1 + ww.g2 - 1])
xg = np.round(xg * 2) / 2
U = source_study_dominance(xg)[len(lines):]
for lab, K in (("Winter wheat, supplied dominance matrix", Dm), ("Winter wheat, source-study coding rebuilt from SNPs", U @ U.T)):
    res, _ = core.projection_test(P, K, B=1000, seed=SEEDS["Winter wheat"])
    rows.append(dict(population=lab, units=len(ww.p1), gca_rank=133, **res)); print(lab, round(res["observed"], 4))

t = pd.DataFrame(rows)
t.to_csv(OUT / "table1_projection.csv", index=False)
print(t.round(4).to_string(index=False))
