"""Tables 2 and 3: A, D and AD models (REML/BLUP) under 100 random 70/30 partitions of the hybrids,
selection comparison of A and AD in each held-out set, and parental rankings with all hybrids.

usage: python scripts/02_random_cv.py [spring_wheat winter_wheat maize rice]
outputs: results/random_cv_<population>_by_partition.csv, results/table2_table3_<population>.csv,
         results/parent_rankings_<population>.csv"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from hybridproj import core, data

OUT = data.ROOT / "results"; OUT.mkdir(exist_ok=True)
pops = sys.argv[1:] or list(data.LOADERS)
for key in pops:
    rows, prow = [], []
    for s in data.LOADERS[key]():
        fa, fd = s.prediction_features()
        km, kd = core.kernel_center(fa), core.kernel_center(fd)
        for trait, y in s.traits.items():
            for k, (tr, te) in enumerate(s.splits, 1):
                fits = core.fit_all(km, kd, y[tr], tr)
                p = core.predict(fits, km, kd, te, tr); obs = y[te]
                rows.append(dict(population=s.population, trait=trait, partition=k, n_train=len(tr), n_test=len(te),
                                 COR_A=core.cor(obs, p["A"]), COR_D=core.cor(obs, p["D"]), COR_AD=core.cor(obs, p["AD"]),
                                 weight_A_in_AD=fits["AD"][0], **core.selection_metrics(obs, p["A"], p["AD"])))
            allh = np.arange(len(y))
            fits = core.fit_all(km, kd, y, allh); g = core.predict(fits, km, kd, allh, allh)
            prow.append(dict(population=s.population, trait=trait, **core.parent_rankings(s.p1, s.p2, g["A"], g["AD"], s.groups)))
            print(s.population, trait, "done", flush=True)
    r = pd.DataFrame(rows); r.to_csv(OUT / f"random_cv_{key}_by_partition.csv", index=False)
    pr = pd.DataFrame(prow); pr.to_csv(OUT / f"parent_rankings_{key}.csv", index=False)
    summ = []
    for (pop, trait), g in r.groupby(["population", "trait"], sort=False):
        d = g.COR_AD - g.COR_A; n = len(d); h = tdist.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        sel = g.obs_top10_AD - g.obs_top10_A
        q = pr[(pr.population == pop) & (pr.trait == trait)].iloc[0]
        summ.append(dict(population=pop, trait=trait, partitions=n, COR_A=g.COR_A.mean(), COR_D=g.COR_D.mean(), COR_AD=g.COR_AD.mean(),
                         AD_minus_A=d.mean(), AD_minus_A_lo=d.mean() - h, AD_minus_A_hi=d.mean() + h,
                         spearman_A_AD=g.spearman_A_AD.mean(), overlap_top10=g.overlap_top10.mean(),
                         obs_top10_A=g.obs_top10_A.mean(), obs_top10_AD=g.obs_top10_AD.mean(),
                         selection_AD_minus_A=sel.mean(), selection_lo=sel.quantile(0.025), selection_hi=sel.quantile(0.975),
                         partitions_AD_better=int((sel > 0).sum()),
                         parent_rank_correlation=min(q[c] for c in q.index if c.endswith("_spearman") and pd.notna(q[c])),
                         parent_top10_overlap=min(q[c] for c in q.index if c.endswith("_top10_overlap"))))
    s = pd.DataFrame(summ); s.to_csv(OUT / f"table2_table3_{key}.csv", index=False)
    pd.set_option("display.width", 250); print(s.round(4).to_string(index=False))
