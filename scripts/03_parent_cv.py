"""Supplementary Table S4: parent-structured validation. In each of 100 partitions two thirds of the parents of each
group are tested; training = 85% of the hybrids with both parents tested (hybrid phenotypes only); test sets with
both (T2), one (T1) or neither (T0) parent tested. Same kernels, models and selection metrics as Tables 2 and 3.

usage: python scripts/03_parent_cv.py [spring_wheat winter_wheat maize rice]
outputs: results/parent_cv_<population>_by_partition.csv, results/tableS4_<population>.csv"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from hybridproj import core, data

OUT = data.ROOT / "results"; OUT.mkdir(exist_ok=True)
SEEDS = {"winter_wheat": 20261124, "maize": 20261125, "rice": 20261126, "spring_wheat": 20261127}
for key in sys.argv[1:] or list(data.LOADERS):
    rows = []
    for i, s in enumerate(data.LOADERS[key]()):
        fa, fd = s.prediction_features()
        km, kd = core.kernel_center(fa), core.kernel_center(fd)
        parts, info = core.parent_partitions(s.p1, s.p2, 100, SEEDS[key] + i)
        for trait, y in s.traits.items():
            for k, P in enumerate(parts, 1):
                tr = P["train"]; fits = core.fit_all(km, kd, y[tr], tr)
                for sc in ("T2", "T1", "T0"):
                    te = P[sc]
                    if len(te) < 10:
                        continue
                    p = core.predict(fits, km, kd, te, tr); obs = y[te]
                    rows.append(dict(population=s.population, trait=trait, scenario=sc, partition=k, n_train=len(tr), n_test=len(te),
                                     COR_A=core.cor(obs, p["A"]), COR_D=core.cor(obs, p["D"]), COR_AD=core.cor(obs, p["AD"]),
                                     **core.selection_metrics(obs, p["A"], p["AD"])))
            print(s.population, trait, "done", flush=True)
    r = pd.DataFrame(rows); r.to_csv(OUT / f"parent_cv_{key}_by_partition.csv", index=False)
    summ = []
    for (pop, trait, sc), g in r.groupby(["population", "trait", "scenario"], sort=False):
        d = g.COR_AD - g.COR_A; n = len(d); h = tdist.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        sel = g.obs_top10_AD - g.obs_top10_A
        summ.append(dict(population=pop, trait=trait, scenario=sc, partitions=n, n_test=g.n_test.mean(), COR_A=g.COR_A.mean(),
                         COR_D=g.COR_D.mean(), COR_AD=g.COR_AD.mean(), AD_minus_A=d.mean(), AD_minus_A_lo=d.mean() - h,
                         AD_minus_A_hi=d.mean() + h, spearman_A_AD=g.spearman_A_AD.mean(), overlap_top10=g.overlap_top10.mean(),
                         selection_AD_minus_A=sel.mean(), selection_lo=sel.quantile(0.025), selection_hi=sel.quantile(0.975),
                         partitions_AD_better=int((sel > 0).sum())))
    s = pd.DataFrame(summ); s.to_csv(OUT / f"tableS4_{key}.csv", index=False)
    pd.set_option("display.width", 250); print(s.round(4).to_string(index=False))
