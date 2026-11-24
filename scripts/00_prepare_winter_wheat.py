"""Winter wheat (Zhao et al. 2015, Exp. I): parental SNP genotypes -> gametic probabilities.

Input  data/winter_wheat/Supplementary_Tables.xlsx  (Dryad, doi:10.5061/dryad.461nc; sheet 'Supplementary Table S2',
       90k SNP array, two allele-presence rows per SNP, columns F001-F120 and M001-M015)
Output data/winter_wheat/ExpI_parents_gametic_p.csv  (parents x SNPs, probability of the first allele)
Check  relationships from these SNPs against the additive relationship matrix supplied with the source study."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from hybridproj.data import DATA
d = DATA / "winter_wheat"
x = pd.read_excel(d / "Supplementary_Tables.xlsx", sheet_name="Supplementary Table S2", header=1, usecols=range(136))
x = x.rename(columns={"Marker/Parent": "marker"}).dropna(subset=["marker"])
x["snp"] = x.marker.str.rsplit(".", n=1).str[0]
par = [c for c in x.columns if c not in ("marker", "snp")]
g = x.groupby("snp", sort=False)
assert (g.size() == 2).all()
first, second = g.nth(0).set_index("snp")[par], g.nth(1).set_index("snp")[par]
p = (first / (first + second).replace(0, np.nan)).T            # parents x SNPs
print(f"{p.shape[1]} SNPs, {p.shape[0]} parents, missing {p.isna().mean().mean():.4f}")
p = p.fillna(p.mean())
p = p.loc[:, (p.mean() > 0) & (p.mean() < 1)]
p.to_csv(d / "ExpI_parents_gametic_p.csv")
# identity check against the supplied additive relationship matrix (hybrids)
ph = pd.read_csv(d / "Yield within Exp I.txt", sep=r"\s+")
A = pd.read_csv(d / "Additive relationship matrix for Exp I.txt", sep=r"\s+", index_col=0).to_numpy(float)
H = np.where(ph.Type != "Line")[0]
xh = p.loc[ph.Female.values[H]].to_numpy() + p.loc[ph.Male.values[H]].to_numpy()
xh -= xh.mean(0)
G = xh @ xh.T
off = ~np.eye(len(H), dtype=bool)
print("correlation with the supplied additive matrix (hybrids, off-diagonal): %.3f" % np.corrcoef(A[np.ix_(H, H)][off], G[off])[0, 1])
