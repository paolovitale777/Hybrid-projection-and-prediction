"""Loaders for the four hybrid populations. Each returns a list of analysis sets
(spring wheat: one per year), with parental identities, parental genotypes per hybrid,
phenotypes and the random 70/30 partitions used in Tables 2 and 3."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, PARTS = ROOT / "data", ROOT / "partitions"
RICE_TRAITS = ["GY", "TGW", "PN", "PH", "SB", "GN", "PL", "PB"]


def _perm_splits(path, n):
    """Partition file: one row per partition, a 1-based permutation of the hybrids; first 70% = training."""
    perms = pd.read_csv(path).to_numpy() - 1
    nt = int(np.floor(0.70 * n))
    return [(p[:nt], p[nt:]) for p in perms]


class PopSet:
    """One analysis set. Genotypes are stored per hybrid as parental codes g1, g2:
    gametic probabilities of the first allele (spring and winter wheat), 0/1 allele codes (maize)
    or -1/+1 codes (rice)."""

    def __init__(self, population, label, p1, p2, g1, g2, traits, splits, coding, groups):
        self.population, self.label, self.p1, self.p2 = population, label, np.asarray(p1), np.asarray(p2)
        self.g1, self.g2, self.traits, self.splits, self.coding, self.groups = g1, g2, traits, splits, coding, groups

    def prediction_features(self):
        """Hybrid additive and dominance features used for the prediction kernels (Tables 2 and 3)."""
        a, b = self.g1, self.g2
        if self.coding == "gametic":                      # expected F1 dosage and heterozygosity
            return a + b, a + b - 2 * a * b
        if self.coding == "01":                           # maize: parental allele codes 0/1
            return a + b, (a != b).astype(float)
        return 0.5 * (a + b), 0.5 * (1.0 - a * b)         # rice: -1/+1 codes

    def projection_features(self):
        """Hybrid additive and dominance/heterozygosity features used for the projection (Table 1)."""
        a, b = self.g1, self.g2
        if self.coding == "gametic":
            return a + b, a + b - 2 * a * b
        return a + b, (a != b).astype(float)              # maize and rice: heterozygous = parental alleles differ


def winter_wheat():
    d = DATA / "winter_wheat"
    ph = pd.read_csv(d / "Yield within Exp I.txt", sep=r"\s+")
    h = ph[ph.Type != "Line"].reset_index(drop=True)
    p = pd.read_csv(d / "ExpI_parents_gametic_p.csv", index_col=0)          # built by scripts/00_prepare_winter_wheat.py
    splits = _perm_splits(PARTS / "winter_wheat_partitions_100.csv", len(h))
    return [PopSet("Winter wheat", "GY", h.Female.values, h.Male.values, p.loc[h.Female].to_numpy(), p.loc[h.Male].to_numpy(),
                   {"GY": h.Yield.to_numpy(float)}, splits, "gametic", ("Female", "Male"))]


def maize():
    d = DATA / "maize"
    dent = pd.read_csv(d / "genetics.114.165860-6.xls", index_col=0)
    flint = pd.read_csv(d / "genetics.114.165860-7.xls", index_col=0)
    ph = pd.read_csv(d / "genetics.114.165860-9.xls", sep="\t")
    dent.index, flint.index = dent.index.astype(str), flint.index.astype(str)
    p1, p2 = ph.dent.astype(str).values, ph.flint.astype(str).values
    splits = _perm_splits(PARTS / "maize_partitions_100.csv", len(ph))
    return [PopSet("Maize", "GY, GM", p1, p2, dent.loc[p1].to_numpy(float), flint.loc[p2].to_numpy(float),
                   {t: ph[t].to_numpy(float) for t in ("GY", "GM")}, splits, "01", ("Dent", "Flint"))]


def rice():
    d = DATA / "rice"
    phe = pd.read_csv(d / "rice_phe.csv")                                    # built by scripts/00_prepare_rice.R
    par = pd.read_csv(d / "rice_parents_10k.csv", index_col=0)
    par.index = par.index.astype(str)
    p1 = phe.hybrid.str.split("/").str[0].values
    p2 = phe.hybrid.str.split("/").str[1].values
    splits = _perm_splits(PARTS / "rice_partitions_100.csv", len(phe))
    return [PopSet("Rice", "8 traits", p1, p2, par.loc[p1].to_numpy(float), par.loc[p2].to_numpy(float),
                   {t: phe[t].to_numpy(float) for t in RICE_TRAITS}, splits, "pm1", ("Inbred line", "Male-sterile line"))]


def spring_wheat():
    d = DATA / "spring_wheat"
    snp = pd.read_csv(d / "spring_wheat_parent_snps.csv.gz", index_col=0)    # markers x parental GIDs, 0/1/2
    snp.columns = snp.columns.astype(str)
    out = []
    for yr in (1, 2):
        c = pd.read_csv(d / f"spring_wheat_year{yr}_cohort.csv")
        n = len(c)
        parts = pd.read_csv(d / f"spring_wheat_year{yr}_partitions.csv")
        splits = []
        for s in parts.test_index_1based:
            te = np.array([int(v) - 1 for v in s.split()])
            splits.append((np.setdiff1d(np.arange(n), te), te))
        f, m = c.F_GID.astype(str).values, c.M_GID.astype(str).values
        out.append(PopSet("Spring wheat", f"GY (Yr {yr})", f, m, snp[f].to_numpy(float).T / 2, snp[m].to_numpy(float).T / 2,
                          {f"GY (Yr {yr})": c.GY.to_numpy(float)}, splits, "gametic", ("Female", "Male")))
    return out


LOADERS = {"spring_wheat": spring_wheat, "winter_wheat": winter_wheat, "maize": maize, "rice": rice}
