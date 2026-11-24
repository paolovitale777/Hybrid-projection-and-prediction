"""Core functions: genomic kernels, REML/BLUP fitting of the A, D and AD models, projection ratio,
permutation test, parent-structured partitions and selection metrics."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import spearmanr


# ----------------------------------------------------------------- kernels and features
def center(x, scale=False):
    """Center marker columns, drop zero-variance columns, optionally scale to unit variance."""
    z = np.asarray(x, dtype=float)
    z = z - z.mean(axis=0, keepdims=True)
    sd = z.std(axis=0)
    keep = sd > 1e-10
    z = z[:, keep]
    return z / sd[keep] if scale else z


def kernel_center(x):
    """Prediction kernel: centered features, ZZ' scaled to unit mean diagonal."""
    z = np.asarray(x, dtype=float)
    z = z - z.mean(axis=0, keepdims=True)
    z = z[:, np.sum(z * z, axis=0) > 1e-8]
    k = z @ z.T
    return k / np.mean(np.diag(k))


# ----------------------------------------------------------------- REML / BLUP
def profile_reml(v, y):
    """Profile restricted log-likelihood (up to constants) of y ~ N(mu 1, sigma2 V)."""
    try:
        cf = cho_factor(v, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf, None
    one = np.ones(y.size)
    vi_y = cho_solve(cf, y, check_finite=False)
    vi_1 = cho_solve(cf, one, check_finite=False)
    den = float(one @ vi_1)
    if not np.isfinite(den) or den <= 0:
        return np.inf, None
    mu = float(one @ vi_y / den)
    resid = y - mu
    vi_r = cho_solve(cf, resid, check_finite=False)
    q = float(resid @ vi_r)
    if not np.isfinite(q) or q <= 0:
        return np.inf, None
    logdet = 2.0 * np.log(np.diag(cf[0])).sum()
    n_eff = y.size - 1
    return float(logdet + np.log(den) + n_eff * np.log(q / n_eff)), (mu, vi_r)


def fit_one(k, y):
    """Single-kernel model y = mu + g + e, g ~ N(0, K s2g); variance ratio by REML."""
    eye = np.eye(y.size)
    opt = minimize_scalar(lambda t: profile_reml(k + np.exp(float(t)) * eye, y)[0],
                          bounds=(-9.0, 9.0), method="bounded", options={"xatol": 2e-3})
    return 1.0, profile_reml(k + float(np.exp(opt.x)) * eye, y)[1]


def fit_two(km, kd, y):
    """Additive-dominance model with kernel w KA + (1 - w) KD; w and the variance ratio by REML."""
    eye = np.eye(y.size)

    def unpack(z):
        return 1.0 / (1.0 + np.exp(-float(z[0]))), np.exp(float(z[1]))

    def obj(z):
        w, lam = unpack(z)
        return profile_reml(w * km + (1 - w) * kd + lam * eye, y)[0]

    opt = minimize(obj, np.array([0.0, 0.0]), method="L-BFGS-B", bounds=[(-8.0, 8.0), (-9.0, 9.0)],
                   options={"ftol": 1e-8, "maxiter": 100})
    w, lam = unpack(opt.x)
    return w, profile_reml(w * km + (1 - w) * kd + lam * eye, y)[1]


def fit_all(km, kd, y, tr):
    """Fit A, D and AD on the training hybrids tr."""
    ka, kdd = km[np.ix_(tr, tr)], kd[np.ix_(tr, tr)]
    return {"A": fit_one(ka, y), "D": fit_one(kdd, y), "AD": fit_two(ka, kdd, y)}


def predict(fits, km, kd, te, tr):
    """BLUP of the hybrids te from the models fitted on tr."""
    out = {}
    for m, (w, (mu, vi_r)) in fits.items():
        if m == "A":
            cross = km[np.ix_(te, tr)]
        elif m == "D":
            cross = kd[np.ix_(te, tr)]
        else:
            cross = w * km[np.ix_(te, tr)] + (1 - w) * kd[np.ix_(te, tr)]
        out[m] = mu + cross @ vi_r
    return out


def cor(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 10 else np.nan


# ----------------------------------------------------------------- projection
def gca_projector(p1, p2):
    """Orthogonal projector onto the centered parental main-effect (GCA-contrast) space."""
    x = np.column_stack([pd.get_dummies(pd.Series(p1)).to_numpy(float), pd.get_dummies(pd.Series(p2)).to_numpy(float)])
    x -= x.mean(axis=0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    u = u[:, : int((s > 1e-8 * s.max()).sum())]
    return u @ u.T, u.shape[1]


def projection_test(P, K, B=1000, seed=123):
    """Projection ratio tr(P K)/tr(K) (main-text Eq. 7) and its permutation null:
    hybrid labels permuted relative to the fixed parental design (rows and columns of K together)."""
    tr = np.trace(K)
    obs = float(np.sum(P * K) / tr)
    rng = np.random.default_rng(seed)
    null = np.empty(B)
    for b in range(B):
        i = rng.permutation(K.shape[0])
        null[b] = np.sum(P * K[np.ix_(i, i)]) / tr
    return dict(observed=obs, null_mean=float(null.mean()), null_lo=float(np.quantile(null, 0.025)),
                null_hi=float(np.quantile(null, 0.975)), null_max=float(null.max()),
                P_emp=float((1 + np.sum(null >= obs)) / (B + 1)), B=B, seed=seed), null


def projection_ratio(P, Z):
    k = Z @ Z.T
    return float(np.sum(P * k) / np.trace(k))


# ----------------------------------------------------------------- parent-structured partitions
def parent_partitions(p1, p2, n_part, seed, frac_tested=2 / 3, train_frac=0.85):
    """Two thirds of the parents of each group tested; training = 85% of the hybrids with both parents
    tested; test sets T2 (both tested), T1 (one tested), T0 (neither tested)."""
    rng = np.random.default_rng(seed)
    u1, u2 = np.unique(p1), np.unique(p2)
    k1 = min(max(int(round(frac_tested * len(u1))), 1), len(u1) - 1)
    k2 = min(max(int(round(frac_tested * len(u2))), 1), len(u2) - 1)
    parts = []
    for _ in range(n_part):
        t1 = rng.choice(u1, k1, replace=False)
        t2 = rng.choice(u2, k2, replace=False)
        s = np.isin(p1, t1).astype(int) + np.isin(p2, t2).astype(int)
        tt = np.where(s == 2)[0]
        rng.shuffle(tt)
        ntr = int(round(train_frac * len(tt)))
        parts.append(dict(train=np.sort(tt[:ntr]), T2=np.sort(tt[ntr:]), T1=np.where(s == 1)[0], T0=np.where(s == 0)[0]))
    return parts, (len(u1), len(u2), k1, k2)


# ----------------------------------------------------------------- selection metrics
def top(v, k):
    return np.argsort(-v, kind="stable")[:k]


def selection_metrics(obs, pa, pad, frac=0.10):
    """Spearman A-AD, top-10% overlap and observed means of the hybrids selected by A and AD."""
    k = int(np.ceil(frac * len(obs)))
    ta, tad = top(pa, k), top(pad, k)
    return dict(spearman_A_AD=float(spearmanr(pa, pad)[0]), overlap_top10=len(np.intersect1d(ta, tad)) / k,
                obs_top10_A=float(obs[ta].mean()), obs_top10_AD=float(obs[tad].mean()), mean_all=float(obs.mean()))


def parent_rankings(p1, p2, ga, gad, labels=("Female", "Male")):
    """Parents scored by the mean genomic value of their hybrids under A and AD."""
    fr = pd.DataFrame({"p1": p1, "p2": p2, "gA": ga, "gAD": gad})
    rec = {}
    for side, lab in zip(("p1", "p2"), labels):
        m = fr.groupby(side)[["gA", "gAD"]].mean()
        k = max(1, int(np.ceil(0.10 * len(m))))
        ra, rad = m.gA.rank(ascending=False), m.gAD.rank(ascending=False)
        rec[f"{lab}_n"] = len(m)
        rec[f"{lab}_spearman"] = float(spearmanr(m.gA, m.gAD)[0]) if len(m) > 2 else np.nan
        rec[f"{lab}_top10_overlap"] = len(set(ra.nsmallest(k).index) & set(rad.nsmallest(k).index)) / k
        rec[f"{lab}_best_A"] = ra.idxmin()
        rec[f"{lab}_best_AD"] = rad.idxmin()
    return rec
