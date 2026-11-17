"""
eval_suite.py - evaluation suite for synthetic rare-class (tail) EHR data.

Compares a synthetic sample S against real data, following the literature review:
  - fidelity is measured against HELD-OUT real rare-class records (R), not the generator's training set (T)
  - every metric can be computed on a real-vs-real pair to get a reference value ("what good looks like")
  - privacy uses T (members) vs R (non-members)

Metric families (sources in docs/eval_methods.md):
  marginal   : KS (numeric), Wasserstein on the target (days), total variation (categorical/binary)
  association: mixed association matrix difference (Pearson / Cramer's V / correlation ratio)
  global     : discriminator AUC (classifier two-sample test), kernel MMD (RBF, median bandwidth)
  diversity  : precision / recall / density / coverage (Naeem et al. 2020), authenticity (Alaa et al. 2022)
  tail       : quantile error, exceedance error, expected shortfall error, GPD shape/scale difference
  subgroup   : mass error (overall and in the extreme tail), presence coverage, within-subgroup coverage
  privacy    : DCR ratio, NNDR, exact copies, distance-based membership inference AUC
  validity   : out-of-range numerics, unseen categories, target not above threshold
  utility    : augmentation gain in average precision with a paired patient-level bootstrap CI

All functions take pandas DataFrames with the same columns. No external services are called.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, genpareto
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# direction used by the stress tests: +1 = larger is worse, -1 = smaller is worse, 0 = distance from an ideal value
# (ideal values in IDEAL); metrics not listed are informational
DIRECTION = {
    "ks_mean": 1, "target_wasserstein": 1, "tv_cat_mean": 1, "tv_cat_max": 1, "bin_prev_err_mean": 1, "bin_prev_err_max": 1,
    "assoc_diff_mean": 1, "disc_auc": 0, "mmd": 1,
    "precision": -1, "density": 0, "coverage": -1, "authenticity": -1,
    "tail_q_err_mean": 1, "exceed_err_mean": 1, "es90_rel_err": 1, "gpd_xi_diff": 1,
    "sub_mass_err_mean": 1, "sub_mass_err_max": 1, "sub_mass_err_extreme": 1, "sub_presence": -1,
    "sub_cov_mean": -1, "sub_cov_min": -1, "sub_min_share_ratio": -1,
    "dcr_ratio": -1, "dcr_p05_ratio": -1, "nndr_median": -1, "exact_copies": 1, "mia_auc": 1,
    "invalid_rate": 1,
}
IDEAL = {"disc_auc": 0.5, "density": 1.0}


class Schema:
    """Column roles. groups: {name: [binary flag columns]} for clinical subgroups (a row may be in several)."""
    def __init__(self, num, cat, bin_, target, threshold, groups=None, blocks=None):
        self.num, self.cat, self.bin, self.target = list(num), list(cat), list(bin_), target
        self.threshold, self.groups, self.blocks = threshold, (groups or {}), (blocks or {})
        self.cols = self.num + self.cat + self.bin


class Encoder:
    """Fit on the generator's training set T. Numeric -> z-score; categorical -> one-hot / sqrt(2) (a mismatch
    contributes distance 1, like one binary flag); binary -> 0/1. Optional blocks (lists of numeric columns,
    e.g. 128-d embeddings) are scaled so each block contributes like ONE feature, so embeddings do not drown
    out the table fields."""
    def __init__(self, schema, T):
        self.s = schema
        self.mu = T[schema.num].mean(); self.sd = T[schema.num].std().replace(0, 1)
        self.levels = {c: sorted(T[c].astype(str).unique()) for c in schema.cat}
        self.block_w = {}
        for b, cols in schema.blocks.items():
            self.block_w[b] = 1.0 / np.sqrt(max(len(cols), 1))

    def __call__(self, D):
        s = self.s
        z = ((D[s.num] - self.mu) / self.sd).astype(float)
        for b, cols in s.blocks.items():
            z[cols] = z[cols] * self.block_w[b]
        parts = [z.values]
        for c in s.cat:
            v = D[c].astype(str).values
            parts.append(np.stack([(v == l) for l in self.levels[c]], 1).astype(float) / np.sqrt(2))
        if s.bin:
            parts.append(D[s.bin].astype(float).values)
        return np.nan_to_num(np.concatenate(parts, 1))


# ---------------------------------------------------------------- marginals and associations
def _tv(a, b):
    p, q = a.astype(str).value_counts(normalize=True), b.astype(str).value_counts(normalize=True)
    idx = p.index.union(q.index)
    return 0.5 * (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum()


def marginal_metrics(s, R, S):
    num_feats = [c for c in s.num if c not in sum(s.blocks.values(), [])]
    ks = [ks_2samp(R[c], S[c]).statistic for c in num_feats] or [np.nan]
    tv = [_tv(R[c], S[c]) for c in s.cat] or [np.nan]
    bp = (R[s.bin].astype(float).mean() - S[s.bin].astype(float).mean()).abs() if s.bin else pd.Series([np.nan])
    return {"ks_mean": float(np.mean(ks)), "target_wasserstein": wasserstein_distance(R[s.target], S[s.target]),
            "tv_cat_mean": float(np.mean(tv)), "tv_cat_max": float(np.max(tv)),
            "bin_prev_err_mean": float(bp.mean()), "bin_prev_err_max": float(bp.max())}


def _cramers_v(x, y):
    t = pd.crosstab(x, y).values
    if t.shape[0] < 2 or t.shape[1] < 2: return np.nan
    n = t.sum(); e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / n
    chi2 = ((t - e) ** 2 / np.where(e == 0, 1, e)).sum()
    return np.sqrt(chi2 / n / (min(t.shape) - 1))


def _corr_ratio(cat, num):
    num = num.astype(float)
    if num.std() == 0: return np.nan
    g = num.groupby(cat.astype(str).values)
    return np.sqrt((g.count() * (g.mean() - num.mean()) ** 2).sum() / ((num - num.mean()) ** 2).sum())


def association_matrix(s, D):
    """Mixed association: Pearson (numeric/binary pairs), Cramer's V (categorical with categorical/binary),
    correlation ratio (categorical with numeric). Embedding blocks are excluded (use MMD/discriminator there)."""
    emb = set(sum(s.blocks.values(), []))
    nums = [c for c in s.num if c not in emb] + s.bin
    cols = nums + s.cat
    M = pd.DataFrame(np.nan, index=cols, columns=cols)
    P = D[nums].astype(float).corr()
    M.loc[nums, nums] = P.values
    for c in s.cat:
        for o in cols:
            if o == c: continue
            if o in s.cat or o in s.bin:
                M.loc[c, o] = M.loc[o, c] = _cramers_v(D[c].astype(str), D[o].astype(str))
            else:
                M.loc[c, o] = M.loc[o, c] = _corr_ratio(D[c], D[o])
    return M


def association_metrics(s, R, S):
    A, B = association_matrix(s, R), association_matrix(s, S)
    d = (A - B).abs().values
    iu = np.triu_indices_from(d, 1)
    v = d[iu]
    # a pair constant in one sample and not the other counts as a full mismatch of its real strength
    v = np.where(np.isnan(v), np.nan_to_num(np.abs(A.values[iu])), v)
    return {"assoc_diff_mean": float(np.nanmean(v))}


# ---------------------------------------------------------------- global distribution tests
def discriminator_auc(XR, XS, seed=0, folds=5, gR=None, gS=None):
    """Classifier two-sample test: out-of-fold AUC of a real-vs-synthetic classifier on equal-size samples.
    0.5 = indistinguishable. Folds are grouped by patient (gR, gS = patient ids; None = every row its own
    group), so the classifier cannot score well by memorising a patient whose other stays are in training."""
    from sklearn.model_selection import StratifiedGroupKFold
    rng = np.random.default_rng(seed); n = min(len(XR), len(XS))
    iR, iS = rng.choice(len(XR), n, replace=False), rng.choice(len(XS), n, replace=False)
    XR, XS = XR[iR], XS[iS]
    gR = np.asarray(gR)[iR].astype(str) if gR is not None else np.array([f"r{i}" for i in iR])
    gS = np.asarray(gS)[iS].astype(str) if gS is not None else np.array([f"s{i}" for i in iS])
    X, y, g = np.vstack([XR, XS]), np.r_[np.zeros(n), np.ones(n)], np.r_[["R_" + x for x in gR], ["S_" + x for x in gS]]
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def mmd_rbf(XR, XS, bandwidth, max_n=1500, seed=0):
    """Biased squared MMD with an RBF kernel; the bandwidth is fixed in advance from real data (median heuristic)."""
    rng = np.random.default_rng(seed)
    if len(XR) > max_n: XR = XR[rng.choice(len(XR), max_n, replace=False)]
    if len(XS) > max_n: XS = XS[rng.choice(len(XS), max_n, replace=False)]
    def k(A, B):
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-np.maximum(d2, 0) / (2 * bandwidth ** 2))
    return float(k(XR, XR).mean() + k(XS, XS).mean() - 2 * k(XR, XS).mean())


def median_bandwidth(X, max_n=1000, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > max_n: X = X[rng.choice(len(X), max_n, replace=False)]
    d = np.sqrt(np.maximum((X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2 * X @ X.T, 0))
    return float(np.median(d[np.triu_indices_from(d, 1)])) or 1.0


# ---------------------------------------------------------------- diversity (Naeem et al. 2020) and authenticity (Alaa et al. 2022)
def _kth_radius(X, k):
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return d[:, k]  # column 0 is the point itself


def prdc(XR, XS, k=5):
    """Precision, recall, density, coverage with k-nearest-neighbour balls. Use equal sample sizes across models."""
    if len(XR) <= k or len(XS) <= k:
        return dict(precision=np.nan, recall=np.nan, density=np.nan, coverage=np.nan)
    rR, rS = _kth_radius(XR, k), _kth_radius(XS, k)
    D = np.sqrt(np.maximum((XS ** 2).sum(1)[:, None] + (XR ** 2).sum(1)[None, :] - 2 * XS @ XR.T, 0))  # syn x real
    return dict(precision=float((D < rR[None, :]).any(1).mean()),
                recall=float((D < rS[:, None]).any(0).mean()),
                density=float((D < rR[None, :]).sum(1).mean() / k),
                coverage=float((D.min(0) < rR).mean()))


def authenticity(XT, XS):
    """Share of synthetic rows farther from their nearest training row than that row is from its own nearest
    other training row (higher = less copy-like). A memorisation diagnostic, not a privacy guarantee."""
    nnT = NearestNeighbors(n_neighbors=2).fit(XT)
    dT = nnT.kneighbors(XT)[0][:, 1]
    d, i = NearestNeighbors(n_neighbors=1).fit(XT).kneighbors(XS)
    return float((d[:, 0] > dT[i[:, 0]]).mean())


# ---------------------------------------------------------------- privacy
def privacy_metrics(XT, XR, XS, seed=0):
    """DCR ratio = median distance (synthetic -> T) / median distance (held-out real R -> T); <1 means closer than
    unseen real patients. NNDR = d1/d2 to T. Membership inference: score each candidate by -distance to the
    synthetic set; members = T rows, non-members = R rows (equal sizes); AUC 0.5 = attacker at chance."""
    nnT = NearestNeighbors(n_neighbors=2).fit(XT)
    dS, _ = nnT.kneighbors(XS); dR, _ = nnT.kneighbors(XR)
    nndr = dS[:, 0] / np.where(dS[:, 1] == 0, np.nan, dS[:, 1])
    rng = np.random.default_rng(seed); n = min(len(XT), len(XR))
    mem, non = XT[rng.choice(len(XT), n, replace=False)], XR[rng.choice(len(XR), n, replace=False)]
    nnS = NearestNeighbors(n_neighbors=1).fit(XS)
    score = -np.r_[nnS.kneighbors(mem)[0][:, 0], nnS.kneighbors(non)[0][:, 0]]
    return {"dcr_ratio": float(np.median(dS[:, 0]) / max(np.median(dR[:, 0]), 1e-9)),
            "dcr_p05_ratio": float(np.quantile(dS[:, 0], .05) / max(np.quantile(dR[:, 0], .05), 1e-9)),
            "nndr_median": float(np.nanmedian(nndr)) if np.isfinite(nndr).any() else 0.0,
            "exact_copies": float((dS[:, 0] < 1e-6).mean()),
            "mia_auc": float(roc_auc_score(np.r_[np.ones(n), np.zeros(n)], score))}


# ---------------------------------------------------------------- tail (extreme value) fidelity
def tail_metrics(s, R, S, qs=(0.5, 0.75, 0.9, 0.95, 0.99)):
    y, z = R[s.target].astype(float).values, S[s.target].astype(float).values
    qerr = [abs(np.quantile(z, q) - np.quantile(y, q)) for q in qs]
    thr = [np.quantile(y, q) for q in (0.5, 0.75, 0.9, 0.95)]
    exc = [abs((z > v).mean() - (y > v).mean()) for v in thr]
    q90 = np.quantile(y, 0.9)
    es_r, es_s = y[y > q90].mean(), (z[z > q90].mean() if (z > q90).any() else np.nan)
    out = {"tail_q_err_mean": float(np.mean(qerr)), "exceed_err_mean": float(np.mean(exc)),
           "es90_rel_err": float(abs(es_s - es_r) / es_r) if np.isfinite(es_s) else 1.0}
    for q, e in zip(qs, qerr): out[f"q{int(q*100)}_err_days"] = float(e)
    try:
        xr = genpareto.fit(y[y > s.threshold] - s.threshold, floc=0)[0]
        xs = genpareto.fit(z[z > s.threshold] - s.threshold, floc=0)[0]
        out["gpd_xi_real"], out["gpd_xi_syn"], out["gpd_xi_diff"] = float(xr), float(xs), float(abs(xs - xr))
    except Exception:
        out["gpd_xi_diff"] = np.nan
    return out


# ---------------------------------------------------------------- subgroups (operational metrics recommended by both reviews)
def subgroup_metrics(s, R, S, XR, XS, k=5, k_min=5, extreme_q=0.9, min_extreme=20, S_full=None, real_shares=None):
    if not s.groups: return {}
    def mass(D):
        f = pd.DataFrame({g: D[c].astype(float).max(1) for g, c in s.groups.items()}, index=D.index)
        f["none"] = (f.sum(1) == 0).astype(float)
        return f
    fR, fS = mass(R), mass(S)
    err = (fR.mean() - fS.mean()).abs()
    cut = R[s.target].quantile(extreme_q)
    eR, eS = R[s.target] >= cut, S[s.target] >= cut
    ext = (fR[eR.values].mean() - fS[eS.values].mean()).abs().mean() if eR.sum() >= min_extreme and eS.sum() >= min_extreme else np.nan
    pres = np.mean([(fS[g] > 0).sum() >= k_min for g in s.groups])
    covs = []
    for g in s.groups:
        mR, mS = fR[g].values > 0, fS[g].values > 0
        kk = int(min(k, mR.sum() - 1, mS.sum() - 1))
        if kk >= 2:   # coverage needs a few neighbours on both sides; tiny groups are reported as NaN
            covs.append(prdc(XR[mR], XS[mS], k=kk)["coverage"])
    out = {"sub_mass_err_mean": float(err.mean()), "sub_mass_err_max": float(err.max()),
           "sub_mass_err_extreme": float(ext), "sub_presence": float(pres),
           "sub_cov_mean": float(np.mean(covs)) if covs else np.nan, "sub_cov_min": float(np.min(covs)) if covs else np.nan}
    for g in fR.columns: out[f"share_{g}"] = float(fS[g].mean())
    if S_full is not None and real_shares is not None:
        # smallest ratio (synthetic share / real share) over groups, on the FULL synthetic sample:
        # 0 = a subgroup is missing entirely, 1 = represented at its real rate
        fF = mass(S_full)
        out["sub_min_share_ratio"] = float(min(fF[g].mean() / max(real_shares[g], 1e-9) for g in s.groups))
    return out


# ---------------------------------------------------------------- validity
def validity_metrics(s, T, S, rules=None):
    """Rows breaking any rule: numeric (non-target) outside the training range, unseen category,
    target not above the rare-class threshold, plus optional custom rules (callables returning a bool mask)."""
    bad = np.zeros(len(S), bool)
    for c in [c for c in s.num if c != s.target and c not in sum(s.blocks.values(), [])]:
        bad |= ((S[c] < T[c].min()) | (S[c] > T[c].max())).values
    for c in s.cat:
        bad |= ~S[c].astype(str).isin(set(T[c].astype(str))).values
    bad |= (S[s.target] <= s.threshold).values
    for r in (rules or []):
        bad |= np.asarray(r(S), bool)
    return {"invalid_rate": float(bad.mean())}


# ---------------------------------------------------------------- one call for everything
def evaluate(s, T, R, S, enc=None, bandwidth=None, k=5, seed=0, rules=None, R_ids=None, S_ids=None, valid_ref=None):
    """T = generator training rows (real rare class), R = held-out real rare class, S = synthetic.
    Fidelity is S vs R; privacy is S vs T with R as non-member control. S is subsampled to len(R) for the
    sample-size-sensitive metrics (PRDC, MMD, discriminator) so every model is compared at the same size."""
    enc = enc or Encoder(s, T)
    rng = np.random.default_rng(seed)
    iq = rng.choice(len(S), min(len(S), len(R)), replace=False)
    S_eq = S.iloc[iq].reset_index(drop=True)
    S_ids_eq = np.asarray(S_ids)[iq] if S_ids is not None else None
    XT, XR, XS, XSe = enc(T), enc(R), enc(S), enc(S_eq)
    bandwidth = bandwidth or median_bandwidth(XR)
    out = {"n_syn": len(S)}
    out.update(marginal_metrics(s, R, S))
    out.update(association_metrics(s, R, S))
    out["disc_auc"] = discriminator_auc(XR, XSe, seed=seed, gR=R_ids, gS=S_ids_eq)
    out["mmd"] = mmd_rbf(XR, XSe, bandwidth, seed=seed)
    out.update(prdc(XR, XSe, k=k))
    out["authenticity"] = authenticity(XT, XS)
    out.update(tail_metrics(s, R, S))
    pooled = pd.concat([T, R], ignore_index=True)
    shares = {g: float(pooled[c].astype(float).max(1).mean()) for g, c in s.groups.items()}
    out.update(subgroup_metrics(s, R, S_eq, XR, XSe, k=k, S_full=S, real_shares=shares))
    out.update(privacy_metrics(XT, XR, XS, seed=seed))
    out.update(validity_metrics(s, valid_ref if valid_ref is not None else T, S, rules))
    return out


# ---------------------------------------------------------------- reference band and detection
def badness(name, value):
    """Turn a metric into 'larger = worse' so detection can use one rule."""
    d = DIRECTION.get(name)
    if d is None or value is None or not np.isfinite(value): return np.nan
    if d == 0: return abs(value - IDEAL[name])
    return value * d


def reference_band(s, T, real_pool, n_ref, reps=20, q=0.95, seed=0, patient_ids=None, **kw):
    """Evaluate `reps` real-vs-real pairs as if one side were synthetic. real_pool must be held out from T.
    The pool is split into two halves BY PATIENT (patient_ids), so one person's stays never sit on both sides.
    Returns the per-metric distribution; the q-quantile of 'badness' is the detection threshold."""
    rng = np.random.default_rng(seed); rows = []
    ids = np.asarray(patient_ids) if patient_ids is not None else np.arange(len(real_pool))
    uniq = np.unique(ids)
    for r in range(reps):
        half = set(rng.permutation(uniq)[:len(uniq) // 2])
        m = np.array([i in half for i in ids])
        R = real_pool[m].reset_index(drop=True)
        S = real_pool[~m].reset_index(drop=True)
        rows.append(evaluate(s, T, R, S, seed=seed + r, R_ids=ids[m], S_ids=ids[~m], **kw))
    ref = pd.DataFrame(rows)
    thr = {c: np.nanquantile([badness(c, v) for v in ref[c]], q) for c in ref.columns if c in DIRECTION}
    return ref, thr


def detected(result, thr, margin=0.0):
    return {c: bool(np.isfinite(badness(c, result.get(c, np.nan))) and badness(c, result[c]) > t + margin)
            for c, t in thr.items() if np.isfinite(t)}


# ---------------------------------------------------------------- utility: augmentation gain
def augmentation_gain(make_X, train, syn, val, test, label="is_plos", group="subject_id", p0=0.5,
                      n_boot=1000, seed=0, aug_weight=None):
    """Train the same classifier on real-only vs real+synthetic; evaluate both on the same real test patients.
    Average precision (AP) and recall at a precision floor p0 (threshold chosen on validation, then frozen).
    Paired bootstrap over test PATIENTS (all their stays together) gives a 95% CI for the AP gain."""
    Xtr, ytr = make_X(train), train[label].values
    Xsy = make_X(syn); Xva, yva = make_X(val), val[label].values; Xte, yte = make_X(test), test[label].values
    cols = Xtr.columns
    Xsy, Xva, Xte = (x.reindex(columns=cols, fill_value=0) for x in (Xsy, Xva, Xte))
    def fit(X, y):
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed).fit(X, y)
    m0 = fit(Xtr, ytr)
    if aug_weight == "balanced":   # baseline: real data only, class weighting instead of synthetic rows
        m1 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed,
                                            class_weight="balanced").fit(Xtr, ytr)
    else:
        m1 = fit(pd.concat([Xtr, Xsy], ignore_index=True), np.r_[ytr, np.ones(len(Xsy), int)])
    out = {}
    preds = {}
    for name, m in [("real", m0), ("aug", m1)]:
        pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        P, Rc, T = precision_recall_curve(yva, pv)
        ok = np.where(P[:-1] >= p0)[0]
        t = T[ok[np.argmax(Rc[:-1][ok])]] if len(ok) else np.inf
        out[f"ap_{name}"] = average_precision_score(yte, pt)
        out[f"recall_at_p{int(p0*100)}_{name}"] = float(((pt >= t) & (yte == 1)).sum() / max(yte.sum(), 1))
        out[f"precision_at_thr_{name}"] = float(yte[pt >= t].mean()) if (pt >= t).any() else np.nan
        preds[name] = pt
    out["ap_gain"] = out["ap_aug"] - out["ap_real"]
    rng = np.random.default_rng(seed)
    pats = test[group].values; up = np.unique(pats); pos = {p: np.where(pats == p)[0] for p in up}
    gains = []
    for _ in range(n_boot):
        idx = np.concatenate([pos[p] for p in rng.choice(up, len(up), replace=True)])
        if yte[idx].sum() == 0: continue
        gains.append(average_precision_score(yte[idx], preds["aug"][idx]) - average_precision_score(yte[idx], preds["real"][idx]))
    out["ap_gain_ci_low"], out["ap_gain_ci_high"] = np.percentile(gains, [2.5, 97.5])
    return out


def summarize(raw, by="generator", metrics=None):
    """Mean, SD and a t-based 95% CI across independent generator seeds."""
    from scipy.stats import t as tdist
    metrics = metrics or [c for c in raw.columns if c not in (by, "seed") and pd.api.types.is_numeric_dtype(raw[c])]
    g = raw.groupby(by)[metrics]
    mean, sd, n = g.mean(), g.std(), g.count()
    half = sd * tdist.ppf(0.975, (n - 1).clip(lower=1)) / np.sqrt(n)
    return pd.concat({"mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half}, axis=1)
