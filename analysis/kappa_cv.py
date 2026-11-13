"""Agreement between automatic features and the single human rater (dataset A).
Pre-specified 1:1 feature-finding pairs. Thresholds chosen by nested (repeated stratified 10-fold)
cross-validation on the training folds (max Youden J), applied to held-out folds; pooled
held-out predictions -> kappa, sensitivity, specificity, balanced accuracy, with bootstrap CI.
Also leave-one-group-out (A-D) transport check and in-sample (optimistic) reference.
Usage: python3 -I kappa_cv.py <merged_features.csv> <out_dir>
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import stats

D = pd.read_csv(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)

PAIRS = [  # (label, auto feature, manual column, positive code, direction: +1 higher feature -> positive)
    ('Crown closed',  'crown_env_cov',   'm_crownopen', 0, +1),
    ('Ground line',   'ground_auto',     'm_ground',    1, +1),
    ('Trunk two lines','trunk_crossings','m_trunk2',    2, +1),
    ('Fruit present', 'small_loops',     'm_fruit',     1, +1),
    ('Leaves present','small_loops',     'm_leaf',      1, +1),
]
CROWN4 = ['crown_env_cov', 'crown_env_md', 'crown_hole_seal', 'crown_env_frac']


def best_thr(x, y):
    """threshold maximising Youden J on training data; predict positive if x >= thr"""
    xs = np.unique(x)
    cands = np.r_[xs[0] - 1e-9, (xs[:-1] + xs[1:]) / 2, xs[-1] + 1e-9]
    best, bt = -2, cands[0]
    for t in cands:
        p = x >= t
        se = (p & (y == 1)).sum() / max((y == 1).sum(), 1)
        sp = (~p & (y == 0)).sum() / max((y == 0).sum(), 1)
        if se + sp - 1 > best:
            best, bt = se + sp - 1, t
    return bt


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
    k = cohen_kappa_score(y, p)
    po = (tp + tn) / len(y)
    return dict(kappa=k, sens=se, spec=sp, bacc=(se + sp) / 2, acc=po, pabak=2 * po - 1,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def boot_ci(y, p, n=2000):
    ks = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        ks.append(cohen_kappa_score(y[b], p[b]))
    return np.percentile(ks, [2.5, 97.5])


def nested_cv(x, y, n_rep=20, k=10):
    """x: 1-d feature (or 2-d matrix for logistic); returns pooled predictions of first repeat,
    and per-repeat kappa list"""
    kap, aucs = [], []
    first = None
    for r in range(n_rep):
        pred = np.zeros(len(y), int); score = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=r).split(x, y):
            if x.ndim == 1:
                t = best_thr(x[tr], y[tr]); pred[te] = (x[te] >= t); score[te] = x[te]
            else:
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x[tr], y[tr])
                s = m.predict_proba(x[tr])[:, 1]; t = best_thr(s, y[tr])
                st = m.predict_proba(x[te])[:, 1]; pred[te] = st >= t; score[te] = st
        kap.append(cohen_kappa_score(y, pred)); aucs.append(roc_auc_score(y, score))
        if first is None:
            first = (pred.copy(), score.copy())
    return first, np.array(kap), np.array(aucs)


rows = []
for lab, f, m, pos, sgn in PAIRS:
    ok = D[f].notna() & D[m].notna()
    x = (sgn * D.loc[ok, f]).to_numpy(float); y = (D.loc[ok, m] == pos).astype(int).to_numpy()
    (pred, score), kap, aucs = nested_cv(x, y)
    met = metrics(y, pred); ci = boot_ci(y, pred)
    # in-sample optimum (optimistic reference)
    t_all = best_thr(x, y); met_in = metrics(y, (x >= t_all).astype(int))
    # leave-one-group-out
    logo = {}
    g = D.loc[ok, 'site'].to_numpy()
    pl = np.zeros(len(y), int)
    for s in np.unique(g):
        tr, te = g != s, g == s
        t = best_thr(x[tr], y[tr]); pl[te] = x[te] >= t
        if len(np.unique(y[te])) == 2:
            logo[s] = dict(n=int(te.sum()), kappa=round(cohen_kappa_score(y[te], pl[te]), 3),
                           auc=round(roc_auc_score(y[te], x[te]), 3), thr=round(float(t), 4))
    met_logo = metrics(y, pl)
    rows.append(dict(finding=lab, feature=f, n=int(ok.sum()), n_pos=int(y.sum()), prevalence=round(y.mean(), 3),
                     AUC_cv=round(aucs.mean(), 3), kappa_cv=round(met['kappa'], 3),
                     kappa_cv_ci=f"{ci[0]:.2f}-{ci[1]:.2f}", kappa_cv_rep_range=f"{kap.min():.2f}-{kap.max():.2f}",
                     sens=round(met['sens'], 3), spec=round(met['spec'], 3), bacc=round(met['bacc'], 3),
                     acc=round(met['acc'], 3), pabak=round(met['pabak'], 3),
                     kappa_insample=round(met_in['kappa'], 3), thr_insample=round(float(t_all), 4),
                     kappa_logo=round(met_logo['kappa'], 3), logo=json.dumps(logo),
                     confusion=f"tp{met['tp']} fp{met['fp']} fn{met['fn']} tn{met['tn']}"))
    print(lab, rows[-1])

# crown closed: 4-feature logistic (secondary)
ok = D['m_crownopen'].notna()
X = D.loc[ok, CROWN4].fillna(D[CROWN4].median()).to_numpy(float); y = (D.loc[ok, 'm_crownopen'] == 0).astype(int).to_numpy()
(pred, score), kap, aucs = nested_cv(X, y)
met = metrics(y, pred); ci = boot_ci(y, pred)
rows.append(dict(finding='Crown closed (4-feature logistic)', feature='+'.join(CROWN4), n=int(ok.sum()), n_pos=int(y.sum()),
                 prevalence=round(y.mean(), 3), AUC_cv=round(aucs.mean(), 3), kappa_cv=round(met['kappa'], 3),
                 kappa_cv_ci=f"{ci[0]:.2f}-{ci[1]:.2f}", kappa_cv_rep_range=f"{kap.min():.2f}-{kap.max():.2f}",
                 sens=round(met['sens'], 3), spec=round(met['spec'], 3), bacc=round(met['bacc'], 3), acc=round(met['acc'], 3),
                 pabak=round(met['pabak'], 3), confusion=f"tp{met['tp']} fp{met['fp']} fn{met['fn']} tn{met['tn']}"))
print(rows[-1])

# branch (ordinal): Spearman + 3-class via two CV thresholds -> weighted kappa
ok = D['m_branch'].notna() & D['junctions_per_len'].notna()
x = D.loc[ok, 'junctions_per_len'].to_numpy(float); yb = D.loc[ok, 'm_branch'].astype(int).to_numpy()
rho = stats.spearmanr(x, yb)
# ordinal thresholds by CV: cut-points chosen to maximise quadratic weighted kappa on train folds (grid on quantiles)
def wk(y, p): return cohen_kappa_score(y, p, weights='quadratic')
qs = np.linspace(0.05, 0.95, 37)
wks = []
first = None
for r in range(20):
    pred = np.zeros(len(yb), int)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=r).split(x, yb):
        best, bc = -2, None
        c = np.quantile(x[tr], qs)
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                p = np.digitize(x[tr], [c[i], c[j]])
                v = wk(yb[tr], p)
                if v > best: best, bc = v, (c[i], c[j])
        pred[te] = np.digitize(x[te], list(bc))
    wks.append(wk(yb, pred))
    if first is None: first = pred.copy()
rows.append(dict(finding='Branches (0 none/1 single/2 double) ordinal', feature='junctions_per_len', n=int(ok.sum()),
                 n_pos=int((yb == 2).sum()), prevalence=round(float((yb == 2).mean()), 3), AUC_cv=np.nan,
                 kappa_cv=round(float(np.mean(wks)), 3), kappa_cv_ci='quadratic weighted', kappa_cv_rep_range=f"{min(wks):.2f}-{max(wks):.2f}",
                 sens=np.nan, spec=np.nan, bacc=np.nan, acc=round(float((first == yb).mean()), 3), pabak=np.nan,
                 confusion=f"spearman rho={rho[0]:.3f} p={rho[1]:.2g}"))
print(rows[-1])

pd.DataFrame(rows).to_csv(os.path.join(out, 'agreement_cv.csv'), index=False)
print('saved')
