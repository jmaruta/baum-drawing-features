import numpy as np, pandas as pd, warnings, json
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')
rng = np.random.default_rng(0)

F = pd.read_csv('feats.csv')
M = pd.read_csv('manual.csv')
B = pd.read_csv('beh.csv', encoding='utf-8-sig')
D = F.merge(M, on='ID', how='left').merge(B, on='ID', how='left')
D['site'] = D.ID.str[0]
ren = {'年齢': 'age', '性': 'sex', '1_最終学歴': 'edu', '1_K6': 'k6', '1_健康状態': 'health',
       '1_友達づきあい1': 'friends', '1_近所づきあい1': 'neigh', '1_外出1': 'outing',
       '1_気分が沈み気が晴れない': 'k6_depressed', '1_価値のない人間': 'k6_worthless',
       '1_絶望的': 'k6_hopeless', '1_何をするのも骨折り': 'k6_effort',
       '1_神経過敏': 'k6_nervous', '1_そわそわ・落ち着かない': 'k6_restless'}
D = D.rename(columns=ren)
D['k6_log'] = np.log1p(D.k6)
D['k6_ge5'] = (D.k6 >= 5).astype(float).where(D.k6.notna())
D['social'] = D[['friends', 'neigh', 'outing']].mean(1)
out = {}
print('N images', len(D), 'with K6', D.k6.notna().sum())

feat_line = ['stroke_w_rel', 'stroke_w_cv', 'ink_dark', 'ink_dark_cv', 'fill_ratio', 'jagged', 'turn_hf',
             'gaps_per_len', 'short_seg_frac', 'frag_small', 'n_strokes']
feat_struct = ['ink_len', 'ink_cover', 'endpoints_per_len', 'junctions_per_len', 'small_loops',
               'crown_env_cov', 'crown_env_md', 'crown_hole_seal', 'crown_trunk_ratio', 'trunk_crossings',
               'ground_auto', 'lr_asym', 'ink_top_half', 'centroid_y']
feat_size = ['aspect_hw']  # margin-dependent, reported only
FEATS = feat_line + feat_struct

# ---------------- 1. validity vs manual coding ----------------
val = []
def auc(y, s):
    ok = y.notna() & s.notna(); return roc_auc_score(y[ok], s[ok]), ok.sum()
for a, m, pos, lab in [('crown_env_cov', 'm_crownopen', 0, '樹冠 閉鎖'), ('crown_hole_seal', 'm_crownopen', 0, '樹冠 閉鎖'),
                       ('trunk_crossings', 'm_trunk2', 2, '幹 二線'), ('ground_auto', 'm_ground', 1, '地平線あり'),
                       ('small_loops', 'm_fruit', 1, '実あり'), ('small_loops', 'm_leaf', 1, '葉あり')]:
    y = (D[m] == pos).astype(float).where(D[m].notna())
    a_, n = auc(y, D[a]); val.append(dict(auto=a, manual=lab, n=int(n), AUC=round(a_, 3)))
# combined crown model (cross-validated)
cc = ['crown_env_cov', 'crown_env_md', 'crown_hole_seal', 'crown_env_frac']
ok = D.m_crownopen.notna()
X = D.loc[ok, cc].fillna(D[cc].median()); y = (D.loc[ok, 'm_crownopen'] == 0).astype(int)
p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()), X, y, cv=StratifiedKFold(10, shuffle=True, random_state=0), method='predict_proba')[:, 1]
val.append(dict(auto='crown 4特徴の組合せ(10-fold CV)', manual='樹冠 閉鎖', n=int(ok.sum()), AUC=round(roc_auc_score(y, p), 3)))
D.loc[ok, 'crown_closed_prob'] = p
r = stats.spearmanr(D.junctions_per_len, D.m_branch, nan_policy='omit')
val.append(dict(auto='junctions_per_len', manual='枝(0なし/1一線/2二線) ρ', n=int(D.m_branch.notna().sum()), AUC=round(r[0], 3)))
out['validity'] = val
print(pd.DataFrame(val))

# ---------------- 2. site dependence ----------------
sd = []
for f in FEATS + feat_size:
    g = [D.loc[D.site == s, f].dropna() for s in 'ABCD']
    H, p = stats.kruskal(*g); eta = (H - 4 + 1) / (len(D) - 4)
    sd.append(dict(feature=f, eta2_site=round(eta, 3), p=p))
sd = pd.DataFrame(sd).sort_values('eta2_site', ascending=False); out['site'] = sd.to_dict('records')
print(sd.head(10))

# ---------------- within-site rank-normalisation ----------------
def inv_norm(x):
    r = x.rank(); n = x.notna().sum(); return pd.Series(stats.norm.ppf((r - 0.5) / n), index=x.index)
for f in FEATS + feat_size:
    D[f + '_z'] = D.groupby('site')[f].transform(inv_norm)
Z = [f + '_z' for f in FEATS]

# ---------------- 3. association screen ----------------
outcomes = {'age': 'ols', 'sex': 'logit', 'edu': 'ols', 'k6_log': 'ols', 'k6_ge5': 'logit',
            'k6_depressed': 'ols', 'k6_worthless': 'ols', 'k6_hopeless': 'ols', 'k6_effort': 'ols',
            'social': 'ols', 'friends': 'ols', 'health': 'ols'}
rows = []
for o, kind in outcomes.items():
    cov = ' + C(site)' + ('' if o == 'age' else ' + age') + ('' if o == 'sex' else ' + sex')
    for f in Z + ['m_crownopen', 'm_trunk2', 'm_leaf', 'm_root', 'm_ground', 'm_branch']:
        d = D[list(dict.fromkeys([o, f, 'age', 'sex', 'site']))].dropna()
        if d[f].nunique() < 2: continue
        fz = (d[f] - d[f].mean()) / d[f].std()
        d = d.assign(x=fz)
        try:
            if kind == 'ols':
                yv = (d[o] - d[o].mean()) / d[o].std(); d = d.assign(y=yv)
                m = smf.ols('y ~ x' + cov, d).fit()
            else:
                d = d.assign(y=d[o].astype(int)); m = smf.logit('y ~ x' + cov, d).fit(disp=0)
            rows.append(dict(outcome=o, feature=f.replace('_z', ''), beta=m.params['x'], p=m.pvalues['x'], n=len(d)))
        except Exception as e:
            pass
A = pd.DataFrame(rows)
from statsmodels.stats.multitest import multipletests
A['q'] = A.groupby('outcome').p.transform(lambda p: multipletests(p, method='fdr_bh')[1])
A.to_csv('assoc.csv', index=False)
print(A.sort_values('p').head(30).to_string())
summ = A.groupby('outcome').apply(lambda g: pd.Series({'n_p05': (g.p < .05).sum(), 'n_q10': (g.q < .10).sum(), 'n_tests': len(g), 'min_p': g.p.min()}))
print(summ); out['assoc_summary'] = summ.reset_index().to_dict('records')

# ---------------- 4. cross-validated prediction ----------------
def cv_auc(X, y, reps=10):
    aucs = []
    for r in range(reps):
        cv = StratifiedKFold(5, shuffle=True, random_state=r)
        m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=4, penalty='l2', max_iter=2000))
        p = cross_val_predict(m, X, y, cv=cv, method='predict_proba')[:, 1]
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs)), float(np.std(aucs))

def cv_r(X, y, reps=10):
    rs = []
    for r in range(reps):
        p = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))), X, y, cv=KFold(5, shuffle=True, random_state=r))
        rs.append(stats.pearsonr(p, y)[0])
    return float(np.mean(rs)), float(np.std(rs))

def perm_auc(X, y, obs, n=200):
    null = []
    for i in range(n):
        yp = rng.permutation(y)
        p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), X, yp, cv=StratifiedKFold(5, shuffle=True, random_state=i), method='predict_proba')[:, 1]
        null.append(roc_auc_score(yp, p))
    return float((np.sum(np.array(null) >= obs) + 1) / (n + 1))

site_d = pd.get_dummies(D.site, drop_first=True).astype(float)
pred = []
def run_bin(name, ycol, extra_cov):
    d = D[[ycol] + Z + extra_cov].join(site_d).dropna()
    y = d[ycol].astype(int).values
    Xc = d[extra_cov + list(site_d.columns)].values
    Xf = d[Z].values
    a0 = cv_auc(Xc, y) if Xc.shape[1] else (0.5, 0)
    a1 = cv_auc(np.c_[Xc, Xf], y)
    a2 = cv_auc(Xf, y)
    pv = perm_auc(Xf, y, a2[0], n=200)
    pred.append(dict(target=name, n=len(y), pos=int(y.sum()), AUC_cov=round(a0[0], 3), AUC_cov_plus_feat=round(a1[0], 3),
                     AUC_feat_only=round(a2[0], 3), sd=round(a2[1], 3), perm_p_feat_only=round(pv, 3)))
    print(pred[-1])
run_bin('性別(陽性対照)', 'sex', ['age'])
run_bin('K6≥5', 'k6_ge5', ['age', 'sex'])
# extreme groups K6=0 vs >=5
D['k6_ext'] = np.where(D.k6 == 0, 0, np.where(D.k6 >= 5, 1, np.nan))
run_bin('K6=0 vs K6≥5', 'k6_ext', ['age', 'sex'])
D['social_low'] = (D.social > D.social.median()).astype(float).where(D.social.notna())
run_bin('社会活動 低(中央値以上)', 'social_low', ['age', 'sex'])
D['old'] = (D.age >= 75).astype(float).where(D.age.notna())
run_bin('75歳以上(陽性対照)', 'old', ['sex'])
out['prediction'] = pred
# continuous
cont = []
for name, ycol in [('年齢', 'age'), ('K6(log)', 'k6_log'), ('学歴', 'edu')]:
    d = D[[ycol] + Z].dropna()
    r_, s_ = cv_r(d[Z].values, d[ycol].values)
    cont.append(dict(target=name, n=len(d), cv_r=round(r_, 3), sd=round(s_, 3)))
print(cont); out['prediction_cont'] = cont

# ---------------- 5. sensitivity: drop faint scans ----------------
faint = D.ink_level < D.ink_level.quantile(0.10)
sens = []
for o in ['k6_log', 'k6_ge5', 'age', 'sex']:
    for f in Z:
        d = D.loc[~faint, list(dict.fromkeys([o, f, 'age', 'sex', 'site']))].dropna()
        cov = ' + C(site)' + ('' if o == 'age' else ' + age') + ('' if o == 'sex' else ' + sex')
        d = d.assign(x=(d[f] - d[f].mean()) / d[f].std())
        try:
            m = smf.logit('y ~ x' + cov, d.assign(y=d[o].astype(int))).fit(disp=0) if o in ('sex', 'k6_ge5') else smf.ols('y ~ x' + cov, d.assign(y=(d[o] - d[o].mean()) / d[o].std())).fit()
            sens.append(dict(outcome=o, feature=f[:-2], beta=m.params['x'], p=m.pvalues['x']))
        except Exception:
            pass
S = pd.DataFrame(sens); S.to_csv('assoc_sens.csv', index=False)
D.to_csv('merged_features.csv', index=False)
json.dump(out, open('results.json', 'w'), ensure_ascii=False, indent=1, default=float)
