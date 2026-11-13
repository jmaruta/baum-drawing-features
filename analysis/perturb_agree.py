"""Agreement original vs perturbed from perturb_long.csv -> ICC(A,1) (=ICC(2,1) absolute agreement), Spearman rho, bias.
Usage: python3 -I perturb_agree.py <perturb_long.csv> <out_dir>
"""
import sys, os
import numpy as np, pandas as pd, pingouin as pg

L = pd.read_csv(sys.argv[1]); out = sys.argv[2]
FEATS = ['stroke_w_rel', 'ink_dark', 'ink_dark_cv', 'stroke_w_cv', 'fill_ratio', 'ink_len', 'ink_cover',
         'endpoints_per_len', 'junctions_per_len', 'n_strokes', 'frag_small', 'gaps_per_len', 'crown_env_cov',
         'crown_env_md', 'crown_env_frac', 'crown_hole_seal', 'small_loops', 'jagged', 'turn_hf', 'short_seg_frac',
         'lr_asym', 'ink_top_half', 'centroid_y', 'trunk_crossings', 'crown_trunk_ratio', 'ground_auto', 'aspect_hw']
O = L[L.kind == 'orig'].set_index('ID')
rows = []
for k in [k for k in L.kind.unique() if k != 'orig']:
    P = L[L.kind == k].set_index('ID'); ids = O.index.intersection(P.index)
    for ft in FEATS:
        a, b = O.loc[ids, ft].astype(float), P.loc[ids, ft].astype(float)
        m = a.notna() & b.notna(); a, b = a[m], b[m]
        if len(a) < 10 or a.std() == 0:
            continue
        d = pd.DataFrame({'t': np.r_[a.index, b.index], 'r': ['o'] * len(a) + ['p'] * len(b), 'v': np.r_[a.values, b.values]})
        icc = pg.intraclass_corr(d, targets='t', raters='r', ratings='v')
        iccA = float(icc.loc[icc.Type == 'ICC(A,1)', 'ICC'].iloc[0]); iccC = float(icc.loc[icc.Type == 'ICC(C,1)', 'ICC'].iloc[0])
        rows.append(dict(perturbation=k, feature=ft, n=int(len(a)), icc_A1=round(iccA, 3), icc_C1=round(iccC, 3),
                         rho=round(a.corr(b, method='spearman'), 3), bias_sd=round(float(np.median(b - a) / (a.std() + 1e-12)), 3)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(out, 'perturb_agreement.csv'), index=False)
order = ['up200', 'down50', 'down25', 'rot+3', 'rot-3', 'blur', 'noise', 'jpeg50', 'lighter', 'darker', 'rescan']
piv = R.pivot(index='feature', columns='perturbation', values='icc_A1')[order]
piv.to_csv(os.path.join(out, 'perturb_icc_matrix.csv'))
print(piv.round(2).to_string()); print('\nmedian ICC per perturbation:\n', piv.median().round(2).to_string())
print('\nmedian ICC per feature (across perturbations):\n', piv.median(axis=1).sort_values().round(2).to_string())
piv2 = R.pivot(index='feature', columns='perturbation', values='rho')[order]
print('\nSpearman rho median per perturbation:\n', piv2.median().round(2).to_string())
