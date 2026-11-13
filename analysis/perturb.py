"""Measurement robustness: re-extract features from perturbed copies of every image and
compare with the original (ICC(2,1) absolute agreement, Spearman rho, median relative change).
Usage: python3 -I perturb.py <features.py dir> <img_root> <out_dir>
"""
import sys, os, glob, io, importlib.util
import numpy as np, pandas as pd, cv2
from PIL import Image
from multiprocessing import Pool

fdir, root, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location('features', os.path.join(fdir, 'features.py'))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)

FEATS = ['stroke_w_rel', 'ink_dark', 'ink_dark_cv', 'stroke_w_cv', 'fill_ratio', 'ink_len', 'ink_cover',
         'endpoints_per_len', 'junctions_per_len', 'n_strokes', 'frag_small', 'gaps_per_len', 'crown_env_cov',
         'crown_env_md', 'crown_env_frac', 'crown_hole_seal', 'small_loops', 'jagged', 'turn_hf', 'short_seg_frac',
         'lr_asym', 'ink_top_half', 'centroid_y', 'trunk_crossings', 'crown_trunk_ratio', 'ground_auto', 'aspect_hw']


def load_rgb(f):
    im = Image.open(f).convert('RGBA'); a = np.asarray(im).astype(np.float32)
    rgb, al = a[..., :3], a[..., 3:] / 255.0
    return np.clip(rgb * al + 255 * (1 - al), 0, 255).astype(np.uint8)


def perturb(img, kind):
    h, w = img.shape[:2]
    if kind == 'orig':
        return img
    if kind == 'down50':
        return cv2.resize(img, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    if kind == 'down25':
        return cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    if kind == 'up200':
        return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    if kind in ('rot+3', 'rot-3'):
        ang = 3 if kind == 'rot+3' else -3
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    if kind == 'lighter':      # faint scan: gamma < 1 on reflectance lifts ink towards paper
        return np.clip(255 * (img / 255.0) ** 0.5, 0, 255).astype(np.uint8)
    if kind == 'darker':
        return np.clip(255 * (img / 255.0) ** 1.8, 0, 255).astype(np.uint8)
    if kind == 'jpeg50':
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == 'noise':
        rng = np.random.default_rng(abs(hash(kind)) % 2**32)
        return np.clip(img.astype(np.float32) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    if kind == 'blur':
        return cv2.GaussianBlur(img, (0, 0), 0.8)
    if kind == 'rescan':       # combination: slight blur + noise + jpeg + mild lightening
        x = cv2.GaussianBlur(img, (0, 0), 0.6)
        rng = np.random.default_rng(1)
        x = np.clip(x.astype(np.float32) + rng.normal(0, 5, x.shape), 0, 255).astype(np.uint8)
        x = np.clip(255 * (x / 255.0) ** 0.8, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    raise ValueError(kind)


KINDS = ['orig', 'down50', 'down25', 'up200', 'rot+3', 'rot-3', 'lighter', 'darker', 'jpeg50', 'noise', 'blur', 'rescan']
tmp = os.path.join(out, 'tmp'); os.makedirs(tmp, exist_ok=True)


def run(f):
    img = load_rgb(f); ID = os.path.basename(f)[:-4]
    rows = []
    for k in KINDS:
        p = os.path.join(tmp, f'{ID}__{k}.png')
        cv2.imwrite(p, perturb(img, k))
        try:
            o, _ = F.features(p)
        except Exception as e:
            o = None
        os.remove(p)
        if o is None:
            continue
        o.update(ID=ID, kind=k); rows.append(o)
    return rows


if __name__ == '__main__':
    fs = sorted(glob.glob(os.path.join(root, '*', '*.png')))
    with Pool(min(8, os.cpu_count())) as pool:
        res = pool.map(run, fs, chunksize=4)
    L = pd.DataFrame([r for rr in res for r in rr])
    L.to_csv(os.path.join(out, 'perturb_long.csv'), index=False)
    # agreement per feature x perturbation
    import pingouin as pg
    O = L[L.kind == 'orig'].set_index('ID')
    rows = []
    for k in KINDS[1:]:
        P = L[L.kind == k].set_index('ID')
        ids = O.index.intersection(P.index)
        for ft in FEATS:
            a, b = O.loc[ids, ft].astype(float), P.loc[ids, ft].astype(float)
            m = a.notna() & b.notna()
            a, b = a[m], b[m]
            if len(a) < 10 or a.std() == 0:
                continue
            d = pd.DataFrame({'t': np.r_[a.index, b.index], 'r': ['o'] * len(a) + ['p'] * len(b), 'v': np.r_[a.values, b.values]})
            try:
                icc = pg.intraclass_corr(d, targets='t', raters='r', ratings='v')
                icc21 = float(icc.loc[icc.Type == 'ICC2', 'ICC'].iloc[0])
            except Exception:
                icc21 = np.nan
            rho = a.corr(b, method='spearman')
            rel = np.median(np.abs(b - a) / (np.abs(a) + 1e-9)) if (np.abs(a) > 1e-9).mean() > 0.5 else np.nan
            bias = float(np.median(b - a) / (a.std() + 1e-12))     # median shift in SD units
            rows.append(dict(perturbation=k, feature=ft, n=int(len(a)), icc21=round(icc21, 3), rho=round(rho, 3),
                             med_rel_change=round(float(rel), 3) if rel == rel else np.nan, bias_sd=round(bias, 3)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, 'perturb_agreement.csv'), index=False)
    piv = R.pivot(index='feature', columns='perturbation', values='icc21')
    piv.to_csv(os.path.join(out, 'perturb_icc_matrix.csv'))
    print(piv.round(2).to_string())
    print('\nmedian ICC per perturbation:\n', piv.median().round(2).to_string())
    print('\nmedian ICC per feature:\n', piv.median(axis=1).sort_values().round(2).to_string())
