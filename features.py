"""Baum (tree-drawing) test: pre-processing and size-invariant feature extraction from scanned drawings.

Usage:
    python features.py <image_folder_or_file> <out_csv> [debug_dir]

<image_folder_or_file>  a PNG/JPG/TIFF file, a folder of images, or a folder of sub-folders of images
<out_csv>               output table, one row per drawing (column ID = file name without extension)
[debug_dir]             optional; writes the binarised drawing of every image for visual checking

The 27 features are defined in the accompanying paper (Table 2) and in README.md.
No training data are needed; cut-offs for classifying qualitative signs must be calibrated
against human coding (see analysis/kappa_cv.py).
"""
import sys, glob, os
import numpy as np, pandas as pd, cv2
from PIL import Image
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy import ndimage as ndi

NORM = 512  # long side (px) of the tree after normalisation


def load_gray(f):
    im = Image.open(f).convert('RGBA')
    a = np.asarray(im).astype(np.float32)
    rgb, al = a[..., :3], a[..., 3:] / 255.0
    g = (rgb * al + 255 * (1 - al)).mean(-1)          # composite on white
    h, w = g.shape
    s = 1000 / max(h, w)
    if s < 1:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return g


def normalise_bg(g):
    """flatten illumination/ scanner shading: divide by large-kernel background estimate"""
    k = max(15, (min(g.shape) // 8) | 1)
    bg = cv2.medianBlur(np.clip(g, 0, 255).astype(np.uint8), min(k, 255) if k <= 255 else 255).astype(np.float32)
    bg = np.maximum(bg, cv2.GaussianBlur(g, (0, 0), k / 2))
    n = np.clip(g / np.maximum(bg, 1) * 255, 0, 255)
    return n


def binarise(n):
    """contrast-adaptive hysteresis threshold.
    darkness is scaled by the image's own ink level (99.7th pct) so that faint pencil scans
    and dark pen scans are treated alike; weak pixels are kept only if connected to strong ones."""
    from skimage.filters import apply_hysteresis_threshold
    dark = cv2.GaussianBlur(255 - n, (0, 0), 0.7)
    hi = np.percentile(dark, 99.7)
    hi = max(hi, 12)
    high = max(0.40 * hi, 10)
    low = max(0.14 * hi, 5)
    b = apply_hysteresis_threshold(dark, low, high)
    return b, hi


def remove_border_lines(b):
    """remove straight paper-edge / ruled lines that hug the image border"""
    h, w = b.shape
    out = b.copy()
    for axis, L in ((0, h), (1, w)):
        prof = b.mean(axis=1 - axis) if axis == 0 else b.mean(axis=0)
    # rows
    rows = b.mean(1)
    for r in np.where(rows > 0.6)[0]:
        if r < 0.12 * h or r > 0.88 * h:
            out[max(0, r - 2):r + 3] = False
    cols = b.mean(0)
    for c in np.where(cols > 0.6)[0]:
        if c < 0.12 * w or c > 0.88 * w:
            out[:, max(0, c - 2):c + 3] = False
    return out


def clean(b):
    h, w = b.shape
    lab = label(b, connectivity=2)
    props = regionprops(lab)
    if not props:
        return b
    # scanner shadows: solid blobs touching the image border (corner triangles, edge bands)
    for p in props:
        r0, c0, r1, c1 = p.bbox
        touch = (r0 == 0) + (c0 == 0) + (r1 == h) + (c1 == w)
        if touch >= 1 and p.solidity > 0.8 and p.area > 0.002 * h * w:
            b[lab == p.label] = False
    lab = label(b, connectivity=2)
    props = regionprops(lab)
    if not props:
        return b
    areas = np.array([p.area for p in props])
    big = props[int(areas.argmax())]
    minr, minc, maxr, maxc = big.bbox
    keep = np.zeros_like(b)
    tot = areas.sum()
    for p in props:
        if p.area < max(6, 0.0005 * tot * 10):   # specks
            continue
        r0, c0, r1, c1 = p.bbox
        # drop tiny compact blobs far outside the main drawing (text, stamps)
        far = (c1 < minc - 0.25 * (maxc - minc)) or (c0 > maxc + 0.25 * (maxc - minc)) or \
              (r1 < minr - 0.25 * (maxr - minr)) or (r0 > maxr + 0.25 * (maxr - minr))
        if far and p.area < 0.05 * tot:
            continue
        keep[lab == p.label] = True
    return keep


def crop(b, n):
    ys, xs = np.where(b)
    if len(ys) == 0:
        return b, n
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return b[y0:y1, x0:x1], n[y0:y1, x0:x1]


def to_norm(b):
    h, w = b.shape
    s = NORM / max(h, w)
    r = cv2.resize(b.astype(np.uint8) * 255, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                   interpolation=cv2.INTER_AREA) > 100
    return r, s


def skel_graph(sk):
    k = np.ones((3, 3), int); k[1, 1] = 0
    nb = ndi.convolve(sk.astype(int), k, mode='constant') * sk
    ends = (nb == 1)
    junc = (nb >= 3)
    return nb, ends, junc


def segments(sk, junc):
    """skeleton split at junctions -> list of ordered pixel paths"""
    s = sk & ~ndi.binary_dilation(junc, np.ones((3, 3)))
    lab = label(s, connectivity=2)
    paths = []
    for p in regionprops(lab):
        if p.area < 8:
            continue
        coords = p.coords
        # order pixels along the path starting from an end
        cs = set(map(tuple, coords))
        nbc = {c: [(c[0] + dy, c[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if (dy or dx) and (c[0] + dy, c[1] + dx) in cs] for c in cs}
        start = next((c for c in cs if len(nbc[c]) == 1), next(iter(cs)))
        path, seen, cur = [start], {start}, start
        while True:
            nxt = [q for q in nbc[cur] if q not in seen]
            if not nxt:
                break
            cur = nxt[0]; seen.add(cur); path.append(cur)
        if len(path) >= 8:
            paths.append(np.array(path, float))
    return paths


def tremor(paths):
    """jaggedness: raw path length / length of the smoothed path (>=1); and
    high-frequency turning energy. Computed on paths >= 20 px at NORM scale."""
    ratios, turn = [], []
    for p in paths:
        if len(p) < 20:
            continue
        sm = ndi.gaussian_filter1d(p, 3, axis=0, mode='nearest')
        raw = np.sum(np.hypot(*np.diff(p, axis=0).T))
        smo = np.sum(np.hypot(*np.diff(sm, axis=0).T))
        if smo > 0:
            ratios.append(raw / smo)
        # turning of lightly smoothed path minus heavily smoothed path
        s1 = ndi.gaussian_filter1d(p, 1.5, axis=0, mode='nearest')
        s4 = ndi.gaussian_filter1d(p, 6, axis=0, mode='nearest')
        def ang(q):
            d = np.diff(q[::2], axis=0); a = np.unwrap(np.arctan2(d[:, 0], d[:, 1])); return a
        a1, a4 = ang(s1), ang(s4)
        m = min(len(a1), len(a4))
        turn.append(np.mean(np.abs(np.diff(a1[:m] - a4[:m]))))
    if not ratios:
        return np.nan, np.nan
    return float(np.median(ratios)), float(np.median(turn))


def crown_feats(bn):
    """crown closure without relying on a perfectly closed contour.
    env_cov: share of the crown envelope (closing r=30px at NORM) that lies on drawn ink
    -> outlined crowns ~1, branch-only crowns lower. hole_seal: enclosed area after
    bridging small gaps and sealing the cut edge."""
    h, w = bn.shape
    up = bn[:max(2, int(h * 0.6))]
    R = 30
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1))
    pad = cv2.copyMakeBorder(up.astype(np.uint8), R, R, R, R, cv2.BORDER_CONSTANT, 0)
    env = ndi.binary_fill_holes(cv2.morphologyEx(pad, cv2.MORPH_CLOSE, k)[R:-R, R:-R] > 0)
    bd = env & ~ndi.binary_erosion(env); bd[-1, :] = False
    dist = ndi.distance_transform_edt(~up)
    o = {}
    o['crown_env_cov'] = float((dist[bd] <= 4).mean()) if bd.any() else np.nan
    o['crown_env_md'] = float(np.median(dist[bd])) if bd.any() else np.nan
    o['crown_env_frac'] = float(env.mean())
    k6 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    c = cv2.morphologyEx(up.astype(np.uint8), cv2.MORPH_CLOSE, k6) > 0
    c[-1, :] = True
    o['crown_hole_seal'] = float((ndi.binary_fill_holes(c) & ~c).sum() / up.size)
    return o


def crossings(row):
    r = row.astype(int)
    return int(np.sum(np.diff(np.r_[0, r, 0]) == 1))


def features(f):
    g = load_gray(f)
    n = normalise_bg(g)
    b, thr = binarise(n)
    b = remove_border_lines(b)
    b = clean(b)
    b, n = crop(b, n)
    H, W = b.shape
    out = {'ink_level': float(thr), 'res_long': int(max(Image.open(f).size)), 'res_tree': int(max(H, W))}
    if b.sum() < 30:
        return None, None
    # ---------- line-quality on (≤1000px) working resolution ----------
    dt = ndi.distance_transform_edt(b)
    sk0 = skeletonize(b)
    sw = 2 * dt[sk0]                       # local stroke width (px)
    out['stroke_w_rel'] = np.median(sw) / max(H, W) * 1000   # stroke width per 1000 px of tree size
    ink = n[b]
    paper = np.median(n[~b]) if (~b).sum() else 255
    out['ink_dark'] = float(paper - np.median(ink))          # pressure proxy
    out['ink_dark_cv'] = float(np.std(255 - ink) / max(np.mean(255 - ink), 1))  # pressure variability
    out['stroke_w_cv'] = float(np.std(sw) / max(np.mean(sw), 1e-6))
    # shading / filled areas: pixels whose local thickness >> median stroke width
    out['fill_ratio'] = float((dt > 2.5 * max(np.median(dt[sk0]), 1)).sum() / b.sum())
    # ---------- structural features at normalised scale ----------
    bn, s = to_norm(b)
    hN, wN = bn.shape
    sk = skeletonize(bn)
    L = sk.sum()
    nb, ends, junc = skel_graph(sk)
    jl = label(junc, connectivity=2).max()
    out['aspect_hw'] = H / W
    out['ink_len'] = L / NORM                                   # total line length / tree size
    out['ink_cover'] = bn.mean()                                # ink density in bbox
    out['endpoints_per_len'] = ends.sum() / max(L, 1) * 100
    out['junctions_per_len'] = jl / max(L, 1) * 100
    lab = label(bn, connectivity=2)
    areas = np.bincount(lab.ravel())[1:]
    out['n_strokes'] = int(lab.max())
    out['frag_small'] = float((areas < 0.002 * bn.sum() * 50).mean()) if len(areas) else 0
    # gaps: endpoint pairs closer than 3% of size but on different components or far along skeleton
    ey, ex = np.where(ends)
    gaps = 0
    if len(ey) > 1:
        P = np.c_[ey, ex]; lbl = lab[ey, ex]
        d = np.hypot(*(P[:, None, :] - P[None, :, :]).transpose(2, 0, 1))
        close = (d < 0.03 * NORM) & (d > 1.5)
        diffc = lbl[:, None] != lbl[None, :]
        gaps = int(np.triu(close & diffc).sum())
    out['gaps_per_len'] = gaps / max(L, 1) * 100
    # enclosed regions (holes) -> crown closure & small loops (leaves/fruit)
    filled = ndi.binary_fill_holes(bn)
    holes = filled & ~bn
    hl = label(holes, connectivity=1)
    ha = np.bincount(hl.ravel())[1:] if hl.max() else np.array([])
    bbox_area = hN * wN
    out.update(crown_feats(bn))
    out['small_loops'] = int(((ha > 15) & (ha < 0.01 * bbox_area)).sum()) if len(ha) else 0
    # tremor / jaggedness
    paths = segments(sk, junc)
    out['jagged'], out['turn_hf'] = tremor(paths)
    out['short_seg_frac'] = float(np.mean([len(p) < 0.03 * NORM for p in paths])) if paths else np.nan
    # ---------- layout inside the drawing (scale-free) ----------
    ys, xs = np.where(bn)
    base = bn[int(hN * 0.85):]
    by, bx = np.where(base)
    axis = np.median(bx) if len(bx) else wN / 2
    out['lr_balance'] = float((xs < axis).sum() / max(len(xs), 1))
    out['lr_asym'] = abs(out['lr_balance'] - 0.5)
    out['ink_top_half'] = float((ys < hN / 2).mean())
    out['centroid_y'] = float(ys.mean() / hN)
    # trunk: crossings of filled shape rows at 10-30% height from the bottom
    rowsel = [int(hN * (1 - q)) for q in (0.12, 0.18, 0.24, 0.30)]
    cr = [crossings(sk[min(r, hN - 1)]) for r in rowsel]
    out['trunk_crossings'] = float(np.median(cr))
    widths = []
    for r in rowsel:
        xx = np.where(filled[min(r, hN - 1)] | bn[min(r, hN - 1)])[0]
        widths.append((xx.max() - xx.min() + 1) if len(xx) else 0)
    trunk_w = np.median(widths) if widths else 0
    top = filled[:hN // 2] | bn[:hN // 2]
    crown_w = max(((np.where(r)[0].max() - np.where(r)[0].min() + 1) if r.any() else 0) for r in top) if top.size else 0
    out['crown_trunk_ratio'] = float(crown_w / max(trunk_w, 1))
    # ground line: long near-horizontal ink in bottom 12%
    bot = bn[int(hN * 0.88):]
    out['ground_auto'] = float(bot.any(0).mean())               # horizontal extent of bottom band
    return out, bn


def main():
    root, outcsv = sys.argv[1], sys.argv[2]
    dbg = sys.argv[3] if len(sys.argv) > 3 else None
    if dbg:
        os.makedirs(dbg, exist_ok=True)
    rows = []
    exts = ('*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.tif', '*.tiff')
    if os.path.isfile(root):
        fs = [root]
    else:
        fs = []
        for e in exts:
            fs += glob.glob(os.path.join(root, e)) + glob.glob(os.path.join(root, '*', e))
        fs = sorted(set(fs))
    if not fs:
        print('no images found under', root); return
    for i, f in enumerate(fs):
        try:
            o, bn = features(f)
        except Exception as e:
            print('ERR', f, e); o, bn = None, None
        if o is None:
            continue
        o['ID'] = os.path.splitext(os.path.basename(f))[0]
        rows.append(o)
        if dbg and bn is not None:
            cv2.imwrite(os.path.join(dbg, o['ID'] + '.png'), (~bn).astype(np.uint8) * 255)
        if i % 50 == 0:
            print(i, flush=True)
    pd.DataFrame(rows).to_csv(outcsv, index=False)
    print('done', len(rows))


if __name__ == '__main__':
    main()
