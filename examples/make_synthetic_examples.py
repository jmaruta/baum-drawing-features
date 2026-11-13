"""Synthetic 'hand-drawn' tree images for the pipeline figure (no participant data).
Usage: python3 -I synth_trees.py <out_dir>"""
import sys, os, numpy as np, cv2
out = sys.argv[1]; os.makedirs(os.path.join(out, 'x'), exist_ok=True)
rng = np.random.default_rng(3)


def stroke(img, pts, width, dark, jitter=1.0, tremor=0.0):
    pts = np.asarray(pts, float)
    # resample densely
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    L = seg[-1]; t = np.linspace(0, L, max(int(L / 1.5), 4))
    x = np.interp(t, seg, pts[:, 0]); y = np.interp(t, seg, pts[:, 1])
    n = len(t)
    low = np.cumsum(rng.normal(0, jitter * 0.15, n)); low -= np.linspace(0, low[-1], n)
    hi = tremor * np.sin(np.linspace(0, n / 2.2, n) + rng.uniform(0, 6)) * rng.uniform(0.6, 1.4)
    nx = np.gradient(y); ny = -np.gradient(x); nn = np.hypot(nx, ny) + 1e-9
    x = x + (low + hi) * nx / nn; y = y + (low + hi) * ny / nn
    p = np.c_[x, y].astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], False, int(255 - dark), int(width), cv2.LINE_AA)


def paper(h, w, shade=0.0):
    base = np.full((h, w), 245, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    base -= shade * 25 * (xx / w)                       # scanner shading
    base += rng.normal(0, 2.5, (h, w))
    return base


def tree1():  # closed crown, pen, ground line
    h, w = 720, 620; img = np.full((h, w), 255, np.uint8)
    cx = 300
    stroke(img, [(cx - 28, 690), (cx - 22, 420), (cx - 40, 330)], 3, 200, 1.0)
    stroke(img, [(cx + 28, 690), (cx + 22, 420), (cx + 40, 330)], 3, 200, 1.0)
    ang = np.linspace(-0.1, np.pi + 0.1, 40)
    crown = [(cx + 200 * np.cos(a) + 18 * np.sin(5 * a), 290 - 150 * np.sin(a) - 14 * np.cos(7 * a)) for a in ang]
    crown = [(cx - 40, 330)] + crown[::-1] + [(cx + 40, 330)]
    stroke(img, crown, 3, 200, 1.2)
    for a in [-1.3, -0.6, 0.0, 0.6, 1.3]:
        stroke(img, [(cx + 10 * a, 330), (cx + 120 * np.sin(a), 330 - 110 * np.cos(a))], 2, 190, 1.0)
    stroke(img, [(40, 695), (580, 692)], 3, 200, 1.0)
    stroke(img, [(cx - 28, 690), (cx - 90, 705)], 3, 200); stroke(img, [(cx + 28, 690), (cx + 85, 708)], 3, 200)
    return img, 0.3


def tree2():  # open crown, branches with twigs, pencil, no ground
    h, w = 760, 560; img = np.full((h, w), 255, np.uint8)
    cx = 270
    stroke(img, [(cx - 18, 740), (cx - 14, 400)], 2, 120, 1.3, 0.6)
    stroke(img, [(cx + 18, 740), (cx + 12, 400)], 2, 120, 1.3, 0.6)
    def branch(p0, ang, L, depth, wdt):
        p1 = (p0[0] + L * np.sin(ang), p0[1] - L * np.cos(ang))
        stroke(img, [p0, p1], wdt, 110, 1.2, 0.8)
        if depth > 0:
            for da in (-0.55, 0.5):
                branch(p1, ang + da + rng.normal(0, 0.1), L * 0.62, depth - 1, max(1, wdt - 1))
    branch((cx - 14, 400), -0.5, 150, 3, 3); branch((cx + 12, 400), 0.45, 150, 3, 3); branch((cx, 400), 0.0, 170, 3, 3)
    return img, 0.6


def tree3():  # faint pencil, shaded crown, small loops (fruit)
    h, w = 700, 600; img = np.full((h, w), 255, np.uint8)
    cx = 300
    stroke(img, [(cx - 24, 680), (cx - 20, 380)], 2, 70, 1.2, 0.5)
    stroke(img, [(cx + 24, 680), (cx + 20, 380)], 2, 70, 1.2, 0.5)
    ang = np.linspace(0, 2 * np.pi, 60)
    crown = [(cx + 170 * np.cos(a) + 10 * np.sin(6 * a), 250 - 130 * np.sin(a)) for a in ang]
    stroke(img, crown, 2, 75, 1.5, 0.5)
    for i in range(14):   # shading strokes
        y0 = 150 + i * 14
        stroke(img, [(cx - 150 + rng.uniform(-10, 10), y0), (cx + 150 + rng.uniform(-10, 10), y0 + 6)], 1, 45, 1.5, 0.3)
    for _ in range(9):    # fruit
        fx, fy = cx + rng.uniform(-120, 120), 250 + rng.uniform(-90, 90)
        a = np.linspace(0, 2 * np.pi, 20)
        stroke(img, [(fx + 11 * np.cos(t), fy + 11 * np.sin(t)) for t in a], 2, 80, 0.6, 0.2)
    stroke(img, [(cx - 24, 680), (cx - 110, 700)], 2, 70); stroke(img, [(cx + 24, 680), (cx + 100, 698)], 2, 70)
    return img, 0.5


for name, fn in [('S-01', tree1), ('S-02', tree2), ('S-03', tree3)]:
    img, shade = fn()
    pap = paper(*img.shape, shade)
    g = np.minimum(pap, img.astype(np.float32) + rng.normal(0, 3, img.shape))
    g = cv2.GaussianBlur(g, (0, 0), 0.9)
    cv2.imwrite(os.path.join(out, 'x', name + '.png'), np.clip(g, 0, 255).astype(np.uint8))
print('ok')
