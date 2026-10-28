#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间分块描述子实验：能否把「带印花白T(V45)」与「纯白T(V41/V43)」区分开
=====================================================================
saveVisitorFrame 以 bbox 为中心按 0.3 比例对称 padding，故可通过
取裁剪图中心 1/1.6≈62.5% 还原出近似的人体框（未触及画面边缘时成立）。

对比两种描述子（组比对均用 best-avg 与 centroid）：
  A) 现状：上/下半身 2 块  颜色(15x8x8)+纹理(18)
  B) 空间：3列 x 2行 = 6 块，每块 颜色(8x4x4)+纹理(8)
"""

import glob
import os

import cv2
import numpy as np

ROOT = "build-CameraViewer-main-Desktop-Debug/save_images/20260929"


def bhatt(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    s1, s2 = a.sum(), b.sum()
    if s1 <= 0 or s2 <= 0:
        return 1.0
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)) / np.sqrt(s1 * s2))))


def color_hist(roi, hb, sb, vb):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [hb, sb, vb], [0, 180, 0, 256, 0, 256])
    cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L2)
    return h.reshape(-1).astype(np.float32)


def tex_hist(roi, nb):
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist = np.zeros(nb, np.float32)
    aa = np.mod(ang, 180.0)
    b = (aa / 180.0 * nb).astype(np.int32)
    np.clip(b, 0, nb - 1, out=b)
    m = mag.ravel()
    v = m > 1e-3
    np.add.at(hist, b.ravel()[v], m[v])
    n = np.linalg.norm(hist)
    if n > 0:
        hist /= n
    return hist


def tight(img):
    """还原近似人体框：中心 62.5%"""
    h, w = img.shape[:2]
    return img[int(h * 0.1875):int(h * 0.8125), int(w * 0.1875):int(w * 0.8125)]


# ---------- A) 2 块（现状） ----------
def featA(img):
    r = tight(img)
    H = r.shape[0]
    half = max(1, H // 2)
    up, lo = r[:half], r[half:]
    return {"c": (color_hist(up, 15, 8, 8), color_hist(lo, 15, 8, 8)),
            "t": (tex_hist(up, 18), tex_hist(lo, 18))}


# ---------- B) 3列x2行 = 6 块 ----------
def featB(img):
    r = tight(img)
    H, Wd = r.shape[:2]
    cells = []
    for ry in range(2):
        for cx in range(3):
            y1, y2 = ry * H // 2, (ry + 1) * H // 2
            x1, x2 = cx * Wd // 3, (cx + 1) * Wd // 3
            cell = r[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            cells.append((color_hist(cell, 8, 4, 4), tex_hist(cell, 8)))
    return cells


def cmpA(a, b):
    def one(x, y):
        c = 1.0 - (0.55 * bhatt(x["c"][0], y["c"][0]) + 0.45 * bhatt(x["c"][1], y["c"][1]))
        t = 1.0 - (0.5 * bhatt(x["t"][0], y["t"][0]) + 0.5 * bhatt(x["t"][1], y["t"][1]))
        cr = 1.0 - (0.55 * bhatt(x["c"][0], y["c"][1]) + 0.45 * bhatt(x["c"][1], y["c"][0]))
        return max(0.0, min(1.0, (0.30 * max(c, cr) + 0.40 * t + 0.30) / 1.0))
    return max(one(a, b), one(b, a))


def cmpB(a, b):
    # a,b: list of (color, tex) cells, 顺序一致
    sims = []
    for ca, ta in a:
        best = 0.0
        for cb, tb in b:
            s = 0.45 * (1 - bhatt(ca, cb)) + 0.55 * (1 - bhatt(ta, tb))
            best = max(best, s)
        sims.append(best)
    return float(np.mean(sims)) if sims else 0.0


def load(name, featfn):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, name, "*.jpg"))):
        img = cv2.imread(p)
        if img is not None:
            out.append(featfn(img))
    return out


def best_avg(A, B, cmpfn):
    tot = 0.0
    for a in A:
        tot += max((cmpfn(a, b) for b in B), default=0.0)
    return tot / len(A) if A else 0.0


def centroid_sim(A, B, tovec, cmpfn):
    ma = np.mean(np.array([tovec(f) for f in A]), axis=0)
    mb = np.mean(np.array([tovec(f) for f in B]), axis=0)
    return cmpfn(tovec_inv(ma) if False else ma, mb) if False else None


def main():
    gA = {n: load(n, featA) for n in ("visitor_V41", "visitor_V43", "visitor_V45")}
    gB = {n: load(n, featB) for n in ("visitor_V41", "visitor_V43", "visitor_V45")}
    print(f"[加载] A特征: " + ", ".join(f"{k}={len(v)}" for k, v in gA.items()))
    print(f"[加载] B特征: " + ", ".join(f"{k}={len(v)}" for k, v in gB.items()))

    pairs = [("V41(人A) vs V43(人A)", "visitor_V41", "visitor_V43"),
             ("V45(人B) vs V43(人A)", "visitor_V45", "visitor_V43"),
             ("V45(人B) vs V41(人A)", "visitor_V45", "visitor_V41")]

    print("\n=== 方案A：现状 2 块描述子（best-avg）===")
    for label, n1, n2 in pairs:
        print(f"   {label}: {best_avg(gA[n1], gA[n2], cmpA):.4f}")

    print("\n=== 方案B：空间 3x2 块描述子（best-avg）===")
    for label, n1, n2 in pairs:
        print(f"   {label}: {best_avg(gB[n1], gB[n2], cmpB):.4f}")

    print("\n=== 方案B'：空间分块 + 逐格对应（不做跨格最优，按相同格子比）===")
    def cmpB2(a, b):
        n = min(len(a), len(b))
        return float(np.mean([0.45 * (1 - bhatt(a[i][0], b[i][0])) + 0.55 * (1 - bhatt(a[i][1], b[i][1]))
                              for i in range(n)])) if n else 0.0
    for label, n1, n2 in pairs:
        print(f"   {label}: {best_avg(gB[n1], gB[n2], cmpB2):.4f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
