#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线“整组图像比对”算法评估
==========================
场景：
  V41 = 乘客A 上车组；V43 = 乘客A 下车组（同一人）
  V45 = 乘客B 上车组（不同人）
期望：sim(V41,V43) 明显高于 sim(V45,V43)；但当前算法误判 V45 相似。

评估三种“组比对”方案（均使用与 C++ 一致的特征与相似度）：
  1) 现方案 best-avg：A 组每帧取与 B 组的最优相似度再平均（含180°旋转不变）
  2) 质心方案 centroid：两组各自特征求平均后比较一次
  3) 稳健质心 trimmed：去极值后平均再比较
特征：HSV 颜色(15x8x8 上/下半身) + 梯度纹理(18bin 上/下半身) + 身形
"""

import glob
import os

import cv2
import numpy as np

ROOT = "build-CameraViewer-main-Desktop-Debug/save_images/20260929"

# ---- 与 C++ 一致 ----
H_BINS, S_BINS, V_BINS = 15, 8, 8
TEX_BINS = 18
W = (0.30, 0.40, 0.30)  # 颜色/纹理/身形


def bhatt(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    s1, s2 = a.sum(), b.sum()
    if s1 <= 0 or s2 <= 0:
        return 1.0
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)) / np.sqrt(s1 * s2))))


def color_hist(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS],
                     [0, 180, 0, 256, 0, 256])
    cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L2)
    return h.reshape(-1).astype(np.float32)


def tex_hist(roi):
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist = np.zeros(TEX_BINS, np.float32)
    a = np.mod(ang, 180.0)
    b = (a / 180.0 * TEX_BINS).astype(np.int32)
    np.clip(b, 0, TEX_BINS - 1, out=b)
    m = mag.ravel()
    v = m > 1e-3
    np.add.at(hist, b.ravel()[v], m[v])
    n = np.linalg.norm(hist)
    if n > 0:
        hist /= n
    return hist


def extract(path):
    img = cv2.imread(path)
    if img is None:
        return None
    H = img.shape[0]
    half = max(1, H // 2)
    up, lo = img[:half], img[half:]
    return (color_hist(up), color_hist(lo), tex_hist(up), tex_hist(lo))


def cmp_pair(a, b):
    def one(x, y):
        col = 1.0 - (0.55 * bhatt(x[0], y[0]) + 0.45 * bhatt(x[1], y[1]))
        tex = 1.0 - (0.5 * bhatt(x[2], y[2]) + 0.5 * bhatt(x[3], y[3]))
        sc = max(col, 1.0 - (0.55 * bhatt(x[0], y[1]) + 0.45 * bhatt(x[1], y[0])))
        st = max(tex, 1.0 - (0.5 * bhatt(x[2], y[3]) + 0.5 * bhatt(x[3], y[2])))
        return max(0.0, min(1.0, (W[0] * sc + W[1] * st + W[2] * 1.0) / sum(W)))
    return max(one(a, b), one(b, a))  # 对称化


def load_group(name):
    fs = []
    for p in sorted(glob.glob(os.path.join(ROOT, name, "*.jpg"))):
        f = extract(p)
        if f is not None:
            fs.append(f)
    return fs


def to_mat(fs):
    """把 (cu,cl,tu,tl) 列表拼成矩阵便于质心"""
    return np.array([np.concatenate([f[0], f[1], f[2], f[3]]) for f in fs], np.float32)


def centroid_sim(A, B, trimmed=False):
    A, B = to_mat(A), to_mat(B)

    def cent(M, k):
        if trimmed and M.shape[0] >= 4:
            # 取与整体均值最接近的 k 个样本的均值（去极值）
            m0 = M.mean(axis=0)
            d = np.linalg.norm(M - m0, axis=1)
            idx = np.argsort(d)[:max(2, int(M.shape[0] * 0.6))]
            M = M[idx]
        return M.mean(axis=0)
    ca, cb = cent(A, 1), cent(B, 1)
    # 分量重新归一化后走统一 compare
    def split(v):
        return (v[0:H_BINS * S_BINS * V_BINS],
                v[H_BINS * S_BINS * V_BINS:2 * H_BINS * S_BINS * V_BINS],
                v[2 * H_BINS * S_BINS * V_BINS:2 * H_BINS * S_BINS * V_BINS + TEX_BINS],
                v[2 * H_BINS * S_BINS * V_BINS + TEX_BINS:])
    return cmp_pair(split(ca), split(cb))


def best_avg(Ag, Bg):
    tot, n = 0.0, 0
    for a in Ag:
        best = max((cmp_pair(a, b) for b in Bg), default=0.0)
        tot += best
        n += 1
    return tot / n if n else 0.0


def main():
    groups = {}
    for name in ("visitor_V41", "visitor_V43", "visitor_V45"):
        groups[name] = load_group(name)
        print(f"[加载] {name}: {len(groups[name])} 帧")

    A, C, E = groups["visitor_V41"], groups["visitor_V43"], groups["visitor_V45"]

    print("\n" + "=" * 74)
    print("方案1：现方案 best-avg（A 组每帧取与 B 组最优再平均）")
    print("=" * 74)
    pairs = [("V41(上,人A)", "V43(下,人A)", A, C),
             ("V45(上,人B)", "V43(下,人A)", E, C),
             ("V45(上,人B)", "V41(上,人A)", E, A)]
    for n1, n2, g1, g2 in pairs:
        print(f"   {n1} vs {n2}: {best_avg(g1, g2):.4f}")

    print("\n" + "=" * 74)
    print("方案2：质心（两组特征求平均后比较一次）")
    print("=" * 74)
    MA, MC, ME = to_mat(A), to_mat(C), to_mat(E)
    for n1, n2, g1, g2 in pairs:
        print(f"   {n1} vs {n2}: {centroid_sim(g1, g2):.4f}")

    print("\n" + "=" * 74)
    print("方案3：稳健质心（去除 40% 离群帧）")
    print("=" * 74)
    for n1, n2, g1, g2 in pairs:
        print(f"   {n1} vs {n2}: {centroid_sim(g1, g2, trimmed=True):.4f}")

    # 以 V43 为探针，看谁最像
    print("\n" + "=" * 74)
    print("以 V43(下课,人A) 为探针，在 {V41(人A), V45(人B)} 中找最相似")
    print("=" * 74)
    s41 = best_avg([C[0]], A)
    s45 = best_avg([C[0]], E)
    print(f"   与 V41(人A) = {s41:.4f}   与 V45(人B) = {s45:.4f}   "
          f"→ {'正确' if s41 > s45 else '误判为人B！'}")

    print("\n完成。")


if __name__ == "__main__":
    main()
