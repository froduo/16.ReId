#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像级独立验证（对应 visitor_V41 / visitor_V43 实际图片）
==========================================================
不依赖运行时落盘特征，而是直接从 save_images/20260929/visitor_V41（上车）
与 visitor_V43（下车）的图像序列中，按 ReIDExtractor::extract() 的算法
重新提取特征，分别用「旧描述子(30x32x32)」与「新描述子(15x8x8)」计算
同一人的上/下车相似度，以验证修复方向。

说明：这些图是带 padding 的人物裁剪图，这里以整图作为人体框（人物居中），
      因此 shape 分量退化为常量，主要考察颜色的可比性（这正是故障点）。
"""

import glob
import os

import cv2
import numpy as np

V41_DIR = "build-CameraViewer-main-Desktop-Debug/save_images/20260929/visitor_V41"
V43_DIR = "build-CameraViewer-main-Desktop-Debug/save_images/20260929/visitor_V43"


def bhatt(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    s1, s2 = a.sum(), b.sum()
    if s1 <= 0 or s2 <= 0:
        return 1.0
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)) / np.sqrt(s1 * s2))))


def color_hist(roi, hb, sb, vb):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [hb, sb, vb],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L2)
    return hist.reshape(-1).astype(np.float32)


TEX_BINS = 18


def tex_hist(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist = np.zeros(TEX_BINS, np.float32)
    a = np.mod(ang, 180.0)
    b = (a / 180.0 * TEX_BINS).astype(np.int32)
    np.clip(b, 0, TEX_BINS - 1, out=b)
    m = mag.ravel()
    valid = m > 1e-3
    np.add.at(hist, b.ravel()[valid], m[valid])
    n = np.linalg.norm(hist)
    if n > 0:
        hist /= n
    return hist


def extract(img, hb, sb, vb):
    H = img.shape[0]
    half = max(1, H // 2)
    up, lo = img[:half], img[half:]
    cu = color_hist(up, hb, sb, vb)
    cl = color_hist(lo, hb, sb, vb)
    tu = tex_hist(up)
    tl = tex_hist(lo)
    return cu, cl, tu, tl


def color_sim(a, b):
    d = 0.55 * bhatt(a[0], b[0]) + 0.45 * bhatt(a[1], b[1])
    return max(0.0, min(1.0, 1.0 - d))


def tex_sim(a, b):
    d = 0.5 * bhatt(a[2], b[2]) + 0.5 * bhatt(a[3], b[3])
    return max(0.0, min(1.0, 1.0 - d))


def swap(f):
    """上下半身交换（复现 180° 旋转不变）"""
    return (f[1], f[0], f[3], f[2])


def fused(a, b, w):
    sc = color_sim(a, b)
    st = tex_sim(a, b)
    s_rot = color_sim(a, swap(b))
    sc = max(sc, s_rot)
    st = max(st, tex_sim(a, swap(b)))
    return (w[0] * sc + w[1] * st + w[2] * 1.0) / sum(w), sc, st


def load_features(d, hb, sb, vb):
    fs = []
    for p in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        img = cv2.imread(p)
        if img is None:
            continue
        fs.append(extract(img, hb, sb, vb))
    return fs


def main():
    print("=" * 86)
    print("图像级验证：visitor_V41(上车) vs visitor_V43(下车) —— 同一人")
    print("=" * 86)

    configs = [
        ("旧: 颜色30x32x32 权重 .6/.3/.1", 30, 32, 32, (0.6, 0.3, 0.1)),
        ("新: 颜色15x8x8   权重 .3/.4/.3", 15, 8, 8, (0.3, 0.4, 0.3)),
        ("新: 颜色15x8x8   权重 .2/.5/.3", 15, 8, 8, (0.2, 0.5, 0.3)),
    ]

    for name, hb, sb, vb, w in configs:
        f41 = load_features(V41_DIR, hb, sb, vb)
        f43 = load_features(V43_DIR, hb, sb, vb)
        # 序列相似度（探针每帧取与模板最优后平均），与运行时一致
        tot, n = 0.0, 0
        cs, ts = 0.0, 0.0
        for p in f43:
            best = -1
            bc = bt = 0
            for t in f41:
                s, c, tt = fused(p, t, w)
                if s > best:
                    best, bc, bt = s, c, tt
            tot += max(best, 0.0)
            cs += bc
            ts += bt
            n += 1
        print(f"\n  {name}")
        print(f"     帧数: V41={len(f41)}  V43={len(f43)}")
        print(f"     平均颜色相似度 = {cs/n:.4f}")
        print(f"     平均纹理相似度 = {ts/n:.4f}")
        print(f"     >>> 融合序列相似度 = {tot/n:.4f}")

    print("\n完成。若『旧』远低于阈值 0.70，而『新』显著提升并越过校准阈值，即证实修复有效。")


if __name__ == "__main__":
    main()
