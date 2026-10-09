#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复方案离线评估
================
在真实运行时数据（visitor_records.json / passenger_book.json）上，
评估不同「颜色直方图分辨率 + 分量权重」组合下：
  - 真实同人（上车模板 P3 ↔ 下车探针 V43）能否成为 Top-1 且超过阈值；
  - 与其它在车乘客（干扰项）的分差是否足够大。

颜色池化通过在已存储的 30x32x32 直方图上做 bin 求和实现，
与 C++ 端直接减小 H_BINS/S_BINS/V_BINS 等价。
"""

import base64
import json
import os

import numpy as np

BASE = os.path.join("build-CameraViewer-main-Desktop-Debug", "visitor_log")
VISITOR_JSON = os.path.join(BASE, "visitor_records.json")
PASSENGER_JSON = os.path.join(BASE, "passenger_book.json")

H0, S0, V0 = 30, 32, 32
HALF0 = H0 * S0 * V0
COLOR_DIM = HALF0 * 2
TEX_BINS = 18
TEX_DIM = TEX_BINS * 2
TOTAL_DIM = COLOR_DIM + TEX_DIM + 2


def decode_mat(s):
    if not s:
        return None
    idx, cnt = -1, 0
    for i, ch in enumerate(s):
        if ch == ',':
            cnt += 1
            if cnt == 4:
                idx = i + 1
                break
    meta = s[:idx].split(',')
    rows, cols = int(meta[0]), int(meta[1])
    arr = np.frombuffer(base64.b64decode(s[idx:]), dtype=np.float32)
    if arr.size != rows * cols:
        return None
    return arr.reshape(rows, cols).astype(np.float32)


def bhatt(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    s1, s2 = a.sum(), b.sum()
    if s1 <= 0 or s2 <= 0:
        return 1.0
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)) / np.sqrt(s1 * s2))))


def pool(v, fh, fs, fv):
    """把展平的 (H0*S0*V0) 直方图池化到 (H0//fh, S0//fs, V0//fv) 并展平"""
    h = v.reshape(H0, S0, V0)
    H2, S2, V2 = H0 // fh, S0 // fs, V0 // fv
    h = h[:H2 * fh, :S2 * fs, :V2 * fv]
    return h.reshape(H2, fh, S2, fs, V2, fv).sum(axis=(1, 3, 5)).reshape(-1)


class Cand:
    def __init__(self, name, pool_f=(1, 1, 1), w=(0.6, 0.3, 0.1)):
        self.name = name
        self.pool_f = pool_f
        self.w = w

    def _color(self, a, b):
        fh, fs, fv = self.pool_f
        # a,b: 1xTOTAL_DIM
        ua, la = pool(a[0, 0:HALF0], fh, fs, fv), pool(a[0, HALF0:COLOR_DIM], fh, fs, fv)
        ub, lb = pool(b[0, 0:HALF0], fh, fs, fv), pool(b[0, HALF0:COLOR_DIM], fh, fs, fv)
        d = 0.55 * bhatt(ua, ub) + 0.45 * bhatt(la, lb)
        return max(0.0, min(1.0, 1.0 - d))

    def _tex(self, a, b):
        d1 = bhatt(a[0, COLOR_DIM:COLOR_DIM + TEX_BINS], b[0, COLOR_DIM:COLOR_DIM + TEX_BINS])
        d2 = bhatt(a[0, COLOR_DIM + TEX_BINS:COLOR_DIM + TEX_DIM],
                   b[0, COLOR_DIM + TEX_BINS:COLOR_DIM + TEX_DIM])
        return max(0.0, min(1.0, 1.0 - 0.5 * (d1 + d2)))

    def _shape(self, a, b):
        def rs(p, q):
            m = max(abs(p), abs(q))
            return 1.0 if m < 1e-6 else max(0.0, 1.0 - abs(p - q) / m)
        return 0.6 * rs(float(a[0, TOTAL_DIM - 2]), float(b[0, TOTAL_DIM - 2])) + \
               0.4 * rs(float(a[0, TOTAL_DIM - 1]), float(b[0, TOTAL_DIM - 1]))

    def fused(self, a, b):
        wc, wt, ws = self.w
        s = (wc * self._color(a, b) + wt * self._tex(a, b) + ws * self._shape(a, b)) / (wc + wt + ws)
        return max(0.0, min(1.0, s))

    def compare(self, a, b):
        # 180° 旋转不变：交换 b 的上下半身（池化后同样交换）
        fh, fs, fv = self.pool_f
        half = (H0 // fh) * (S0 // fs) * (V0 // fv)
        tb = (TEX_BINS)
        b2 = b.copy()
        b2[0, 0:half] = b[0, half:2 * half]
        b2[0, half:2 * half] = b[0, 0:half]
        b2[0, COLOR_DIM:COLOR_DIM + tb] = b[0, COLOR_DIM + tb:COLOR_DIM + TEX_DIM]
        b2[0, COLOR_DIM + tb:COLOR_DIM + TEX_DIM] = b[0, COLOR_DIM:COLOR_DIM + tb]
        return max(self.fused(a, b), self.fused(a, b2))


def seq_sim(cand, probe, templ):
    tot, n = 0.0, 0
    for p in probe:
        best = max((cand.compare(p, t) for t in templ), default=0.0)
        tot += best
        n += 1
    return tot / n if n else 0.0


def main():
    with open(VISITOR_JSON, encoding="utf-8") as f:
        vd = json.load(f)
    varr = vd.get("visitors", vd) if isinstance(vd, dict) else vd
    V = {o.get("passengerId", o.get("visitorId", -1)): decode_mat(o.get("reidFeature", "")) for o in varr}

    with open(PASSENGER_JSON, encoding="utf-8") as f:
        pd = json.load(f)
    P = {}
    for o in pd.get("passengers", []):
        pid = o.get("passenger_id", -1)
        fs = [decode_mat(s) for s in o.get("features", [])]
        P[pid] = [x for x in fs if x is not None]

    f41, f43 = V.get(41), V.get(43)

    cands = [
        Cand("基线(30x32x32, .6/.3/.1)      ", (1, 1, 1), (0.6, 0.3, 0.1)),
        Cand("池化(15x8x8, .6/.3/.1)        ", (2, 4, 4), (0.6, 0.3, 0.1)),
        Cand("池化(15x8x8, .3/.4/.3)        ", (2, 4, 4), (0.3, 0.4, 0.3)),
        Cand("池化(15x8x8, .2/.5/.3)        ", (2, 4, 4), (0.2, 0.5, 0.3)),
        Cand("池化(10x8x8, .2/.5/.3)        ", (3, 4, 4), (0.2, 0.5, 0.3)),
        Cand("池化(6x8x8,  .2/.5/.3)        ", (5, 4, 4), (0.2, 0.5, 0.3)),
    ]

    print("=" * 92)
    print("A) 真实同人单帧相似度：V41(上车) vs V43(下车)")
    print("=" * 92)
    for c in cands:
        print(f"   {c.name} → {c.compare(f41, f43):.4f}")

    print("\n" + "=" * 92)
    print("B) 以 V43 下车特征为探针，与在车乘客整段上车序列比对（真实同人应为 P3）")
    print("=" * 92)
    for c in cands:
        rows = []
        for pid, feats in P.items():
            rows.append((seq_sim(c, [f43], feats), pid))
        rows.sort(reverse=True)
        rank = [pid for _, pid in rows].index(3) + 1
        top1 = rows[0]
        print(f"   {c.name} | Top1={('P%d' % top1[1]):<4} sim={top1[0]:.4f}"
              f" | P3 sim={[s for s,p in rows if p==3][0]:.4f} rank={rank}"
              f" | 相邻分差={top1[0]-([s for s,p in rows if p==3][0]):+.4f}")
        print(f"        Top5: " + ", ".join(f"P{p}={s:.3f}" for s, p in rows[:5]))

    print("\n" + "=" * 92)
    print("C) 全体访客两两相似度分布（评估可分性/阈值）")
    print("=" * 92)
    ids = sorted(V.keys())
    for c in cands:
        vals = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = V[ids[i]], V[ids[j]]
                if a is None or b is None:
                    continue
                vals.append(c.compare(a, b))
        v = np.array(vals)
        print(f"   {c.name} | mean={v.mean():.3f} p90={np.percentile(v,90):.3f} "
              f"p95={np.percentile(v,95):.3f} p99={np.percentile(v,99):.3f} max={v.max():.3f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
