#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线 reID 深度诊断：定位颜色分量失效的根因
============================================
在 offline_reid_analyze.py 基础上，进一步分析：
  1. 校验 Python 端 Bhattacharyya 与 cv2.compareHist 是否一致；
  2. 统计颜色直方图的稀疏度 / 能量分布（分 bin 数是否过细）；
  3. 对颜色直方图做 bin 池化（降分辨率）后重算相似度，验证“bins 过细 → 描述子塌陷”假设；
  4. 对比“同方向同人”（相邻帧）与“异方向同人”（上/下车）的颜色相似度。
"""

import base64
import json
import os

import cv2
import numpy as np

BASE = os.path.join("build-CameraViewer-main-Desktop-Debug", "visitor_log")
VISITOR_JSON = os.path.join(BASE, "visitor_records.json")
PASSENGER_JSON = os.path.join(BASE, "passenger_book.json")

H_BINS, S_BINS, V_BINS = 30, 32, 32
COLOR_HALF = H_BINS * S_BINS * V_BINS
COLOR_DIM = COLOR_HALF * 2
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
    if idx < 0:
        return None
    meta = s[:idx].split(',')
    rows, cols = int(meta[0]), int(meta[1])
    raw = base64.b64decode(s[idx:])
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size != rows * cols:
        return None
    return arr.reshape(rows, cols).astype(np.float32)


def bhatt(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    s1, s2 = a.sum(), b.sum()
    if s1 <= 0 or s2 <= 0:
        return 1.0
    s12 = np.sum(np.sqrt(a * b))
    return float(np.sqrt(max(0.0, 1.0 - s12 / np.sqrt(s1 * s2))))


def color_sim_from_parts(up_a, lo_a, up_b, lo_b, method="bhatt"):
    """模拟 compareColor：0.55*up + 0.45*lo"""
    d1 = bhatt(up_a, up_b)
    d2 = bhatt(lo_a, lo_b)
    d = 0.55 * d1 + 0.45 * d2
    return max(0.0, min(1.0, 1.0 - d))


def pool3(v, fh, fs, fv):
    """把 (H*S*V) 一维直方图池化到 (H/fh, S/fs, V/fv)"""
    h = v.reshape(H_BINS, S_BINS, V_BINS)
    H2, S2, V2 = H_BINS // fh, S_BINS // fs, V_BINS // fv
    h = h[:H2 * fh, :S2 * fs, :V2 * fv]
    h = h.reshape(H2, fh, S2, fs, V2, fv).sum(axis=(1, 3, 5))
    return h.reshape(-1)


def main():
    # ---------- 0. 校验 Bhattacharyya 与 OpenCV ----------
    rng = np.random.default_rng(0)
    ta = rng.random(500).astype(np.float32)
    tb = rng.random(500).astype(np.float32)
    py = bhatt(ta, tb)
    cv_ = cv2.compareHist(ta.reshape(-1, 1), tb.reshape(-1, 1), cv2.HISTCMP_BHATTACHARYYA)
    print("=" * 78)
    print("[校验] Bhattacharyya 复现 vs OpenCV")
    print(f"   python={py:.6f}  opencv={cv_:.6f}  diff={abs(py-cv_):.2e}"
          f"  {'✔一致' if abs(py-cv_) < 1e-6 else '✘不一致'}")
    print("=" * 78)

    with open(VISITOR_JSON, "r", encoding="utf-8") as f:
        vd = json.load(f)
    varr = vd.get("visitors", vd) if isinstance(vd, dict) else vd
    V = {o.get("passengerId", o.get("visitorId", -1)): decode_mat(o.get("reidFeature", "")) for o in varr}

    with open(PASSENGER_JSON, "r", encoding="utf-8") as f:
        pd = json.load(f)
    P = {}
    for o in pd.get("passengers", []):
        pid = o.get("passenger_id", -1)
        P[pid] = [decode_mat(s) for s in o.get("features", [])]
        P[pid] = [x for x in P[pid] if x is not None]

    f41, f43 = V.get(41), V.get(43)
    p3 = P.get(3, [])

    # ---------- 1. 稀疏度统计 ----------
    print("\n[统计] 颜色直方图稀疏度（非零 bin 占比，越小越稀疏）")
    for name, f in [("V41(上车)", f41), ("V43(下车)", f43)] + \
                   [(f"P3[{i}](上车)", p3[i]) for i in (0, len(p3) // 2, len(p3) - 1) if p3]:
        up = f[0, 0:COLOR_HALF]
        lo = f[0, COLOR_HALF:COLOR_DIM]
        for half, tag in ((up, "上"), (lo, "下")):
            nz = np.count_nonzero(half)
            print(f"   {name:<14} {tag}半身: 非零bin={nz:6d}/30720 ({100.0*nz/30720:5.2f}%)  "
                  f"max={half.max():.4f}  能量前10bin占比={np.sort(half)[::-1][:10].sum():.3f}")

    # ---------- 2. 同方向 vs 异方向 颜色相似度 ----------
    print("\n[对比] 颜色分量相似度（原始 30x32x32）")
    if f41 is not None and f43 is not None:
        u41, l41 = f41[0, 0:COLOR_HALF], f41[0, COLOR_HALF:COLOR_DIM]
        u43, l43 = f43[0, 0:COLOR_HALF], f43[0, COLOR_HALF:COLOR_DIM]
        print(f"   异方向同人 V41(上)  vs V43(下) : {color_sim_from_parts(u41, l41, u43, l43):.4f}")
    if len(p3) >= 2:
        a, b = p3[0], p3[min(3, len(p3) - 1)]
        print(f"   同方向同人 P3[0]   vs P3[{min(3,len(p3)-1)}] (相邻帧): "
              f"{color_sim_from_parts(a[0,0:COLOR_HALF], a[0,COLOR_HALF:COLOR_DIM], b[0,0:COLOR_HALF], b[0,COLOR_HALF:COLOR_DIM]):.4f}")
        # 序列平均异方向
        if f43 is not None:
            sims = [color_sim_from_parts(p[0,0:COLOR_HALF], p[0,COLOR_HALF:COLOR_DIM], u43, l43) for p in p3]
            print(f"   异方向同人 平均(P3各帧 vs V43): {np.mean(sims):.4f}  (逐帧 max={max(sims):.4f})")

    # ---------- 3. bin 池化实验 ----------
    print("\n[实验] 颜色直方图降分辨率（bin 池化）后的相似度")
    print("   —— 若池化后相似度显著上升，则说明 bin 过细导致描述子塌陷")
    combos = [(1, 1, 1), (2, 2, 2), (3, 4, 4), (5, 4, 4), (6, 8, 8), (10, 8, 8), (15, 8, 8), (30, 4, 4), (30, 8, 4)]
    print(f"   {'H×S×V':>10} {'#bins':>7} | {'V41↔V43':>8} {'P3同向':>8} {'P3↔V43均':>9}")
    if f41 is not None and f43 is not None:
        u41, l41 = f41[0, 0:COLOR_HALF], f41[0, COLOR_HALF:COLOR_DIM]
        u43, l43 = f43[0, 0:COLOR_HALF], f43[0, COLOR_HALF:COLOR_DIM]
        for fh, fs, fv in combos:
            a_u, a_l = pool3(u41, fh, fs, fv), pool3(l41, fh, fs, fv)
            b_u, b_l = pool3(u43, fh, fs, fv), pool3(l43, fh, fs, fv)
            cross = color_sim_from_parts(a_u, a_l, b_u, b_l)
            same = float('nan')
            if len(p3) >= 2:
                x, y = p3[0], p3[min(3, len(p3) - 1)]
                same = color_sim_from_parts(pool3(x[0,0:COLOR_HALF],fh,fs,fv), pool3(x[0,COLOR_HALF:COLOR_DIM],fh,fs,fv),
                                            pool3(y[0,0:COLOR_HALF],fh,fs,fv), pool3(y[0,COLOR_HALF:COLOR_DIM],fh,fs,fv))
            crossavg = float('nan')
            if p3:
                vs = []
                for p in p3:
                    vs.append(color_sim_from_parts(pool3(p[0,0:COLOR_HALF],fh,fs,fv), pool3(p[0,COLOR_HALF:COLOR_DIM],fh,fs,fv),
                                                   b_u, b_l))
                crossavg = float(np.mean(vs))
            nb = (H_BINS//fh)*(S_BINS//fs)*(V_BINS//fv)*2
            print(f"   {fh}x{fs}x{fv:<3} {nb:>7} | {cross:8.4f} {same:8.4f} {crossavg:9.4f}")

    # ---------- 4. 分量贡献 ----------
    print("\n[分解] V41 vs V43 融合相似度分解（原始权重 0.6/0.3/0.1）")
    if f41 is not None and f43 is not None:
        def comps(x, y):
            cu = color_sim_from_parts(x[0,0:COLOR_HALF], x[0,COLOR_HALF:COLOR_DIM],
                                      y[0,0:COLOR_HALF], y[0,COLOR_HALF:COLOR_DIM])
            def tex(t):
                return bhatt(x[0,COLOR_DIM:COLOR_DIM+TEX_BINS], y[0,COLOR_DIM:COLOR_DIM+TEX_BINS]), \
                       bhatt(x[0,COLOR_DIM+TEX_BINS:COLOR_DIM+TEX_DIM], y[0,COLOR_DIM+TEX_BINS:COLOR_DIM+TEX_DIM])
            d1,d2 = tex(0)
            st = 1 - 0.5*(d1+d2)
            sa, sb = x[0,TOTAL_DIM-2], y[0,TOTAL_DIM-2]
            h1, h2 = x[0,TOTAL_DIM-1], y[0,TOTAL_DIM-1]
            rs = (lambda p,q: 1.0 if max(abs(p),abs(q))<1e-6 else max(0.0,1-abs(p-q)/max(abs(p),abs(q))))
            ss = 0.6*rs(sa,sb)+0.4*rs(h1,h2)
            return cu, st, ss
        c,t,s = comps(f41,f43)
        fused = 0.6*c+0.3*t+0.1*s
        print(f"   颜色={c:.4f}  纹理={t:.4f}  身形={s:.4f}  →  融合={fused:.4f}")
        print(f"   若提高纹理/身形权重(0.3/0.5/0.2): {0.3*c+0.5*t+0.2*s:.4f}")
        print(f"   若提高纹理/身形权重(0.2/0.5/0.3): {0.2*c+0.5*t+0.3*s:.4f}")

    # ---------- 5. 大样本：同方向相邻访客 vs 跨方向 ----------
    print("\n[大样本] 全体访客相似度分位（用于阈值设定参考）")
    ids = sorted(V.keys())
    vals = []
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = V[ids[i]], V[ids[j]]
            if a is None or b is None:
                continue
            # 完整 compare（含旋转不变）
            def full_cmp(x, y):
                c1 = color_sim_from_parts(x[0,0:COLOR_HALF],x[0,COLOR_HALF:COLOR_DIM],
                                          y[0,0:COLOR_HALF],y[0,COLOR_HALF:COLOR_DIM])
                c2 = color_sim_from_parts(x[0,COLOR_HALF:COLOR_DIM],x[0,0:COLOR_HALF],
                                          y[0,0:COLOR_HALF],y[0,COLOR_HALF:COLOR_DIM])
                # (简化：仅颜色旋转不变，纹理/身形近似取1)
                return max(c1, c2)
            vals.append(full_cmp(a,b))
    vals = np.array(vals)
    for q in (50, 90, 95, 99):
        print(f"   {q}th percentile = {np.percentile(vals, q):.4f}")
    print(f"   max = {vals.max():.4f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
