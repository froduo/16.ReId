#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线 reID 对比分析工具
======================
目标：复现程序中 ReIDExtractor::compare() 的相似度计算，分析
      visitor_V41（上车）与 visitor_V43（下车）为同一人却无法匹配的原因。

数据来源（程序运行时真实落盘数据，最忠实）：
  - visitor_records.json : 每个访客 V## 的 reID 单帧特征（base64 编码的 cv::Mat）
  - passenger_book.json  : 每个上车乘客 P## 的整段上车特征序列（多帧）

用法:
  python3 offline_reid_analyze.py
"""

import base64
import json
import os
import sys

import numpy as np

BASE = os.path.join("build-CameraViewer-main-Desktop-Debug", "visitor_log")
VISITOR_JSON = os.path.join(BASE, "visitor_records.json")
PASSENGER_JSON = os.path.join(BASE, "passenger_book.json")

# ===== 与 reidextractor.h 完全一致的维度常量 =====
H_BINS, S_BINS, V_BINS = 30, 32, 32
COLOR_HALF = H_BINS * S_BINS * V_BINS          # 30720
COLOR_DIM = COLOR_HALF * 2                     # 61440
TEX_BINS = 18
TEX_DIM = TEX_BINS * 2                         # 36
SHAPE_DIM = 2
TOTAL_DIM = COLOR_DIM + TEX_DIM + SHAPE_DIM    # 61478

# 分量权重（ReIDExtractor 默认）
W_COLOR, W_TEXTURE, W_SHAPE = 0.60, 0.30, 0.10


# ============================================================================
# VisitorLog::matToBase64 / base64ToMat 的解码对应实现
# 格式: "rows,cols,type,channels," + base64(raw bytes)
# ============================================================================
def decode_mat(s: str):
    if not s:
        return None
    # 找到第 4 个逗号
    idx = -1
    cnt = 0
    for i, ch in enumerate(s):
        if ch == ',':
            cnt += 1
            if cnt == 4:
                idx = i + 1
                break
    if idx < 0:
        return None
    meta = s[:idx].split(',')
    rows, cols, _type, _ch = int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3])
    raw = base64.b64decode(s[idx:])
    # type 5 = CV_32F, 1 channel
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size != rows * cols:
        return None
    return arr.reshape(rows, cols).astype(np.float32)


# ============================================================================
# cv::compareHist(..., HISTCMP_BHATTACHARYYA) 的精确复现
#   Bh = sqrt( max(0, 1 - sum(sqrt(a_i*b_i)) / sqrt(sum(a)*sum(b)) ) )
# ============================================================================
def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    s1 = a.sum()
    s2 = b.sum()
    if s1 <= 0 or s2 <= 0:
        return 1.0
    s12 = np.sum(np.sqrt(a * b))
    val = 1.0 - s12 / np.sqrt(s1 * s2)
    if val < 0:
        val = 0.0
    return float(np.sqrt(val))


def compare_color(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    half = a.shape[1] // 2
    a1, a2 = a[:, :half], a[:, half:]
    b1, b2 = b[:, :half], b[:, half:]
    d1 = bhattacharyya(a1, b1)
    d2 = bhattacharyya(a2, b2)
    d = 0.55 * d1 + 0.45 * d2
    return max(0.0, min(1.0, 1.0 - d))


def compare_texture(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.5
    half = a.shape[1] // 2
    a1, a2 = a[:, :half], a[:, half:]
    b1, b2 = b[:, :half], b[:, half:]
    d1 = bhattacharyya(a1, b1)
    d2 = bhattacharyya(a2, b2)
    d = 0.5 * d1 + 0.5 * d2
    return max(0.0, min(1.0, 1.0 - d))


def _rel_sim(x: float, y: float) -> float:
    m = max(abs(x), abs(y))
    if m < 1e-6:
        return 1.0
    return max(0.0, 1.0 - abs(x - y) / m)


def compare_shape(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size < 2 or b.size < 2:
        return 0.5
    s = 0.6 * _rel_sim(float(a[0, 0]), float(b[0, 0])) + 0.4 * _rel_sim(float(a[0, 1]), float(b[0, 1]))
    return max(0.0, min(1.0, s))


def components(x: np.ndarray, y: np.ndarray):
    """返回 (simColor, simTex, simShape, fused)"""
    sc = compare_color(x[:, 0:COLOR_DIM], y[:, 0:COLOR_DIM])
    st = compare_texture(x[:, COLOR_DIM:COLOR_DIM + TEX_DIM],
                         y[:, COLOR_DIM:COLOR_DIM + TEX_DIM])
    ss = compare_shape(x[:, COLOR_DIM + TEX_DIM:TOTAL_DIM],
                       y[:, COLOR_DIM + TEX_DIM:TOTAL_DIM])
    fused = (W_COLOR * sc + W_TEXTURE * st + W_SHAPE * ss) / (W_COLOR + W_TEXTURE + W_SHAPE)
    return sc, st, ss, fused


def swap_halves(v: np.ndarray) -> np.ndarray:
    """复现 compare() 中的‘上下半身交换’(180°旋转不变)"""
    o = v.copy()
    c_half = COLOR_HALF
    t_half = TEX_BINS
    o[:, 0:c_half] = v[:, c_half:COLOR_DIM]
    o[:, c_half:COLOR_DIM] = v[:, 0:c_half]
    o[:, COLOR_DIM:COLOR_DIM + t_half] = v[:, COLOR_DIM + t_half:COLOR_DIM + TEX_DIM]
    o[:, COLOR_DIM + t_half:COLOR_DIM + TEX_DIM] = v[:, COLOR_DIM:COLOR_DIM + t_half]
    return o


def compare(x: np.ndarray, y: np.ndarray, detail=False):
    """复现 ReIDExtractor::compare()；detail=True 时额外返回分量分解"""
    if x is None or y is None:
        return (0.0, None) if detail else 0.0
    if x.shape[1] != TOTAL_DIM or y.shape[1] != TOTAL_DIM:
        return (0.0, None) if detail else 0.0
    sc, st, ss, fused0 = components(x, y)
    y_rot = swap_halves(y)
    sc2, st2, ss2, fused1 = components(x, y_rot)
    sim = max(fused0, fused1)
    if detail:
        return sim, dict(sc=sc, st=st, ss=ss, fused=fused0,
                         sc_rot=sc2, st_rot=st2, ss_rot=ss2, fused_rot=fused1)
    return sim


def sequence_similarity(probe_list, templ_list):
    """复现 PassengerBook::sequenceSimilarity / IdentityMatchWorker"""
    if not probe_list or not templ_list:
        return 0.0
    total = 0.0
    counted = 0
    for p in probe_list:
        if p is None:
            continue
        best = 0.0
        for t in templ_list:
            if t is None:
                continue
            s = compare(p, t)
            if s > best:
                best = s
        total += best
        counted += 1
    return total / counted if counted else 0.0


# ============================================================================
def load_visitors():
    print(f"[加载] {VISITOR_JSON}")
    with open(VISITOR_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 结构: {"visitors":[...]} 或直接数组 —— 兼容处理
    arr = data.get("visitors", data) if isinstance(data, dict) else data
    out = {}
    for o in arr:
        vid = o.get("passengerId", o.get("visitorId", -1))
        rec = {
            "id": vid,
            "boardTime": o.get("boardingTime", o.get("enterTime", "")),
            "alightTime": o.get("alightingTime", o.get("exitTime", "")),
            "isInside": o.get("isInside", None),
            "direction": o.get("direction", ""),
            "feat": decode_mat(o.get("reidFeature", "")),
        }
        out[vid] = rec
    print(f"       共 {len(out)} 条访客记录")
    return out


def load_passengers():
    print(f"[加载] {PASSENGER_JSON}")
    with open(PASSENGER_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    arr = data.get("passengers", [])
    out = {}
    for o in arr:
        pid = o.get("passenger_id", -1)
        feats = [decode_mat(s) for s in o.get("features", [])]
        feats = [x for x in feats if x is not None]
        out[pid] = {
            "id": pid,
            "boardTime": o.get("board_time", ""),
            "image": o.get("board_image", ""),
            "feats": feats,
        }
    print(f"       共 {len(out)} 条在车乘客记录")
    return out


def main():
    visitors = load_visitors()
    try:
        passengers = load_passengers()
    except FileNotFoundError:
        passengers = {}

    A, B = 41, 43  # V41 上车 / V43 下车
    va = visitors.get(A)
    vb = visitors.get(B)

    print("\n" + "=" * 78)
    print(f"目标访客：V{A}（上车） vs V{B}（下车）")
    print("=" * 78)
    for v in (va, vb):
        if v is None:
            print(f"  V?v 缺失")
            continue
        print(f"  V{v['id']}: 上车={v['boardTime']} 下车={v['alightTime']} "
              f"direction={v['direction']} inside={v['isInside']} "
              f"feat={'有' if v['feat'] is not None else '无'}"
              f"{'' if v['feat'] is None else ' shape=' + str(v['feat'].shape)}")

    if va and vb and va["feat"] is not None and vb["feat"] is not None:
        sim, det = compare(va["feat"], vb["feat"], detail=True)
        print(f"\n  >>> V{A} vs V{B} 单帧相似度 = {sim:.4f}")
        print(f"      [未旋转] 颜色={det['sc']:.4f} 纹理={det['st']:.4f} 身形={det['ss']:.4f} "
              f"融合={det['fused']:.4f}")
        print(f"      [180°旋转不变] 颜色={det['sc_rot']:.4f} 纹理={det['st_rot']:.4f} "
              f"身形={det['ss_rot']:.4f} 融合={det['fused_rot']:.4f}")

    # ---- 找出 V41 上车时对应的台账乘客 P（时间最接近） ----
    print("\n" + "=" * 78)
    print("台账（上车乘客）整段序列 vs V43 下车特征")
    print("=" * 78)
    if vb and vb["feat"] is not None and passengers:
        rows = []
        for pid, p in passengers.items():
            probe = [vb["feat"]]
            sim = sequence_similarity(probe, p["feats"])
            # 也给出逐帧最大值与均值（更细）
            vals = [compare(vb["feat"], t) for t in p["feats"]]
            mx = max(vals) if vals else 0.0
            avg = float(np.mean(vals)) if vals else 0.0
            rows.append((sim, mx, avg, pid, p["boardTime"], len(p["feats"])))
        rows.sort(reverse=True)
        print(f"  {'seqSim':>7} {'max':>7} {'avg':>7}  P##   boardTime            frames")
        for sim, mx, avg, pid, bt, n in rows:
            print(f"  {sim:7.4f} {mx:7.4f} {avg:7.4f}  P{pid:<4} {bt:<20} {n}")

        # 与阈值比较
        THR = 0.70
        print(f"\n  程序阈值 = {THR:.2f}；上述最大值 = {rows[0][0]:.4f} "
              f"→ {'可达' if rows[0][0] >= THR else '不可达（必然判定为异常下车）'}")

    # ---- 全体访客两两相似度分布，寻找是否存在同人可比对的更高分 ----
    print("\n" + "=" * 78)
    print("全体访客两两单帧相似度分布（用于评估阈值的可分性）")
    print("=" * 78)
    ids = [i for i in sorted(visitors) if visitors[i]["feat"] is not None]
    n = len(ids)
    print(f"  有效特征访客数 = {n}")
    if n >= 2:
        lim = 60  # 避免过大
        sel = ids[-lim:] if n > lim else ids
        mat = np.zeros((len(sel), len(sel)), dtype=np.float32)
        for i, a in enumerate(sel):
            for j, b in enumerate(sel):
                if j < i:
                    mat[i, j] = mat[j, i]
                else:
                    mat[i, j] = compare(visitors[a]["feat"], visitors[b]["feat"])
        off = mat[~np.eye(len(sel), dtype=bool)]
        print(f"  相似度: min={off.min():.3f} max={off.max():.3f} "
              f"mean={off.mean():.3f} median={np.median(off):.3f}")
        # 展示最高的一批配对，看是否集中在真正同人（相邻时间/上下车）
        pairs = []
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                pairs.append((mat[i, j], sel[i], sel[j]))
        pairs.sort(reverse=True)
        print("  最相似的前 10 对：")
        for s, a, b in pairs[:10]:
            print(f"     V{a} <-> V{b}: {s:.4f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
