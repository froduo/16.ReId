#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepMatchThreshold 标定脚本（GPU 版）
====================================
复现 C++ FolderMatchWorker::bestAvgSimilarity 的「目录级 best-avg」相似度：
  对 probe(下车) 每张图，取与 template(上车) 目录所有特征的最大余弦，再取平均。

用人工确认的 manual_pairs.json（同一人 上车↔下车）计算「同一人」分布，
用 board × 非配对 alight 交叉配对计算「不同人」分布，据此给出 DeepMatchThreshold 建议。

特征提取与训练/导出/C++ DeepReID 完全一致：
  BGR→RGB → resize(128,256) → /255 → ImageNet 归一化 → featuremaps → GAP → L2。

用法（yolotest 环境，GPU）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python calibrate_deep_threshold.py \
      --weights /media/tianheng/test/16.reId/passenger_reid_dataset/osnet_best.pth \
      --sessions <session1> <session2> \
      --pairs /media/tianheng/test/16.reId/manual_pairs.json
"""
import argparse
import importlib.util
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IN_W, IN_H = 128, 256
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
CROP_SUBDIRS = ("乘客图片", "重新检测_乘客图片")

OSNET_PY = "/media/tianheng/test/deep-person-reid/torchreid/models/osnet.py"


def load_model(weights):
    spec = importlib.util.spec_from_file_location("osnet_mod", OSNET_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.osnet_x1_0(num_classes=1, pretrained=False, loss="softmax")
    if weights and os.path.isfile(weights):
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[model] 加载 {weights}: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[model] 警告：未找到权重，使用随机初始化")
    return model.eval().cuda()


def image_files(path, max_images):
    files = sorted(
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith(IMAGE_EXTS)
    )
    return files if max_images <= 0 else files[:max_images]


def dir_feats(model, dirpath, max_images, batch=64):
    """返回目录下所有图像的 L2 归一化 512 维特征（N,512），GPU 批推理。"""
    files = image_files(dirpath, max_images)
    feats = []
    buf = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
        buf.append(img)
        if len(buf) >= batch:
            feats.append(_embed_batch(model, buf))
            buf = []
    if buf:
        feats.append(_embed_batch(model, buf))
    if not feats:
        return None
    return np.concatenate(feats, axis=0).astype(np.float32)


def _embed_batch(model, imgs_rgb):
    x = []
    for img in imgs_rgb:
        t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        t = (t - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
        x.append(t)
    x = torch.stack(x).cuda()
    with torch.no_grad():
        fm = model.featuremaps(x)                      # [N,512,16,8]
        v = F.adaptive_avg_pool2d(fm, 1).flatten(1)    # [N,512]
        v = F.normalize(v, dim=1)
    return v.cpu().numpy()


def best_avg(probe_feats, templ_feats):
    """probe 每张取与 template 目录的最大余弦，再平均（= C++ bestAvgSimilarity）。"""
    sims = probe_feats @ templ_feats.T
    return float(sims.max(axis=1).mean())


def find_person_dir(sessions, person):
    for s in sessions:
        d = os.path.join(s, person)
        if not os.path.isdir(d):
            continue
        for sub in CROP_SUBDIRS:
            sd = os.path.join(d, sub)
            if os.path.isdir(sd) and image_files(sd, 1):
                return sd
        return d
    return None


def main():
    ap = argparse.ArgumentParser(description="标定 DeepMatchThreshold")
    ap.add_argument("--weights", required=True, help="osnet_best.pth 权重路径")
    ap.add_argument("--sessions", nargs="+", required=True, help="session 目录列表")
    ap.add_argument("--pairs", required=True, help="manual_pairs.json 路径")
    ap.add_argument("--max-images", type=int, default=100,
                    help="每个目录最多使用的图片数（0=不限）")
    args = ap.parse_args()

    with open(args.pairs, encoding="utf-8") as f:
        pairs = json.load(f)

    model = load_model(args.weights)

    boards = [p["board"] for p in pairs]
    alights = [p["alight"] for p in pairs]

    # 加载所有目录特征（board / alight 各一份）
    board_feats, alight_feats = {}, {}
    for p in pairs:
        b, a = p["board"], p["alight"]
        if b not in board_feats:
            d = find_person_dir(args.sessions, b)
            board_feats[b] = (d, dir_feats(model, d, args.max_images) if d else None)
        if a not in alight_feats:
            d = find_person_dir(args.sessions, a)
            alight_feats[a] = (d, dir_feats(model, d, args.max_images) if d else None)

    # 打印目录命中情况
    print("=" * 72)
    print(f"权重: {args.weights}")
    print(f"配对总数: {len(pairs)}")
    missing = [b for b in boards if board_feats[b][1] is None] + \
              [a for a in alights if alight_feats[a][1] is None]
    print(f"缺失目录数: {len(missing)} {missing if missing else ''}")
    print("=" * 72)

    pos, neg = [], []
    for i, p in enumerate(pairs):
        b, a = p["board"], p["alight"]
        bf = board_feats[b][1]
        af = alight_feats[a][1]
        if bf is None or af is None:
            continue
        pos.append(best_avg(af, bf))  # 同一人：probe=下车 alight, template=上车 board

        # 负样本：本 board 与所有「非配对」的 alight 比较
        for j, p2 in enumerate(pairs):
            if j == i:
                continue
            af2 = alight_feats[p2["alight"]][1]
            if af2 is None:
                continue
            neg.append(best_avg(af2, bf))

    pos = np.array(pos, dtype=np.float32)
    neg = np.array(neg, dtype=np.float32)

    # 明细：正样本按 sim 升序
    pos_detail = []
    for i, p in enumerate(pairs):
        b, a = p["board"], p["alight"]
        bf, af = board_feats[b][1], alight_feats[a][1]
        if bf is None or af is None:
            continue
        pos_detail.append((best_avg(af, bf), b, a))
    pos_detail.sort()
    print("\n[正样本明细] (board=上车 ↔ alight=下车) sim 升序")
    for s, b, a in pos_detail:
        print(f"  {s:.4f}  {b}  ↔  {a}")

    # 明细：负样本 top15（高相似度的「不同人」可能是重复 ID / 相似乘客）
    neg_detail = []
    for i, p in enumerate(pairs):
        bf = board_feats[p["board"]][1]
        if bf is None:
            continue
        for j, p2 in enumerate(pairs):
            if j == i:
                continue
            af2 = alight_feats[p2["alight"]][1]
            if af2 is None:
                continue
            neg_detail.append((best_avg(af2, bf), p["board"], p2["alight"]))
    neg_detail.sort(reverse=True)
    print("\n[负样本 TOP15] (board ↔ 非配对 alight) sim 降序")
    for s, b, a in neg_detail[:15]:
        print(f"  {s:.4f}  {b}  ↔  {a}")

    print(f"\n[同一人] 正样本相似度  n={len(pos)}")
    print(f"  min={pos.min():.4f}  max={pos.max():.4f}  mean={pos.mean():.4f}  std={pos.std():.4f}")
    print(f"\n[不同人] 负样本相似度  n={len(neg)}")
    print(f"  min={neg.min():.4f}  max={neg.max():.4f}  mean={neg.mean():.4f}  std={neg.std():.4f}")

    # 阈值建议：若干候选及其在正/负样本上的漏检/误检
    print("\n[阈值候选]  阈值 | 正样本漏检(判为不同) | 负样本误检(判为同一)")
    lo, hi = float(neg.max()), float(pos.min())
    print(f"  正负样本间隔: [{lo:.4f}, {hi:.4f}]  gap={hi - lo:+.4f}")
    candidates = sorted(set([
        round(lo, 3), round(hi, 3),
        round((lo + hi) / 2, 3),
        round(float(neg.mean()) + float(neg.std()), 3),
        0.50, 0.55, 0.60, 0.65, 0.70,
    ]))
    for t in candidates:
        fn = float((pos < t).mean()) if len(pos) else 0.0
        fp = float((neg >= t).mean()) if len(neg) else 0.0
        print(f"    {t:.3f}   |   {fn:.3f}   |   {fp:.3f}")

    print("\n建议：取正样本 min 与负样本 max 的中点，兼顾漏检与误检。")
    print(f"      DeepMatchThreshold 建议值 ≈ {round((lo + hi) / 2, 3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
