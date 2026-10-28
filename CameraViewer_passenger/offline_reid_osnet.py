#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度 ReID（OSNet x1.0 / MSMT17 预训练）离线评估
================================================
检验：用深度嵌入替换手工颜色/纹理特征后，能否把
      V41/V43（人A，纯白T） 与 V45（人B，印花白T）区分开。

依赖：本机已有权重
  ~/.cache/torch/checkpoints/osnet_x1_0_msmt17_combineall_256x128_....pth
     以及 ../deep-person-reid 仓库（torchreid）
"""

import glob
import importlib.util
import os

import cv2
import numpy as np
import torch

ROOT = "build-CameraViewer-main-Desktop-Debug/save_images/20260929"
CKPT = os.path.expanduser(
    "~/.cache/torch/checkpoints/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
OSNET_PY = os.path.abspath("../deep-person-reid/torchreid/models/osnet.py")

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model():
    # 直接按文件加载 osnet.py（仅依赖 torch），避免 torchreid 包级导入的额外依赖
    spec = importlib.util.spec_from_file_location("osnet_mod", OSNET_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.osnet_x1_0(num_classes=4101, pretrained=False, loss="softmax")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  权重加载: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def preprocess(img, tight):
    if tight:
        h, w = img.shape[:2]
        img = img[int(h * 0.1875):int(h * 0.8125), int(w * 0.1875):int(w * 0.8125)]
    img = cv2.resize(img, (128, 256))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    return t


@torch.no_grad()
def embed(model, img, tight=True):
    t = preprocess(img, tight)
    f = model(t, return_featuremaps=True)
    f = f.view(-1).numpy().astype(np.float32)
    n = np.linalg.norm(f)
    return f / n if n > 0 else f


def cosine(a, b):
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def load_group(model, name, tight=True):
    feats = []
    for p in sorted(glob.glob(os.path.join(ROOT, name, "*.jpg"))):
        img = cv2.imread(p)
        if img is None:
            continue
        feats.append(embed(model, img, tight))
    return feats


def best_avg(A, B):
    tot = 0.0
    for a in A:
        tot += max((cosine(a, b) for b in B), default=0.0)
    return tot / len(A) if A else 0.0


def centroid(A, B):
    ca = np.mean(np.array(A), axis=0)
    cb = np.mean(np.array(B), axis=0)
    ca /= np.linalg.norm(ca)
    cb /= np.linalg.norm(cb)
    return cosine(ca, cb)


def main():
    print("构建 OSNet x1.0 (MSMT17)...")
    model = build_model()

    for tight in (False, True):
        tag = "紧致框(中心62.5%)" if tight else "整图"
        g = {n: load_group(model, n, tight) for n in
             ("visitor_V41", "visitor_V43", "visitor_V45")}
        print("\n" + "=" * 70)
        print(f"[{tag}] 特征: " + ", ".join(f"{k}={len(v)}" for k, v in g.items()))
        print("=" * 70)
        print("  best-avg 组比对：")
        print(f"     V41(人A) vs V43(人A) = {best_avg(g['visitor_V41'], g['visitor_V43']):.4f}")
        print(f"     V45(人B) vs V43(人A) = {best_avg(g['visitor_V45'], g['visitor_V43']):.4f}")
        print(f"     V45(人B) vs V41(人A) = {best_avg(g['visitor_V45'], g['visitor_V41']):.4f}")
        print("  质心(Centroid) 组比对：")
        print(f"     V41(人A) vs V43(人A) = {centroid(g['visitor_V41'], g['visitor_V43']):.4f}")
        print(f"     V45(人B) vs V43(人A) = {centroid(g['visitor_V45'], g['visitor_V43']):.4f}")
        print(f"     V45(人B) vs V41(人A) = {centroid(g['visitor_V45'], g['visitor_V41']):.4f}")
        # 判定
        d_same = centroid(g['visitor_V41'], g['visitor_V43'])
        d_diff = centroid(g['visitor_V45'], g['visitor_V43'])
        print(f"  → 同人={d_same:.3f} 异人={d_diff:.3f} 可分={d_same - d_diff:+.3f} "
              f"{'✔可区分' if d_same > d_diff + 0.05 else '✘仍不可分'}")

    print("\n完成。")


if __name__ == "__main__":
    main()
