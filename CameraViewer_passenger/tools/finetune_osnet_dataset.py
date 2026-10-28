#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSNet x1.0 ReID 微调训练脚本（适配 passenger_reid_dataset 目录结构）
===================================================================
- 主干：deep-person-reid/torchreid/models/osnet.py::osnet_x1_0（与 torchreid 兼容）
- 初始化：--weights 指定 head_finetuned.pth 继续微调（去掉分类头），否则 MSMT17
- 损失：CrossEntropy(LabelSmoothing=0.1) + Batch-Hard Triplet
- 采样：RandomIdentitySampler（P 身份 × K 图）
- 数据：passenger_reid_dataset/passenger_XXX/（每个目录 = 一个身份）
- 导出：ONNX 输入 (1,3,256,128) -> 输出 (1,512,16,8) 特征图，与 C++ DeepReID 兼容

用法（yolotest 环境，torch 2.1.2）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python finetune_osnet_dataset.py \
      --root /media/tianheng/test/16.reId/CameraViewer_passenger/passenger_reid_dataset \
      --weights /media/tianheng/test/16.reId/CameraViewer_passenger/passenger_reid_dataset/osnet_best.pth \
      --epochs 60 --p 8 --k 4
"""
import argparse
import importlib.util
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
OSNET_PY = "/media/tianheng/test/deep-person-reid/torchreid/models/osnet.py"
CKPT_MSMT = os.path.expanduser(
    "~/.cache/torch/checkpoints/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_osnet(num_classes, weights=None):
    """从 torchreid osnet.py 加载 osnet_x1_0；优先加载 --weights（去分类头）。"""
    spec = importlib.util.spec_from_file_location("osnet_mod", OSNET_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.osnet_x1_0(num_classes=num_classes, pretrained=False, loss="softmax")

    if weights and os.path.isfile(weights):
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[osnet] 从 {weights} 加载(去分类头): missing={len(missing)} unexpected={len(unexpected)}")
    elif os.path.exists(CKPT_MSMT):
        ck = torch.load(CKPT_MSMT, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[osnet] 从 MSMT17 加载: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[osnet] 未找到预训练权重，使用随机初始化")
    return model


class ReidDataset(Dataset):
    """每个类别目录 = 一个身份。接受 class_dirs 列表（便于 train/val 划分）。"""

    def __init__(self, class_dirs, transform):
        self.transform = transform
        self.samples = []
        for lbl, cdir in enumerate(class_dirs):
            for f in sorted(os.listdir(cdir)):
                if f.lower().endswith(IMAGE_EXTS):
                    self.samples.append((os.path.join(cdir, f), lbl))
        self.cache = []
        for p, l in self.samples:
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.cache.append((img, l))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        img, lbl = self.cache[idx]
        return self.transform(img), lbl


class RandomIdentitySampler(Sampler):
    """每个 batch 采 P 个身份，每个身份 K 张图（batch_sampler 语义）。"""

    def __init__(self, labels, p, k):
        self.p, self.k = p, k
        self.index_dic = defaultdict(list)
        for i, l in enumerate(labels):
            self.index_dic[l].append(i)
        self.pids = list(self.index_dic.keys())
        self.length = (len(labels) // (p * k)) * p * k

    def __iter__(self):
        batches = []
        for _ in range(self.length // (self.p * self.k)):
            batch = []
            pids = random.sample(self.pids, min(self.p, len(self.pids)))
            for pid in pids:
                idxs = self.index_dic[pid]
                if len(idxs) >= self.k:
                    batch.extend(random.sample(idxs, self.k))
                else:
                    batch.extend(random.choices(idxs, k=self.k))
            batches.append(batch)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.length // (self.p * self.k)


def build_transform(train=True):
    t = [transforms.ToPILImage(), transforms.Resize((256, 128))]
    if train:
        t += [transforms.RandomHorizontalFlip(),
              transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)]
    t += [transforms.ToTensor(), transforms.Normalize(MEAN.tolist(), STD.tolist())]
    if train:
        t += [transforms.RandomErasing(p=0.3, value="random")]
    return transforms.Compose(t)


def batch_hard_triplet(feats, labels, margin=0.3):
    d = 1.0 - feats @ feats.t()
    n = labels.size(0)
    same = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    eye = torch.eye(n, dtype=torch.bool, device=labels.device)
    pos = same & ~eye
    neg = ~same
    ap = (d * pos.float()).max(dim=1).values
    valid = pos.any(dim=1)
    an = (d.masked_fill(~neg, 1e6)).min(dim=1).values
    loss = F.relu(ap - an + margin)
    return loss[valid].mean() if valid.any() else feats.sum() * 0.0


@torch.no_grad()
def extract_feats(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        fm = model.featuremaps(x)
        v = F.adaptive_avg_pool2d(fm, 1).flatten(1)
        v = F.normalize(v, dim=1)
        feats.append(v.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def eval_similarity(feats, labels):
    sim = feats @ feats.t()
    same = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    eye = torch.eye(len(labels), dtype=torch.bool)
    pos = same & ~eye
    neg = ~same
    s_pos = sim[pos].mean().item() if pos.any() else 0.0
    s_neg = sim[neg].mean().item() if neg.any() else 0.0
    ctrs = {}
    for c in labels.unique():
        ctrs[int(c)] = feats[labels == c].mean(0)
    correct = total = 0
    ids = list(ctrs.keys())
    for c in ids:
        v = ctrs[c]
        dists = {o: float((v @ ctrs[o]) / (v.norm() * ctrs[o].norm() + 1e-8)) for o in ids}
        best = max(dists, key=dists.get)
        total += 1
        correct += int(best == c)
    return s_pos, s_neg, correct / max(total, 1)


def collect_class_dirs(root):
    dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))]
    return dirs


def split_dirs(dirs, val_ratio=0.2):
    dirs = list(dirs)
    random.shuffle(dirs)
    val_n = max(1, int(len(dirs) * val_ratio))
    val_n = min(val_n, len(dirs) - 1) if len(dirs) > 1 else 0
    val_dirs = dirs[:val_n]
    train_dirs = dirs[val_n:]
    return train_dirs, val_dirs


def export_onnx(model, out_path):
    model.eval().cpu()

    class FeatWrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            # 与 DeepReID 兼容：输出 4D 特征图 [1,512,16,8]；GAP + L2 在 C++ 侧完成
            return self.m.featuremaps(x)

    wrap = FeatWrap(model)
    dummy = torch.randn(1, 3, 256, 128)
    torch.onnx.export(wrap, dummy, out_path,
                      input_names=["images"], output_names=["features"],
                      opset_version=12, dynamic_axes=None)
    print(f"[export] onnx -> {out_path}")
    try:
        import onnx
        m = onnx.load(out_path)
        print("  input :", [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim])
        print("  output:", [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])
    except Exception as e:
        print("  (inspect skipped:", e, ")")


def main():
    ap = argparse.ArgumentParser(description="OSNet x1.0 ReID 微调（passenger_reid_dataset）")
    ap.add_argument("--root", default="/media/tianheng/test/16.reId/CameraViewer_passenger/passenger_reid_dataset")
    ap.add_argument("--weights", default="/media/tianheng/test/16.reId/CameraViewer_passenger/passenger_reid_dataset/osnet_best.pth")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--out", default=None, help="输出 ONNX 路径（默认 <root>/osnet_x1_0_finetuned.onnx）")
    args = ap.parse_args()

    class_dirs = collect_class_dirs(args.root)
    if not class_dirs:
        print(f"[data] 错误：{args.root} 下没有乘客目录（passenger_XXX）")
        return 1
    train_dirs, val_dirs = split_dirs(class_dirs, args.val_ratio)
    print(f"[data] 身份数={len(class_dirs)} train身份={len(train_dirs)} val身份={len(val_dirs)}")
    if len(class_dirs) < 100:
        print(f"[data] 警告：身份数仅 {len(class_dirs)}，建议攒够 100~200 再训练，否则容易过拟合")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = ReidDataset(train_dirs, build_transform(True))
    val_ds = ReidDataset(val_dirs, build_transform(False)) if val_dirs else train_ds
    num_classes = len(train_dirs)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} classes={num_classes} device={device}")

    labels = [l for _, l in train_ds.cache]
    p = max(1, min(args.p, num_classes))
    sampler = RandomIdentitySampler(labels, p, args.k)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = load_osnet(num_classes, weights=args.weights).to(device)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    best_pth = os.path.join(args.root, "osnet_best.pth")
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = tc = tt = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            fm = model.featuremaps(x)
            v = F.normalize(F.adaptive_avg_pool2d(fm, 1).flatten(1), dim=1)
            l_ce = ce(logits, y)
            l_tri = batch_hard_triplet(v, y)
            loss = l_ce + l_tri
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item(); tc += l_ce.item(); tt += l_tri.item()
        sched.step()
        n = max(len(train_loader), 1)
        msg = f"ep {ep:3d}/{args.epochs} loss={tl/n:.4f} ce={tc/n:.4f} tri={tt/n:.4f}"
        if ep % 5 == 0 or ep == args.epochs:
            f, l = extract_feats(model, val_loader, device)
            sp, sn, acc = eval_similarity(f, l)
            msg += f" | val same={sp:.3f} diff={sn:.3f} gap={sp-sn:+.3f} idacc={acc:.3f}"
            score = sp - sn
            if score > best:
                best = score
                torch.save(model.state_dict(), best_pth)
        print(msg, flush=True)

    if os.path.exists(best_pth):
        model.load_state_dict(torch.load(best_pth, map_location="cpu"))
        print(f"[export] 使用最佳权重: {best_pth} (gap={best:.4f})")
    out_onnx = args.out or os.path.join(args.root, "osnet_x1_0_finetuned.onnx")
    export_onnx(model, out_onnx)
    print("完成。将导出的 onnx 复制到 CameraViewer/model/ 并更新 config.ini 的 DeepModelPath，")
    print("再删除旧 .engine 让程序自动重建 TensorRT 引擎。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


