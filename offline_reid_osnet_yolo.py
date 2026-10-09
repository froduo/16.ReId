#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度 ReID(OSNet) + YOLO 紧致人体框 的评估：验证 V41/V43(同人) 与 V45(异人) 可分性。"""
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


def build_osnet():
    spec = importlib.util.spec_from_file_location("osnet_mod", OSNET_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.osnet_x1_0(num_classes=4101, pretrained=False, loss="softmax")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("state_dict", sd), strict=True)
    return m.eval()


@torch.no_grad()
def embed(model, crop):
    img = cv2.resize(crop, (128, 256))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    f = model(t, return_featuremaps=True)          # 1,512,16,8
    f = f.mean(dim=(2, 3)).view(-1).numpy()        # 全局平均池化 -> 512
    n = np.linalg.norm(f)
    return f / n if n > 0 else f


def cos(a, b):
    return float(np.clip(np.dot(a, b), -1, 1))


DET_ONNX = "CameraViewer/build/model/yolo26n.onnx"


class DnnDetector:
    """复刻 PersonDetector 的 OpenCV-DNN 前/后处理（yolo26n.onnx, 640 输入）"""

    def __init__(self, path, conf=0.45, nms=0.5):
        self.net = cv2.dnn.readNetFromONNX(path)
        self.conf, self.nms, self.size = conf, nms, 640

    def detect(self, img):
        w, h = img.shape[1], img.shape[0]
        s = min(self.size / w, self.size / h)
        pw, ph = int(w * s), int(h * s)
        dw, dh = (self.size - pw) // 2, (self.size - ph) // 2
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (self.size, self.size),
                                     (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        if out.ndim == 3:
            m = out[0]
            if m.shape[0] < m.shape[1]:
                m = m.T                      # -> (N, 4+nc)
        else:
            m = out
        boxes, confs = [], []
        for row in m:
            scores = row[4:]
            cid = int(np.argmax(scores))
            cf = float(scores[cid])
            if cid == 0 and cf >= self.conf:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                cx = (cx - dw) / s
                cy = (cy - dh) / s
                bw, bh = bw / s, bh / s
                x1, y1 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
                boxes.append([x1, y1, int(bw), int(bh)])
                confs.append(cf)
        idx = cv2.dnn.NMSBoxes(boxes, confs, self.conf, self.nms)
        return [boxes[i] for i in (idx.flatten().tolist() if len(idx) else [])]


def main():
    if os.path.exists("yolo11n.pt"):
        from ultralytics import YOLO
        _y = YOLO("yolo11n.pt")

        class _U:
            def detect(self, img):
                r = _y.predict(img, classes=[0], conf=0.25, verbose=False)[0]
                out = []
                for b in r.boxes.xyxy.tolist():
                    x1, y1, x2, y2 = b
                    out.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                return out
        det = _U()
        print("使用 YOLO(yolo11n.pt) 检测人体框")
    else:
        det = DnnDetector(DET_ONNX)
        print("使用本地 yolo26n.onnx 检测人体框")
    osnet = build_osnet()

    groups = {}
    for name in ("visitor_V41", "visitor_V43", "visitor_V45"):
        feats, nbox = [], 0
        for p in sorted(glob.glob(os.path.join(ROOT, name, "*.jpg"))):
            img = cv2.imread(p)
            if img is None:
                continue
            dets = det.detect(img)
            if not dets:
                continue
            # 取面积最大的人
            bx = max(dets, key=lambda r: r[2] * r[3])
            x, y, bw, bh = bx
            crop = img[max(0, y):y + bh, max(0, x):x + bw]
            if crop.size == 0:
                continue
            feats.append(embed(osnet, crop))
            nbox += 1
        groups[name] = feats
        print(f"[{name}] 有效人体框 {nbox} 帧")

    def cent(fs):
        c = np.mean(np.array(fs), axis=0)
        return c / np.linalg.norm(c)

    A, C, E = cent(groups["visitor_V41"]), cent(groups["visitor_V43"]), cent(groups["visitor_V45"])
    print("\n=== 深度ReID(OSNet) 组质心 余弦 ===")
    print(f"  V41(人A) vs V43(人A) 同人 = {cos(A, C):.4f}")
    print(f"  V45(人B) vs V43(人A) 异人 = {cos(E, C):.4f}")
    print(f"  V45(人B) vs V41(人A) 异人 = {cos(E, A):.4f}")

    def best_avg(ga, gb):
        return float(np.mean([max(cos(x, y) for y in gb) for x in ga]))
    print("\n=== best-avg ===")
    print(f"  同人 V41-V43 = {best_avg(groups['visitor_V41'], groups['visitor_V43']):.4f}")
    print(f"  异人 V45-V43 = {best_avg(groups['visitor_V45'], groups['visitor_V43']):.4f}")


if __name__ == "__main__":
    main()
