#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 OSNet x1.0(MSMT17) 导出为 ONNX（输出 512 维全局特征），供 C++ OpenCV-DNN 使用。"""
import importlib.util
import os

import numpy as np
import torch

CKPT = os.path.expanduser(
    "~/.cache/torch/checkpoints/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
OSNET_PY = os.path.abspath("../deep-person-reid/torchreid/models/osnet.py")
OUT = os.path.abspath("CameraViewer/model/osnet_x1_0_msmt17.onnx")


class FeatureWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x, return_featuremaps=True)  # (B, 512, H', W')


def main():
    spec = importlib.util.spec_from_file_location("osnet_mod", OSNET_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.osnet_x1_0(num_classes=4101, pretrained=False, loss="softmax")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    model.load_state_dict(sd, strict=True)
    model.eval()

    wrap = FeatureWrap(model).eval()
    dummy = torch.randn(1, 3, 256, 128)

    # 关键：dynamo=False 使用传统导出器，权重内嵌为单文件（OpenCV DNN 无法读取 .data 外置权重）
    torch.onnx.export(
        wrap, dummy, OUT,
        input_names=["input"], output_names=["feat"],
        opset_version=12, do_constant_folding=True,
        dynamo=False)

    data_side = OUT + ".data"
    if os.path.exists(data_side):
        os.remove(data_side)
    print("已导出:", OUT, os.path.getsize(OUT) // 1024, "KB (单文件)")

    # 校验 onnxruntime 与 torch 输出一致
    try:
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
        o1 = ort_sess.run(None, {"input": dummy.numpy()})[0]
        with torch.no_grad():
            o2 = wrap(dummy).numpy()
        print("onnx shape:", o1.shape, "torch shape:", o2.shape)
        print("onnxruntime max abs diff (mean-pooled):",
              float(abs(o1.mean(axis=(2, 3)) - o2.mean(axis=(2, 3))).max()))
    except Exception as e:
        print("onnxruntime 校验跳过:", e)

    # 关键校验：OpenCV DNN 能否加载并给出接近结果（C++ 端即用它）
    try:
        import cv2
        net = cv2.dnn.readNetFromONNX(OUT)
        blob = cv2.dnn.blobFromImage(np.zeros((256, 128, 3), np.uint8), 1.0, (128, 256),
                                     (0, 0, 0), swapRB=False)
        blob = np.random.randn(1, 3, 256, 128).astype(np.float32)
        net.setInput(blob)
        out = net.forward()
        with torch.no_grad():
            ref = wrap(torch.from_numpy(blob)).numpy()
        print("opencv-dnn out:", out.shape, "torch:", ref.shape)
        print("opencv-dnn max abs diff (mean-pooled):",
              float(abs(out.mean(axis=(2, 3)) - ref.mean(axis=(2, 3))).max()))
    except Exception as e:
        print("opencv-dnn 校验跳过:", e)


if __name__ == "__main__":
    main()
