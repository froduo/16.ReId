#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理 SMB 共享上的乘客视频：
  1) 先逐个下载全部剩余视频到本地；
  2) 再以 --parallel 个进程并行识别（每个视频一个子进程），生成 YOLO 数据集；
  3) 处理完成后删除本地临时视频。

用法:
  python3 -u tools/batch_process_smb.py [--parallel 4] [--every 3]

每个视频输出到:  passenger_dataset_all/<视频名(不含扩展名)>/
                  (内含 images/ + labels/ + data.yaml)
每个子进程日志:   passenger_dataset_all/<视频名>.log
"""

import argparse
import os
import subprocess
import sys
import time

import smbclient

SERVER = "192.168.200.8"
USERNAME = "tianheng"
PASSWORD = "tianheng123"
SHARE_DIR = r"\\192.168.200.8\vision\25.乘客视频\extracted"

LOCAL_DIR = "passenger_videos"
OUTPUT_ROOT = "passenger_dataset_all"
MODEL = "model/passenger_yolov8n_trt10_x86_20261013_180200.engine"
DETECT_SCRIPT = "tools/passenger_detect_trt.py"

# 已单独处理过的视频（不再重复处理）
SKIP = {"ch01_20260930_063419_074220topspeed.mp4"}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def list_todo():
    smbclient.register_session(SERVER, username=USERNAME, password=PASSWORD)
    entries = sorted(e.name for e in smbclient.scandir(SHARE_DIR,
                                                       username=USERNAME,
                                                       password=PASSWORD)
                     if e.name.lower().endswith(".mp4"))
    return [n for n in entries if n not in SKIP]


def download(name):
    src = SHARE_DIR + "\\" + name
    local = os.path.join(LOCAL_DIR, name)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        log(f"本地已存在，跳过下载: {name} ({os.path.getsize(local)} 字节)")
        return local
    smbclient.register_session(SERVER, username=USERNAME, password=PASSWORD)
    log(f"下载: {name}")
    t0 = time.time()
    with smbclient.open_file(src, "rb", share_access="r",
                             username=USERNAME, password=PASSWORD) as f:
        with open(local, "wb") as o:
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                o.write(chunk)
    log(f"下载完成: {name} ({os.path.getsize(local)} 字节, {time.time() - t0:.0f}s)")
    return local


def output_done(stem):
    return (os.path.isdir(os.path.join(OUTPUT_ROOT, stem, "labels")) and
            os.path.isfile(os.path.join(OUTPUT_ROOT, stem, "data.yaml")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=4, help="并行识别进程数（默认 4）")
    ap.add_argument("--every", type=int, default=3, help="每 N 帧检测一次（默认 3）")
    args = ap.parse_args()

    todo = list_todo()
    log(f"共享目录待处理 {len(todo)} 个视频，并行度 {args.parallel}，every={args.every}")

    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ===== 阶段 1：逐个下载全部剩余视频 =====
    jobs = []
    for name in todo:
        local = download(name)
        jobs.append((name, local))

    # ===== 阶段 2：并行识别（最多 --parallel 个并发子进程）=====
    running = {}   # name -> Popen
    pending = list(jobs)

    while pending or running:
        # 清理已完成的子进程
        for name in list(running.keys()):
            p = running[name]
            if p.poll() is not None:
                del running[name]
                log(f"完成 code={p.returncode}: {name}")
                # 删除本地临时视频，节省磁盘
                local = os.path.join(LOCAL_DIR, name)
                try:
                    os.remove(local)
                    log(f"已删除临时视频: {local}")
                except OSError as e:
                    log(f"删除临时视频失败（忽略）: {e}")

        # 启动新任务直到达到并行上限
        while pending and len(running) < args.parallel:
            name, local = pending.pop(0)
            stem = os.path.splitext(name)[0]
            outdir = os.path.join(OUTPUT_ROOT, stem)
            if output_done(stem):
                log(f"跳过（输出已存在）: {name}")
                try:
                    os.remove(local)
                except OSError:
                    pass
                continue
            os.makedirs(outdir, exist_ok=True)
            logfile = os.path.join(OUTPUT_ROOT, stem + ".log")
            lf = open(logfile, "w")
            cmd = [sys.executable, "-u", DETECT_SCRIPT,
                   "--input", local,
                   "--output", outdir,
                   "--model", MODEL,
                   "--conf", "0.30",
                   "--every", str(args.every)]
            log(f"启动识别[{len(running) + 1}/{args.parallel}]: {name}")
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            running[name] = p

        if running:
            time.sleep(5)

    log("全部视频处理完成")


if __name__ == "__main__":
    main()
