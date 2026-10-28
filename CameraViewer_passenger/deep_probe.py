#!/usr/bin/env python3
"""深入分析相机 API"""
import requests
import re
import json
import base64

IP = "192.168.99.10"
BASE = f"http://{IP}"
session = requests.Session()
session.auth = ("admin", "hf123456")

# 读取 JS
with open('/tmp/foorir_main.js') as f:
    js = f.read()

# 1. 找到所有包含 ImageAndVideo 的上下文
print("=== ImageAndVideo 上下文 ===")
for m in re.finditer(r'.{0,100}ImageAndVideo.{0,100}', js):
    print(f"  ...{m.group()}...")
    print()

# 2. 找到 POST/GET 请求的所有 URL
print("\n=== 请求 URL ===")
for m in re.finditer(r'(?:He|xe|fa|xe)\.(?:POST|GET|PUT)\("([^"]+)"', js):
    url = m.group(1)
    print(f"  {url}")

# 3. 找到所有以 / 开头的路径
print("\n=== 所有 API 路径 ===")
paths = re.findall(r'"(/[A-Z][^"]*)"', js)
for p in sorted(set(paths)):
    print(f"  {p}")

# 4. 尝试 POST 到 /ImageAndVideo/CapturePictureBase64
print("\n=== 测试 CapturePictureBase64 ===")
for prefix in ['', '/api', '/v1']:
    url = f"{BASE}{prefix}/ImageAndVideo/CapturePictureBase64"
    try:
        r = session.post(url, timeout=5)
        print(f"  POST {url} -> {r.status_code} {r.headers.get('Content-Type','')} {r.text[:100]}")
    except Exception as e:
        print(f"  POST {url} -> ERR: {e}")

# 5. 测试 camera API
print("\n=== 测试 Camera API ===")
for path in ['/api/camera/heartBeat', '/api/camera/dataUpload', '/api/passengerFlow', '/api/camera/heart']:
    url = f"{BASE}{path}"
    try:
        r = session.get(url, timeout=3)
        print(f"  GET {url} -> {r.status_code} {r.text[:100]}")
    except Exception as e:
        print(f"  GET {url} -> ERR: {e}")
    try:
        r = session.post(url, json={}, timeout=3)
        print(f"  POST {url} -> {r.status_code} {r.text[:100]}")
    except Exception as e:
        print(f"  POST {url} -> ERR: {e}")

# 6. 测试 RtmpPull API
print("\n=== 测试 RtmpPull ===")
for path in ['/ImageAndVideo/RtmpPullGet', '/ImageAndVideo/RtmpPullSet']:
    for prefix in ['', '/api', '/v1']:
        url = f"{BASE}{prefix}{path}"
        try:
            r = session.get(url, timeout=3)
            print(f"  GET {url} -> {r.status_code} {r.text[:100]}")
        except Exception as e:
            print(f"  GET {url} -> ERR: {e}")

# 7. 测试 RTSP URL
print("\n=== 测试 RTSP live/camera ===")
import subprocess
try:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-rtsp_transport", "tcp",
         "-i", f"rtsp://{IP}:8554/live/camera",
         "-show_entries", "stream=codec_name,width,height",
         "-of", "json"],
        capture_output=True, text=True, timeout=10
    )
    print(f"  stdout: {result.stdout[:300]}")
    print(f"  stderr: {result.stderr[:300]}")
except Exception as e:
    print(f"  ERR: {e}")

# 8. 测试 HTTP-FLV / HLS
print("\n=== 测试 HTTP-FLV/HLS ===")
for path in ['/live/camera.flv', '/live/camera/hls.m3u8', '/live/camera.live.flv']:
    url = f"http://{IP}:8000{path}"
    try:
        r = requests.get(url, timeout=3, stream=True)
        print(f"  GET {url} -> {r.status_code} {r.headers.get('Content-Type','')} {r.text[:50]}")
        r.close()
    except Exception as e:
        print(f"  GET {url} -> ERR: {e}")
