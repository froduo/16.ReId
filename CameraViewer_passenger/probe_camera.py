#!/usr/bin/env python3
"""探测 HX-CCD23 相机的 API 端点和流地址"""
import requests
import re
import socket
import subprocess
import sys

CAMERA_IP = "192.168.99.10"
BASE = f"http://{CAMERA_IP}"

session = requests.Session()
session.auth = ("admin", "hf123456")

def check_port(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    result = s.connect_ex((CAMERA_IP, port))
    s.close()
    return result == 0

# 1. 扫描端口
print("=" * 60)
print(" 1. 端口扫描")
print("=" * 60)
for port in [80, 443, 554, 1935, 8000, 8080, 8554, 8888, 9000]:
    if check_port(port):
        print(f"  端口 {port}: 开放")

# 2. 获取首页
print("\n" + "=" * 60)
print(" 2. Web 首页")
print("=" * 60)
try:
    resp = session.get(BASE, timeout=5)
    print(f"  状态码: {resp.status_code}")
    # 提取 JS 文件
    js_files = re.findall(r'src="([^"]*\.js[^"]*)"', resp.text)
    print(f"  JS 文件: {js_files}")
except Exception as e:
    print(f"  错误: {e}")

# 3. 获取主 JS 文件
print("\n" + "=" * 60)
print(" 3. 主 JS 文件分析")
print("=" * 60)
try:
    resp = session.get(f"{BASE}/static/js/foorir-CO3NoLKg.js", timeout=10)
    content = resp.text
    print(f"  JS 文件大小: {len(content)} 字节")
    
    # 提取 API 路径
    apis = re.findall(r'["\'](/[a-zA-Z0-9/_\-\.]+(?:\?[^"\']*)?)["\']', content)
    unique_apis = sorted(set(apis))
    print(f"\n  发现 {len(unique_apis)} 个 API 端点:")
    for api in unique_apis:
        print(f"    {api}")
    
    # 查找流相关字符串
    print(f"\n  --- 流/视频相关 ---")
    streams = re.findall(r'["\']([^"\']*(?:rtsp|rtmp|stream|webrtc|live|video|snap|capture|rtc|hls|flv|m3u8|on_demand)[^"\']*)["\']', content, re.IGNORECASE)
    for s in sorted(set(streams)):
        print(f"    {s}")
        
except Exception as e:
    print(f"  错误: {e}")

# 4. 测试 API 端点
print("\n" + "=" * 60)
print(" 4. 测试常见 API")
print("=" * 60)
test_urls = [
    f"{BASE}/api/getDeviceInfo",
    f"{BASE}/api/getSystemInfo",
    f"{BASE}/api/getStreamUrl",
    f"{BASE}/api/rtsp",
    f"{BASE}/api/video",
    f"{BASE}/api/capture",
    f"{BASE}/api/snapshot",
    f"{BASE}/cgi-bin/snapshot.cgi",
    f"{BASE}/tmpfs/snap.jpg",
    f"{BASE}/capture",
    f"{BASE}/api/getRtspAddr",
    f"{BASE}:8000/api/getRtspAddr",
]
for url in test_urls:
    try:
        resp = session.get(url, timeout=3)
        status = resp.status_code
        ct = resp.headers.get("Content-Type", "")
        body_preview = resp.text[:100] if "json" in ct or "text" in ct else f"[binary {len(resp.content)} bytes]"
        print(f"  {status} {url}")
        if status == 200:
            print(f"       Content-Type: {ct}")
            print(f"       Body: {body_preview}")
    except Exception as e:
        print(f"  ERR {url}: {e}")

# 5. 测试端口 8000
print("\n" + "=" * 60)
print(" 5. 端口 8000 探测")
print("=" * 60)
for url in [
    f"http://{CAMERA_IP}:8000/",
    f"http://{CAMERA_IP}:8000/api",
    f"http://{CAMERA_IP}:8000/api/getRtspAddr",
    f"http://{CAMERA_IP}:8000/index.html",
]:
    try:
        resp = requests.get(url, timeout=3)
        print(f"  {resp.status_code} {url}")
        if resp.status_code == 200:
            print(f"       Body: {resp.text[:150]}")
    except Exception as e:
        print(f"  ERR {url}: {e}")

# 6. RTSP URL 测试
print("\n" + "=" * 60)
print(" 6. RTSP 流地址测试")
print("=" * 60)
rtsp_urls = [
    f"rtsp://{CAMERA_IP}:8554/live/main",
    f"rtsp://{CAMERA_IP}:8554/live/sub",
    f"rtsp://{CAMERA_IP}:8554/live/0",
    f"rtsp://{CAMERA_IP}:8554/live/1",
    f"rtsp://{CAMERA_IP}:8554/stream0",
    f"rtsp://{CAMERA_IP}:8554/stream1",
    f"rtsp://{CAMERA_IP}:8554/0",
    f"rtsp://{CAMERA_IP}:8554/1",
    f"rtsp://{CAMERA_IP}:8554/live",
    f"rtsp://admin:hf123456@{CAMERA_IP}:8554/live/main",
    f"rtsp://admin:hf123456@{CAMERA_IP}:8554/live/sub",
]
for url in rtsp_urls:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-rtsp_transport", "tcp", 
             "-analyzeduration", "3000000", "-probesize", "3000000",
             "-i", url, "-show_entries", "stream=codec_name,width,height",
             "-of", "json"],
            capture_output=True, text=True, timeout=8
        )
        if result.returncode == 0 and "codec_name" in result.stdout:
            print(f"  ✓ {url}")
            print(f"       {result.stdout.strip()[:200]}")
        else:
            err = result.stderr.strip().split('\n')[-1] if result.stderr else "unknown"
            print(f"  ✗ {url} -> {err}")
    except subprocess.TimeoutExpired:
        print(f"  ⏱ {url} -> 超时")
    except Exception as e:
        print(f"  ERR {url}: {e}")

print("\n探测完成")
