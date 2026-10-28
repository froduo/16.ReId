#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动恰3DV6 AI HTTP 新对接协议（客流时段聚合上行）—— 验证工具

用途：
  1) --binary  运行 CameraViewer 的 --selftest-uplink（真实 C++ 代码路径），
               校验协议不变量与「精准识别某一个人的上车/下车」。
  2) --loopback 本地起一个模拟服务端（严格按协议实现接收规则），
               把报文真正 HTTP POST 过去，验证：2xx+非空体、revision 取大、幂等。
  3) --live     连接真实平台 footfall.dongqia.com：验证账号 dongqia01/123456 登录，
               并探测文档约定的上行端点是否可接收。

示例：
  python3 verify_dongqia_uplink.py --all
  python3 verify_dongqia_uplink.py --binary build-dongqia/CameraViewer
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

# ----------------------------------------------------------------------------
# 协议常量
# ----------------------------------------------------------------------------
BASE_URL = "https://footfall.dongqia.com"
INTERVAL_PATH = "/api/passenger-flow/interval-aggregate"
PERSON_PATH = "/api/passenger-flow/person-event"
USERNAME = "dongqia01"
PASSWORD = "123456"
LOGIN_PATH = "/api/iot-auth/auth/token"

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
_results = []


def check(name, ok, detail=""):
    _results.append((name, ok))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


# ----------------------------------------------------------------------------
# 1) C++ 自检：真实代码路径
# ----------------------------------------------------------------------------
def run_binary_selftest(binary):
    print("\n=== 1) C++ 自检（真实代码路径）:", binary, "===")
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    try:
        out = subprocess.run([os.path.abspath(binary), "--selftest-uplink"],
                             capture_output=True, text=True, timeout=120, env=env)
    except FileNotFoundError:
        return check("可执行文件存在", False, binary)
    except subprocess.TimeoutExpired:
        return check("自检未超时", False)

    text = out.stdout
    print(text.rstrip())
    if out.returncode != 0 and not text:
        print("  stderr:", out.stderr.strip()[:600])
        return check("自检进程正常退出", False, f"rc={out.returncode}")

    check("自检整体结果 = PASS", "SELFTEST_RESULT=PASS" in text)

    # 精准识别某人：V1 上车 1 次 / 下车 1 次
    check("V1 上车/下车计数正确", "SUMMARY_V1=V1 上车1次/下车1次" in text)

    # 单人事件：boarding 方向 B->A，alighting 方向 A->B，且 visitor=1
    check("V1 boarding 事件方向 B->A",
          "PE|visitor=1|type=boarding|dir=B->A" in text)
    check("V1 alighting 事件方向 A->B",
          "PE|visitor=1|type=alighting|dir=A->B" in text)

    # 报文 JSON 可解析且恒等式成立
    line = next((l for l in text.splitlines() if l.startswith("INTERVAL_JSON=")), None)
    if not line:
        return check("存在 INTERVAL_JSON", False)
    try:
        msg = json.loads(line.split("=", 1)[1])
    except json.JSONDecodeError as e:
        return check("INTERVAL_JSON 可解析", False, str(e))

    check("schema_version=1.0", msg.get("schema_version") == "1.0")
    check("type=interval_aggregate", msg.get("type") == "interval_aggregate")
    check("end_ts_s = start_ts_s + 60",
          msg["end_ts_s"] == msg["start_ts_s"] + 60)
    raw = msg["raw_stats"]["count"]
    fin = msg["final_stats"]["count"]
    rep = msg["final_stats"]["repeat"]
    non = msg["non_customer"]["count"]
    check(f"恒等式 raw({raw}) = final({fin}) + repeat({rep}) + nonC({non})",
          raw == fin + rep + non)
    check("timezone 三键齐全",
          set(msg["timezone"].keys()) == {"name", "utc_offset_seconds", "is_dst"})
    return True


# ----------------------------------------------------------------------------
# 2) 本地模拟服务端：验证传输语义（revision 取大 / 幂等 / 非空体）
# ----------------------------------------------------------------------------
class MockStore:
    def __init__(self):
        self.rows = {}          # (master, batch, start) -> (revision, body)
        self.accepted = 0
        self.dropped = 0
        self.lock = threading.Lock()

    def upsert(self, msg):
        key = (msg["master_sn"], msg["batch_date"], msg["start_ts_s"])
        rev = msg["revision"]
        with self.lock:
            if key not in self.rows:
                self.rows[key] = (rev, msg)
                self.accepted += 1
                return "insert"
            if rev > self.rows[key][0]:
                self.rows[key] = (rev, msg)
                self.accepted += 1
                return "overwrite"
            self.dropped += 1
            return "drop"


STORE = MockStore()


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静音
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            msg = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"bad json"}')
            return
        action = STORE.upsert(msg)
        body = json.dumps({"ok": True, "action": action}).encode("utf-8")
        # 协议要求：2xx 且响应体非空
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_interval(start_ts, revision, enter=1, leave=0, pass_by=0, turn_back=0):
    return {
        "schema_version": "1.0", "type": "interval_aggregate",
        "master_sn": "CAM-LOOPBACK", "device_sns": ["CAM-LOOPBACK"],
        "timezone": {"name": "Asia/Shanghai", "utc_offset_seconds": 28800, "is_dst": False},
        "batch_date": "2026-09-23", "stat_basis": "enter",
        "revision": revision, "interval_s": 60,
        "start_ts_s": start_ts, "end_ts_s": start_ts + 60,
        "event_counts": {"enter": enter, "leave": leave,
                         "pass_by": pass_by, "turn_back": turn_back},
        "raw_stats": {"count": enter,
                      "breakdowns": {"height_category": [{"height_category": "not_measured", "count": enter}] if enter else [],
                                     "age_range|gender": [{"age_range": "not_measured", "gender": "not_measured", "count": enter}] if enter else []}},
        "final_stats": {"count": enter, "repeat": 0,
                        "breakdowns": {"height_category": [{"height_category": "not_measured", "count": enter}] if enter else [],
                                       "age_range|gender": [{"age_range": "not_measured", "gender": "not_measured", "count": enter}] if enter else []}},
        "non_customer": {"count": 0, "breakdowns": {"person_type": []}},
        "dwell_stats": {"count": 0, "breakdowns": {"dwell_range": []}}
    }


def post_json(url, obj, timeout=10):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def run_loopback():
    print("\n=== 2) 本地模拟服务端：传输语义验证 ===")
    srv = HTTPServer(("127.0.0.1", 8077), MockHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    url = f"http://127.0.0.1:8077{INTERVAL_PATH}"
    start = 1790149800

    try:
        # a) 首次入库
        st, body = post_json(url, make_interval(start, 1000))
        check("首次报文 2xx", 200 <= st < 300)
        check("响应体非空（协议要求）", len(body) > 0)

        # b) 更小 revision 应被丢弃
        post_json(url, make_interval(start, 900))
        # c) 更大 revision 应覆盖
        post_json(url, make_interval(start, 1500, enter=2))

        key = ("CAM-LOOPBACK", "2026-09-23", start)
        rev, msg = STORE.rows[key]
        check("revision 取大（保留 1500）", rev == 1500, f"actual={rev}")
        check("整件替换（enter 更新为 2）", msg["event_counts"]["enter"] == 2)

        # d) 幂等：重复投递终态不变
        before = STORE.rows[key]
        post_json(url, make_interval(start, 1500, enter=2))
        post_json(url, make_interval(start, 1500, enter=2))
        check("幂等：重复投递终态不变", STORE.rows[key] == before)

        # e) 同一分钟不同 revision 只留最新
        check("丢弃计数 > 0（存在被丢弃的迟到件）", STORE.dropped > 0,
              f"dropped={STORE.dropped}")
    finally:
        srv.shutdown()
    return True


# ----------------------------------------------------------------------------
# 3) 真实平台连通性 / 账号验证
# ----------------------------------------------------------------------------
def post_json_auth(url, obj, token=None, timeout=15, basic=None):
    data = json.dumps(obj).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if basic:
        import base64
        headers["Authorization"] = "Basic " + base64.b64encode(
            f"{basic[0]}:{basic[1]}".encode()).decode()
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return -1, str(e)


def run_live():
    print("\n=== 3) 真实平台验证:", BASE_URL, "===")
    # 3.1 账号登录
    st, body = post_json_auth(BASE_URL + LOGIN_PATH,
                              {"username": USERNAME, "password": PASSWORD})
    token = None
    try:
        j = json.loads(body)
        ok = (j.get("code") == 200 and bool(j.get("data")))
        token = j.get("data")
    except Exception:
        ok = False
    check("账号登录 (POST /api/iot-auth/auth/token) 成功", ok, f"http={st}")

    # 3.2 文档约定的上行接收端点可达性
    payload = make_interval(int(time.time()) // 60 * 60, int(time.time()))
    st2, body2 = post_json_auth(BASE_URL + INTERVAL_PATH, payload, token=token,
                                basic=(USERNAME, PASSWORD))
    reachable = 200 <= st2 < 300
    print(f"  [i] 上行端点 {INTERVAL_PATH} -> http {st2}")
    print(f"      body: {body2[:160]}")
    if reachable:
        check("上行接收端点可用", True)
    else:
        print("  [!] 上行端点当前不可接收（服务端尚未部署该接收端点或路径不同）。")
        print("      协议 0 节说明：路径与服务端地址均可在设备侧配置，"
              "需服务端实现接收端点后再联调。")
    return True


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build-dongqia/CameraViewer")
    ap.add_argument("--binary-only", action="store_true")
    ap.add_argument("--loopback", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all or args.binary_only or not (args.loopback or args.live):
        run_binary_selftest(args.binary)
    if args.all or args.loopback:
        run_loopback()
    if args.all or args.live:
        run_live()

    total = len(_results)
    passed = sum(1 for _, ok in _results if ok)
    print(f"\n================ 汇总: {passed}/{total} 通过 ================")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
