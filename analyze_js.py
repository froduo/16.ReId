#!/usr/bin/env python3
import re
with open('/tmp/foorir_main.js') as f:
    c = f.read()
print('=== API ===')
for x in sorted(set(re.findall(r'"/api/[^"]*"', c))): print(x)
print('=== RTSP ===')
for x in sorted(set(re.findall(r'"[^"]*rtsp[^"]*"', c, re.I))): print(x)
print('=== STREAM/LIVE ===')
for x in sorted(set(re.findall(r'"[^"]*(?:stream|webrtc|live|video|snap|capture|hls|flv|m3u8)[^"]*"', c, re.I))): print(x)
print('=== RTMP/PUSH ===')
for x in sorted(set(re.findall(r'"[^"]*(?:rtmp|push|publish)[^"]*"', c, re.I))): print(x)
print('=== 关键函数名 ===')
for x in sorted(set(re.findall(r'(?:function\s+|var\s+|const\s+|let\s+)(\w*(?:stream|video|live|play|rtsp|webrtc|rtc|camera|snap|capture)\w*)', c, re.I))): print(f'  {x}')
