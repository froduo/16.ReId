#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 match_result.json 的人工核对 HTML 对比页
================================================
把每对匹配的两个 person 的「乘客图片」左右并排展示，方便人工判断是否同一人。
高置信(sim>=0.85)绿色边框，低置信红色边框（需重点复核）。

用法（yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python render_match_review.py \
      --match /media/tianheng/test/16.reId/CameraViewer_passenger/match_result.json \
      --sessions <session目录1> <session目录2> ... \
      --out /media/tianheng/test/16.reId/CameraViewer_passenger/match_review.html \
      --max-imgs 8
"""
import argparse
import json
import os

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def sample_crops(sess_map, name, session, max_imgs):
    """从 person 的「乘客图片」均匀采样 max_imgs 张代表图。"""
    sess_dir = sess_map.get(session)
    if not sess_dir:
        return []
    crop_dir = os.path.join(sess_dir, name, '乘客图片')
    if not os.path.isdir(crop_dir):
        return []
    files = sorted(f for f in os.listdir(crop_dir) if f.lower().endswith(IMAGE_EXTS))
    if len(files) > max_imgs:
        step = len(files) / max_imgs
        files = [files[int(i * step)] for i in range(max_imgs)]
    return [os.path.join(crop_dir, f) for f in files]


def img_tag(path, w=80, h=160):
    return (f'<img src="file://{path}" width="{w}" height="{h}" '
            f'style="object-fit:cover;margin:2px;border:1px solid #999">')


def main():
    ap = argparse.ArgumentParser(description='生成匹配人工核对 HTML 对比页')
    ap.add_argument('--match', required=True, help='match_result.json 路径')
    ap.add_argument('--sessions', nargs='+', required=True, help='一个或多个 session 目录')
    ap.add_argument('--out', default='/media/tianheng/test/16.reId/CameraViewer_passenger/match_review.html')
    ap.add_argument('--max-imgs', type=int, default=8, help='每侧展示的代表图数量')
    ap.add_argument('--exclude', default=None,
                    help='已确认配对 JSON（confirmed_pairs.json），排除这些配对，只展示「非同一人」待重新匹配')
    args = ap.parse_args()

    data = json.load(open(args.match, encoding='utf-8'))
    matches = data['matches']

    # 排除已确认的配对（person_a↔person_b 任一顺序都算已确认）
    if args.exclude and os.path.isfile(args.exclude):
        confirmed = json.load(open(args.exclude, encoding='utf-8'))
        confirmed_set = set()
        for c in confirmed:
            confirmed_set.add((c['person_a'], c['person_b']))
            confirmed_set.add((c['person_b'], c['person_a']))
        before = len(matches)
        matches = [m for m in matches
                   if (m['person_a'], m['person_b']) not in confirmed_set]
        print(f'排除已确认 {before - len(matches)} 对，剩余 {len(matches)} 对「非同一人」待重新匹配')

    sess_map = {os.path.basename(s.rstrip('/')): s for s in args.sessions}

    rows = []
    for i, m in enumerate(matches, 1):
        a, b = m['person_a'], m['person_b']
        sim = m['similarity']
        if sim >= 0.85:
            cls, flag = 'high', '高置信'
        elif sim >= 0.70:
            cls, flag = 'mid', '中置信'
        else:
            cls, flag = 'low', '需复核'
        a_imgs = sample_crops(sess_map, a, m['a_session'], args.max_imgs)
        b_imgs = sample_crops(sess_map, b, m['b_session'], args.max_imgs)
        a_html = ''.join(img_tag(p) for p in a_imgs)
        b_html = ''.join(img_tag(p) for p in b_imgs)
        rows.append(
            f'<div class="pair {cls}" id="pair-{i}">\n'
            f'  <div class="head"><b>#{i}</b> [<span class="{cls}">{flag}</span>] '
            f'{a}({m["a_dir"]}) ↔ {b}({m["b_dir"]}) <b>sim={sim:.4f}</b></div>\n'
            f'  <div class="side">{a_html}</div>\n'
            f'  <div class="side">{b_html}</div>\n'
            f'  <div class="confirm">'
            f'<button onclick="confirmPair({i},\'{a}\',\'{m["a_dir"]}\',\'{b}\',\'{m["b_dir"]}\',{sim:.4f})">✓ 同一人</button> '
            f'<button onclick="rejectPair({i})">✗ 非同一人</button></div>\n'
            f'</div>'
        )

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>person_N 匹配人工核对（{len(matches)} 对）</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  h1 {{ font-size: 20px; }}
  p.tip {{ color: #666; font-size: 13px; }}
  .toolbar {{ position: sticky; top: 0; background: #fff; padding: 10px 0; border-bottom: 2px solid #333; z-index: 10; }}
  .toolbar button {{ padding: 8px 16px; font-size: 14px; margin-right: 8px; cursor: pointer; }}
  .pair {{ border: 1px solid #ddd; margin: 12px 0; padding: 10px; }}
  .pair.high {{ border-left: 6px solid #2e7d32; background: #f1f8e9; }}
  .pair.mid {{ border-left: 6px solid #f9a825; background: #fff8e1; }}
  .pair.low {{ border-left: 6px solid #c62828; background: #fbe9e7; }}
  .pair.confirmed {{ outline: 3px solid #2e7d32; background: #e8f5e9; }}
  .pair.rejected {{ opacity: 0.4; }}
  .head {{ font-size: 14px; margin-bottom: 6px; }}
  .head .high {{ color: #2e7d32; font-weight: bold; }}
  .head .mid {{ color: #f9a825; font-weight: bold; }}
  .head .low {{ color: #c62828; font-weight: bold; }}
  .side {{ display: inline-block; vertical-align: top; margin-right: 16px; }}
  .confirm {{ margin-top: 8px; }}
  .confirm button {{ padding: 4px 12px; margin-right: 6px; cursor: pointer; }}
  #out {{ width: 100%; height: 160px; font-family: monospace; font-size: 12px; margin-top: 8px; }}
</style>
</head>
<body>
<h1>person_N 匹配人工核对（共 {len(matches)} 对）</h1>
<p class="tip">绿色=高置信(sim≥0.85)，黄色=中置信(0.70~0.85)，红色=需重点复核(sim&lt;0.70)。<br>
每对：左边=person_a，右边=person_b。请判断左右是否同一人，点「✓ 同一人」确认（用于导出 OSNet 训练数据集）。</p>
<div class="toolbar">
  <button onclick="exportJson()">导出「同一人」配对 JSON（用于重建数据集）</button>
  <span id="progress" style="margin-left:12px;color:#333;"></span>
</div>
{chr(10).join(rows)}
<h2>确认「同一人」的配对 JSON（复制保存为 confirmed_pairs.json）</h2>
<textarea id="out" readonly></textarea>
<script>
const confirmed = {{}};
function confirmPair(i, a, adir, b, bdir, sim) {{
  confirmed[i] = {{person_a: a, a_dir: adir, person_b: b, b_dir: bdir, similarity: sim}};
  const card = document.getElementById('pair-' + i);
  card.classList.add('confirmed');
  card.classList.remove('rejected');
  updateProgress();
}}
function rejectPair(i) {{
  delete confirmed[i];
  const card = document.getElementById('pair-' + i);
  card.classList.remove('confirmed');
  card.classList.add('rejected');
  updateProgress();
}}
function updateProgress() {{
  const total = document.querySelectorAll('.pair').length;
  const done = Object.keys(confirmed).length;
  document.getElementById('progress').textContent = '已确认同一人 ' + done + ' / ' + total;
}}
function exportJson() {{
  document.getElementById('out').value = JSON.stringify(Object.values(confirmed), null, 2);
}}
updateProgress();
</script>
</body>
</html>'''

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'已生成: {args.out}（{len(matches)} 对）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
