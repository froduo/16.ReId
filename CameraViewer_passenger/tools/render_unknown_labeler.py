#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「方向人工标注」HTML 工具
============================
扫描 session 目录下方向为「未判定」的 person（可选包含「折返」），
为每个 person 展示采样代表图 + 方向单选按钮，供人工判断真实方向：
  上车 / 下车 / 上车折返 / 下车折返 / 其他

标注结果可导出 JSON，配合 apply_direction_labels.py 回写到 person 目录的
match.txt，让后续评估/训练使用人工标注的方向。

用法（yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python render_unknown_labeler.py \
      --session <session目录> --out <输出.html> --max-imgs 10 [--include-invalid]
"""
import argparse
import os
import re

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
PERSON_RE = re.compile(r'^person_\d+_')


def read_direction(person_dir):
    mt = os.path.join(person_dir, 'match.txt')
    if os.path.isfile(mt):
        try:
            with open(mt, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('方向'):
                        return line.split(':', 1)[1].strip()
        except Exception:
            pass
    return '未判定'


def sample_images(crop_dir, max_imgs):
    if not os.path.isdir(crop_dir):
        return []
    files = sorted(f for f in os.listdir(crop_dir) if f.lower().endswith(IMAGE_EXTS))
    # max_imgs=0 表示展示全部（按时间顺序）；>0 时均匀采样到 max_imgs 张
    if max_imgs and len(files) > max_imgs:
        step = len(files) / max_imgs
        files = [files[int(i * step)] for i in range(max_imgs)]
    return [os.path.join(crop_dir, f) for f in files]


def collect_persons(session, max_imgs, include_invalid):
    persons = []
    if not os.path.isdir(session):
        return persons
    for name in sorted(os.listdir(session)):
        pdir = os.path.join(session, name)
        if not os.path.isdir(pdir) or not PERSON_RE.match(name):
            continue
        direction = read_direction(pdir)
        if direction != '未判定':
            if not (include_invalid and direction.endswith('折返')):
                continue
        imgs = sample_images(os.path.join(pdir, '乘客图片'), max_imgs)
        if not imgs:
            continue
        crop = os.path.join(pdir, '乘客图片')
        n_img = 0
        if os.path.isdir(crop):
            n_img = len([f for f in os.listdir(crop) if f.lower().endswith(IMAGE_EXTS)])
        persons.append({'name': name, 'direction': direction,
                        'imgs': imgs, 'n_img': n_img})
    return persons


HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>方向人工标注（未判定/折返 person）</title>
<style>
  body { font-family: sans-serif; margin: 16px; max-width: 1400px; }
  h1 { font-size: 20px; }
  p.tip { color: #666; font-size: 13px; }
  .toolbar { position: sticky; top: 0; background: #fff; padding: 10px 0; border-bottom: 2px solid #333; z-index: 10; }
  .toolbar button { padding: 8px 16px; font-size: 14px; margin-right: 8px; cursor: pointer; }
  .person { border: 2px solid #ccc; border-radius: 8px; margin: 12px 0; padding: 10px; background: #fafafa; }
  .person.invalid { border-color: #fb8c00; background: #fff8e1; }
  .person .head { font-size: 14px; font-weight: bold; margin-bottom: 6px; }
  .person .imgs { overflow-x: auto; white-space: nowrap; padding: 4px 0; }
  .person .imgs img { width: 52px; height: 104px; object-fit: cover; margin: 2px; border: 1px solid #ccc; display: inline-block; }
  .person .opts label { display: inline-block; margin-right: 14px; font-size: 14px; cursor: pointer; }
  .person.done { border-color: #2e7d32; background: #e8f5e9; }
  #out { width: 100%; height: 160px; font-family: monospace; font-size: 12px; margin-top: 8px; }
  .stat { font-size: 14px; margin: 6px 0; }
</style>
</head>
<body>
<h1>方向人工标注（未判定/折返 person）</h1>
<div class="stat">共 <b>__TOTAL__</b> 个 person 待标注（未判定 __UNKNOWN__ / 折返 __INVALID__）</div>
<p class="tip">
  请根据每个 person 的「乘客图片」判断其真实方向：
  <b>上车</b>=从下往上穿越门线；<b>下车</b>=从上往下穿越门线；
  <b>上车折返</b>=先上车未离开视野即反向下车；<b>下车折返</b>=先下车未离开视野即反向上车；
  <b>其他</b>=碎片/无法判断/多人混杂。标注后卡片变绿，可导出 JSON。
</p>
<div class="toolbar">
  <button onclick="exportJson()">导出标注结果 JSON</button>
  <button onclick="selectAllUnknown()">未标注的全部置为「其他」</button>
  <span id="progress" style="margin-left:12px;color:#333;"></span>
</div>
<div id="list">__PERSONS__</div>
<h2>标注结果 JSON（复制保存为 labels.json，供 apply_direction_labels.py 回写）</h2>
<textarea id="out" readonly></textarea>
<script>
const labels = {};
function onChoose(name, dir) {
  labels[name] = dir;
  const card = document.getElementById('card-' + name);
  if (card) card.classList.add('done');
  updateProgress();
}
function selectAllUnknown() {
  document.querySelectorAll('.person').forEach(card => {
    const name = card.dataset.name;
    if (!labels[name]) {
      labels[name] = '其他';
      const r = card.querySelector('input[value="其他"]');
      if (r) r.checked = true;
      card.classList.add('done');
    }
  });
  updateProgress();
}
function updateProgress() {
  const total = document.querySelectorAll('.person').length;
  const done = Object.keys(labels).length;
  document.getElementById('progress').textContent = '已标注 ' + done + ' / ' + total;
}
function exportJson() {
  document.getElementById('out').value = JSON.stringify(labels, null, 2);
}
updateProgress();
</script>
</body>
</html>
'''


def render_person(p):
    cls = 'person invalid' if p['direction'].endswith('折返') else 'person'
    imgs = ''.join(f'<img src="file://{path}" loading="lazy">' for path in p['imgs'])
    opts = ''
    for d in ['上车', '下车', '上车折返', '下车折返', '其他']:
        opts += (f'<label><input type="radio" name="{p["name"]}" value="{d}" '
                 f'onchange="onChoose(\'{p["name"]}\',\'{d}\')"> {d}</label>')
    return (f'<div class="{cls}" id="card-{p["name"]}" data-name="{p["name"]}">\n'
            f'  <div class="head">{p["name"]}（当前={p["direction"]}，{p["n_img"]} 张图）</div>\n'
            f'  <div class="imgs">{imgs}</div>\n'
            f'  <div class="opts">{opts}</div>\n'
            f'</div>')


def main():
    ap = argparse.ArgumentParser(description='生成方向人工标注 HTML 工具')
    ap.add_argument('--session', required=True, help='session 目录')
    ap.add_argument('--out', default='/media/tianheng/test/16.reId/CameraViewer_passenger/auto_pipeline/unknown_labeler.html')
    ap.add_argument('--max-imgs', type=int, default=0, help='每个 person 展示图片数（0=全部，横向一行滚动查看全过程）')
    ap.add_argument('--include-invalid', action='store_true',
                    help='同时包含「上车折返/下车折返」person（用于复核折返是否误判）')
    args = ap.parse_args()

    persons = collect_persons(args.session, args.max_imgs, args.include_invalid)
    unknown = sum(1 for p in persons if p['direction'] == '未判定')
    invalid = len(persons) - unknown
    print(f'扫描到 {len(persons)} 个待标注 person（未判定 {unknown}，折返 {invalid}）')

    cards = '\n'.join(render_person(p) for p in persons)
    html = (HTML.replace('__PERSONS__', cards)
                .replace('__TOTAL__', str(len(persons)))
                .replace('__UNKNOWN__', str(unknown))
                .replace('__INVALID__', str(invalid)))
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'已生成标注工具: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


