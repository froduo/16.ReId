#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「person 乘客审核」HTML 工具
================================
加载 session 目录下【所有】 person_N，每一行显示该 person 的全部乘客图片（横向滚动），
人工确认：
  1) 同一文件夹是否是同一个乘客（是 / 否）
  2) 若「否」→ 输入混入的乘客个数（如 2、3）
  3) 若「是」→ 选择真实方向：上车 / 下车 / 上车折返 / 下车折返 / 门口逗留 / 其他

标注结果可导出 JSON，供程序后续优化（拆分混入多人的文件夹 / 修正方向）。

用法（yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python render_person_audit.py \
      --session <session目录> --out <审核.html> [--max-imgs 0]
"""
import argparse
import os
import re

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
PERSON_RE = re.compile(r'^person_\d+_')

DIRECTIONS = ['上车', '下车', '上车折返', '下车折返', '门口逗留', '其他']


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


def collect_persons(session, max_imgs):
    persons = []
    if not os.path.isdir(session):
        return persons
    for name in sorted(os.listdir(session)):
        pdir = os.path.join(session, name)
        if not os.path.isdir(pdir) or not PERSON_RE.match(name):
            continue
        crop = os.path.join(pdir, '乘客图片')
        imgs = sample_images(crop, max_imgs)
        if not imgs:
            continue
        n_img = 0
        if os.path.isdir(crop):
            n_img = len([f for f in os.listdir(crop) if f.lower().endswith(IMAGE_EXTS)])
        persons.append({'name': name, 'direction': read_direction(pdir),
                        'imgs': imgs, 'n_img': n_img})
    return persons


def collect_all(root, max_imgs):
    """遍历 save_images 根目录下所有 session 子目录，收集所有 person"""
    sessions = []
    if not os.path.isdir(root):
        return sessions
    for name in sorted(os.listdir(root)):
        sdir = os.path.join(root, name)
        if not os.path.isdir(sdir):
            continue
        persons = collect_persons(sdir, max_imgs)
        if persons:
            sessions.append({'name': name, 'persons': persons})
    return sessions


HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>person 乘客审核</title>
<style>
  body { font-family: sans-serif; margin: 16px; max-width: 1400px; }
  h1 { font-size: 20px; }
  p.tip { color: #666; font-size: 13px; }
  .toolbar { position: sticky; top: 0; background: #fff; padding: 10px 0; border-bottom: 2px solid #333; z-index: 10; }
  .toolbar button { padding: 8px 16px; font-size: 14px; margin-right: 8px; cursor: pointer; }
  .person { border: 2px solid #ccc; border-radius: 8px; margin: 12px 0; padding: 10px; background: #fafafa; }
  .person .head { font-size: 14px; font-weight: bold; margin-bottom: 6px; }
  .person .imgs { overflow-x: auto; white-space: nowrap; padding: 4px 0; }
  .person .imgs img { width: 56px; height: 112px; object-fit: cover; margin: 2px; border: 1px solid #ccc; display: inline-block; }
  .person .ctrl { margin-top: 8px; font-size: 14px; }
  .person .ctrl .sub { margin-top: 6px; display: none; }
  .person .ctrl label { display: inline-block; margin-right: 14px; cursor: pointer; }
  .person.done { border-color: #2e7d32; background: #e8f5e9; }
  .person.multi { border-color: #e53935; background: #ffebee; }
  .person .cnt input { width: 50px; font-size: 14px; }
  #out { width: 100%; height: 160px; font-family: monospace; font-size: 12px; margin-top: 8px; }
  .stat { font-size: 14px; margin: 6px 0; }
  .session { font-size: 16px; margin: 20px 0 6px; border-bottom: 2px solid #999; padding-bottom: 4px; }
  #nav { margin: 8px 0; font-size: 13px; line-height: 1.9; }
  #nav a { margin-right: 12px; text-decoration: none; color: #1565c0; white-space: nowrap; }
</style>
</head>
<body>
<h1>person 乘客审核（共 __TOTAL__ 个 person / __NSESS__ 个 session）</h1>
<div class="stat">请逐行判断：<b>① 同一文件夹是否是同一个乘客</b>（否→输入乘客个数）；<b>② 若是同一乘客</b>→ 选择真实方向。</div>
<p class="tip">
  方向：<b>上车</b>=从下往上穿越门线；<b>下车</b>=从上往下穿越门线；
  <b>上车折返</b>=先上车未离开视野即反向下车；<b>下车折返</b>=先下车未离开视野即反向上车；
  <b>门口逗留</b>=在门线附近停留未穿越；<b>其他</b>=碎片/无法判断。
</p>
<div class="toolbar">
  <button onclick="exportJson()">导出审核结果 JSON</button>
  <button onclick="autoFillOthers()">未审核的默认置为「同一乘客 + 其他」</button>
  <span id="progress" style="margin-left:12px;color:#333;"></span>
</div>
<div id="nav">__NAV__</div>
<div id="list">__PERSONS__</div>
<h2>审核结果 JSON（复制保存为 audit_result.json）</h2>
<textarea id="out" readonly></textarea>
<script>
const state = {};
function markDone(name, cls) {
  const card = document.getElementById('card-' + name);
  if (card) { card.classList.remove('done', 'multi'); if (cls) card.classList.add(cls); }
  updateProgress();
}
function chooseSame(name, isSame) {
  state[name] = state[name] || {};
  state[name].same = isSame;
  const card = document.getElementById('card-' + name);
  if (card) {
    card.querySelector('.sub-same').style.display = isSame ? 'block' : 'none';
    card.querySelector('.sub-multi').style.display = isSame ? 'none' : 'block';
  }
  if (isSame) markDone(name, 'done'); else markDone(name, 'multi');
}
function chooseDirection(name, dir) {
  state[name] = state[name] || {};
  state[name].direction = dir;
  markDone(name, 'done');
}
function setCount(name) {
  state[name] = state[name] || {};
  const v = document.getElementById('cnt-' + name).value;
  state[name].count = parseInt(v, 10) || 0;
  markDone(name, 'multi');
}
function autoFillOthers() {
  document.querySelectorAll('.person').forEach(card => {
    const name = card.dataset.name;
    if (!state[name]) { state[name] = {same: true, direction: '其他'}; }
    if (state[name].same === undefined) { state[name].same = true; state[name].direction = '其他'; }
    if (state[name].same) {
      if (!state[name].direction) state[name].direction = '其他';
      card.classList.add('done'); card.classList.remove('multi');
      card.querySelector('.sub-same').style.display = 'block';
    }
  });
  updateProgress();
}
function updateProgress() {
  const total = document.querySelectorAll('.person').length;
  const done = Object.keys(state).length;
  document.getElementById('progress').textContent = '已审核 ' + done + ' / ' + total;
}
function exportJson() {
  const out = {};
  document.querySelectorAll('.person').forEach(card => {
    const uid = card.dataset.name;
    if (state[uid]) {
      out[uid] = Object.assign({}, state[uid],
        { session: card.dataset.session || '', person: card.dataset.person || uid });
    }
  });
  document.getElementById('out').value = JSON.stringify(out, null, 2);
}
updateProgress();
</script>
</body>
</html>
'''


def render_person(session_name, p):
    uid = f'{session_name}__{p["name"]}'
    imgs = ''.join(f'<img src="file://{path}" loading="lazy">' for path in p['imgs'])
    dir_opts = ''.join(
        f'<label><input type="radio" name="dir-{uid}" value="{d}" '
        f'onchange="chooseDirection(\'{uid}\',\'{d}\')"> {d}</label>'
        for d in DIRECTIONS)
    return (f'<div class="person" id="card-{uid}" data-name="{uid}" '
            f'data-session="{session_name}" data-person="{p["name"]}">\n'
            f'  <div class="head">[{session_name}] {p["name"]}（当前方向={p["direction"]}，{p["n_img"]} 张图）</div>\n'
            f'  <div class="imgs">{imgs}</div>\n'
            f'  <div class="ctrl">\n'
            f'    <b>同一乘客？</b>\n'
            f'    <button onclick="chooseSame(\'{uid}\', true)">✓ 是</button>\n'
            f'    <button onclick="chooseSame(\'{uid}\', false)">✗ 否（混入多人）</button>\n'
            f'    <div class="sub sub-same">\n'
            f'      方向：{dir_opts}\n'
            f'    </div>\n'
            f'    <div class="sub sub-multi cnt">\n'
            f'      混入乘客个数：<input type="number" id="cnt-{uid}" min="2" max="10" value="2" '
            f'onchange="setCount(\'{uid}\')"> 人\n'
            f'    </div>\n'
            f'  </div>\n'
            f'</div>')


def main():
    ap = argparse.ArgumentParser(description='生成 person 乘客审核 HTML 工具')
    ap.add_argument('--session', default=None, help='单个 session 目录（与 --root 二选一）')
    ap.add_argument('--root', default=None, help='save_images 根目录，自动遍历所有 session')
    ap.add_argument('--out', default='/media/tianheng/test/16.reId/CameraViewer_passenger/auto_pipeline/person_audit.html')
    ap.add_argument('--max-imgs', type=int, default=0, help='每个 person 展示图片数（0=全部）')
    args = ap.parse_args()

    if args.session:
        sessions = [{'name': os.path.basename(os.path.normpath(args.session)),
                     'persons': collect_persons(args.session, args.max_imgs)}]
    else:
        root = args.root or '/media/tianheng/test/16.reId/build-CameraViewer_passenger-Desktop-Debug/save_images'
        sessions = collect_all(root, args.max_imgs)

    sessions = [s for s in sessions if s['persons']]
    total = sum(len(s['persons']) for s in sessions)
    print(f'扫描到 {len(sessions)} 个 session，共 {total} 个 person')

    cards, nav = [], []
    for s in sessions:
        sid = f"sess-{s['name']}"
        nav.append(f'<a href="#{sid}">{s["name"]}（{len(s["persons"])}）</a>')
        cards.append(f'<h2 class="session" id="{sid}">{s["name"]}（{len(s["persons"])} 个 person）</h2>')
        cards.append('\n'.join(render_person(s['name'], p) for p in s['persons']))

    html = (HTML.replace('__PERSONS__', '\n'.join(cards))
                .replace('__NAV__', ' · '.join(nav))
                .replace('__TOTAL__', str(total))
                .replace('__NSESS__', str(len(sessions))))
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'已生成审核工具: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

