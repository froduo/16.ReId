#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「人工配对」交互式 HTML 工具
================================
- 扫描 session 目录下的 person_N，按方向分为「上车」和「下车」两组
- 左边显示上车 person，右边显示下车 person（各采样几张代表图）
- 人工点击左边一个 + 右边一个 = 配对，可撤销
- 配对结果在页面底部实时显示，可复制为 JSON（供后续导出数据集）

用法（yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python render_pairing_tool.py \
      --sessions <session目录1> <session目录2> ... \
      --out /media/tianheng/test/16.reId/CameraViewer_passenger/pairing_tool.html \
      --max-imgs 6 [--prefill 预配对结果.json]

--prefill 可传 manual_pairs.json 或 match_result.json（程序预配对结果），
会预填到配对列表供人工核对/修正（双击卡片撤销、点击补配）。
"""
import argparse
import json
import os
import re

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
PERSON_RE = re.compile(r'^person_\d+_')


def read_direction(person_dir):
    """读取 person 目录下 match.txt 的「方向」字段。"""
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
    """从「乘客图片」均匀采样 max_imgs 张代表图。"""
    if not os.path.isdir(crop_dir):
        return []
    files = sorted(f for f in os.listdir(crop_dir) if f.lower().endswith(IMAGE_EXTS))
    if len(files) > max_imgs:
        step = len(files) / max_imgs
        files = [files[int(i * step)] for i in range(max_imgs)]
    return [os.path.join(crop_dir, f) for f in files]


def collect_persons(sessions, max_imgs):
    persons = []
    for sess in sessions:
        if not os.path.isdir(sess):
            print(f'[warn] session 不存在: {sess}')
            continue
        for name in sorted(os.listdir(sess)):
            pdir = os.path.join(sess, name)
            if not os.path.isdir(pdir) or not PERSON_RE.match(name):
                continue
            imgs = sample_images(os.path.join(pdir, '乘客图片'), max_imgs)
            if not imgs:
                continue
            persons.append({
                'name': name,
                'direction': read_direction(pdir),
                'imgs': imgs,
            })
    return persons


HTML_HEAD = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>人工配对：上车 ↔ 下车</title>
<style>
  body { font-family: sans-serif; margin: 16px; }
  h1 { font-size: 18px; }
  p.tip { color: #666; font-size: 13px; }
  .cols { display: flex; gap: 20px; }
  .col { flex: 1; min-width: 0; }
  .col h2 { font-size: 15px; }
  .person { border: 2px solid #ccc; border-radius: 6px; margin: 8px 0; padding: 8px; cursor: pointer; background: #fff; }
  .person .name { font-size: 12px; font-weight: bold; margin-bottom: 4px; }
  .person img { width: 78px; height: 156px; object-fit: cover; margin: 2px; border: 1px solid #eee; }
  .person.selected { border-color: #f57c00; background: #fff3e0; box-shadow: 0 0 6px #f57c00; }
  .person.paired { /* 边框颜色由 JS 按配对组着色，不再变灰 */ }
  .col#mid-col { flex: 0 0 150px; }
  #mid { border: 1px solid #e0e0e0; border-radius: 4px; padding: 8px; background: #fafafa; font-size: 13px; }
  #mid b { color: #43a047; }
  #board, #alight { height: 72vh; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 4px; padding: 4px; background: #fafafa; }
  #pairs { margin-top: 16px; border-top: 2px solid #333; padding-top: 8px; }
  #pairs .row { font-size: 13px; margin: 3px 0; }
  #pairs .row button { margin-left: 10px; }
  textarea { width: 100%; height: 120px; font-size: 12px; margin-top: 8px; }
  button { cursor: pointer; }
</style>
</head>
<body>
<h1>人工配对：上车（左） ↔ 下车（右）</h1>
<p class="tip">
  已预填程序自动配对结果（彩色边框），请逐对核对：<br>
  1) 确认是同一人 → 点该对右侧「✓ 确认」，该对从界面消失并记入待导出结果；<br>
  2) 不是同一人 → 点「撤销」或双击卡片，解除配对；<br>
  3) 补配：点击左边一个「上车」+ 右边一个「下车」；<br>
  4) 全部确认后点「导出 JSON」，仅导出已「✓ 确认」的配对。
</p>
<div class="cols">
  <div class="col" id="board-col"><h2>上车（board）</h2><div id="board"></div></div>
  <div class="col" id="mid-col"><h2>确认</h2>
    <div id="mid">
      <p>已确认：<b id="confirm-count">0</b> 对</p>
      <p>待核对：<b id="pending-count">0</b> 对</p>
    </div>
  </div>
  <div class="col" id="alight-col"><h2>下车（alight）</h2><div id="alight"></div></div>
</div>
<div id="pairs">
  <h2>待核对配对（点「✓ 确认」后移入导出结果）</h2>
  <div id="pair-list"></div>
  <button onclick="exportJson()">导出 JSON</button>
  <button onclick="clearPairs()">清空</button>
  <textarea id="out" readonly placeholder="点击「导出 JSON」后在这里复制结果（仅含已✓确认的配对）"></textarea>
</div>
<script>
const PERSONS = __PERSONS__;
const MAX_IMGS = __MAX_IMGS__;
const PREFILL = __PREFILL__;
'''

HTML_TAIL = '''
const boardPersons = PERSONS.filter(p => p.direction === '上车');
const alightPersons = PERSONS.filter(p => p.direction === '下车');
let selectedBoard = null;
let pairs = PREFILL.slice();        // 未确认配对
let confirmedPairs = [];            // 已「✓ 确认」配对（最终导出）
const PALETTE = ['#e53935', '#1e88e5', '#43a047', '#fb8c00', '#8e24aa', '#00acc1',
                 '#3949ab', '#f4511e', '#6d4c41', '#00897b', '#c0ca33', '#d81b60'];

function imgTags(imgs) {
  const tags = [];
  for (let i = 0; i < MAX_IMGS; i++) {
    if (i < imgs.length) {
      tags.push(`<img src="file://${imgs[i]}">`);
    } else {
      tags.push('<img style="visibility:hidden">');
    }
  }
  return tags.join('');
}

function findPairIndex(name) {
  return pairs.findIndex(x => x.board === name || x.alight === name);
}

function isConfirmed(name) {
  return confirmedPairs.some(x => x.board === name || x.alight === name);
}

function pairColor(pi) {
  return PALETTE[pi % PALETTE.length];
}

function sortKey(name) {
  const pi = findPairIndex(name);
  return pi >= 0 ? pi : 1e9;  // 已配对按配对序号，未配对排最后
}

function sortedPersons(list) {
  return list.slice().sort((a, b) => sortKey(a.name) - sortKey(b.name));
}

function personCard(p, side) {
  if (isConfirmed(p.name)) return '';   // 已确认的卡片从界面隐藏
  const pi = findPairIndex(p.name);
  const isPaired = pi >= 0;
  const sel = (side === 'board' && selectedBoard === p.name) ? ' selected' : '';
  let extra = '';
  if (isPaired) {
    extra = ` style="border-color:${pairColor(pi)}; box-shadow:0 0 0 2px ${pairColor(pi)}"`;
  }
  const onclick = side === 'board' ? `clickBoard('${p.name}')` : `clickAlight('${p.name}')`;
  return `<div class="person${sel}"${extra} onclick="${onclick}" ondblclick="dblclickPerson('${p.name}')"><div class="name">${p.name}</div>${imgTags(p.imgs)}</div>`;
}

function render() {
  document.getElementById('board').innerHTML = sortedPersons(boardPersons).map(p => personCard(p, 'board')).join('');
  document.getElementById('alight').innerHTML = sortedPersons(alightPersons).map(p => personCard(p, 'alight')).join('');
  document.getElementById('pair-list').innerHTML = pairs.map((x, i) =>
    `<div class="row" style="border-left:4px solid ${pairColor(i)}; padding-left:6px">${i + 1}. ${x.board} ↔ ${x.alight}${(x.sim !== undefined && x.sim !== null) ? ` <span style="color:#888">sim=${x.sim}</span>` : ''} <button onclick="confirmPair(${i})" style="color:#fff;background:#43a047;border:none;border-radius:3px;padding:2px 8px">✓ 确认</button> <button onclick="removePair(${i})">撤销</button></div>`
  ).join('');
  document.getElementById('confirm-count').textContent = confirmedPairs.length;
  document.getElementById('pending-count').textContent = pairs.length;
}

function clickBoard(name) {
  selectedBoard = (selectedBoard === name) ? null : name;
  render();
}

function clickAlight(name) {
  if (selectedBoard) {
    pairs.push({ board: selectedBoard, alight: name });
    selectedBoard = null;
    render();
  }
}

function dblclickPerson(name) {
  const pi = findPairIndex(name);
  if (pi >= 0) {
    pairs.splice(pi, 1);
    render();
  }
}

function confirmPair(i) {
  confirmedPairs.push(pairs[i]);
  pairs.splice(i, 1);
  render();
}

function removePair(i) { pairs.splice(i, 1); render(); }

function clearPairs() {
  pairs = [];
  confirmedPairs = [];
  selectedBoard = null;
  render();
  document.getElementById('out').value = '';
}

function exportJson() {
  document.getElementById('out').value = JSON.stringify(confirmedPairs.map(x => ({ board: x.board, alight: x.alight })), null, 2);
}

render();
</script>
</body>
</html>
'''


def load_prefill(path, persons_by_name):
    """从预配对结果（manual_pairs.json 或 match_result.json）提取 board/alight 配对，
    只保留双方都在当前 person 集合、且方向正确（board=上车, alight=下车）的配对。"""
    if not path or not os.path.isfile(path):
        return []
    data = json.load(open(path, encoding='utf-8'))
    rows = []
    if isinstance(data, dict) and isinstance(data.get('matches'), list):
        # match_result.json：每项含 person_a/a_dir/person_b/b_dir/similarity
        for m in data['matches']:
            a, ad = m.get('person_a'), m.get('a_dir')
            b, bd = m.get('person_b'), m.get('b_dir')
            board = alight = None
            if ad == '上车' and bd == '下车':
                board, alight = a, b
            elif bd == '上车' and ad == '下车':
                board, alight = b, a
            if board and alight:
                rows.append({'board': board, 'alight': alight, 'sim': m.get('similarity')})
    elif isinstance(data, list):
        # manual_pairs.json：[{board, alight}]
        for x in data:
            if isinstance(x, dict) and x.get('board') and x.get('alight'):
                rows.append({'board': x['board'], 'alight': x['alight']})

    pairs = []
    for x in rows:
        b, a = x['board'], x['alight']
        pb, pa = persons_by_name.get(b), persons_by_name.get(a)
        if not (pb and pa):
            continue
        if pb['direction'] != '上车' or pa['direction'] != '下车':
            continue
        item = {'board': b, 'alight': a}
        if x.get('sim') is not None:
            item['sim'] = round(float(x['sim']), 4)
        pairs.append(item)
    return pairs


def main():
    ap = argparse.ArgumentParser(description='生成人工配对交互式 HTML 工具')
    ap.add_argument('--sessions', nargs='+', required=True, help='一个或多个 session 目录')
    ap.add_argument('--out', default='/media/tianheng/test/16.reId/CameraViewer_passenger/pairing_tool.html')
    ap.add_argument('--max-imgs', type=int, default=10, help='每个 person 展示的代表图数量（至少 10 张）')
    ap.add_argument('--prefill', default=None,
                    help='预配对结果 JSON（manual_pairs.json 或 match_result.json），预填供人工核对/修正')
    ap.add_argument('--exclude', default=None,
                    help='已确认配对 JSON（confirmed_pairs.json），排除其中涉及的 person，只保留待自由匹配的「非同一人」')
    args = ap.parse_args()

    persons = collect_persons(args.sessions, args.max_imgs)

    # 排除已确认配对涉及的 person（这些 person 已正确配对，不再参与自由匹配）
    if args.exclude and os.path.isfile(args.exclude):
        confirmed = json.load(open(args.exclude, encoding='utf-8'))
        excluded = set()
        for c in confirmed:
            excluded.add(c['person_a'])
            excluded.add(c['person_b'])
        before = len(persons)
        persons = [p for p in persons if p['name'] not in excluded]
        print(f'排除已确认配对涉及的 {len(excluded)} 个 person，剩余 {len(persons)} 个（原 {before}）')

    board = sum(1 for p in persons if p['direction'] == '上车')
    alight = sum(1 for p in persons if p['direction'] == '下车')
    other = len(persons) - board - alight
    print(f'待自由匹配 person：上车 {board}，下车 {alight}，其他 {other}')

    persons_by_name = {p['name']: p for p in persons}
    prefill = load_prefill(args.prefill, persons_by_name)
    if prefill:
        print(f'预填配对 {len(prefill)} 对（来自 {args.prefill}）')

    html = (HTML_HEAD.replace('__PERSONS__', json.dumps(persons, ensure_ascii=False))
                    .replace('__MAX_IMGS__', str(args.max_imgs))
                    .replace('__PREFILL__', json.dumps(prefill, ensure_ascii=False)) + HTML_TAIL)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'已生成配对工具: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

