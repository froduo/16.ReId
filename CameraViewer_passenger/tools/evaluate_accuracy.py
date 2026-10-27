#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReID 检测准确度评估脚本
========================
读取一个 session 目录（含 person_N 子目录）+ 独立比对结果 match_result.json，
统计方向分布、匹配情况、相似度分布，给出「当前准确度判断」与「下一步完善方案」。

用法：
  /home/tianheng/miniconda3/envs/yolotest/bin/python evaluate_accuracy.py \
      --session <session目录> --match <match_result.json> --out <报告.txt>
"""
import argparse
import json
import os
import re
import statistics

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


def read_match_obj(person_dir):
    mt = os.path.join(person_dir, 'match.txt')
    obj, sim = None, None
    if os.path.isfile(mt):
        try:
            with open(mt, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('匹配对象'):
                        obj = line.split(':', 1)[1].strip()
                    elif line.startswith('相似度'):
                        try:
                            sim = float(line.split(':', 1)[1].strip())
                        except Exception:
                            pass
        except Exception:
            pass
    return obj, sim


def collect_persons(session):
    persons = []
    for name in sorted(os.listdir(session)):
        pdir = os.path.join(session, name)
        if not os.path.isdir(pdir) or not PERSON_RE.match(name):
            continue
        crop_dir = os.path.join(pdir, '乘客图片')
        n_img = 0
        if os.path.isdir(crop_dir):
            n_img = len([f for f in os.listdir(crop_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        persons.append({
            'name': name,
            'direction': read_direction(pdir),
            'match_obj': read_match_obj(pdir)[0],
            'match_sim': read_match_obj(pdir)[1],
            'n_img': n_img,
        })
    return persons


def percentile(vals, p):
    if not vals:
        return float('nan')
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser(description='ReID 检测准确度评估')
    ap.add_argument('--session', required=True, help='session 目录（含 person_N）')
    ap.add_argument('--match', required=True, help='match_person_folders.py 输出的 JSON')
    ap.add_argument('--out', required=True, help='评估报告输出路径（.txt）')
    args = ap.parse_args()

    persons = collect_persons(args.session)
    if not persons:
        print('错误：未找到 person_N 目录')
        return 1

    match = {}
    if os.path.isfile(args.match):
        with open(args.match, encoding='utf-8') as f:
            match = json.load(f)

    dirs = {}
    for p in persons:
        dirs[p['direction']] = dirs.get(p['direction'], 0) + 1

    matched_persons = [p for p in persons if p['match_sim'] is not None]
    sims = [p['match_sim'] for p in matched_persons]

    indep_sims = [m['similarity'] for m in match.get('matches', [])]
    indep_matches = match.get('matches', [])
    indep_unmatched = match.get('unmatched', [])
    total_persons = match.get('total_persons', len(persons))

    import datetime
    lines = []
    A = lines.append
    A('=' * 72)
    A('ReID 检测准确度评估报告')
    A('=' * 72)
    A('生成时间: %s' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    A('session 目录: %s' % args.session)
    A('独立比对结果: %s' % args.match)
    A('')
    A('【1. 乘客统计】')
    A('  总 person 数: %d' % len(persons))
    for d in sorted(dirs.keys()):
        A('    %s: %d' % (d, dirs[d]))
    A('')
    A('【2. 方向合理性】')
    board = dirs.get('上车', 0)
    alight = dirs.get('下车', 0)
    invalid = dirs.get('上车折返', 0) + dirs.get('下车折返', 0)
    A('  上车 %d 人 / 下车 %d 人 / 折返 %d 人 / 未判定 %d 人'
      % (board, alight, invalid, dirs.get('未判定', 0)))
    if board > 0 and alight > 0:
        ratio = abs(board - alight) / max(board, alight)
        A('  上下车数量差异比例: %.1f%% %s'
          % (ratio * 100, '(合理)' if ratio < 0.2 else '(偏大，可能存在漏检/误判)'))
    A('')
    A('【3. 程序自身匹配相似度分布】（person 目录 match.txt）')
    if sims:
        A('  已匹配 %d 人，相似度: 均值=%.3f 中位数=%.3f min=%.3f max=%.3f'
          % (len(sims), statistics.mean(sims), statistics.median(sims), min(sims), max(sims)))
        hi = sum(1 for s in sims if s >= 0.85)
        mid = sum(1 for s in sims if 0.70 <= s < 0.85)
        lo = sum(1 for s in sims if 0.60 <= s < 0.70)
        A('  高置信(≥0.85): %d 人 | 中置信(0.70~0.85): %d 人 | 低置信(0.60~0.70): %d 人' % (hi, mid, lo))
    else:
        A('  （无匹配记录）')
    A('')
    A('【4. 独立比对结果】（match_person_folders.py，osnet_best.pth）')
    A('  参与比对 %d 人，匹配 %d 对，未匹配 %d 人' % (total_persons, len(indep_matches), len(indep_unmatched)))
    if indep_sims:
        A('  匹配相似度: 均值=%.3f 中位数=%.3f P25=%.3f P75=%.3f'
          % (statistics.mean(indep_sims), statistics.median(indep_sims),
             percentile(indep_sims, 0.25), percentile(indep_sims, 0.75)))
        hi = sum(1 for s in indep_sims if s >= 0.85)
        mid = sum(1 for s in indep_sims if 0.70 <= s < 0.85)
        lo = sum(1 for s in indep_sims if 0.60 <= s < 0.70)
        A('  高置信(≥0.85): %d 对 | 中置信(0.70~0.85): %d 对 | 低置信(0.60~0.70): %d 对' % (hi, mid, lo))
    A('')
    A('【5. 低置信匹配明细】（需人工复核）')
    low_matches = [m for m in indep_matches if m['similarity'] < 0.70]
    if low_matches:
        for m in low_matches:
            A('  %s(%s) <-> %s(%s) sim=%.4f'
              % (m['person_a'], m['a_dir'], m['person_b'], m['b_dir'], m['similarity']))
    else:
        A('  （无低置信匹配）')
    A('')
    A('【6. 准确度判断】')
    if indep_sims and len(indep_sims) >= 5:
        med = statistics.median(indep_sims)
        p25 = percentile(indep_sims, 0.25)
        if med >= 0.80 and p25 >= 0.70:
            verdict = '良好：匹配相似度整体较高，模型区分度充足'
        elif med >= 0.70:
            verdict = '一般：匹配相似度中等，建议继续积累数据微调'
        else:
            verdict = '偏弱：匹配相似度偏低，模型区分度不足，建议优先补数据/调阈值'
        A('  独立比对匹配相似度中位数 %.3f → %s' % (med, verdict))
    if board > 0 and alight > 0:
        r = abs(board - alight) / max(board, alight)
        if r > 0.25:
            A('  注意：上下车数量差异偏大，提示存在多检/漏检/折返误判，需排查跟踪与门线判定')
        else:
            A('  上下车数量基本平衡，检测计数基本可靠')
    A('')
    A('【7. 下一步完善方案建议】')
    A('  1) 低置信匹配(<0.70)建议人工在「匹配细节」界面复核，标记正确/错误后重新导出训练数据继续微调；')
    A('  2) 若上下车数量差异大：检查 person_* 目录中「未判定/折返」样本，优化门线折返判定与 tracker 身份分离；')
    A('  3) 若匹配相似度整体偏低：增加训练身份数（当前 173 身份），或调低 DeepMatchThreshold 观察召回；')
    A('  4) 建议把「人工确认正确」的匹配对用程序内「导出 OSNet 训练数据」按钮导出，追加训练后重跑本流程对比 gap。')
    A('=' * 72)

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

