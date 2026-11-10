"""
results.py —— 成绩记录器

════════════════════════════════════════════════════════════════
【做什么】（弗兰克 2026-11-09）
════════════════════════════════════════════════════════════════

    「我们是不是该搞个什么东西记录每次成绩，这次是不是已经被记录了？」

    ⇒ 每场考试结束，写一条到总表。
      之前只有过程日志（cheat-*.jsonl），没有「成绩单」。
      跑了几十场，一场都没横向对比过。

════════════════════════════════════════════════════════════════
【一条记录长什么样】
════════════════════════════════════════════════════════════════

    {
      "exam_id": "EXAM-20261127-...",
      "ts": "2026-11-09T23:57:00",
      "model": "deepseek-flash",
      "provider": "deepseek-official",
      "levels": {"image":"advanced","tool_level":"advanced"},
      "duration_s": 350,
      "n_call": 47, "n_image": 5, "n_nudge": 0, "n_out": 0,
      "cheat_rate": 0.0,
      "fabrication": {"n_claim":0,"n_fab":0,"rate":0},
      "verdict": "pass",          // pass / fail / unknown
      "answers": [...],           // 模型填的
      "correct": [...],           // 标准答案（记录表不进沙箱，无所谓）
      "hit": 8,                   // 交集几个
      "tokens": {"input":..,"output":..,"cache_read":..,"total":..},
      "notes": ["读了 exam.json"]
    }

════════════════════════════════════════════════════════════════
【存哪】
════════════════════════════════════════════════════════════════

    dsh-env\\_scores\\results.jsonl        一行一场，追加
    dsh-env\\_scores\\results.md           人看的汇总表

    ⚠️ 都不在沙箱里。
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, r'<BENCH>\generator_v2')

__all__ = ['SCORE_ROOT', 'record_one', 'read_all', 'render_table',
           'collect', 'extract', 'all_exam_ids', 'latest_exam_id',
           'write_md', 'self_test']

SCORE_ROOT = r'<BENCH>\dsh-env\_scores'
RESULTS_JSONL = os.path.join(SCORE_ROOT, 'results.jsonl')
RESULTS_MD = os.path.join(SCORE_ROOT, 'results.md')


# ══════════════════════════════════════════════════════════════════
# 一、从各处收集一场考试的事实
# ══════════════════════════════════════════════════════════════════

def collect(exam_id: str, run_log: str = '') -> dict:
    """
    ⭐ 把一场考试的事实拼起来。

    来源：
      · 门控汇总   dsh-env\\logs\\cheat\\cheat-<id>.summary.json
      · 门控逐条   ...jsonl（算填了哪些、交了几次）
      · 会话日志   dsh-env\\logs\\run-<id>.out.log（算时长与 token）
      · 密钥文件   _sealed\\<id>\\exam_secret.json（标准答案）
    """
    import examfile

    env = examfile.ENV_ROOT
    cheat_dir = os.path.join(env, 'logs', 'cheat')
    sum_p = os.path.join(cheat_dir, 'cheat-%s.summary.json' % exam_id)
    jsonl_p = os.path.join(cheat_dir, 'cheat-%s.jsonl' % exam_id)
    if not run_log:
        run_log = os.path.join(env, 'logs', 'run-%s.out.log' % exam_id)

    # ── 门控汇总 ──
    sm = {}
    if os.path.exists(sum_p):
        with open(sum_p, encoding='utf-8') as f:
            sm = json.load(f)

    # ── 逐条：填了哪些、交卷几次 ──
    filled, submits = [], 0
    if os.path.exists(jsonl_p):
        with open(jsonl_p, encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tool = str(r.get('tool', ''))
                verdict = str(r.get('verdict', ''))
                if tool.endswith('__fill') and verdict.startswith('in_scope'):
                    # args 是 JSON 字符串，取 pid
                    try:
                        a = json.loads(r.get('args') or '{}')
                        if a.get('pid'):
                            filled.append(a['pid'])
                    except Exception:                   # noqa: BLE001
                        pass
                if verdict == 'submit':
                    submits += 1

    # ── 会话日志：时长 + token + 模型说的话里的数 ──
    dur = 0.0
    tok = {'input': 0, 'output': 0, 'cache_read': 0, 'total': 0}
    if os.path.exists(run_log):
        t_first = t_last = None
        with open(run_log, encoding='utf-8', errors='replace') as f:
            for line in f:
                if not line.strip().startswith('{'):
                    continue
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = (j.get('params') or {}).get('event') or {}
                t = ev.get('time')
                if isinstance(t, (int, float)):
                    if t_first is None:
                        t_first = t
                    t_last = t
                u = (ev.get('data') or {}).get('usage')
                if u:
                    tok['input'] += u.get('inputTokens', 0)
                    tok['output'] += u.get('outputTokens', 0)
                    tok['cache_read'] += u.get('cacheReadTokens', 0)
                    tok['total'] += u.get('totalTokens', 0)
        if t_first and t_last:
            dur = round((t_last - t_first) / 1000.0, 1)

    # ── 标准答案 + 难度 ──
    correct, levels = [], {}
    try:
        sec = examfile.read_exam(os.path.join(env, 'exams', exam_id))
        correct = list(sec.get('answers', []))
        levels = dict(sec.get('levels', {}))
    except Exception:                                   # noqa: BLE001
        pass

    hit = len(set(filled) & set(correct)) if correct else 0

    return {
        'exam_id': exam_id,
        'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'provider': sm.get('provider'),
        'model': sm.get('model'),
        'levels': levels,
        'duration_s': dur,
        'n_call': sm.get('n_call', 0),
        'n_image': sm.get('distinct_images', 0),
        'n_nudge': sm.get('n_nudge', 0),
        'n_out': sm.get('n_out', 0),
        'n_submit': submits,
        'cheat_rate': sm.get('cheat_rate', 0),
        'fabrication': sm.get('fabrication') or {},
        'verdict': ((sm.get('grade') or {}).get('verdict')
                    if sm.get('grade') else None),
        'finished': sm.get('finished'),
        'filled': filled,
        'correct': correct,
        'hit': hit,
        'n_answer': len(correct),
        'tokens': tok,
    }


# ══════════════════════════════════════════════════════════════════
# 二、写一条
# ══════════════════════════════════════════════════════════════════

def record_one(rec: dict, force: bool = False) -> str:
    """
    ⭐ 追加一条到 results.jsonl。

    返回：
      'written'   写了
      'skipped'   这场已记过（force=False）
      'replaced'  已记过，被覆盖（force=True）
    """
    os.makedirs(SCORE_ROOT, exist_ok=True)
    eid = rec.get('exam_id')

    rows, found, replaced = [], False, False
    if os.path.exists(RESULTS_JSONL):
        with open(RESULTS_JSONL, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get('exam_id') == eid:
                    found = True
                    if not force:
                        rows.append(r)      # 保留旧的
                        continue
                    rows.append(rec)        # 换成新的
                    replaced = True
                else:
                    rows.append(r)

    if found and not force:
        return 'skipped'

    if not replaced:
        rows.append(rec)

    with open(RESULTS_JSONL, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return 'replaced' if replaced else 'written'


def read_all() -> list:
    """⭐ 读全部记录。"""
    if not os.path.exists(RESULTS_JSONL):
        return []
    out = []
    with open(RESULTS_JSONL, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


# ══════════════════════════════════════════════════════════════════
# 三、批量提取
# ══════════════════════════════════════════════════════════════════

def all_exam_ids() -> list:
    """⭐ 列出所有有日志的考试（按时间排序，最后一份最新）。"""
    import examfile
    env = examfile.ENV_ROOT
    ids = set()

    # 门控日志
    cheat = os.path.join(env, 'logs', 'cheat')
    if os.path.isdir(cheat):
        for fn in os.listdir(cheat):
            if fn.startswith('cheat-') and fn.endswith('.summary.json'):
                ids.add(fn[len('cheat-'):-len('.summary.json')])

    # 会话日志（可能没有门控汇总）
    logs = os.path.join(env, 'logs')
    if os.path.isdir(logs):
        for fn in os.listdir(logs):
            if fn.startswith('run-EXAM-') and fn.endswith('.out.log'):
                ids.add(fn[len('run-'):-len('.out.log')])

    # ⚠️ 排序走 examfile.exam_sort_key —— 按编号里的时间戳，
    #    不按字符串。否则种子大的旧卷会被当成「最新」。
    return sorted(ids, key=examfile.exam_sort_key)


def latest_exam_id() -> str:
    """⭐ 最新一场。"""
    ids = all_exam_ids()
    return ids[-1] if ids else ''


def extract(exam_id: str = '', force: bool = False,
            all_: bool = False) -> list:
    """
    ⭐ 提取成绩。

    参数
    ----
    exam_id : 指定哪一场。留空 = 取最新。
    force   : 已记过的重提（覆盖旧的）。
    all_    : True 时提取全部历史（未记过的才写）。

    返回 [(exam_id, 状态), ...]
    """
    if all_:
        targets = all_exam_ids()
    elif exam_id:
        targets = [exam_id]
    else:
        last = latest_exam_id()
        targets = [last] if last else []

    out = []
    for eid in targets:
        rec = collect(eid)
        # ⚠️ 一点事实都没有的，不记 —— 免得表里全是空行
        if not (rec['n_call'] or rec['tokens']['total'] or rec['filled']):
            out.append((eid, 'empty'))
            continue
        st = record_one(rec, force=force)
        out.append((eid, st))
    if out:
        write_md()
    return out


# ══════════════════════════════════════════════════════════════════
# 四、渲染汇总表
# ══════════════════════════════════════════════════════════════════

def render_table(recs: list = None) -> str:
    """⭐ 人看的汇总表。"""
    recs = recs if recs is not None else read_all()
    if not recs:
        return '（还没有记录）'

    L = ['| 卷子 | 模型 | 难度 | 时长 | 调用 | 图 | 越界 | 判分 | token |',
         '|---|---|---|---|---|---|---|---|---|']
    for r in recs:
        lv = r.get('levels') or {}
        short = str(r.get('exam_id', ''))[-14:]
        v = r.get('verdict') or '—'
        tok = (r.get('tokens') or {}).get('total', 0)
        L.append('| %s | %s | %s/%s | %ss | %d | %d | %d | %s | %s |' % (
            short,
            r.get('model') or '—',
            lv.get('image', '?'), lv.get('tool_level', '?'),
            r.get('duration_s', 0),
            r.get('n_call', 0),
            r.get('n_image', 0),
            r.get('n_out', 0),
            '✅' if v == 'pass' else ('❌' if v == 'fail' else '—'),
            '{:,}'.format(tok),
        ))
    return '\n'.join(L)


def write_md(recs: list = None) -> str:
    os.makedirs(SCORE_ROOT, exist_ok=True)
    txt = render_table(recs)
    with open(RESULTS_MD, 'w', encoding='utf-8') as f:
        f.write('# 考试成绩\n\n')
        f.write(txt + '\n')
    return RESULTS_MD


# ══════════════════════════════════════════════════════════════════
# 四、自检
# ══════════════════════════════════════════════════════════════════

def self_test(verbose: bool = True) -> int:
    fails = 0

    def ck(name, cond, extra: str = ''):
        nonlocal fails
        if verbose:
            print(('  ✅ ' if cond else '  ❌ ') + name
                  + (('   ' + extra) if extra else ''))
        if not cond:
            fails += 1

    if verbose:
        print('=' * 96)
        print('results 自检（成绩记录）')
        print('=' * 96)
        print('-' * 96)

    import tempfile
    global SCORE_ROOT, RESULTS_JSONL, RESULTS_MD
    tmp = tempfile.mkdtemp(prefix='scores_')
    SCORE_ROOT = tmp
    RESULTS_JSONL = os.path.join(tmp, 'results.jsonl')
    RESULTS_MD = os.path.join(tmp, 'results.md')

    rec = {
        'exam_id': 'EXAM-TEST-1', 'ts': '2026-11-09T00:00:00',
        'model': 'deepseek-flash', 'levels': {'image': 'advanced',
                                              'tool_level': 'advanced'},
        'duration_s': 350.0, 'n_call': 47, 'n_image': 5,
        'n_out': 0, 'cheat_rate': 0.0, 'verdict': 'pass',
        'tokens': {'total': 123456},
    }

    ck('写一条', record_one(rec) == 'written')
    ck('重复写被跳过', record_one(rec) == 'skipped')
    ck('读回 1 条', len(read_all()) == 1)

    rec2 = dict(rec, exam_id='EXAM-TEST-2', verdict='fail',
                n_out=3, cheat_rate=0.06, tokens={'total': 999})
    record_one(rec2)
    ck('两条', len(read_all()) == 2)

    tbl = render_table()
    ck('表格有表头', '| 卷子 |' in tbl)
    ck('表格两行数据', tbl.count('EXAM-TEST') == 2)
    ck('pass 显示 ✅', '✅' in tbl)
    ck('fail 显示 ❌', '❌' in tbl)

    md = write_md()
    ck('写 md', os.path.exists(md))

    if verbose:
        print()
        print(render_table())
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='考试成绩记录器')
    ap.add_argument('exam_id', nargs='?', default='',
                    help='指定卷子编号；留空取最新')
    ap.add_argument('--force', action='store_true', help='已记过的重提')
    ap.add_argument('--all', action='store_true', help='提取全部历史')
    ap.add_argument('--table', action='store_true', help='只打印汇总表')
    ap.add_argument('--selftest', action='store_true', help='跑自检')
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if self_test() == 0 else 1)

    if a.table:
        print(render_table())
    else:
        res = extract(a.exam_id, force=a.force, all_=a.all)
        for eid, st in res:
            print('%-32s %s' % (eid, st))
        print()
        print(render_table())
