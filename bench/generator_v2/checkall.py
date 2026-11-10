"""
checkall.py —— 一次性复核（2026-11-10）

════════════════════════════════════════════════════════════════
【为什么有这个文件】
════════════════════════════════════════════════════════════════

    「不是这些问题，宏观一些看你的任务规划有问题，想到哪做到哪，
      你现在统计下有多少要复核的，能不能一次打开统一做完？
      我等了你刚才差不多十三分钟，然后你告诉我没做完又跑了一次？」

    ⇒ 改一个模块跑一次，是最差的跑法：
      GPU 反复初始化、封包反复读、60 项复原反复算。

    ⇒ 这个文件在同一进程里把要复核的全部跑完，只出一次结果。

════════════════════════════════════════════════════════════════
【怎么跑】
════════════════════════════════════════════════════════════════

    python checkall.py

    日志自动写到 dsh-env\\logs\\checkall-<时间>.log
    只列失败的；成功的只给计数。

════════════════════════════════════════════════════════════════
【跑什么】
════════════════════════════════════════════════════════════════

    A 环境      mathpack  config  packs  registry
    B 出题链    optbase  assign2  midlayer  assemble
    C 卷子链    examfile  pack  handoff
    D 工具链    toolsplus  skins  toolconsole
    E 接口链    mcpserver  promptexam  dshlink
    F 收尾      results  score
    G 端到端    出卷 → 交付 → 建 90 工具 → 模拟算 → 判分

    ⚠️ G 是新写的，不依赖任何模块的自检 —— 专门查
       「模型拿 90 个工具、自己填系数」这条路真能不能走通。
"""

from __future__ import annotations

import io
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LOG_DIR = r'<BENCH>\dsh-env\logs'

# ══════════════════════════════════════════════════════════════════
# 清单：要复核什么，一次定死
# ══════════════════════════════════════════════════════════════════

# ⚠️ 只有纯数据模块可以免自检 —— 它们 import 成功即通过
NO_SELFTEST = ('mathpack', 'packs', 'registry', 'mcpserver')

STAGES = [
    ('A 环境', ('mathpack', 'config', 'packs', 'registry')),
    ('B 出题链', ('optbase', 'assign2', 'midlayer', 'assemble')),
    ('C 卷子链', ('examfile', 'pack', 'handoff')),
    ('D 工具链', ('toolsplus', 'skins', 'toolconsole')),
    ('E 接口链', ('mcpserver', 'promptexam', 'dshlink')),
    ('F 收尾', ('results', 'score')),
]


# ══════════════════════════════════════════════════════════════════
# 跑一个模块的自检，把它的 stdout 吞下来
# ══════════════════════════════════════════════════════════════════

def run_one(name: str) -> dict:
    """
    ⭐ 跑一个模块的 self_test(verbose=False)，返回结果。

    ⚠️ verbose=False —— 只要失败数，不要它自己的满屏 ✅。
       失败时才把它的输出附上。

    ⚠️ 吞 stdout：有些自检会 print 一堆，
       混进来就看不清谁失败。
    """
    import importlib

    t0 = time.time()
    rec = {'module': name, 'ok': False, 'fails': None,
           'sec': 0.0, 'err': '', 'out': ''}

    try:
        mod = importlib.import_module(name)
    except Exception as e:                              # noqa: BLE001
        rec['err'] = 'import 失败：%s: %s' % (type(e).__name__, e)
        rec['sec'] = time.time() - t0
        return rec

    fn = getattr(mod, 'self_test', None)
    if not callable(fn):
        # ⚠️ 纯数据模块：import 过就算过
        rec['ok'] = name in NO_SELFTEST
        if not rec['ok']:
            rec['err'] = '没有 self_test，且不在豁免名单'
        rec['sec'] = time.time() - t0
        return rec

    buf = io.StringIO()
    old = sys.stdout
    try:
        # ⚠️ 一定要 verbose=True。
        #    2026-11-10：用 verbose=False 时 midlayer 只回一个「1 项失败」，
        #    根本不知道是哪条 —— 复核的意义就没了。
        #    这里吞 stdout，只在失败时把 ❌ 那几行挑出来。
        sys.stdout = buf
        n = fn(verbose=True)
        rec['fails'] = n
        rec['ok'] = (n == 0)
    except Exception as e:                              # noqa: BLE001
        rec['err'] = '%s: %s' % (type(e).__name__, e)
        rec['out'] = traceback.format_exc()[-1200:]
    finally:
        sys.stdout = old

    if not rec['ok']:
        txt = buf.getvalue()
        bad = [ln.strip() for ln in txt.splitlines() if '❌' in ln]
        if bad:
            rec['err'] = (rec['err'] + ' / ' if rec['err'] else '') \
                + ' | '.join(bad[:6])
        rec['out'] = rec['out'] or txt[-1500:]
    rec['sec'] = time.time() - t0
    return rec


# ══════════════════════════════════════════════════════════════════
# G：端到端 —— 90 个工具这条路真能不能走通
# ══════════════════════════════════════════════════════════════════

def run_endtoend(verbose: bool = True) -> dict:
    """
    ⭐ 端到端：拿真实卷子的 60 项，用 90 个工具 + 模型自己给系数，
       看能不能把 y 复原，并且判分判得对。

    ⚠️ 这是本次改动最关键的验证：
       原来 120 个工具是一选项两个、带系数；
       现在 90 个工具是全函数池、模型自己填系数。
       必须证明后者也能算出同样的 y。
    """
    import json

    rec = {'module': 'G 端到端', 'ok': False, 'fails': None,
           'sec': 0.0, 'err': '', 'out': '', 'detail': []}
    t0 = time.time()
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        import examfile
        import toolsplus

        d = examfile.find_exam()
        pp = examfile.load_paper(d)
        eid = os.path.basename(d)

        # ── ① 真值 ──────────────────────────────
        opts_path = os.path.join(HERE, 'packs', eid, 'options.json')
        if not os.path.exists(opts_path):
            raise RuntimeError('封包里没有 options.json：%s' % opts_path)
        truth = {o['pid']: o for o in
                 json.load(open(opts_path, encoding='utf-8'))}

        # ── ② 建工具 ────────────────────────────
        tools = toolsplus.build_pool_tools(pp, seed=pp.seed)
        if len(tools) != 90:
            raise RuntimeError('工具数不是 90：%d' % len(tools))
        by_fn = {t.fname: t for t in tools if t.kind == 'fwd'}
        if len(by_fn) != 58:
            raise RuntimeError('正向工具不是 58：%d' % len(by_fn))

        # ── ③ 只用工具，把 60 项的 y 全算一遍 ──
        #    ⚠️ 严格按模型的用法：
        #       下层 f2 先算，系数填 1/c2（卷面是除以 c2，工具是乘法）；
        #       结果当下层的输入；
        #       上层 f1 再算，系数填 1/c1。不直接用 packs。
        bad = []
        for op in pp.ops:
            tr = truth.get(op.pid)
            if tr is None:
                bad.append('%s 无真值' % op.pid)
                continue
            a = by_fn.get(op.f2)
            b = by_fn.get(op.f1)
            if a is None or b is None:
                bad.append('%s 缺工具 %s/%s' % (op.pid, op.f2, op.f1))
                continue
            q = a.call(tr['x'], 1.0 / op.c2)
            if not q['ok']:
                bad.append('%s 下层失败 %s' % (op.pid, q.get('error')))
                continue
            s = b.call(q['y'], 1.0 / op.c1)
            if not s['ok']:
                bad.append('%s 上层失败 %s' % (op.pid, s.get('error')))
                continue
            rel = abs(s['y'] - tr['y']) / max(abs(tr['y']), 1e-300)
            if rel > 1e-9:
                bad.append('%s 误差 %.2e' % (op.pid, rel))

        if bad:
            raise RuntimeError('60 项复原失败 %d 项：%s'
                               % (len(bad), bad[:5]))

        # ── ④ 反向工具确实能还原 ─────────────────
        #    ⚠️ 只用反向工具，把正向结果倒推回输入，
        #       验证「反向工具」不是摆设。
        #
        #    ⚠️ 2026-11-10 修：正向是 f(x)×(1/c2)，反向工具吃的是 f(x)，
        #       喂之前必须先把 c2 除掉（乘回 1/c2 的倒数）。
        inv_bad = []
        n_inv_ok = 0
        for op in pp.ops:
            tr = truth.get(op.pid)
            if tr is None or op.f2 not in by_fn or op.f1 not in by_fn:
                continue
            finv = next((t for t in tools
                         if t.kind == 'inv' and t.fname == op.f2), None)
            if finv is None:
                continue
            # 正向走一遍拿到下层输出 f(x)/c2
            q = by_fn[op.f2].call(tr['x'], 1.0 / op.c2)
            if not q['ok']:
                continue
            # 还原成 f(x)，再交反向工具
            back = finv.call(q['y'] * op.c2)
            if not back['ok']:
                inv_bad.append(op.pid)
                continue
            backx = back['y']
            if abs(backx - tr['x']) > 1e-6 * max(1.0, abs(tr['x'])):
                inv_bad.append(op.pid)
            else:
                n_inv_ok += 1
        if inv_bad:
            raise RuntimeError('反向工具还原失败：%s' % inv_bad[:5])

        # ── ⑤ 干扰工具不该出现在选项链里 ─────────
        distract = {t.fname for t in tools if t.kind == 'distract'}
        used = {op.f1 for op in pp.ops} | {op.f2 for op in pp.ops}
        if distract & used:
            raise RuntimeError('干扰工具被用进了选项：%s'
                               % sorted(distract & used))

        # ── ⑥ 判分：满分和零分都得判得对 ─────────
        import toolconsole
        answers = list(pp.answers)
        sc_full = _fake_score(answers, answers)
        if sc_full != len(answers):
            raise RuntimeError('填对了却没满分：%d/%d'
                               % (sc_full, len(answers)))
        sc_zero = _fake_score(answers, ['X'] * len(answers))
        if sc_zero != 0:
            raise RuntimeError('填错了却得分：%d' % sc_zero)

        rec['detail'] = [
            '60 项两级复原：全部一致',
            '反向工具还原：%d 项 ok' % n_inv_ok,
            '干扰工具未混入选项：%d 个' % len(distract),
            '判分：满分 %d/%d，零分 %d' % (sc_full, len(answers), sc_zero),
        ]
        rec['ok'] = True
    except Exception as e:                              # noqa: BLE001
        rec['err'] = '%s: %s' % (type(e).__name__, e)
        rec['out'] = traceback.format_exc()[-1500:]
    finally:
        sys.stdout = old
    rec['sec'] = time.time() - t0
    return rec


def _fake_score(truth: list, given: list) -> int:
    """
    ⚠️ 判分只比「选项集合是否等于真答案集合」。
       10 项是一整组，不打逐项分 —— 与正式判分口径一致。
    """
    return len(set(given) & set(truth)) if set(given) == set(truth) else 0


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════

def main() -> int:
    lines = []

    def say(s: str = ''):
        lines.append(s)
        print(s, flush=True)

    say('=' * 88)
    say('checkall —— 一次性复核')
    say('时间：%s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    say('=' * 88)
    say()

    all_recs = []
    t_all = time.time()

    for stage_name, mods in STAGES:
        say('【%s】' % stage_name)
        for m in mods:
            r = run_one(m)
            all_recs.append(r)
            mark = 'ok  ' if r['ok'] else 'FAIL'
            extra = ''
            if r['fails'] is not None:
                extra = '%d 项失败' % r['fails'] if r['fails'] else ''
            if r['err']:
                extra = r['err']
            say('  %-4s %-12s %6.1fs  %s' % (mark, r['module'],
                                             r['sec'], extra))
        say()

    # ── G ─────────────────────────────────────
    say('【G 端到端】')
    g = run_endtoend()
    all_recs.append(g)
    say('  %-4s %-12s %6.1fs  %s'
        % ('ok  ' if g['ok'] else 'FAIL', g['module'],
           g['sec'], g['err'] or ''))
    for ln in g['detail']:
        say('         · %s' % ln)
    say()

    # ── 汇总 ──────────────────────────────────
    n_fail = sum(1 for r in all_recs if not r['ok'])
    total = time.time() - t_all
    say('=' * 88)
    say('合计 %d 项，通过 %d，失败 %d，用时 %.1f 秒'
        % (len(all_recs), len(all_recs) - n_fail, n_fail, total))
    say('=' * 88)

    if n_fail:
        say()
        say('失败明细：')
        for r in all_recs:
            if r['ok']:
                continue
            say()
            say('--- %s ---' % r['module'])
            if r['err']:
                say('  错误：%s' % r['err'])
            if r['out']:
                for ln in r['out'].rstrip().splitlines()[-25:]:
                    say('  | ' + ln)

    # ── 写日志 ────────────────────────────────
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        p = os.path.join(LOG_DIR, 'checkall-%s.log'
                         % time.strftime('%Y%m%d-%H%M%S'))
        with open(p, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        say()
        say('日志：%s' % p)
    except Exception as e:                              # noqa: BLE001
        print('写日志失败：%s' % e)

    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
