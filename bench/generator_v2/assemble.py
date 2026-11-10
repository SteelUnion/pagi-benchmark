"""
assemble —— 组装考卷：60 项生成 → 打乱 → 出题面

════════════════════════════════════════════════════════════════
【这一步做什么】（弗兰克 2026-11-09）
════════════════════════════════════════════════════════════════

    「123 可能要一起做」（60 项出满、打乱、组装）

    ① 生成 60 项（10 真 + 50 干扰），每项带自己的函数和系数
    ② 打乱（每组 <= 3 个真值，与原序相同 <= 30%）
    ③ 出题面（题干两行 + 60 个选项）

    ⚠️ 封包（画图）不在这一步，留到下一步。

════════════════════════════════════════════════════════════════
【题面长什么样】
════════════════════════════════════════════════════════════════

    条件
        A000 = 5.53056800524
        Z    = 35939655.1925

        求：A000 × [?] × [?] × … × [?] = Z
            （10 个空，各填一个选项的代码）

    选项（60 个）
        O001
          <词> = 25.5781382677
          Y-O001 = (exp((( <词> )^2) ÷ 100)) ÷ 100
        ...

    ⚠️ 题面上不出现：
        · y 的值（模型要求）
        · 函数编号（内部用）

════════════════════════════════════════════════════════════════
【验检什么】
════════════════════════════════════════════════════════════════

    ① 60 项都构造成功
    ② 前 10 项在数据里是真值，打乱后散开
    ③ 120 个词互不重复
    ④ 每个选项的式子能算出 y（独立验算）
    ⑤ 10 个真值的编码连乘 = Z
    ⑥ 每组真值 <= 3
"""

from __future__ import annotations

import math
import sys

import torch

sys.path.insert(0, r'<BENCH>\generator_v2')

import assign2
import config
import figs
import logutil
import midlayer
import optbase
import packs
import shuffle1
from mathpack import DT, DEV

__all__ = ['Paper', 'assemble', 'render_paper', 'self_test']


# ══════════════════════════════════════════════════════════════════
# 一、一道考卷
# ══════════════════════════════════════════════════════════════════

class Paper:
    """
    ⭐ 一道完整的考卷。

    ⚠️ order 是新的展示顺序：第 i 个位置放原来的第 order[i] 项。
    """

    __slots__ = ('a000', 'z', 'ops', 'mos', 'order', 'true_pos',
                 'group_true', 'sim_orig', 'n_try', 'answers', 'seed',
                 'figs', 'fig_dir', 'levels', 'exam_id', 'tool_level')

    def __init__(self):
        self.a000 = float('nan')
        self.z = float('nan')
        self.ops = []
        self.mos = []
        self.order = []
        self.true_pos = []
        self.group_true = []
        self.sim_orig = 1.0
        self.n_try = 0
        self.answers = []           # 打乱后，真值所在的选项 pid
        self.seed = 0
        self.figs = []              # 每项一张图（None = 该档不出图）
        self.fig_dir = ''
        self.levels = {}            # 生成本题用的难度档位
        self.exam_id = ''           # 问卷编号（封包时定，DSH 侧据此建目录）
        self.tool_level = 'basic'   # 工具档位（basic / advanced）

    # ── 判分 ──────────────────────────────────────
    def check(self, pids: list[str], tol: float = 1e-9) -> dict:
        """
        ⭐ 判分：把模型交的 10 个 pid 对应的 y 连乘，看等不等于 Z。
        """
        ymap = {op.pid: op.y for op in self.ops}
        if any(p not in ymap for p in pids):
            return {'ok': False, 'why': '有无效编码',
                    'product': float('nan'), 'err': float('inf')}
        prod = self.a000
        for p in pids:
            prod *= ymap[p]
        err = abs(prod - self.z) / abs(self.z)
        return {'ok': err < tol and sorted(pids) == sorted(self.answers),
                'product': prod, 'z': self.z, 'err': err,
                'set_ok': sorted(pids) == sorted(self.answers)}


# ══════════════════════════════════════════════════════════════════
# 二、组装
# ══════════════════════════════════════════════════════════════════

def assemble(seed: int = 20261109, lg=None,
             fig_dir: str = '', make_figs: bool = True) -> Paper:
    """
    ⭐ 一路做到底：造值 → 造选项 → 换词 → 打乱。

    参数 lg：logutil.Log 实例（可省）
    """
    pp = Paper()
    pp.seed = seed

    # ── ① 造值（GPU）──────────────────────────────
    batch = assign2.build_batch(seed=seed)
    pp.a000 = batch['y0']
    pp.z = batch['y11']
    if lg:
        lg.head(seed=seed, y0=pp.a000, y11=pp.z,
                ys_true=batch['ys_true'].tolist(),
                n_true=batch['n_true'], n_fake=batch['n_fake'])

    # ── ② 造选项（每项独立随机函数）─────────────────
    ops = optbase.build_from_batch(batch, seed=seed)
    pp.ops = ops

    # ⚠️⚠️ 2026-11-09 修（弗兰克）：
    #   「答案编号不展示，我们答题器上面写的是选项或者英文，1-60，
    #     你怎么搞的，你出题的时候怎么处处透题啊。」
    #
    #   ⇒ 原 Bug：pid = opt_pid(idx)，真值永远是 idx 0..9，
    #     所以 答案永远是 O001~O010。
    #     答案文件在沙箱里时模型读到它；
    #     即使读不到，猜「前 10 个」也 100% 中。
    #
    #   ⇒ 修：把 pid 的分配打乱。
    #     真值散落在 60 个编码里的任意 10 个，位置不可预测。
    #
    #   ⚠️ 编号形式保留 O001 风格（与图名 fig_O001 对应），
    #      但不含任何顺序信息 —— 它是乱序分配的。
    import random as _rnd
    _perm = list(range(len(ops)))
    _rnd.Random(seed * 104729 + 7).shuffle(_perm)
    for k, op in enumerate(ops):
        op.pid = optbase.opt_pid(_perm[k])

    # ── ③ 换词 ────────────────────────────────────
    mos = midlayer.build_midlayer(ops, seed=seed)
    pp.mos = mos

    # ── ④ 打乱 ────────────────────────────────────
    sh = shuffle1.shuffle_options(seed=seed)
    pp.order = sh['order']
    pp.true_pos = sh['true_pos']
    pp.group_true = sh['group_true']
    pp.sim_orig = sh['sim_orig']
    pp.n_try = sh['n_try']

    # 打乱后，答案 = 真值所在位置上的选项 pid
    #
    # ⚠️⚠️ 2026-11-09 修：
    #   order[p] = 「新位置 p 上放的是原来第几项」
    #   true_pos = 「真值在新顺序里的哪些位置」
    #   ⇒ 答案 = ops[order[p]] for p in true_pos
    #   （原来写成 ops[p]，把「位置编号」当成了「原始下标」，错了）
    pp.answers = [ops[sh['order'][p]].pid for p in sh['true_pos']]

    # ── ⑤ 出图（按难度档位）──────────────────────
    #
    # ⚠️ 弗兰克：「题目就是保留 x 的变量名，但不输出 x 的值，
    #    x 应该有 1-60 编号，图片根据编号决定。」
    pp.levels = dict(config.DIFFICULTY)
    pp.tool_level = config.difficulty('tool_level')
    if make_figs:
        import os as _os
        if not fig_dir:
            fig_dir = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)), 'figs',
                'paper_%d' % seed)
        pp.fig_dir = fig_dir
        # ⚠️ 2026-11-10：按展示顺序给图命名（第 k 行 → 00k.png）。
        #    弗兰克：「我们打乱了正确选项的顺序，图片的顺序也要跟着动。」
        pp.figs = figs.draw_all(ops, mos, fig_dir, seed=seed,
                                order=pp.order)
    else:
        pp.figs = [None] * len(ops)

    # 日志
    if lg:
        for i, op in enumerate(ops):
            rec = {'pid': op.pid, 'ok': op.ok, 'why': op.why,
                   'is_true': i < batch['n_true']}
            rec.update(op.trace)
            lg.rec(**rec)
        lg.summary(n_ops=len(ops), n_ok=sum(1 for o in ops if o.ok),
                   group_true=pp.group_true, sim_orig=pp.sim_orig,
                   n_try=pp.n_try)
    return pp


# ══════════════════════════════════════════════════════════════════
# 三、出题面
# ══════════════════════════════════════════════════════════════════

def render_paper(pp: Paper, show_answer: bool = False,
                 n_show: int = 60) -> str:
    """
    ⭐ 把考卷渲染成文字。

    ⚠️ 弗兰克 2026-11-09：
    「题目就是保留 x 的变量名，但不输出 x 的值，x 应该有 1-60 编号，
      图片根据编号决定，然后题面函数有函数自身和中英文描述两版。」

    ⇒ 每个选项给出：
         · 选项代码（O001，也是图的编号）
         · x 的变量名（词）
         · x 的图（按代码找）
         · y 的编码 + 表达式（含函数名与参数）
    ⚠️ 不出现 x 的数值（图片档）
    """
    L = []
    img_lv = pp.levels.get('image', 'advanced')

    L.append('=' * 72)
    L.append('条件')
    L.append('=' * 72)
    L.append('  A000 = %.12g' % pp.a000)
    L.append('  Z    = %.12g' % pp.z)
    L.append('')
    L.append('  求：A000 × [ ? ] × [ ? ] × … × [ ? ] = Z')
    L.append('      （%d 个空，各填一个选项代码）' % len(pp.answers))
    L.append('')

    if img_lv != 'none':
        # ⚠️⚠️ 2026-11-10 弗兰克：
        #    「图片地址提示改文绝对地址，同时在提示词注明
        #      沙箱根目录绝对地址。」
        #
        #    实测：写相对路径 figs/001.png，模型还是试 001.png、
        #    060.png、T-001.png、090.png —— 浪费十几次调用。
        #    ⇒ 直接给绝对路径，零转换。
        #    ⚠️ 卷面里记的是相对路径占位，handoff 落盘时换成绝对。
        L.append('  说明：每个选项的 x 值在对应的图里。')
        L.append('        「图」那一行给的就是该图的完整路径，直接用它读图。')
        if img_lv == 'advanced':
            L.append('        图上是科学计数法：正负号与 10 的指数已给出，')
            L.append('        尾数逐位画成图案，每排图案个数 = 该位的数字。')
        else:
            L.append('        图上直接写着数值。')
        L.append('')

    L.append('=' * 72)
    L.append('选项（共 %d 个）' % len(pp.ops))
    L.append('=' * 72)

    for k in range(min(n_show, len(pp.order))):
        i = pp.order[k]
        op = pp.ops[i]
        mo = pp.mos[i]
        L.append('  %s' % mo.pid)
        L.append('    x 的名称：%s' % mo.x_word)
        if img_lv == 'none':
            L.append('    %s = %.14g' % (mo.x_word, mo.x_value))
        else:
            # ⚠️⚠️ 2026-11-10 弗兰克：
            #    「图片地址提示改文绝对地址，同时在提示词注明
            #      沙箱根目录绝对地址。」
            #
            #    实测：写相对路径 figs/001.png，模型还是试 001.png、
            #    060.png、T-001.png、090.png —— 浪费十几次调用。
            #    ⇒ 直接给绝对路径，零转换。
            #
            #    ⚠️ 卷面里记的绝对路径要等交付之后才有效，
            #       所以这里先用相对路径占位，pack/handoff 落盘时换成绝对。
            L.append('    图：figs/%s' % figs.fig_name_by_index(k + 1))
        line = mo.text.split('\n')[-1].strip()
        L.append('    %s' % line)
        L.append('')

    if show_answer:
        L.append('=' * 72)
        L.append('答案（内部用，不进题面）')
        L.append('=' * 72)
        ymap = {op.pid: op.y for op in pp.ops}
        prod = pp.a000
        for p_ in sorted(pp.answers):
            L.append('  %-8s y = %.12g' % (p_, ymap[p_]))
            prod *= ymap[p_]
        L.append('  连乘 = %.12g   Z = %.12g   err = %.2e'
                 % (prod, pp.z, abs(prod - pp.z) / abs(pp.z)))
        L.append('  每组真值数：%s' % pp.group_true)
        L.append('  与原序相同：%.1f%%   重抽 %d 次'
                 % (100 * pp.sim_orig, pp.n_try))
        L.append('  难度档位：%s' % pp.levels)
    return '\n'.join(L)


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
        print('assemble 自检（60 项 → 打乱 → 题面）')
        print('=' * 96)
        print('-' * 96)

    with logutil.open_log('assemble') as lg:
        pp = assemble(seed=20261109, lg=lg)
        lgpath = lg.path

    # ① 60 项
    ck('60 个选项', len(pp.ops) == 60, '%d' % len(pp.ops))
    n_ok = sum(1 for o in pp.ops if o.ok)
    ck('60 项全部构造成功', n_ok == 60, '%d/60' % n_ok)

    # ② 在数据里前 10 是真值
    ck('A000 / Z 是有限数',
       math.isfinite(pp.a000) and math.isfinite(pp.z),
       'A000=%.6g Z=%.6g' % (pp.a000, pp.z))

    # ③ 120 个词不重复
    words = [mo.x_word for mo in pp.mos]
    ck('60 个 x 的词互不重复', len(set(words)) == 60)
    ck('词都是非空', all(words))

    # ④ 打乱约束
    ck('每组真值 <= 3', max(pp.group_true) <= 3,
       str(pp.group_true))
    ck('与原序相同 <= 30%', pp.sim_orig <= 0.3001,
       '%.1f%%' % (100 * pp.sim_orig))

    # ⑤ 答案是 10 个
    ck('答案 10 个', len(pp.answers) == 10)
    ck('答案不重复', len(set(pp.answers)) == 10)

    # ⑥ 连乘 = Z
    ymap = {op.pid: op.y for op in pp.ops}
    prod = pp.a000
    for p in pp.answers:
        prod *= ymap[p]
    err = abs(prod - pp.z) / abs(pp.z)
    ck('10 个答案连乘 = Z', err < 1e-9, 'err=%.2e' % err)

    # ⑦ 每个选项的式子能算出 y（独立验算）
    bad = []
    n_eval = 0
    for op in pp.ops:
        try:
            p1, p2 = packs.get(op.f1), packs.get(op.f2)
            a = float(p2._fwd(torch.tensor([op.x], device=DEV,
                                           dtype=DT))[0])
            v2 = a / op.c2
            yv = float(p1._fwd(torch.tensor([v2], device=DEV,
                                            dtype=DT))[0]) / op.c1
            n_eval += 1
            if abs(yv - op.y) / abs(op.y) > 1e-6:
                bad.append(op.pid)
        except Exception:                              # noqa: BLE001
            bad.append(op.pid)
    ck('每个选项的式子能算出 y', not bad, '验了 %d 项，错 %s'
       % (n_eval, bad[:3]))

    # ⑧ 判分器
    r = pp.check(pp.answers)
    ck('填对 → 判通过', r['ok'], 'err=%.2e' % r['err'])
    wrong = [op.pid for op in pp.ops if op.pid not in pp.answers][:10]
    r2 = pp.check(wrong)
    ck('填错 → 判不通过', not r2['ok'])

    # ⑨ 题面不泄题
    txt = render_paper(pp, show_answer=False)
    ck('题面里不出现「答案」字样', '答案' not in txt)
    leak = [op.pid for op in pp.ops if ('%.6g' % op.y) in txt]
    ck('题面里不出现任何 y 的值', not leak, str(leak[:3]))

    if verbose:
        print()
        print('  每组真值：', pp.group_true)
        print('  与原序相同：%.1f%%   重抽 %d 次'
              % (100 * pp.sim_orig, pp.n_try))
        print('  真值位置：', pp.true_pos)
        print('  日志：', lgpath)
        print()
        print('-' * 96)
        print(render_paper(pp, show_answer=True, n_show=3))
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
